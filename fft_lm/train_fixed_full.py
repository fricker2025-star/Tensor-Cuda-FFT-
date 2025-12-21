"""fft_lm.train_fixed_full

Fixed spectral mixer training on TinyStories with non-greedy sampling.

Key fix (already validated): MIX IN FREQUENCY DOMAIN.

This script:
  - trains on a large corpus via random window sampling (no 1000-seq overfit)
  - uses a JPEG/progressive frequency cutoff schedule
  - generates with temperature + top-k + repetition penalty

Run:
  python train_fixed_full.py

Notes:
  - Uses byte-level modeling (vocab=256).
  - Default seq_len=1024 (changeable).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
import argparse
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TrainConfig:
    # data
    data_path: str = "tinystories_train.txt"
    # model
    vocab_size: int = 256
    d_model: int = 512
    n_layers: int = 6
    seq_len: int = 1024  # context length
    kernel_len: int = 128  # causal conv kernel length (per block)
    ffn_mult: int = 2  # feedforward expansion factor (improves capacity)
    # FREQUENCY-NATIVE MODE (experimental)
    frequency_native: bool = False  # Use phase activations instead of time-domain FFN
    use_fp32: bool = False  # Force FP32 for frequency operations (better for complex arithmetic)
    # BICAMERAL MODE (two-hemisphere architecture)
    bicameral: bool = False  # Use dual-path: frequency (global) + time (local)
    # training
    batch_size: int = 8
    accum_steps: int = 1  # gradient accumulation micro-steps per optimizer step
    epochs: int = 200
    steps_per_epoch: int = 250  # random-window batches per epoch
    # LR tuned for stability on full corpus (user-requested)
    lr: float = 2e-4
    weight_decay: float = 5e-4
    grad_clip: float = 1.0
    # progressive frequency schedule
    jpeg_low: int = 128
    jpeg_mid: int = 512
    jpeg_high: int = 1024
    jpeg_transition: int = 32  # soft roll-off bins to reduce Gibbs ringing
    # generation
    temperature: float = 0.8
    # Prefer nucleus sampling (top-p) over top-k for byte-level stability.
    top_p: float = 0.9
    top_k: int = 0  # optional backstop; 0 disables
    repetition_penalty: float = 1.25
    repetition_window: int = 256
    max_run_length: int = 6  # hard anti-stutter: disallow > N identical bytes
    # Presence/frequency penalties are OFF by default for byte-level models;
    # they can push probability mass into rare bytes (digits/punct attractors).
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    ban_cr: bool = True  # ban '\r'
    ascii_only: bool = True  # allow \n and printable ASCII only
    max_new: int = 400
    # misc
    seed: int = 1337
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = True
    # checkpointing
    ckpt_path: str = "fixed_spectral_ckpt.pt"
    save_every_epochs: int = 5
    # evaluation / anti-parroting
    val_windows: int = 2048
    val_batches: int = 20
    parroting_snip_len: int = 64
    parroting_stride: int = 16
    parroting_snips: int = 64  # number of snippets to test
    log_every_steps: int = 50  # per-epoch step progress (prevents "stuck" feeling)

    # Sawtooth LR schedule (cosine annealing with stage-aligned restarts)
    # IMPORTANT: Stage durations must align with expected cutoff raises!
    # Cutoff progression: 128 (epoch 0) → 512 (epoch 1+) - SIMPLE 2-STAGE
    stage1_epochs: int = 1  # Epoch 0: cutoff=128, learn basic character frequencies
    stage2_epochs: int = 3  # Epochs 1-3: cutoff=512, full resolution learning
    stage1_lr_mult: float = 1.0
    stage1_min_mult: float = 0.1
    stage2_lr_mult: float = 1.0  # RESTART to full LR when cutoff raises at epoch 1
    stage2_min_mult: float = 0.1
    # Stage 3: epoch 4+, continue at cutoff=512
    stage3_lr_mult: float = 1.0
    stage3_min_mult: float = 0.05


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_corpus_as_u8(path: str, *, sanitize_ascii: bool) -> torch.Tensor:
    # Read bytes (utf-8 file) and keep raw bytes; this is intentional.
    # We model bytes, so we just take file bytes modulo 256.
    with open(path, "rb") as f:
        data = f.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    if sanitize_ascii:
        # Keep \n and printable ASCII. Everything else -> space.
        keep = (arr == 10) | ((arr >= 32) & (arr <= 126))
        arr = np.where(keep, arr, 32).astype(np.uint8)
    # Keep on CPU; we sample windows and then move to GPU.
    return torch.from_numpy(arr.copy())  # make contiguous, owns memory


def conv_freq_bins(seq_len: int, kernel_len: int) -> int:
    """rFFT bin count used by the causal FFT-conv.

    We compute linear convolution via zero-padding to n_fft = next_pow2(seq_len + kernel_len - 1).
    The rFFT has n_fft//2 + 1 bins.
    """
    n_fft = 1
    need = int(seq_len + kernel_len - 1)
    while n_fft < need:
        n_fft *= 2
    return int(n_fft // 2 + 1)


def make_val_starts(n_bytes: int, seq_len: int, count: int, seed: int) -> torch.Tensor:
    """Deterministic validation window start indices."""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    hi = max(1, n_bytes - (seq_len + 1) - 1)
    return torch.randint(0, hi, (count,), generator=g)


@torch.no_grad()
def eval_loss(
    model: nn.Module,
    corpus_u8: torch.Tensor,
    starts: torch.Tensor,
    cfg: TrainConfig,
    cutoff: int | None,
) -> float:
    """Compute an approximate validation loss on fixed windows."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    n = int(corpus_u8.numel())
    # sample a subset of starts each call
    idx = torch.randperm(starts.numel())[: cfg.val_batches * cfg.batch_size]
    sel = starts[idx]
    losses = []
    # Precompute arange on CPU once
    ar = torch.arange(cfg.seq_len + 1, dtype=torch.long)
    for i in range(0, sel.numel(), cfg.batch_size):
        s = sel[i : i + cfg.batch_size]
        if s.numel() < cfg.batch_size:
            break
        # Vectorized gather (much faster than Python list/stack)
        idx2 = s[:, None].to(torch.long) + ar[None, :]
        batch = corpus_u8[idx2]
        bx = batch[:, :-1].to(torch.long)
        by = batch[:, 1:].to(torch.long)
        bx = bx.to(cfg.device, non_blocking=True)
        by = by.to(cfg.device, non_blocking=True)
        logits = model(bx, cutoff=cutoff)
        loss = loss_fn(logits.reshape(-1, cfg.vocab_size), by.reshape(-1))
        losses.append(float(loss.item()))
    return float(sum(losses) / max(1, len(losses)))


def parroting_score(corpus_bytes: bytes, gen_bytes: bytes, cfg: TrainConfig) -> float:
    """Heuristic: fraction of random fixed-length snippets from the generation that occur verbatim in corpus.

    High score means the model is likely copying/memorizing; low score suggests novelty.
    """
    if len(gen_bytes) < cfg.parroting_snip_len + 1:
        return 0.0
    # ignore the prompt prefix by skipping the first 32 bytes
    start0 = min(32, len(gen_bytes) - cfg.parroting_snip_len)
    candidates = list(range(start0, len(gen_bytes) - cfg.parroting_snip_len, cfg.parroting_stride))
    if not candidates:
        return 0.0
    # deterministic sampling
    rng = np.random.default_rng(123)
    picks = rng.choice(candidates, size=min(cfg.parroting_snips, len(candidates)), replace=False)
    hits = 0
    for p in picks:
        snip = gen_bytes[p : p + cfg.parroting_snip_len]
        if corpus_bytes.find(snip) != -1:
            hits += 1
    return hits / float(len(picks))


def jpeg_cutoff(epoch: int, cfg: TrainConfig, freq_bins: int) -> int:
    # Expand horizon: low -> mid -> high -> full.
    # Values are in "frequency bins" units. For rfft(seq_len), bins = seq_len//2+1.
    if epoch < 20:
        target = cfg.jpeg_low
    elif epoch < 50:
        target = cfg.jpeg_mid
    elif epoch < 100:
        target = cfg.jpeg_high
    else:
        target = freq_bins
    return int(min(target, freq_bins))


def sawtooth_lr(global_step: int, epoch: int, cfg: TrainConfig, *, cutoff_raised: bool = False) -> float:
    """Cosine annealing with restarts aligned to the curriculum stages.

    base LR = cfg.lr. Within each stage, LR decays from (base*stage_lr_mult)
    down to (base*stage_min_mult).
    
    Args:
        global_step: Global optimizer step number
        epoch: Current epoch
        cfg: Training config
        cutoff_raised: If True, FORCE LR restart to maximum (Shock & Awe protocol)
    """
    s_per = int(cfg.steps_per_epoch)
    e1 = int(cfg.stage1_epochs)
    e2 = int(cfg.stage1_epochs + cfg.stage2_epochs)

    if epoch < e1:
        stage_start = 0
        stage_epochs = max(1, e1)
        lr_mult = cfg.stage1_lr_mult
        min_mult = cfg.stage1_min_mult
    elif epoch < e2:
        stage_start = e1 * s_per
        stage_epochs = max(1, int(cfg.stage2_epochs))
        lr_mult = cfg.stage2_lr_mult
        min_mult = cfg.stage2_min_mult
    else:
        stage_start = e2 * s_per
        stage_epochs = max(1, int(cfg.epochs) - e2)
        lr_mult = cfg.stage3_lr_mult
        min_mult = cfg.stage3_min_mult

    # SHOCK & AWE: If cutoff was just raised, restart LR to peak
    if cutoff_raised:
        return float(cfg.lr * lr_mult)

    stage_total_steps = max(1, stage_epochs * s_per)
    local_step = max(0, int(global_step) - int(stage_start))
    progress = min(1.0, local_step / float(stage_total_steps))

    # cosine from 1 -> 0
    cos01 = 0.5 * (1.0 + math.cos(math.pi * progress))
    mult = float(min_mult + (lr_mult - min_mult) * cos01)
    return float(cfg.lr * mult)


def lr_stage_params(epoch: int, cfg: TrainConfig) -> tuple[str, float, float]:
    """Return (stage_name, lr_mult, min_mult) for logging."""
    e1 = int(cfg.stage1_epochs)
    e2 = int(cfg.stage1_epochs + cfg.stage2_epochs)
    if epoch < e1:
        return ("stage1", float(cfg.stage1_lr_mult), float(cfg.stage1_min_mult))
    if epoch < e2:
        return ("stage2", float(cfg.stage2_lr_mult), float(cfg.stage2_min_mult))
    return ("stage3", float(cfg.stage3_lr_mult), float(cfg.stage3_min_mult))


def curriculum_cutoff(epoch: int, cfg: TrainConfig, freq_bins: int) -> int:
    """Spectral Curriculum Learning (simple 2-stage).

    Stage 1 (epoch 0-4): 128 bins  (basic character frequencies)
    Stage 2 (epoch 5+):  512 bins  (full resolution - Nyquist limit)

    Skip 256 - unnecessary middle step. Jump straight to full resolution.
    Values are capped to available freq_bins.
    """
    if epoch < 5:
        target = 128
    else:
        target = 512  # Jump straight to full resolution
    return int(min(target, freq_bins))


def adaptive_cutoff(
    epoch: int,
    current_cutoff: int,
    loss_history: list[float],
    freq_bins: int,
    *,
    min_epoch_before_raise: int = 1,
    plateau_window: int = 50,
    plateau_threshold: float = 0.005,
) -> tuple[int, bool]:
    """Dynamic cutoff based on loss plateau detection (The Plateau Rule).
    
    Args:
        epoch: Current epoch number
        current_cutoff: Current frequency cutoff
        loss_history: Recent loss values (last N optimizer steps)
        freq_bins: Maximum available frequency bins
        min_epoch_before_raise: Minimum epoch before first cutoff raise (default 1)
        plateau_window: Number of recent losses to check for plateau
        plateau_threshold: Max relative change to consider a plateau
        
    Returns:
        (new_cutoff, cutoff_raised): New cutoff value and whether it was raised
    """
    # Never raise before minimum epoch
    if epoch < min_epoch_before_raise:
        return current_cutoff, False
    
    # Already at maximum
    if current_cutoff >= freq_bins:
        return current_cutoff, False
    
    # Need enough history to detect plateau
    if len(loss_history) < plateau_window:
        return current_cutoff, False
    
    # Check for plateau: compare recent average vs older average
    recent = loss_history[-plateau_window:]
    if len(recent) < 2:
        return current_cutoff, False
    
    # Calculate trend: if loss is not dropping significantly, it's a plateau
    first_half = recent[: plateau_window // 2]
    second_half = recent[plateau_window // 2 :]
    
    avg_first = sum(first_half) / len(first_half)
    avg_second = sum(second_half) / len(second_half)
    
    # Relative improvement
    if avg_first > 0:
        rel_improvement = (avg_first - avg_second) / avg_first
    else:
        rel_improvement = 0.0
    
    # Plateau detected: loss not improving enough
    if rel_improvement < plateau_threshold:
        # Simple curriculum: 128 -> 512 (skip 256)
        # Never go below 128 (64 is too blurry, no point)
        if current_cutoff < 512:
            new_cutoff = 512  # Jump straight to full resolution
        else:
            new_cutoff = freq_bins  # Already at Nyquist
        
        new_cutoff = min(new_cutoff, freq_bins)
        return new_cutoff, new_cutoff > current_cutoff
    
    return current_cutoff, False


def plateau_cutoff(
    current_cutoff: int,
    recent_loss: float,
    freq_bins: int,
    best_loss_at_cutoff: float,
    steps_without_improvement: int,
    *,
    patience: int = 50,
    improvement_threshold: float = 0.01,
) -> tuple[int, bool, float, int]:
    """PLATEAU-BASED cutoff: Unlock when STUCK, not when winning!
    
    CORRECT PHILOSOPHY:
    - Let the model MASTER the current frequency band
    - Only unlock when it PLATEAUS (can't improve anymore)
    - This ensures solid foundations before adding complexity
    
    Like weight training: Don't add weight until you can't do more reps!
    
    Args:
        current_cutoff: Current frequency cutoff
        recent_loss: Recent average loss (last 10 steps)
        freq_bins: Maximum available frequency bins (Nyquist limit: seq_len//2 + 1)
        best_loss_at_cutoff: Best loss achieved at current cutoff
        steps_without_improvement: Counter for plateau detection
        patience: How many steps without improvement before unlock (default 50)
        improvement_threshold: Minimum improvement to reset counter (default 0.01)
        
    Returns:
        (new_cutoff, cutoff_raised, new_best_loss, new_counter)
    """
    # Already at maximum (Nyquist limit)
    if current_cutoff >= freq_bins:
        return current_cutoff, False, best_loss_at_cutoff, steps_without_improvement
    
    # Check if we improved
    if recent_loss < best_loss_at_cutoff - improvement_threshold:
        # NEW PERSONAL BEST! Keep training at this level
        return current_cutoff, False, recent_loss, 0  # Reset counter
    else:
        # No improvement, increment stall counter
        new_counter = steps_without_improvement + 1
        
        # Check if we've plateaued (stuck for too long)
        if new_counter >= patience:
            # PLATEAU DETECTED! Unlock to help model improve further
            # Simple curriculum: 128 -> 512 (Nyquist limit, full resolution)
            # Skip 256 - it's an unnecessary middle step
            if current_cutoff < 512:
                new_cutoff = 512  # Jump straight to full resolution
            else:
                new_cutoff = freq_bins  # Already at Nyquist
            
            # Cap at Nyquist limit
            new_cutoff = min(new_cutoff, freq_bins)
            
            if new_cutoff > current_cutoff:
                # Reset best loss for new cutoff (expect spike then improvement)
                return new_cutoff, True, float('inf'), 0
        
        return current_cutoff, False, best_loss_at_cutoff, new_counter


class FixedSpectralBlock(nn.Module):
    """A single *causal* spectral mixing block.

    IMPORTANT: The earlier non-causal frequency filter leaks FUTURE tokens during training
    (because FFT mixes the whole window). That gives artificially low train loss and
    degenerates at generation time.

    Fix: implement a causal linear convolution via FFT (zero-padded), using a one-sided
    time-domain kernel.
    """

    def __init__(self, d_model: int, seq_len: int, kernel_len: int, transition_bins: int, dropout: float = 0.1):
        super().__init__()
        # Pre-norm tends to stabilize spectral ops.
        self.ln = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

        self.seq_len = seq_len
        self.kernel_len = kernel_len
        self.transition_bins = int(max(1, transition_bins))
        # Learnable *causal* kernel in time domain: k[0..K-1]
        self.kernel = nn.Parameter(torch.zeros(kernel_len))
        nn.init.normal_(self.kernel, mean=0.0, std=0.001)  # identity-ish

        # Per-channel gain so channels aren't forced to share the exact same filter
        self.gain = nn.Parameter(torch.ones(d_model))

        # ------------------------------------------------------------
        # GATING ("Valve")
        # ------------------------------------------------------------
        # We gate the frequency-domain signal to prevent resonant attractors
        # (e.g., "888888" loops) from dominating.
        #
        # Two gates:
        #  1) Per-frequency gate: learns which spectral bands to suppress globally.
        #  2) Context gate: learns per-channel gating conditioned on current hidden state.
        #
        # Gate values are real in [0,1] and multiply complex spectra (scales real+imag).

        # Max rFFT bins given the largest FFT length this block will use.
        # n_fft is next power-of-2 >= (T + K - 1). For maximum T=seq_len.
        max_lin_conv = int(seq_len + kernel_len - 1)
        max_n_fft = 1
        while max_n_fft < max_lin_conv:
            max_n_fft *= 2
        self.max_freq_bins = max_n_fft // 2 + 1

        # Per-frequency gate logits, initialized "mostly open".
        self.gate_freq_logits = nn.Parameter(torch.ones(self.max_freq_bins) * 2.0)  # sigmoid ~0.88

        # Context-conditioned gate (per channel), initialized to "mostly open".
        self.gate_ctx = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.gate_ctx.weight)
        nn.init.constant_(self.gate_ctx.bias, 2.0)  # sigmoid ~0.88

        # Pointwise feedforward (adds capacity for spelling/vocab)
        hidden = d_model * 2
        self.ffn_ln = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )
        # init small so residual path dominates early
        for m in self.ffn:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, cutoff: int | None = None) -> torch.Tensor:
        """x: [B, T, C] real.

        cutoff: number of rfft bins to keep (progressive training). If provided,
                bins >= cutoff are zeroed.
        """
        residual = x
        x = self.ln(x)

        B, T, C = x.shape
        # Use linear convolution via zero-padding to avoid circular wrap (future leakage).
        n_fft = 1
        while n_fft < (T + self.kernel_len - 1):
            n_fft *= 2

        # Build padded causal kernel (real)
        k = torch.zeros(n_fft, device=x.device, dtype=x.dtype)
        k[: self.kernel_len] = self.kernel
        k_freq = torch.fft.rfft(k)  # [F]

        # FFT input along time
        x_pad = F.pad(x, (0, 0, 0, n_fft - T))  # [B, n_fft, C]
        x_freq = torch.fft.rfft(x_pad, dim=1)  # [B, F, C]

        # Apply frequency response (broadcast), plus per-channel gain
        y_freq = x_freq * k_freq.unsqueeze(0).unsqueeze(-1) * self.gain.unsqueeze(0).unsqueeze(0)

        # ----------------
        # Gating (Valve)
        # ----------------
        Fbins = y_freq.size(1)

        # Per-frequency gate
        g_freq = torch.sigmoid(self.gate_freq_logits[:Fbins]).to(dtype=y_freq.real.dtype)  # [F]

        # Context gate (per channel): use pooled hidden state
        pooled = x.mean(dim=1)  # [B, C]
        g_ctx = torch.sigmoid(self.gate_ctx(pooled)).to(dtype=y_freq.real.dtype)  # [B, C]

        # Apply both
        y_freq = y_freq * g_freq.unsqueeze(0).unsqueeze(-1) * g_ctx.unsqueeze(1)

        # Progressive frequency horizon (JPEG schedule): soft roll-off to reduce Gibbs ringing
        if cutoff is not None:
            cutoff_idx = min(int(cutoff), Fbins)
            if cutoff_idx < Fbins:
                trans = min(self.transition_bins, cutoff_idx)  # can't exceed cutoff
                # 1 up to (cutoff_idx-trans), cosine down to 0 at cutoff_idx, 0 beyond.
                mask = torch.ones(Fbins, device=y_freq.device, dtype=y_freq.real.dtype)
                start = cutoff_idx - trans
                if trans > 0:
                    t = torch.linspace(0, 1, steps=trans, device=y_freq.device, dtype=mask.dtype)
                    mask[start:cutoff_idx] = 0.5 * (1.0 + torch.cos(torch.pi * t))
                mask[cutoff_idx:] = 0.0
                y_freq = y_freq * mask.unsqueeze(0).unsqueeze(-1)

        y_pad = torch.fft.irfft(y_freq, n=n_fft, dim=1)  # [B, n_fft, C]
        # Linear conv output length: T + K - 1; we take first T (causal)
        y = y_pad[:, :T, :]

        y = self.drop(y)
        x = residual + y

        # FFN residual
        ff_in = self.ffn_ln(x)
        x = x + self.drop(self.ffn(ff_in))
        return x


class FixedSpectralLM(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        
        # Choose architecture based on flags
        if cfg.bicameral:
            from fft_lm.bicameral import BicameralBlock
            block_class = BicameralBlock
            print("[BICAMERAL] Dual-hemisphere: Frequency (global) + Time (local)")
        elif cfg.frequency_native:
            from fft_lm.frequency_native import FrequencyNativeBlock
            block_class = FrequencyNativeBlock
            print("[FREQUENCY-NATIVE] Phase activations, no time-domain roundtrips")
        else:
            block_class = FixedSpectralBlock
            print("[STANDARD] Time-domain FFN with GELU")
        
        self.blocks = nn.ModuleList(
            [
                block_class(
                    cfg.d_model,
                    seq_len=cfg.seq_len,
                    kernel_len=cfg.kernel_len,
                    transition_bins=cfg.jpeg_transition,
                    dropout=0.1,
                )
                for _ in range(cfg.n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)
        # Weight-tying via matmul with embedding weight

    def forward(self, x: torch.Tensor, cutoff: int | None = None) -> torch.Tensor:
        """x: [B, T] long -> logits [B, T, V]."""
        h = self.forward_hidden(x, cutoff=cutoff)
        logits = torch.matmul(h, self.embed.weight.t())
        return logits

    def forward_hidden(self, x: torch.Tensor, cutoff: int | None = None) -> torch.Tensor:
        """Return final hidden states (useful for chunk heads).

        Args:
            x: [B, T] long
        Returns:
            h: [B, T, C] float
        """
        h = self.embed(x)
        for blk in self.blocks:
            h = blk(h, cutoff=cutoff)
        h = self.ln_f(h)
        return h


@torch.no_grad()
def generate(
    model: FixedSpectralLM,
    prompt: str,
    cfg: TrainConfig,
    device: str,
    *,
    cutoff: int | None = None,
) -> str:
    model.eval()

    # bytes for prompt
    ctx = [b for b in prompt.encode("utf-8", errors="ignore")]
    if len(ctx) == 0:
        ctx = [32]

    def apply_top_p(logits_1d: torch.Tensor, p: float) -> torch.Tensor:
        # returns masked logits
        sorted_logits, sorted_idx = torch.sort(logits_1d, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cdf = torch.cumsum(probs, dim=-1)
        # keep at least 1 token
        keep = cdf <= p
        keep[0] = True
        cutoff_pos = keep.sum().item()
        masked = logits_1d.clone()
        masked[:] = -float("inf")
        masked[sorted_idx[:cutoff_pos]] = logits_1d[sorted_idx[:cutoff_pos]]
        return masked

    for _ in range(cfg.max_new):
        x = torch.tensor([ctx[-cfg.seq_len :]], dtype=torch.long, device=device)
        logits = model(x, cutoff=cutoff)
        next_logits = logits[0, -1].float()

        # repetition penalty (stronger, longer window)
        recent = ctx[-cfg.repetition_window :]
        for tok in set(recent):
            next_logits[tok] = next_logits[tok] / cfg.repetition_penalty

        # presence/frequency penalties (OpenAI-style) to break attractors
        if cfg.presence_penalty or cfg.frequency_penalty:
            # count occurrences in recent window
            counts = {}
            for t in recent:
                counts[t] = counts.get(t, 0) + 1
            for tok, c in counts.items():
                next_logits[tok] = next_logits[tok] - cfg.presence_penalty - cfg.frequency_penalty * float(c)

        # optionally ban control chars (esp. '\r')
        if cfg.ascii_only:
            # allow newline and printable ASCII; ban everything else
            banned = torch.ones_like(next_logits, dtype=torch.bool)
            banned[10] = False
            banned[32:127] = False
            next_logits[banned] = -float("inf")
        if cfg.ban_cr:
            next_logits[13] = -float("inf")

        # hard anti-stutter: if last N bytes are identical, ban that byte for 1 step
        if len(ctx) >= cfg.max_run_length:
            run_byte = ctx[-1]
            if all(b == run_byte for b in ctx[-cfg.max_run_length :]):
                next_logits[run_byte] = -float("inf")

        # temperature
        next_logits = next_logits / cfg.temperature

        # nucleus (top-p) sampling
        if cfg.top_p is not None and cfg.top_p < 1.0:
            next_logits = apply_top_p(next_logits, cfg.top_p)

        # optional top-k backstop
        if cfg.top_k and cfg.top_k > 0:
            k = min(cfg.top_k, next_logits.numel())
            v, _ = torch.topk(next_logits, k)
            next_logits[next_logits < v[-1]] = -float("inf")

        probs = F.softmax(next_logits, dim=-1)
        nxt = torch.multinomial(probs, 1).item()
        ctx.append(int(nxt))

    # decode bytes (may include non-ascii). Return unicode with replacement.
    return bytes(ctx).decode("utf-8", errors="replace")


def safe_console(s: str) -> str:
    """Make a string safe to print on Windows cp1252 consoles."""
    # Convert unprintable/unencodable chars into backslash escapes.
    return s.encode("unicode_escape", errors="backslashreplace").decode("ascii", errors="ignore")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--accum-steps", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--kernel-len", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--presence-penalty", type=float, default=None)
    parser.add_argument("--frequency-penalty", type=float, default=None)
    parser.add_argument("--log-every-steps", type=int, default=None)
    parser.add_argument("--no-sawtooth", action="store_true")
    parser.add_argument("--stage3-lr-mult", type=float, default=None)
    parser.add_argument("--stage3-min-mult", type=float, default=None)
    parser.add_argument("--ckpt-path", type=str, default=None)
    parser.add_argument("--val-batches", type=int, default=None)
    parser.add_argument("--eval-every-epochs", type=int, default=None)
    parser.add_argument("--no-val", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = TrainConfig()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.steps_per_epoch is not None:
        cfg.steps_per_epoch = args.steps_per_epoch
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.accum_steps is not None:
        cfg.accum_steps = max(1, int(args.accum_steps))
    if args.seq_len is not None:
        cfg.seq_len = args.seq_len
    if args.kernel_len is not None:
        cfg.kernel_len = args.kernel_len
    if args.lr is not None:
        cfg.lr = args.lr
    if args.top_p is not None:
        cfg.top_p = args.top_p
    if args.top_k is not None:
        cfg.top_k = args.top_k
    if args.temperature is not None:
        cfg.temperature = args.temperature
    if args.repetition_penalty is not None:
        cfg.repetition_penalty = args.repetition_penalty
    if args.presence_penalty is not None:
        cfg.presence_penalty = args.presence_penalty
    if args.frequency_penalty is not None:
        cfg.frequency_penalty = args.frequency_penalty
    if args.log_every_steps is not None:
        cfg.log_every_steps = args.log_every_steps
    if args.ckpt_path is not None:
        cfg.ckpt_path = args.ckpt_path
    if args.stage3_lr_mult is not None:
        cfg.stage3_lr_mult = float(args.stage3_lr_mult)
    if args.stage3_min_mult is not None:
        cfg.stage3_min_mult = float(args.stage3_min_mult)
    if args.val_batches is not None:
        cfg.val_batches = max(1, int(args.val_batches))
    if args.eval_every_epochs is not None:
        cfg.eval_every_epochs = max(1, int(args.eval_every_epochs))
    set_seed(cfg.seed)

    if cfg.device == "cpu":
        raise SystemExit("CUDA required for this run")

    if not os.path.exists(cfg.data_path):
        raise SystemExit(f"Missing dataset file: {cfg.data_path}")

    print("=" * 70)
    print("TRAIN FIXED SPECTRAL MIXER (FULL DATA, NON-GREEDY SAMPLING)")
    print("=" * 70)
    print(f"Device: {cfg.device}")
    print(f"Data:   {cfg.data_path}")
    print(f"SeqLen: {cfg.seq_len}")
    eff_batch = cfg.batch_size * cfg.accum_steps
    print(f"Batch:  {cfg.batch_size} (micro)  x accum {cfg.accum_steps}  => effective {eff_batch}")
    print(f"Epochs: {cfg.epochs} (optimizer steps/epoch={cfg.steps_per_epoch})")
    print(f"LR:     {cfg.lr} (wd={cfg.weight_decay})")
    print(f"CKPT:   {cfg.ckpt_path}")
    if not args.no_sawtooth:
        print(
            f"LR sched: sawtooth cosine restarts (stage-aligned)"
            f"  s1(e0-{cfg.stage1_epochs-1}) mult {cfg.stage1_lr_mult}->{cfg.stage1_min_mult}"
            f"  s2(e{cfg.stage1_epochs}-{cfg.stage1_epochs+cfg.stage2_epochs-1}) mult {cfg.stage2_lr_mult}->{cfg.stage2_min_mult}"
            f"  s3(e{cfg.stage1_epochs+cfg.stage2_epochs}+) mult {cfg.stage3_lr_mult}->{cfg.stage3_min_mult}"
        )
    print(f"Gen:    temp={cfg.temperature} top_p={cfg.top_p} top_k={cfg.top_k} rep={cfg.repetition_penalty} win={cfg.repetition_window} maxrun={cfg.max_run_length}")
    print(f"        presence={cfg.presence_penalty} freq={cfg.frequency_penalty} ascii_only={cfg.ascii_only} ban_cr={cfg.ban_cr}")

    corpus_u8 = load_corpus_as_u8(cfg.data_path, sanitize_ascii=cfg.ascii_only)
    n = int(corpus_u8.numel())
    print(f"Corpus bytes: {n:,}")

    # precompute fixed validation windows + corpus blob for parroting test
    val_starts = make_val_starts(n, cfg.seq_len, cfg.val_windows, cfg.seed + 1)
    corpus_blob = bytes(corpus_u8.numpy().tobytes())

    model = FixedSpectralLM(cfg).to(cfg.device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Params: {params:,} (~{params/1e6:.2f}M)")
    print("=" * 70)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)

    def load_state_flexible(m: nn.Module, state: dict) -> tuple[list, list]:
        """Load checkpoint even if a few tensors changed shape (e.g., seq_len change).

        We *only* auto-resize known safe tensors. Everything else with a mismatch is skipped.
        """
        cur = m.state_dict()
        to_load = {}
        resized = []
        skipped = []
        for k, v in state.items():
            if k not in cur:
                continue
            if cur[k].shape == v.shape:
                to_load[k] = v
                continue
            # Safe resize: per-frequency gate logits
            if k.endswith("gate_freq_logits") and v.ndim == 1 and cur[k].ndim == 1:
                tgt = cur[k].clone()
                n = min(tgt.numel(), v.numel())
                tgt[:n] = v[:n]
                to_load[k] = tgt
                resized.append((k, tuple(v.shape), tuple(tgt.shape)))
            else:
                skipped.append((k, tuple(v.shape), tuple(cur[k].shape)))

        m.load_state_dict(to_load, strict=False)
        return resized, skipped

    start_epoch = 0
    if args.resume and os.path.exists(cfg.ckpt_path):
        ckpt = torch.load(cfg.ckpt_path, map_location="cpu")
        try:
            model.load_state_dict(ckpt["model"], strict=True)
        except RuntimeError as e:
            print("[warn] strict checkpoint load failed (likely seq_len/kernel_len changed).")
            print("[warn] falling back to flexible load; mismatched tensors will be resized/skipped.")
            resized, skipped = load_state_flexible(model, ckpt["model"])
            if resized:
                print(f"[warn] resized {len(resized)} tensors (example: {resized[0][0]} {resized[0][1]}-> {resized[0][2]})")
            if skipped:
                print(f"[warn] skipped {len(skipped)} tensors due to shape mismatch")
            # NOTE: if too many tensors are skipped, it's better to start fresh.

        # Optimizer/scaler states are generally NOT compatible across shape changes.
        # Only load them if shapes likely match; otherwise start fresh.
        try:
            opt.load_state_dict(ckpt["opt"])
            if "scaler" in ckpt and cfg.amp and ckpt.get("scaler") is not None:
                scaler.load_state_dict(ckpt["scaler"])
        except Exception:
            print("[warn] optimizer/scaler state not loaded (shape changed). Starting optimizer fresh.")
        start_epoch = int(ckpt.get("epoch", 0))
        print(f"Resumed from {cfg.ckpt_path} at epoch {start_epoch}")

    # IMPORTANT: cutoff schedule must match the FFT size used by causal FFT-conv.
    # For seq_len=512,kernel_len=128 => n_fft=1024 => 513 bins.
    freq_bins = conv_freq_bins(cfg.seq_len, cfg.kernel_len)
    t0 = time.time()

    def save_ckpt(epoch_idx: int) -> None:
        torch.save(
            {
                "epoch": epoch_idx,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "scaler": scaler.state_dict() if cfg.amp else None,
                "cfg": cfg.__dict__,
            },
            cfg.ckpt_path,
        )

    last_epoch = start_epoch
    try:
        for epoch in range(start_epoch, cfg.epochs):
            last_epoch = epoch + 1
            model.train()
            # Spectral Curriculum Learning: cutoff is locked per stage.
            cutoff = curriculum_cutoff(epoch, cfg, freq_bins)

            # Gradient accumulation
            opt.zero_grad(set_to_none=True)
            losses = []  # per optimizer step
            running = 0.0
            running_lr = 0.0

            micro_steps_total = cfg.steps_per_epoch * cfg.accum_steps
            for micro in range(micro_steps_total):
                opt_step = micro // cfg.accum_steps  # 0..steps_per_epoch-1

                # LR schedule is defined on OPTIMIZER steps, not micro-steps
                global_opt_step = epoch * cfg.steps_per_epoch + opt_step
                if not args.no_sawtooth:
                    lr_now = sawtooth_lr(global_opt_step, epoch, cfg)
                    for pg in opt.param_groups:
                        pg["lr"] = lr_now
                else:
                    lr_now = opt.param_groups[0]["lr"]

                # random windows
                starts = torch.randint(0, n - (cfg.seq_len + 1) - 1, (cfg.batch_size,))
                batch_x = torch.stack([corpus_u8[s : s + cfg.seq_len] for s in starts]).to(torch.long)
                batch_y = torch.stack([corpus_u8[s + 1 : s + cfg.seq_len + 1] for s in starts]).to(torch.long)
                batch_x = batch_x.to(cfg.device, non_blocking=True)
                batch_y = batch_y.to(cfg.device, non_blocking=True)

                with torch.autocast("cuda", enabled=cfg.amp):
                    logits = model(batch_x, cutoff=cutoff)
                    loss = loss_fn(logits.reshape(-1, cfg.vocab_size), batch_y.reshape(-1))
                    loss = loss / float(cfg.accum_steps)

                scaler.scale(loss).backward()

                # Step optimizer at accumulation boundary
                if (micro + 1) % cfg.accum_steps == 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)

                    # track *unscaled* loss for logging (multiply back)
                    li = float(loss.item()) * float(cfg.accum_steps)
                    losses.append(li)
                    running += li
                    running_lr += float(lr_now)

                    if cfg.log_every_steps and (opt_step + 1) % cfg.log_every_steps == 0:
                        avg_step = running / cfg.log_every_steps
                        avg_lr = running_lr / cfg.log_every_steps
                        running = 0.0
                        running_lr = 0.0
                        print(
                            f"  step {opt_step+1:5d}/{cfg.steps_per_epoch}  avg_loss={avg_step:.4f}  lr={avg_lr:.6g}  cutoff={cutoff}/{freq_bins}",
                            flush=True,
                        )

            avg = sum(losses) / len(losses)
            elapsed = time.time() - t0

            do_eval = (not args.no_val) and ((epoch + 1) % cfg.eval_every_epochs == 0 or epoch == start_epoch)
            if do_eval:
                vloss = eval_loss(model, corpus_u8, val_starts, cfg, cutoff=cutoff)
                gap = avg - vloss
            else:
                vloss = float('nan')
                gap = float('nan')
            stage_name, lr_mult, min_mult = lr_stage_params(epoch, cfg)
            lr_floor = cfg.lr * min_mult
            lr_peak = cfg.lr * lr_mult
            print(
                f"Epoch {epoch+1:3d}/{cfg.epochs}  train={avg:.4f}  val={vloss:.4f}  gap={gap:+.4f}"
                f"  cutoff={cutoff}/{freq_bins}"
                f"  lr_stage={stage_name} [{lr_peak:.2e}->{lr_floor:.2e}]"
                f"  elapsed={elapsed/60:.1f}m"
            )

            if do_eval and (epoch + 1) % 25 == 0:
                print("-" * 70)
                # Use stage cutoff for stable samples.
                sample = generate(model, "Once upon a time", cfg, cfg.device, cutoff=cutoff)
                print(safe_console(sample))
                # parroting test
                score = parroting_score(corpus_blob, sample.encode("utf-8", errors="ignore"), cfg)
                print(f"[parroting_score] {score:.2f} (0=novel, 1=copied)")
                print("-" * 70)

            if (epoch + 1) % cfg.save_every_epochs == 0:
                save_ckpt(epoch + 1)

    finally:
        # always save the latest checkpoint
        if last_epoch > 0:
            save_ckpt(last_epoch)

    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
