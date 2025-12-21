"""generate_chunked_overlap_save.py

Exact "overlap-save" chunk generation for the spectral backbone.

This implements the user's requested Option 2:
  - Generate CHUNK bytes at a time using the chunk head.
  - Apply a quantization barrier (bytes are discrete ints 0..255).
  - Update backbone state using *exact* FFT-based convolution per chunk via
    overlap-save with fixed FFT size (matching training), including:
      - causal convolution kernel
      - per-frequency gate (gate_freq_logits)
      - context gate (gate_ctx)
      - per-channel gain
      - FFN residual

Key: we DO NOT recompute the backbone over the full history every token.
We update state per chunk (still sequential across chunks).

Complexity:
  O(layers * FFT(n_fft_full) per chunk)
  where n_fft_full = next_pow2(seq_len + kernel_len - 1).

Run:
  python generate_chunked_overlap_save.py --ckpt chunklm_ckpt_1024.pt --prompt "Once upon a time" --chunks 30
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from fft_lm import train_fixed_full as tff
from fft_lm.chunk_head import ChunkLM
from fft_lm.ckpt_io import load_checkpoint


def apply_top_p(logits_1d: torch.Tensor, p: float) -> torch.Tensor:
    sorted_logits, sorted_idx = torch.sort(logits_1d, descending=True)
    probs = F.softmax(sorted_logits, dim=-1)
    cdf = torch.cumsum(probs, dim=-1)
    keep = cdf <= p
    keep[0] = True
    cutoff = int(keep.sum().item())
    masked = torch.full_like(logits_1d, -float("inf"))
    masked[sorted_idx[:cutoff]] = logits_1d[sorted_idx[:cutoff]]
    return masked


@torch.no_grad()
def init_layer_states(backbone: tff.FixedSpectralLM, x_ids: torch.Tensor) -> dict:
    """Initialize per-layer context buffers from a full forward over the context window.

    We store, per layer:
      - ctx_ln: [1, T, C] the layernormed inputs to the conv (for gating mean + overlap)
      - ctx_sum: [1, C] running sum of ctx_ln over time
    """
    device = x_ids.device
    B, T = x_ids.shape
    assert B == 1

    h = backbone.embed(x_ids)
    states = []
    for blk in backbone.blocks:
        # pre-norm inputs used by conv and by context gate pooled mean
        ln_in = blk.ln(h)  # [1,T,C]
        ctx_sum = ln_in.sum(dim=1)  # [1,C]
        states.append({"ctx_ln": ln_in.contiguous(), "ctx_sum": ctx_sum.contiguous()})

        # run the actual block forward to advance h
        h = blk(h, cutoff=None)
    h = backbone.ln_f(h)
    return {"h_last": h[:, -1, :].contiguous(), "layers": states}


@torch.no_grad()
def overlap_save_block_update(
    blk: tff.FixedSpectralBlock,
    layer_state: dict,
    h_chunk: torch.Tensor,
    *,
    n_fft_full: int,
    kernel_len: int,
    cache: dict | None = None,
) -> tuple[torch.Tensor, dict]:
    """Update a single block for a chunk using overlap-save.

    Inputs:
      h_chunk: [1, B, C] current hidden chunk entering this block.
      layer_state: contains ctx_ln [1,T,C] and ctx_sum [1,C] for this block.

    Returns:
      h_chunk_out: [1, B, C]
      updated layer_state
    """
    device = h_chunk.device
    B = h_chunk.size(1)
    T = layer_state["ctx_ln"].size(1)

    # 1) compute ln inputs for the new chunk (this is what conv uses)
    ln_chunk = blk.ln(h_chunk)  # [1,B,C]

    # 2) update context buffer (sliding window of length T)
    # drop oldest B and append new B
    ctx_ln = layer_state["ctx_ln"]
    if B >= T:
        # degenerate: keep last T of ln_chunk
        ctx_ln_new = ln_chunk[:, -T:, :]
    else:
        ctx_ln_new = torch.cat([ctx_ln[:, B:, :], ln_chunk], dim=1)
    ctx_sum_new = ctx_ln_new.sum(dim=1)

    # pooled context for gate
    pooled = (ctx_sum_new / float(ctx_ln_new.size(1))).to(dtype=torch.float32)  # [1,C]
    g_ctx = torch.sigmoid(blk.gate_ctx(pooled.to(dtype=next(blk.gate_ctx.parameters()).dtype))).to(torch.float32)  # [1,C]

    Fbins = (n_fft_full // 2 + 1)

    # per-frequency gate (cacheable at inference)
    if cache is not None and "g_freq" in cache:
        g_freq = cache["g_freq"]
    else:
        g_freq = torch.sigmoid(blk.gate_freq_logits[:Fbins]).to(torch.float32)  # [F]

    # 3) overlap-save convolution using fixed FFT size
    # build segment of length L = (K-1)+B from last K-1 ctx_ln_new and current ln_chunk
    overlap = ctx_ln_new[:, - (kernel_len - 1 + B) : -B, :] if kernel_len > 1 else ctx_ln_new[:, :0, :]
    if kernel_len > 1:
        overlap = ctx_ln_new[:, -(kernel_len - 1 + B) : -B, :]
    else:
        overlap = ctx_ln_new[:, :0, :]
    x_seg = torch.cat([overlap, ln_chunk], dim=1)  # [1, K-1+B, C]
    L = x_seg.size(1)

    # pad to n_fft_full
    if L < n_fft_full:
        x_pad = F.pad(x_seg, (0, 0, 0, n_fft_full - L))
    else:
        x_pad = x_seg[:, :n_fft_full, :]

    # FFT along time
    x_pad_f = x_pad.to(torch.float32)
    x_freq = torch.fft.rfft(x_pad_f, dim=1)  # [1,F,C] complex64

    # kernel freq (cacheable at inference)
    if cache is not None and "k_freq" in cache:
        k_freq = cache["k_freq"]
    else:
        k = torch.zeros(n_fft_full, device=device, dtype=torch.float32)
        k[:kernel_len] = blk.kernel.to(torch.float32)
        k_freq = torch.fft.rfft(k)  # [F]

    # apply kernel, gain, gates
    if cache is not None and "gain" in cache:
        gain = cache["gain"]
    else:
        gain = blk.gain.to(torch.float32)
    gain = gain.view(1, 1, -1)
    y_freq = x_freq * k_freq.view(1, -1, 1) * gain
    y_freq = y_freq * g_freq.view(1, -1, 1) * g_ctx.view(1, 1, -1)

    # iFFT back
    y_pad = torch.fft.irfft(y_freq, n=n_fft_full, dim=1)  # [1,n_fft,C] float32

    # take outputs for the new chunk positions: indices [K-1 : K-1+B]
    start = kernel_len - 1
    y_chunk = y_pad[:, start : start + B, :].to(h_chunk.dtype)  # [1,B,C]

    # residual + FFN (dropout off at eval)
    h_out = h_chunk + y_chunk
    ff_in = blk.ffn_ln(h_out)
    h_out = h_out + blk.ffn(ff_in)

    new_state = {"ctx_ln": ctx_ln_new.contiguous(), "ctx_sum": ctx_sum_new.contiguous()}
    return h_out, new_state


@torch.no_grad()
def update_backbone_chunk(backbone: tff.FixedSpectralLM, states: dict, new_ids: list[int]) -> dict:
    """Advance backbone state by a chunk of newly generated token IDs."""
    device = states["h_last"].device
    B = len(new_ids)
    x_ids = torch.tensor([new_ids], dtype=torch.long, device=device)  # [1,B]
    h_chunk = backbone.embed(x_ids)  # [1,B,C]

    n_fft_full = tff.conv_freq_bins(backbone.cfg.seq_len, backbone.cfg.kernel_len) * 2 - 2
    # (since bins = n_fft//2+1) => n_fft = (bins-1)*2
    kernel_len = backbone.cfg.kernel_len

    caches = states.get("caches")
    for li, blk in enumerate(backbone.blocks):
        h_chunk, new_layer_state = overlap_save_block_update(
            blk,
            states["layers"][li],
            h_chunk,
            n_fft_full=n_fft_full,
            kernel_len=kernel_len,
            cache=(caches[li] if caches is not None else None),
        )
        states["layers"][li] = new_layer_state

    # final norm
    h_chunk = backbone.ln_f(h_chunk)
    states["h_last"] = h_chunk[:, -1, :].contiguous()
    return states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="chunklm_ckpt_1024.pt")
    ap.add_argument("--prompt", nargs="+", default=["Once", "upon", "a", "time"])
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--kernel-len", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--chunks", type=int, default=30)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--rep", type=float, default=1.15)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise SystemExit("CUDA required")

    ck = load_checkpoint(args.ckpt, map_location="cpu")

    # prefer checkpoint cfg when present
    cfg = tff.TrainConfig(seq_len=args.seq_len, kernel_len=args.kernel_len)
    if "cfg" in ck and isinstance(ck["cfg"], dict):
        for k, v in ck["cfg"].items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        # CLI override remains authoritative
        cfg.seq_len = args.seq_len
        cfg.kernel_len = args.kernel_len
    cfg.device = device
    backbone = tff.FixedSpectralLM(cfg).to(device)
    # If checkpoint recorded chunk size, respect it unless CLI forces a value.
    ckpt_chunk = int(ck.get("chunk", args.chunk))
    model = ChunkLM(backbone, chunk=ckpt_chunk).to(device)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()

    # Precompute per-layer caches for exact overlap-save updates.
    n_fft_full = tff.conv_freq_bins(cfg.seq_len, cfg.kernel_len) * 2 - 2
    Fbins = n_fft_full // 2 + 1
    caches = []
    for blk in backbone.blocks:
        # kernel FFT
        k = torch.zeros(n_fft_full, device=device, dtype=torch.float32)
        k[: cfg.kernel_len] = blk.kernel.detach().to(torch.float32)
        k_freq = torch.fft.rfft(k)  # [F] complex64
        # frequency gate
        g_freq = torch.sigmoid(blk.gate_freq_logits.detach()[:Fbins]).to(torch.float32)
        gain = blk.gain.detach().to(torch.float32)
        caches.append({"k_freq": k_freq, "g_freq": g_freq, "gain": gain})

    prompt = " ".join(args.prompt)
    ctx = list(prompt.encode("utf-8", errors="ignore"))
    if not ctx:
        ctx = [32]

    # Build initial context window (pad/truncate) for state init
    if len(ctx) < cfg.seq_len:
        init_ids = [32] * (cfg.seq_len - len(ctx)) + ctx
    else:
        init_ids = ctx[-cfg.seq_len :]
    x0 = torch.tensor([init_ids], dtype=torch.long, device=device)

    states = init_layer_states(backbone, x0)
    # Attach caches to the state dict so update_backbone_chunk can access them.
    states["caches"] = caches

    generated = init_ids[:]  # include padding; sampling uses recent window anyway

    for _ in range(args.chunks):
        # predict chunk logits from current hidden state (head only)
        flat = model.head(states["h_last"].float())  # [1, chunk*256]
        logits = flat.view(1, args.chunk, 256)[0]

        new_bytes = []
        for i in range(args.chunk):
            l = logits[i].float()
            for tok in set(generated[-256:]):
                l[tok] = l[tok] / args.rep
            l = l / args.temperature
            l = apply_top_p(l, args.top_p)
            probs = F.softmax(l, dim=-1)
            b = int(torch.multinomial(probs, 1).item())
            b = max(0, min(255, b))
            new_bytes.append(b)

        # print chunk
        print(tff.safe_console(bytes(new_bytes).decode("utf-8", errors="replace")), end="", flush=True)

        generated.extend(new_bytes)
        # exact overlap-save update (one backbone pass per chunk)
        states = update_backbone_chunk(backbone, states, new_bytes)

    print()


if __name__ == "__main__":
    main()
