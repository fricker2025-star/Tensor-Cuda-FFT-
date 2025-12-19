"""stream_generate_fast.py

Fast generation for the *causal FFT-conv* model by turning the convolution into a
streaming FIR recurrence (stateful ring buffer) during inference.

Why:
  - Naive generation recomputes full forward over the whole context every token
    (O(T^2)).
  - Our blocks include a causal convolution with finite kernel_len, so we can
    stream with O(kernel_len) per token, independent of context length.

Notes:
  - This keeps outputs as discrete bytes (no float drift). "Quantization barrier"
    is inherently satisfied because we sample integer byte IDs.
  - Supports chunked emission (e.g., 16 bytes per chunk) for lower Python overhead.

Run:
  python stream_generate_fast.py --ckpt fixed_spectral_ckpt_1024.pt --prompt "Once upon a time" --max-new 400
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from fft_lm import train_fixed_full as tff


class StreamingConvBlock:
    """Streaming version of one FixedSpectralBlock.

    Implements the causal convolution part as a ring buffer FIR.
    Keeps FFN + norms identical to the trained module.
    """

    def __init__(self, block: tff.FixedSpectralBlock, *, kernel_len: int):
        self.block = block
        self.kernel_len = kernel_len

        # Cache time-domain kernel and gain.
        # block.kernel: [K]
        k = block.kernel.detach().to(torch.float32)  # [K]
        if k.numel() != kernel_len:
            # tolerate mismatch by trunc/pad
            kk = torch.zeros(kernel_len, dtype=torch.float32)
            n = min(kernel_len, k.numel())
            kk[:n] = k[:n]
            k = kk
        # Reverse for dot with buffer in chronological order.
        self.k_rev = torch.flip(k, dims=[0])  # [K]

        self.gain = block.gain.detach().to(torch.float32)  # [C]

        # Optional gates (present in current architecture)
        self.has_freq_gate = hasattr(block, "gate_freq_logits")
        self.has_ctx_gate = hasattr(block, "gate_ctx")

    def init_state(self, batch: int, d_model: int, device: str, dtype: torch.dtype):
        # ring buffer of last K inputs to convolution (after pre-norm)
        buf = torch.zeros(batch, self.kernel_len, d_model, device=device, dtype=dtype)
        return {"buf": buf}

    @torch.no_grad()
    def step(self, x_t: torch.Tensor, state: dict) -> torch.Tensor:
        """Advance one token.

        x_t: [B, C] float
        returns: [B, C]
        """
        # Pre-norm
        xn = self.block.ln(x_t)

        # Update buffer: shift left and append xn
        buf = state["buf"]
        buf = torch.roll(buf, shifts=-1, dims=1)
        buf[:, -1, :] = xn
        state["buf"] = buf

        # FIR convolution: y = sum_i k[i] * x_{t-i}
        # buf is [B, K, C] oldest->newest; k_rev aligns newest weight with last element.
        y = (buf * self.k_rev.to(buf.dtype).view(1, self.kernel_len, 1)).sum(dim=1)

        # Per-channel gain
        y = y * self.gain.to(y.dtype).view(1, -1)

        # Context gate (per-channel), approximated using current token representation.
        if self.has_ctx_gate:
            g_ctx = torch.sigmoid(self.block.gate_ctx(xn)).to(y.dtype)
            y = y * g_ctx

        # NOTE: per-frequency gate is not representable in finite FIR form without
        # expanding kernel length to n_fft. We intentionally ignore it here to keep
        # streaming O(K). If needed, we can approximate by folding it into an
        # expanded time-domain kernel and truncating.

        # Residual + (no dropout at eval)
        x = x_t + y

        # FFN residual
        ff_in = self.block.ffn_ln(x)
        x = x + self.block.ffn(ff_in)
        return x


class StreamingLM:
    def __init__(self, model: tff.FixedSpectralLM, *, kernel_len: int):
        self.model = model
        self.kernel_len = kernel_len
        self.blocks = [StreamingConvBlock(b, kernel_len=kernel_len) for b in model.blocks]

    def init_state(self, batch: int, device: str):
        d_model = self.model.embed.weight.shape[1]
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        return {
            "blocks": [blk.init_state(batch, d_model, device, dtype) for blk in self.blocks],
        }

    @torch.no_grad()
    def warmup(self, prefix_ids: torch.Tensor, state: dict) -> None:
        """Feed prefix through the streaming model to fill convolution states."""
        # prefix_ids: [B, T]
        B, T = prefix_ids.shape
        for t in range(T):
            x = self.model.embed(prefix_ids[:, t])  # [B, C]
            for i, blk in enumerate(self.blocks):
                x = blk.step(x, state["blocks"][i])
            # we don't need logits during warmup

    @torch.no_grad()
    def next_logits(self, last_id: torch.Tensor, state: dict) -> torch.Tensor:
        """Step one token and return logits for next-token distribution.

        last_id: [B] long, the current token id being fed
        returns logits: [B, V]
        """
        x = self.model.embed(last_id)  # [B, C]
        for i, blk in enumerate(self.blocks):
            x = blk.step(x, state["blocks"][i])
        x = self.model.ln_f(x)
        logits = x @ self.model.embed.weight.t()  # weight-tied
        return logits


def sample_next(logits_1d: torch.Tensor, temperature: float, top_p: float, top_k: int, rep_penalty: float, recent: list[int]):
    logits = logits_1d.float().clone()
    # repetition penalty
    for tok in set(recent[-256:]):
        logits[tok] = logits[tok] / rep_penalty
    logits = logits / temperature

    # nucleus
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cdf = torch.cumsum(probs, dim=-1)
        keep = cdf <= top_p
        keep[0] = True
        cutoff = int(keep.sum().item())
        mask = torch.full_like(logits, -float("inf"))
        mask[sorted_idx[:cutoff]] = logits[sorted_idx[:cutoff]]
        logits = mask

    if top_k and top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.numel()))
        logits[logits < v[-1]] = -float("inf")

    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="fixed_spectral_ckpt_1024.pt")
    ap.add_argument("--prompt", nargs="+", default=["Once", "upon", "a", "time"])
    ap.add_argument("--max-new", type=int, default=400)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--kernel-len", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.92)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--rep", type=float, default=1.25)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise SystemExit("CUDA required")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = tff.TrainConfig()
    if "cfg" in ckpt and isinstance(ckpt["cfg"], dict):
        for k, v in ckpt["cfg"].items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    model = tff.FixedSpectralLM(cfg).to(device)
    # strict load: requires matching architecture
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    prompt = " ".join(args.prompt)
    prefix = list(prompt.encode("utf-8", errors="ignore"))
    if not prefix:
        prefix = [32]

    # streaming model
    slm = StreamingLM(model, kernel_len=args.kernel_len)
    state = slm.init_state(batch=1, device=device)

    # warmup with all but last token
    if len(prefix) > 1:
        warm = torch.tensor([prefix[:-1]], dtype=torch.long, device=device)
        slm.warmup(warm, state)

    out = prefix[:]  # bytes
    last = torch.tensor([out[-1]], dtype=torch.long, device=device)

    # chunked emission (reduces Python overhead)
    remaining = args.max_new
    while remaining > 0:
        take = min(args.chunk, remaining)
        for _ in range(take):
            logits = slm.next_logits(last, state)[0]
            nxt = sample_next(
                logits,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                rep_penalty=args.rep,
                recent=out,
            )
            out.append(nxt)
            last = torch.tensor([nxt], dtype=torch.long, device=device)
        remaining -= take

        # "Quantization barrier" is naturally satisfied: out is integers.

    print(out[: len(prefix)].__class__)  # keep linter quiet
    print(bytes(out).decode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()
