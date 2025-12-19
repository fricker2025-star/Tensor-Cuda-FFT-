"""generate_chunked.py

Chunk-by-chunk generation using a trained ChunkLM checkpoint.

This is the "Piston Engine": generate CHUNK bytes, append, repeat.
Quantization barrier is inherent because we sample integer bytes.

Run:
  python generate_chunked.py --ckpt chunklm_ckpt.pt --prompt "Once upon a time" --chunks 20
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from fft_lm import train_fixed_full as tff
from fft_lm.chunk_head import ChunkLM


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="chunklm_ckpt.pt")
    ap.add_argument("--prompt", nargs="+", default=["Once", "upon", "a", "time"])
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--kernel-len", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--chunks", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--rep", type=float, default=1.15)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise SystemExit("CUDA required")

    ck = torch.load(args.ckpt, map_location="cpu")
    cfg = tff.TrainConfig(seq_len=args.seq_len, kernel_len=args.kernel_len)
    cfg.device = device

    backbone = tff.FixedSpectralLM(cfg).to(device)
    model = ChunkLM(backbone, chunk=args.chunk).to(device)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()

    prompt = " ".join(args.prompt)
    ctx = list(prompt.encode("utf-8", errors="ignore"))
    if not ctx:
        ctx = [32]

    # pad/truncate
    if len(ctx) < cfg.seq_len:
        ctx = [32] * (cfg.seq_len - len(ctx)) + ctx
    else:
        ctx = ctx[-cfg.seq_len :]

    generated = ctx[:]  # keep full history

    def quantization_barrier(int_bytes: list[int]) -> list[int]:
        """Hard reset: ensure clean discrete bytes (0..255).

        In this generator, we already sample integer IDs, but we enforce the barrier
        explicitly to guarantee no float contamination, and to clamp out-of-range.
        """
        outb = []
        for b in int_bytes:
            b = int(b)
            if b < 0:
                b = 0
            elif b > 255:
                b = 255
            outb.append(b)
        return outb

    for c in range(args.chunks):
        x = torch.tensor([generated[-cfg.seq_len :]], dtype=torch.long, device=device)
        cutoff_bins = tff.curriculum_cutoff(10**9, cfg, tff.conv_freq_bins(cfg.seq_len, cfg.kernel_len))  # full detail
        logits = model(x, cutoff=cutoff_bins)[0]  # [chunk,256]

        new_bytes = []
        for i in range(args.chunk):
            l = logits[i].float()
            # repetition penalty from recent
            for tok in set(generated[-256:]):
                l[tok] = l[tok] / args.rep
            l = l / args.temperature
            l = apply_top_p(l, args.top_p)
            probs = F.softmax(l, dim=-1)
            b = int(torch.multinomial(probs, 1).item())
            new_bytes.append(b)

        # QUANTIZATION BARRIER (piston reset)
        new_bytes = quantization_barrier(new_bytes)

        # Clean handoff: only ever append clean ints
        generated.extend(new_bytes)

        # print chunk (safe for Windows console)
        chunk_text = bytes(new_bytes).decode("utf-8", errors="replace")
        print(tff.safe_console(chunk_text), end="", flush=True)

    print()


if __name__ == "__main__":
    main()
