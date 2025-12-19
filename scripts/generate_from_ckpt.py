"""generate_from_ckpt.py

Load the latest checkpoint produced by train_fixed_full.py and run text generation.

Examples:
  python generate_from_ckpt.py --prompt "Once upon a time" --max-new 400
  python generate_from_ckpt.py --prompt "There was a little girl" --top-p 0.9 --temperature 0.9
"""

from __future__ import annotations

import argparse
import os

import torch

from fft_lm import train_fixed_full as tff


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="fixed_spectral_ckpt.pt")
    ap.add_argument("--prompt", nargs="+", default=["Once", "upon", "a", "time"])
    ap.add_argument("--max-new", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--repetition-penalty", type=float, default=1.6)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--kernel-len", type=int, default=128)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--cutoff", type=int, default=None, help="Override cutoff bins (debug)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise SystemExit("CUDA required")

    if not os.path.exists(args.ckpt):
        raise SystemExit(f"Checkpoint not found: {args.ckpt}")

    ckpt = torch.load(args.ckpt, map_location="cpu")

    # Start from saved cfg if present, else defaults.
    cfg = tff.TrainConfig()
    if "cfg" in ckpt and isinstance(ckpt["cfg"], dict):
        for k, v in ckpt["cfg"].items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    # Override core model shape from CLI (useful if ckpt lacks cfg)
    cfg.seq_len = args.seq_len
    cfg.kernel_len = args.kernel_len
    cfg.d_model = args.d_model
    cfg.n_layers = args.n_layers

    # Override generation knobs
    cfg.max_new = args.max_new
    cfg.temperature = args.temperature
    cfg.top_p = args.top_p
    cfg.top_k = args.top_k
    cfg.repetition_penalty = args.repetition_penalty

    # Build model and load
    model = tff.FixedSpectralLM(cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    prompt = " ".join(args.prompt)
    # Use schedule cutoff if checkpoint epoch is present and < full horizon,
    # otherwise generate at full horizon.
    cutoff = args.cutoff
    if cutoff is None and "epoch" in ckpt and isinstance(ckpt["epoch"], int) and ckpt["epoch"] > 0:
        freq_bins = tff.conv_freq_bins(cfg.seq_len, cfg.kernel_len)
        cutoff = tff.curriculum_cutoff(ckpt["epoch"] - 1, cfg, freq_bins)
    out = tff.generate(model, prompt, cfg, device, cutoff=cutoff)
    print(tff.safe_console(out))


if __name__ == "__main__":
    main()
