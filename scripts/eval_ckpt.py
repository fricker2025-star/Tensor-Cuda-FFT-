"""eval_ckpt.py

Quick evaluation for the latest checkpoint:
  - prints checkpoint epoch
  - computes validation loss on fixed windows
  - generates a few prompts with the sampler
  - estimates parroting_score (verbatim snippet match rate)

Run:
  python eval_ckpt.py
  python eval_ckpt.py --prompt "Once upon a time"
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
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.92)
    ap.add_argument("--repetition-penalty", type=float, default=1.8)
    ap.add_argument("--val-batches", type=int, default=20)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise SystemExit("CUDA required")

    if not os.path.exists(args.ckpt):
        raise SystemExit(f"Missing ckpt: {args.ckpt}")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    epoch = ckpt.get("epoch")
    print("=" * 70)
    print(f"Checkpoint: {args.ckpt}")
    print(f"ckpt_epoch: {epoch}")
    print("=" * 70)

    # Build cfg (prefer saved cfg)
    cfg = tff.TrainConfig()
    if "cfg" in ckpt and isinstance(ckpt["cfg"], dict):
        for k, v in ckpt["cfg"].items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    # Override generation params
    cfg.max_new = args.max_new
    cfg.temperature = args.temperature
    cfg.top_p = args.top_p
    cfg.repetition_penalty = args.repetition_penalty
    cfg.device = device
    cfg.val_batches = args.val_batches

    model = tff.FixedSpectralLM(cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    # Load corpus (sanitized if cfg.ascii_only)
    corpus_u8 = tff.load_corpus_as_u8(cfg.data_path, sanitize_ascii=cfg.ascii_only)
    n = int(corpus_u8.numel())
    val_starts = tff.make_val_starts(n, cfg.seq_len, cfg.val_windows, cfg.seed + 1)

    # Evaluate val loss using full horizon (cutoff=None) and current schedule horizon
    freq_bins = tff.conv_freq_bins(cfg.seq_len, cfg.kernel_len)
    # if ckpt epoch is known, use schedule at that epoch-1
    sched_cutoff = None
    if isinstance(epoch, int) and epoch > 0:
        # Use the same strict curriculum as training.
        sched_cutoff = tff.curriculum_cutoff(epoch - 1, cfg, freq_bins)

    vloss_full = tff.eval_loss(model, corpus_u8, val_starts, cfg, cutoff=None)
    vloss_sched = tff.eval_loss(model, corpus_u8, val_starts, cfg, cutoff=sched_cutoff) if sched_cutoff is not None else vloss_full
    print(f"val_loss(full_horizon): {vloss_full:.4f}")
    if sched_cutoff is not None:
        print(f"val_loss(schedule_cutoff={sched_cutoff}/{freq_bins}): {vloss_sched:.4f}")

    # Generate and parroting score
    corpus_blob = bytes(corpus_u8.numpy().tobytes())
    prompt = " ".join(args.prompt)
    out = tff.generate(model, prompt, cfg, device, cutoff=sched_cutoff)
    score = tff.parroting_score(corpus_blob, out.encode("utf-8", errors="ignore"), cfg)
    print("-" * 70)
    print(tff.safe_console(out))
    print(f"[parroting_score] {score:.2f} (0=novel, 1=copied)")
    print("=" * 70)


if __name__ == "__main__":
    main()
