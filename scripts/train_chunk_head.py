"""train_chunk_head.py

Train a CHUNK-PREDICTING head on top of the causal spectral backbone.

This is the missing link for the "Piston Engine":
  - predict 16 bytes at once (or any chunk size)
  - sample a chunk
  - quantization barrier is inherent (sampled ints)
  - slide window and repeat

This avoids token-by-token O(T^2) generation because we only run the backbone
once per chunk.

Run:
  python train_chunk_head.py --seq-len 1024 --chunk 16 --steps-per-epoch 1000 --batch-size 4 --accum-steps 8 --lr 0.0002
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.nn as nn

from fft_lm import train_fixed_full as tff
from fft_lm.chunk_head import ChunkLM, vectorized_windows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="tinystories_train.txt")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--kernel-len", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--steps-per-epoch", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--accum-steps", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--ckpt", default="chunklm_ckpt.pt")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--no-sawtooth", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise SystemExit("CUDA required")
    if not os.path.exists(args.data):
        raise SystemExit(f"Missing data: {args.data}")

    # reuse backbone config
    cfg = tff.TrainConfig(
        data_path=args.data,
        seq_len=args.seq_len,
        kernel_len=args.kernel_len,
        lr=args.lr,
        weight_decay=args.wd,
        batch_size=args.batch_size,
        accum_steps=args.accum_steps,
        steps_per_epoch=args.steps_per_epoch,
        epochs=args.epochs,
        ckpt_path=args.ckpt,
    )
    cfg.device = device
    tff.set_seed(cfg.seed)

    corpus_u8 = tff.load_corpus_as_u8(cfg.data_path, sanitize_ascii=True)
    n = int(corpus_u8.numel())
    hi = n - (cfg.seq_len + args.chunk) - 1
    if hi <= 0:
        raise SystemExit("Dataset too small for requested seq_len+chunk")

    backbone = tff.FixedSpectralLM(cfg).to(device)
    model = ChunkLM(backbone, chunk=args.chunk).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)
    loss_fn = nn.CrossEntropyLoss()

    start_epoch = 0
    if args.resume and os.path.exists(cfg.ckpt_path):
        ck = torch.load(cfg.ckpt_path, map_location="cpu")
        model.load_state_dict(ck["model"], strict=True)
        opt.load_state_dict(ck["opt"])
        if ck.get("scaler") is not None and cfg.amp:
            scaler.load_state_dict(ck["scaler"])
        start_epoch = int(ck.get("epoch", 0))
        print(f"Resumed {cfg.ckpt_path} at epoch {start_epoch}")

    def save(epoch_idx: int):
        torch.save(
            {
                "epoch": epoch_idx,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "scaler": scaler.state_dict() if cfg.amp else None,
                "cfg": cfg.__dict__,
                "chunk": args.chunk,
            },
            cfg.ckpt_path,
        )

    print("=" * 70)
    print("TRAIN CHUNK HEAD (Piston Engine)")
    print("=" * 70)
    print(f"device={device} seq_len={cfg.seq_len} kernel_len={cfg.kernel_len} chunk={args.chunk}")
    print(f"micro_batch={cfg.batch_size} accum={cfg.accum_steps} effective={cfg.batch_size*cfg.accum_steps}")
    print(f"steps/epoch={cfg.steps_per_epoch} epochs={cfg.epochs} lr={cfg.lr} wd={cfg.weight_decay}")
    if not args.no_sawtooth:
        print(
            f"LR sched: sawtooth cosine restarts (stage-aligned)"
            f"  s1(e0-{cfg.stage1_epochs-1}) mult {cfg.stage1_lr_mult}->{cfg.stage1_min_mult}"
            f"  s2(e{cfg.stage1_epochs}-{cfg.stage1_epochs+cfg.stage2_epochs-1}) mult {cfg.stage2_lr_mult}->{cfg.stage2_min_mult}"
            f"  s3(e{cfg.stage1_epochs+cfg.stage2_epochs}+) mult {cfg.stage3_lr_mult}->{cfg.stage3_min_mult}"
        )
    print(f"corpus_bytes={n:,}")
    params = sum(p.numel() for p in model.parameters())
    print(f"params={params:,} (~{params/1e6:.2f}M)")
    print("=" * 70)

    t0 = time.time()
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        cutoff_bins = tff.curriculum_cutoff(epoch, cfg, tff.conv_freq_bins(cfg.seq_len, cfg.kernel_len))

        opt.zero_grad(set_to_none=True)
        losses = []
        running = 0.0

        micro_total = cfg.steps_per_epoch * cfg.accum_steps
        for micro in range(micro_total):
            opt_step = micro // cfg.accum_steps

            # LR schedule on optimizer steps
            if not args.no_sawtooth:
                global_opt_step = epoch * cfg.steps_per_epoch + opt_step
                lr_now = tff.sawtooth_lr(global_opt_step, epoch, cfg)
                for pg in opt.param_groups:
                    pg["lr"] = lr_now

            starts = torch.randint(0, hi, (cfg.batch_size,), dtype=torch.long)
            bx, by = vectorized_windows(corpus_u8, starts, cfg.seq_len, args.chunk)
            bx = bx.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)

            with torch.autocast("cuda", enabled=cfg.amp):
                logits = model(bx, cutoff=cutoff_bins)  # [B,chunk,256]
                loss = loss_fn(logits.reshape(-1, 256), by.reshape(-1))
                loss = loss / float(cfg.accum_steps)

            scaler.scale(loss).backward()

            if (micro + 1) % cfg.accum_steps == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

                li = float(loss.item()) * float(cfg.accum_steps)
                losses.append(li)
                running += li
                if (opt_step + 1) % args.log_every == 0:
                    lr_disp = opt.param_groups[0]["lr"]
                    print(f"  step {opt_step+1}/{cfg.steps_per_epoch} loss={running/args.log_every:.4f} lr={lr_disp:.2e} cutoff={cutoff_bins}")
                    running = 0.0

        avg = sum(losses) / max(1, len(losses))
        print(f"Epoch {epoch+1}/{cfg.epochs} avg_loss={avg:.4f} cutoff={cutoff_bins} elapsed={(time.time()-t0)/60:.1f}m")
        if (epoch + 1) % 5 == 0:
            save(epoch + 1)

    save(cfg.epochs)
    print("DONE")


if __name__ == "__main__":
    main()
