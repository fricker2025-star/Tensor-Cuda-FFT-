"""train_frequency_native.py

Train the FREQUENCY-NATIVE architecture with phase activations.

This is the experimental "galaxy brain" version where:
- All operations stay in frequency domain
- Phase rotations replace time-domain nonlinearities
- Gradients flow through spectral space
- Custom autograd for O(1) gradient computation

Run:
  python -m scripts.train_frequency_native --seq-len 1024 --kernel-len 128 --chunk 16 \
    --batch-size 4 --accum-steps 8 --steps-per-epoch 1000 --epochs 50 --lr 0.0002 \
    --ckpt chunklm_freq_native_ckpt.pt
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.nn as nn

from fft_lm import train_fixed_full as tff
from fft_lm.chunk_head import ChunkLM
from fft_lm.ckpt_io import save_checkpoint, load_checkpoint


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
    ap.add_argument("--ckpt", default="chunklm_freq_native_ckpt.pt")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--no-sawtooth", action="store_true")
    ap.add_argument("--compile", action="store_true", help="Use torch.compile() for faster training (PyTorch 2.0+)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise SystemExit("CUDA required")
    if not os.path.exists(args.data):
        raise SystemExit(f"Missing data: {args.data}")

    # Config with frequency-native mode ENABLED
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
        frequency_native=True,  # 🚀 ENABLE FREQUENCY-NATIVE MODE
        use_fp32=True,  # Use FP32 for complex arithmetic (faster, less casting)
        amp=False,  # Disable AMP for frequency-native (complex ops don't mix well with FP16)
    )
    cfg.device = device
    tff.set_seed(cfg.seed)

    corpus_u8 = tff.load_corpus_as_u8(cfg.data_path, sanitize_ascii=True)
    n = int(corpus_u8.numel())
    hi = n - (cfg.seq_len + args.chunk) - 1
    if hi <= 0:
        raise SystemExit("Dataset too small for requested seq_len+chunk")

    # Build frequency-native model
    backbone = tff.FixedSpectralLM(cfg).to(device)
    model = ChunkLM(backbone, chunk=args.chunk, use_ema=False).to(device)
    
    # Apply torch.compile for fused operations and custom kernels
    # NOTE: torch.compile has issues with complex tensors (stride mismatches)
    # Skipping for now - the complex64 optimization alone is enough
    if args.compile:
        print("=" * 70)
        print("[INFO] torch.compile() requested but DISABLED for frequency-native")
        print("Reason: PyTorch compile has stride issues with complex tensors")
        print("The complex64 dtype optimization is still active!")
        print("=" * 70)
        # Uncomment when PyTorch fixes complex tensor compile support:
        # model = torch.compile(model, mode="reduce-overhead")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)
    loss_fn = nn.CrossEntropyLoss()

    start_epoch = 0
    if args.resume and os.path.exists(cfg.ckpt_path):
        ck = load_checkpoint(cfg.ckpt_path, map_location="cpu")
        model.load_state_dict(ck["model"], strict=True)
        opt.load_state_dict(ck["opt"])
        if ck.get("scaler") is not None and cfg.amp:
            scaler.load_state_dict(ck["scaler"])
        start_epoch = int(ck.get("epoch", 0))
        print(f"Resumed {cfg.ckpt_path} at epoch {start_epoch}")

    def save(epoch_idx: int):
        save_checkpoint(
            {
                "epoch": epoch_idx,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "scaler": scaler.state_dict() if cfg.amp else None,
                "cfg": cfg.__dict__,
                "chunk": args.chunk,
                "frequency_native": True,
            },
            cfg.ckpt_path,
        )

    print("=" * 70)
    print("TRAIN FREQUENCY-NATIVE CHUNK LM")
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
    last_saved_epoch = start_epoch
    
    # Adaptive cutoff state
    freq_bins = tff.conv_freq_bins(cfg.seq_len, cfg.kernel_len)
    if start_epoch >= 1:
        current_cutoff = tff.curriculum_cutoff(start_epoch, cfg, freq_bins)
        print(f"Resuming at epoch {start_epoch}: inferred cutoff={current_cutoff}")
    else:
        current_cutoff = 128
        print(f"Starting at cutoff=128 (basic syntax/characters)")
    
    loss_history = []
    cutoff_was_raised_this_step = False
    
    try:
        for epoch in range(start_epoch, cfg.epochs):
            model.train()
            
            # Check for plateau and potentially raise cutoff at start of epoch
            prev_cutoff = current_cutoff
            current_cutoff, cutoff_raised = tff.adaptive_cutoff(
                epoch, 
                current_cutoff, 
                loss_history,
                freq_bins,
                min_epoch_before_raise=1,
                plateau_window=50,
                plateau_threshold=0.005,
            )
            
            if cutoff_raised:
                print(f"🚀 CUTOFF RAISED: {prev_cutoff} -> {current_cutoff} (loss plateaued)")
                loss_history = []

            opt.zero_grad(set_to_none=True)
            losses = []
            running = 0.0

            micro_total = cfg.steps_per_epoch * cfg.accum_steps
            for micro in range(micro_total):
                opt_step = micro // cfg.accum_steps
                
                if opt_step > 0:
                    cutoff_was_raised_this_step = False
                else:
                    cutoff_was_raised_this_step = cutoff_raised

                # LR schedule - RESTART when cutoff raised
                if not args.no_sawtooth:
                    global_opt_step = epoch * cfg.steps_per_epoch + opt_step
                    lr_now = tff.sawtooth_lr(global_opt_step, epoch, cfg, cutoff_raised=cutoff_was_raised_this_step)
                    for pg in opt.param_groups:
                        pg["lr"] = lr_now

                starts = torch.randint(0, hi, (cfg.batch_size,), dtype=torch.long)
                from fft_lm.chunk_head import vectorized_windows
                bx, by = vectorized_windows(corpus_u8, starts, cfg.seq_len, args.chunk)
                bx = bx.to(device, non_blocking=True)
                by = by.to(device, non_blocking=True)

                with torch.autocast("cuda", enabled=cfg.amp):
                    logits = model(bx, cutoff=current_cutoff)
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
                    loss_history.append(li)
                    running += li
                    if (opt_step + 1) % args.log_every == 0:
                        lr_disp = opt.param_groups[0]["lr"]
                        print(f"  step {opt_step+1}/{cfg.steps_per_epoch} loss={running/args.log_every:.4f} lr={lr_disp:.2e} cutoff={current_cutoff}")
                        running = 0.0

            avg = sum(losses) / max(1, len(losses))
            print(f"Epoch {epoch+1}/{cfg.epochs} avg_loss={avg:.4f} cutoff={current_cutoff}/{freq_bins} elapsed={(time.time()-t0)/60:.1f}m")
            if (epoch + 1) % 5 == 0:
                save(epoch + 1)
                last_saved_epoch = epoch + 1

    except KeyboardInterrupt:
        print("\n[Ctrl+C] Saving checkpoint...")
        save(max(last_saved_epoch, start_epoch))
        print(f"Saved to {cfg.ckpt_path}")
        raise
    finally:
        save(max(last_saved_epoch, start_epoch))

    print("DONE - FREQUENCY-NATIVE TRAINING COMPLETE")


if __name__ == "__main__":
    main()
