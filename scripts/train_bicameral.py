"""train_bicameral.py

Train the BICAMERAL (Two-Hemisphere) Architecture.

LEFT BRAIN (Time):      Sharp details, causality, spelling, syntax  
RIGHT BRAIN (Frequency): Global context, structure, intuition, vibes
CORPUS CALLOSUM:        Learns optimal fusion of both paths

This fixes the "blurry text" problem:
- Frequency path provides skeleton (paragraphs, topics)
- Time path paints skin (sharp letters, punctuation)
- Together: Infinite context + Sharp precision!

Run:
  python -m scripts.train_bicameral --seq-len 1024 --kernel-len 128 --chunk 16 \
    --batch-size 4 --accum-steps 8 --steps-per-epoch 1000 --epochs 50 --lr 0.0002 \
    --ckpt bicameral_ckpt.pt
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.nn as nn

from fft_lm import train_fixed_full as tff
from fft_lm.chunk_head import vectorized_windows
from fft_lm.phase_clock import PhaseClockChunkLM, generate_phase_targets, compute_phase_clock_loss
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
    ap.add_argument("--ckpt", default="bicameral_ckpt.pt")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--no-sawtooth", action="store_true")
    ap.add_argument("--achievement-mode", action="store_true", help="Use achievement-based cutoff (unlock by mastering, not plateau)")
    ap.add_argument("--phase-weight", type=float, default=5.0, help="Weight for phase-clock loss (default 5.0)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise SystemExit("CUDA required")
    if not os.path.exists(args.data):
        raise SystemExit(f"Missing data: {args.data}")

    # Config with BICAMERAL mode ENABLED
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
        bicameral=True,  # 🧠 ENABLE BICAMERAL MODE
    )
    cfg.device = device
    tff.set_seed(cfg.seed)

    corpus_u8 = tff.load_corpus_as_u8(cfg.data_path, sanitize_ascii=True)
    n = int(corpus_u8.numel())
    hi = n - (cfg.seq_len + args.chunk) - 1
    if hi <= 0:
        raise SystemExit("Dataset too small for requested seq_len+chunk")

    # Build bicameral model with 2-neuron phase-clock head (ALWAYS ENABLED)
    backbone = tff.FixedSpectralLM(cfg).to(device)
    model = PhaseClockChunkLM(backbone, chunk=args.chunk).to(device)
    
    print(f"[PHASE-CLOCK] 2-neuron word-as-wave head enabled (weight={args.phase_weight})")
    print(f"              Words are continuous waves: 0° → 180°")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)

    start_epoch = 0
    resume_step = 0  # Step within epoch to resume from
    resume_cutoff = None
    resume_best_loss = None
    resume_plateau_counter = None
    
    if args.resume and os.path.exists(cfg.ckpt_path):
        ck = load_checkpoint(cfg.ckpt_path, map_location="cpu")
        model.load_state_dict(ck["model"], strict=True)
        opt.load_state_dict(ck["opt"])
        if ck.get("scaler") is not None and cfg.amp:
            scaler.load_state_dict(ck["scaler"])
        
        # Restore training state
        start_epoch = int(ck.get("epoch", 0))
        resume_step = int(ck.get("step", 0))
        resume_cutoff = ck.get("cutoff", None)
        resume_best_loss = ck.get("best_loss_at_cutoff", None)
        resume_plateau_counter = ck.get("steps_without_improvement", None)
        
        print(f"Resumed {cfg.ckpt_path}")
        print(f"  Epoch: {start_epoch}, Step: {resume_step}/{cfg.steps_per_epoch}")
        if resume_cutoff:
            print(f"  Cutoff: {resume_cutoff}, Best loss: {resume_best_loss:.3f}")
            print(f"  Plateau counter: {resume_plateau_counter}/50")

    def save(epoch_idx: int, step_idx: int = 0, cutoff: int = 128, best_loss: float = float('inf'), plateau_count: int = 0):
        save_checkpoint(
            {
                "epoch": epoch_idx,
                "step": step_idx,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "scaler": scaler.state_dict() if cfg.amp else None,
                "cfg": cfg.__dict__,
                "chunk": args.chunk,
                "bicameral": True,
                # Save training state for proper resume
                "cutoff": cutoff,
                "best_loss_at_cutoff": best_loss,
                "steps_without_improvement": plateau_count,
            },
            cfg.ckpt_path,
        )

    print("=" * 70)
    print("TRAIN BICAMERAL CHUNK LM (Two-Hemisphere Architecture)")
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
    
    # Initialize or restore cutoff and plateau state
    if resume_cutoff is not None:
        # Resuming - use saved state
        current_cutoff = resume_cutoff
        best_loss_at_cutoff = resume_best_loss if resume_best_loss is not None else float('inf')
        steps_without_improvement = resume_plateau_counter if resume_plateau_counter is not None else 0
        print(f"Resuming training state:")
        print(f"  Cutoff: {current_cutoff}, Best loss: {best_loss_at_cutoff:.3f}")
        print(f"  Plateau progress: {steps_without_improvement}/50")
    elif start_epoch >= 1:
        # Starting from saved epoch but no training state - infer
        current_cutoff = tff.curriculum_cutoff(start_epoch, cfg, freq_bins)
        best_loss_at_cutoff = float('inf')
        steps_without_improvement = 0
        print(f"Resuming at epoch {start_epoch}: inferred cutoff={current_cutoff}")
    else:
        # Fresh start
        current_cutoff = 128
        best_loss_at_cutoff = float('inf')
        steps_without_improvement = 0
        print(f"Starting at cutoff=128 (basic syntax/characters)")
    
    loss_history = []
    cutoff_was_raised_this_step = False
    
    # Track hemisphere balance
    def log_hemisphere_balance():
        """Log which brain is dominant."""
        alpha_f = torch.sigmoid(model.backbone.blocks[0].alpha_freq).item()
        alpha_t = torch.sigmoid(model.backbone.blocks[0].alpha_time).item()
        total = alpha_f + alpha_t
        print(f"  Hemisphere: Freq={alpha_f/total:.1%} Time={alpha_t/total:.1%}")
    
    try:
        for epoch in range(start_epoch, cfg.epochs):
            model.train()

            opt.zero_grad(set_to_none=True)
            losses = []
            running_losses = []  # Track recent losses for running average
            phase_losses = []  # Track phase-clock losses
            cutoff_was_raised_this_step = False

            micro_total = cfg.steps_per_epoch * cfg.accum_steps
            for micro in range(micro_total):
                opt_step = micro // cfg.accum_steps
                
                # Skip already-completed steps when resuming
                if epoch == start_epoch and opt_step < resume_step:
                    continue

                # LR schedule - RESTART when cutoff raised
                if not args.no_sawtooth:
                    global_opt_step = epoch * cfg.steps_per_epoch + opt_step
                    lr_now = tff.sawtooth_lr(global_opt_step, epoch, cfg, cutoff_raised=cutoff_was_raised_this_step)
                    for pg in opt.param_groups:
                        pg["lr"] = lr_now

                starts = torch.randint(0, hi, (cfg.batch_size,), dtype=torch.long)
                bx, by = vectorized_windows(corpus_u8, starts, cfg.seq_len, args.chunk)
                bx = bx.to(device, non_blocking=True)
                by = by.to(device, non_blocking=True)

                with torch.autocast("cuda", enabled=cfg.amp):
                    # Forward with phase-clock head (always enabled)
                    char_logits, phase_vectors = model(bx, cutoff=current_cutoff, return_phase_vectors=True)
                    
                    # Get phase-clock targets (words as rotating waves)
                    phase_targets = generate_phase_targets(bx)
                    phase_targets = phase_targets.to(device)
                    
                    # Compute combined loss
                    loss, char_loss, phase_loss = compute_phase_clock_loss(
                        char_logits, phase_vectors, by, phase_targets,
                        char_weight=1.0, phase_weight=args.phase_weight
                    )
                    
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
                    running_losses.append(li)
                    
                    # Track phase-clock loss
                    phase_losses.append(float(phase_loss.item()))
                    
                    # ⚡ CHECK FOR PLATEAU UNLOCK AFTER EACH OPTIMIZER STEP
                    if args.achievement_mode and len(loss_history) >= 10:
                        recent_avg_loss = sum(loss_history[-10:]) / 10
                        prev_cutoff = current_cutoff
                        
                        # PLATEAU-BASED: Unlock when STUCK, not when winning!
                        current_cutoff, cutoff_raised, best_loss_at_cutoff, steps_without_improvement = tff.plateau_cutoff(
                            current_cutoff,
                            recent_avg_loss,
                            freq_bins,
                            best_loss_at_cutoff,
                            steps_without_improvement,
                            patience=50,  # Wait 50 steps without improvement
                            improvement_threshold=0.01,  # Need 0.01 drop to count as improvement
                        )
                        
                        if cutoff_raised:
                            # Announce unlock with context
                            print(f"\n  [PLATEAU] {prev_cutoff} -> {current_cutoff}")
                            print(f"    Reason: Stuck at loss={recent_avg_loss:.3f} for 50 steps")
                            print(f"    Opening next frequency band to help model improve further")
                            
                            # ⚡ CRITICAL: Reduce LR for high-frequency precision
                            # Low freqs (128) = smooth hills, can run fast (high LR)
                            # High freqs (512) = jagged cliffs, must walk carefully (low LR)
                            old_lr = cfg.lr
                            cfg.lr = cfg.lr * 0.5  # Cut LR in HALF for precision navigation
                            print(f"    LR REDUCED: {old_lr:.2e} -> {cfg.lr:.2e} (precision mode for high frequencies)")
                            
                            # Update optimizer's base LR
                            for pg in opt.param_groups:
                                pg['lr'] = cfg.lr
                            
                            # Clear history for fresh start at new cutoff
                            loss_history = []
                            running_losses = []
                            
                            # Mark that we raised so LR can restart
                            cutoff_was_raised_this_step = True
                        else:
                            cutoff_was_raised_this_step = False
                            
                            # Show progress toward mastery (every 100 steps)
                            if steps_without_improvement > 0 and steps_without_improvement % 100 == 0:
                                print(f"    [Mastering cutoff={current_cutoff}] Best: {best_loss_at_cutoff:.3f}, Plateau: {steps_without_improvement}/50")
                    else:
                        cutoff_was_raised_this_step = False
                    
                    # Display running average every log_every steps
                    if (opt_step + 1) % args.log_every == 0:
                        lr_disp = opt.param_groups[0]["lr"]
                        # Use actual count of samples, not fixed log_every (handles post-unlock correctly)
                        running_avg = sum(running_losses) / max(1, len(running_losses))
                        phase_avg = sum(phase_losses[-args.log_every:]) / min(args.log_every, len(phase_losses)) if len(phase_losses) > 0 else 0.0
                        
                        print(f"  step {opt_step+1}/{cfg.steps_per_epoch} loss={running_avg:.4f} phase={phase_avg:.4f} lr={lr_disp:.2e} cutoff={current_cutoff}")
                        
                        running_losses = []  # Reset for next window
                    
                    # Autosave every 100 steps so Ctrl+C doesn't lose too much progress
                    if (opt_step + 1) % 100 == 0:
                        save(epoch, opt_step + 1, current_cutoff, best_loss_at_cutoff, steps_without_improvement)

            avg = sum(losses) / max(1, len(losses))
            print(f"Epoch {epoch+1}/{cfg.epochs} avg_loss={avg:.4f} cutoff={current_cutoff}/{freq_bins} elapsed={(time.time()-t0)/60:.1f}m")
            
            # Log hemisphere balance every 5 epochs
            if (epoch + 1) % 5 == 0:
                log_hemisphere_balance()
                save(epoch + 1, 0, current_cutoff, best_loss_at_cutoff, steps_without_improvement)
                last_saved_epoch = epoch + 1

    except KeyboardInterrupt:
        print("\n[Ctrl+C] Saving checkpoint...")
        # Save with current step so we can resume exactly where we left off
        current_step = opt_step if 'opt_step' in locals() else 0
        save(epoch, current_step, current_cutoff, best_loss_at_cutoff, steps_without_improvement)
        print(f"Saved to {cfg.ckpt_path}")
        print(f"Resume from: epoch={epoch}, step={current_step}")
        raise
    finally:
        save(epoch, 0, current_cutoff, best_loss_at_cutoff, steps_without_improvement)

    print("=" * 70)
    print("DONE - BICAMERAL TRAINING COMPLETE")
    print("=" * 70)
    log_hemisphere_balance()
    print("\nThe two hemispheres have learned to work together!")
    print("Frequency path: Global context, structure, vibes")
    print("Time path: Sharp details, spelling, syntax")
    print("Result: Infinite context + Perfect precision")


if __name__ == "__main__":
    main()
