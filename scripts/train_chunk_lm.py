"""scripts.train_chunk_lm

Single-file training entrypoint that contains BOTH:
  - the chunk head ("mouth")
  - the end-to-end training loop updating backbone + head together

It imports the backbone implementation from `fft_lm.train_fixed_full`.

Run from repo root:
  python -m scripts.train_chunk_lm --seq-len 1024 --kernel-len 128 --chunk 16 --batch-size 4 --accum-steps 8 \
    --steps-per-epoch 1000 --epochs 50 --lr 0.0002 --ckpt chunklm_ckpt_1024.pt --log-every 10
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.nn as nn

from fft_lm import train_fixed_full as tff
from fft_lm.spectral_ssm import SpectralEMA, EMAConfig
from fft_lm.ckpt_io import save_checkpoint, load_checkpoint


class ChunkLM(nn.Module):
    """Backbone + non-autoregressive chunk head (predicts N future bytes at once)."""

    def __init__(
        self,
        backbone: tff.FixedSpectralLM,
        chunk: int,
        *,
        use_ema: bool = False,
        ema_chunk_len: int = 16,
        ema_rho_init: float = 0.95,
        ema_mode: str = "aligned",
    ):
        super().__init__()
        self.backbone = backbone
        self.chunk = int(chunk)
        d_model = backbone.embed.weight.shape[1]
        self.head = nn.Linear(d_model, 256 * self.chunk)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.head.bias)

        self.use_ema = bool(use_ema)
        self.ema_chunk_len = int(ema_chunk_len)
        if self.use_ema:
            n_freqs = self.ema_chunk_len // 2 + 1
            self.ema = SpectralEMA(EMAConfig(n_freqs=n_freqs, rho_init=ema_rho_init, mode=ema_mode))
            self.ema_proj = nn.Linear(2 * n_freqs, d_model)
            nn.init.normal_(self.ema_proj.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.ema_proj.bias)

    def forward(self, x: torch.Tensor, cutoff: int | None = None) -> torch.Tensor:
        # hidden from backbone
        h = self.backbone.forward_hidden(x, cutoff=cutoff)  # [B,T,C]
        last = h[:, -1, :]

        if self.use_ema:
            B, T = x.shape
            L = self.ema_chunk_len
            S = T // L
            if S > 0:
                xx = x[:, : S * L].reshape(B, S, L).to(torch.float32)
                xx = (xx / 127.5) - 1.0
                fft_chunks = torch.fft.rfft(xx, dim=-1)  # [B,S,F] complex
                ema_state = self.ema.scan(fft_chunks)    # [B,F] complex
                feat = torch.view_as_real(ema_state).reshape(B, -1)
                last = last + self.ema_proj(feat.to(last.dtype))

        flat = self.head(last)
        return flat.view(x.size(0), self.chunk, 256)


def vectorized_windows(corpus_u8: torch.Tensor, starts: torch.Tensor, seq_len: int, chunk: int):
    """Gather x:[B,seq_len], y:[B,chunk] from CPU tensor."""
    ar = torch.arange(seq_len + chunk, dtype=torch.long)
    idx = starts[:, None].to(torch.long) + ar[None, :]
    batch = corpus_u8[idx]
    x = batch[:, :seq_len]
    y = batch[:, seq_len:]
    return x.to(torch.long), y.to(torch.long)


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
    ap.add_argument("--use-ema", action="store_true")
    ap.add_argument("--ema-chunk-len", type=int, default=16)
    ap.add_argument("--ema-rho-init", type=float, default=0.95)
    ap.add_argument("--ema-mode", type=str, default="aligned", choices=["aligned", "polar"])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise SystemExit("CUDA required")
    if not os.path.exists(args.data):
        raise SystemExit(f"Missing data: {args.data}")

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

    # Auto-detect EMA from checkpoint if resuming
    use_ema = args.use_ema
    ema_chunk_len = args.ema_chunk_len
    ema_rho_init = args.ema_rho_init
    ema_mode = args.ema_mode
    
    if args.resume and os.path.exists(cfg.ckpt_path):
        ck = load_checkpoint(cfg.ckpt_path, map_location="cpu")
        # Check if checkpoint has EMA parameters
        model_keys = list(ck["model"].keys())
        ckpt_has_ema = any("ema" in k for k in model_keys)
        if ckpt_has_ema:
            print(f"Detected EMA in checkpoint, enabling EMA for model")
            use_ema = True
    
    backbone = tff.FixedSpectralLM(cfg).to(device)
    model = ChunkLM(
        backbone,
        chunk=args.chunk,
        use_ema=use_ema,
        ema_chunk_len=ema_chunk_len,
        ema_rho_init=ema_rho_init,
        ema_mode=ema_mode,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)
    loss_fn = nn.CrossEntropyLoss()

    start_epoch = 0
    if args.resume and os.path.exists(cfg.ckpt_path):
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
            },
            cfg.ckpt_path,
        )

    print("=" * 70)
    print("TRAIN CHUNK LM (single script: body + mouth)")
    print("=" * 70)
    print(f"device={device} seq_len={cfg.seq_len} kernel_len={cfg.kernel_len} chunk={args.chunk}")
    print(f"micro_batch={cfg.batch_size} accum={cfg.accum_steps} effective={cfg.batch_size*cfg.accum_steps}")
    print(f"steps/epoch={cfg.steps_per_epoch} epochs={cfg.epochs} lr={cfg.lr} wd={cfg.weight_decay}")
    print(f"ema: use={args.use_ema} mode={args.ema_mode} ema_chunk_len={args.ema_chunk_len} ema_rho_init={args.ema_rho_init}")
    if not args.no_sawtooth:
        print(
            f"LR sched: sawtooth cosine restarts (stage-aligned)"
            f"  s1(e0-{cfg.stage1_epochs-1}) mult {cfg.stage1_lr_mult}->{cfg.stage1_min_mult}"
            f"  s2(e{cfg.stage1_epochs}-{cfg.stage1_epochs+cfg.stage2_epochs-1}) mult {cfg.stage2_lr_mult}->{cfg.stage2_min_mult}"
            f"  s3(e{cfg.stage1_epochs+cfg.stage2_epochs}+) mult {cfg.stage3_lr_mult}->{cfg.stage3_min_mult}"
        )
    print(f"corpus_bytes={n:,}")
    print(f"params={sum(p.numel() for p in model.parameters()):,}")
    print(f"ckpt={cfg.ckpt_path}")
    print("=" * 70)

    t0 = time.time()
    last_saved_epoch = start_epoch
    
    # Adaptive cutoff state
    freq_bins = tff.conv_freq_bins(cfg.seq_len, cfg.kernel_len)
    
    # IMPORTANT: When resuming, infer the current cutoff from the epoch we're at
    # Don't regress to a lower cutoff than what the model has already learned!
    if start_epoch >= 1:
        # Model has already proven it can handle higher frequencies
        # Start at the curriculum level it should be at
        current_cutoff = tff.curriculum_cutoff(start_epoch, cfg, freq_bins)
        print(f"Resuming at epoch {start_epoch}: inferred cutoff={current_cutoff} (don't regress!)")
    else:
        # Start at 128 (64 is too blurry, no point)
        current_cutoff = 128
        print(f"Starting at cutoff=128 (basic syntax/characters)")
    
    loss_history = []  # Track all optimizer step losses for plateau detection
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
                print(f"🚀 CUTOFF RAISED: {prev_cutoff} -> {current_cutoff} (loss plateaued, opening frequency gates)")
                # Clear old history to restart plateau detection
                loss_history = []

            opt.zero_grad(set_to_none=True)
            losses = []
            running = 0.0

            micro_total = cfg.steps_per_epoch * cfg.accum_steps
            for micro in range(micro_total):
                opt_step = micro // cfg.accum_steps
                
                # Reset cutoff_raised flag after first step of epoch
                if opt_step > 0:
                    cutoff_was_raised_this_step = False
                else:
                    cutoff_was_raised_this_step = cutoff_raised

                # LR schedule on optimizer steps - RESTART when cutoff raised
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
                    loss_history.append(li)  # Track for plateau detection
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

    print("DONE")


if __name__ == "__main__":
    main()
