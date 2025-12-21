"""generate_chunk_simple.py

Simple chunk generation that auto-detects EMA from checkpoint.

Run:
  python -m scripts.generate_chunk_simple --ckpt chunklm_ckpt.pt --prompt "Once upon a time" --chunks 30
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from fft_lm import train_fixed_full as tff
from fft_lm.chunk_head import ChunkLM
from fft_lm.phase_clock import PhaseClockChunkLM


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
    ap.add_argument("--prompt", type=str, default="Once upon a time")
    ap.add_argument("--chunks", type=int, default=30)
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--top-p", type=float, default=0.92)
    ap.add_argument("--rep", type=float, default=1.15)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise SystemExit("CUDA required")

    print(f"Loading checkpoint: {args.ckpt}")
    ck = torch.load(args.ckpt, map_location="cpu")
    
    # Get config from checkpoint
    cfg_dict = ck.get("cfg", {})
    
    # Auto-detect architecture from checkpoint
    model_keys = list(ck["model"].keys())
    use_ema = any("ema" in k for k in model_keys)
    is_bicameral = ck.get("bicameral", False) or any("alpha_freq" in k for k in model_keys)
    is_frequency_native = ck.get("frequency_native", False) or (
        any("phase_weights" in k for k in model_keys) and not is_bicameral
    )
    has_phase_clock = any("phase_head" in k for k in model_keys)
    
    cfg = tff.TrainConfig(
        seq_len=cfg_dict.get("seq_len", 1024),
        kernel_len=cfg_dict.get("kernel_len", 128),
        frequency_native=is_frequency_native,
        bicameral=is_bicameral,
    )
    cfg.device = device
    
    chunk = ck.get("chunk", 16)
    
    arch_name = "BICAMERAL" if is_bicameral else ("FREQ-NATIVE" if is_frequency_native else "STANDARD")
    if has_phase_clock:
        arch_name += "+PHASE"
    print(f"Detected: seq_len={cfg.seq_len}, kernel_len={cfg.kernel_len}, chunk={chunk}")
    print(f"Architecture: {arch_name}, use_ema={use_ema}")

    backbone = tff.FixedSpectralLM(cfg).to(device)
    
    # Use phase-clock model if detected, otherwise standard
    if has_phase_clock:
        model = PhaseClockChunkLM(backbone, chunk=chunk).to(device)
    else:
        model = ChunkLM(
            backbone, 
            chunk=chunk,
            use_ema=use_ema,
            ema_chunk_len=16,
            ema_rho_init=0.95,
            ema_mode="aligned",
        ).to(device)
    
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Generating with prompt: '{args.prompt}'")
    print(f"Settings: temp={args.temperature}, top_p={args.top_p}, rep_penalty={args.rep}")
    print("=" * 70)

    ctx = list(args.prompt.encode("utf-8", errors="ignore"))
    if not ctx:
        ctx = [32]

    # Pad to seq_len
    if len(ctx) < cfg.seq_len:
        ctx = [32] * (cfg.seq_len - len(ctx)) + ctx
    else:
        ctx = ctx[-cfg.seq_len :]

    generated = ctx[:]

    # Print the prompt
    print(args.prompt, end="", flush=True)

    for c in range(args.chunks):
        x = torch.tensor([generated[-cfg.seq_len :]], dtype=torch.long, device=device)
        
        # Use full frequency detail for generation
        freq_bins = tff.conv_freq_bins(cfg.seq_len, cfg.kernel_len)
        cutoff_bins = freq_bins  # Full detail
        
        with torch.no_grad():
            logits = model(x, cutoff=cutoff_bins)[0]  # [chunk,256]

        new_bytes = []
        for i in range(chunk):
            l = logits[i].float()
            
            # Repetition penalty
            recent = generated[-256:]
            for tok in set(recent):
                l[tok] = l[tok] / args.rep
            
            # ASCII filter (allow newline + printable)
            mask = torch.ones_like(l, dtype=torch.bool)
            mask[10] = False  # newline
            mask[32:127] = False  # printable ASCII
            l[mask] = -float("inf")
            
            # Temperature
            l = l / args.temperature
            
            # Top-p
            l = apply_top_p(l, args.top_p)
            
            probs = F.softmax(l, dim=-1)
            b = int(torch.multinomial(probs, 1).item())
            new_bytes.append(b)

        generated.extend(new_bytes)

        # Print chunk
        chunk_text = bytes(new_bytes).decode("utf-8", errors="replace")
        print(chunk_text, end="", flush=True)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
