"""generate_chunked_stateful.py

Stateful chunk generation ("Piston Engine"):
  - Predict a CHUNK of bytes from the current hidden state (one forward through the head)
  - Quantization barrier (bytes are ints)
  - Re-encode in hidden space: feed the clean bytes through a streaming backbone update

This avoids re-running the backbone over the full context every chunk.

NOTE:
  The streaming backbone here implements the *finite* causal conv kernel (kernel_len)
  exactly. If you enabled any frequency-domain-only effects that extend the impulse
  response beyond kernel_len, those won't be captured.

Run:
  python generate_chunked_stateful.py --ckpt chunklm_ckpt_1024.pt --prompt "Once upon a time" --chunks 30
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


class StreamingBackbone:
    """Streaming version of the backbone that maintains per-layer conv state."""

    def __init__(self, backbone: tff.FixedSpectralLM):
        self.backbone = backbone
        self.blocks = backbone.blocks
        self.kernel_len = backbone.cfg.kernel_len
        self.d_model = backbone.cfg.d_model

        # Precompute reversed kernels per block (float32 for stability)
        self.k_rev = []
        self.gain = []
        for blk in self.blocks:
            k = blk.kernel.detach().float()
            if k.numel() != self.kernel_len:
                kk = torch.zeros(self.kernel_len)
                n = min(self.kernel_len, k.numel())
                kk[:n] = k[:n]
                k = kk
            self.k_rev.append(torch.flip(k, dims=[0]))
            self.gain.append(blk.gain.detach().float())

        # Use the backbone's parameter dtype to avoid dtype mismatches in LayerNorm.
        self.param_dtype = next(backbone.parameters()).dtype

    def init_state(self, device: str, dtype: torch.dtype):
        # ring buffers per block of normalized inputs
        bufs = [torch.zeros(1, self.kernel_len, self.d_model, device=device, dtype=dtype) for _ in self.blocks]
        return {"bufs": bufs}

    @torch.no_grad()
    def step_hidden(self, x_t: torch.Tensor, state: dict, cutoff: int | None = None) -> torch.Tensor:
        """Advance one token through all blocks.

        x_t: [1, C] float
        returns: [1, C] float
        """
        h = x_t
        for li, blk in enumerate(self.blocks):
            residual = h
            hn = blk.ln(h)

            # update buffer
            buf = state["bufs"][li]
            buf = torch.roll(buf, shifts=-1, dims=1)
            buf[:, -1, :] = hn
            state["bufs"][li] = buf

            # FIR conv
            k = self.k_rev[li].to(device=h.device, dtype=h.dtype).view(1, self.kernel_len, 1)
            y = (buf * k).sum(dim=1)
            y = y * self.gain[li].to(device=h.device, dtype=h.dtype).view(1, -1)

            # optional context gate (per token)
            if hasattr(blk, "gate_ctx"):
                g_ctx = torch.sigmoid(blk.gate_ctx(hn)).to(y.dtype)
                y = y * g_ctx

            # residual
            h = residual + y
            # FFN
            ff_in = blk.ffn_ln(h)
            h = h + blk.ffn(ff_in)

        h = self.backbone.ln_f(h)
        return h


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

    ck = torch.load(args.ckpt, map_location="cpu")

    # Build cfg/backbone
    cfg = tff.TrainConfig(seq_len=args.seq_len, kernel_len=args.kernel_len)
    cfg.device = device
    backbone = tff.FixedSpectralLM(cfg).to(device)
    model = ChunkLM(backbone, chunk=args.chunk).to(device)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()

    # Streaming state
    sb = StreamingBackbone(backbone)
    dtype = sb.param_dtype
    state = sb.init_state(device=device, dtype=dtype)

    prompt = " ".join(args.prompt)
    ctx = list(prompt.encode("utf-8", errors="ignore"))
    if not ctx:
        ctx = [32]

    # Warmup: feed prompt bytes to fill conv state and get last hidden
    h_last = torch.zeros(1, cfg.d_model, device=device, dtype=dtype)
    for b in ctx:
        emb = backbone.embed(torch.tensor([b], device=device)).to(dtype)
        h_last = sb.step_hidden(emb, state)

    generated = ctx[:]

    # chunk loop
    for _ in range(args.chunks):
        # head-only: predict chunk logits from current hidden
        flat = model.head(h_last.float())  # [1, chunk*256]
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
            # quantization barrier
            b = max(0, min(255, b))
            new_bytes.append(b)

        # print and update
        print(tff.safe_console(bytes(new_bytes).decode("utf-8", errors="replace")), end="", flush=True)
        generated.extend(new_bytes)

        # Re-encode loop in hidden space: feed clean bytes through streaming backbone
        for b in new_bytes:
            emb = backbone.embed(torch.tensor([b], device=device)).to(dtype)
            h_last = sb.step_hidden(emb, state)

    print()


if __name__ == "__main__":
    main()
