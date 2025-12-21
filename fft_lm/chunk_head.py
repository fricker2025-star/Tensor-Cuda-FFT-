"""fft_lm.chunk_head

Chunk-predicting head on top of the spectral backbone.

This enables "piston engine" generation: predict N future bytes at once.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from fft_lm.spectral_ssm import SpectralEMA, EMAConfig


class ChunkLM(nn.Module):
    """Backbone + non-autoregressive chunk head."""

    def __init__(
        self,
        backbone: nn.Module,
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

        self.use_ema = bool(use_ema)
        self.ema_chunk_len = int(ema_chunk_len)
        if self.use_ema:
            n_freqs = self.ema_chunk_len // 2 + 1
            self.ema = SpectralEMA(EMAConfig(n_freqs=n_freqs, rho_init=ema_rho_init, mode=ema_mode))
            self.ema_proj = nn.Linear(2 * n_freqs, d_model)
            nn.init.normal_(self.ema_proj.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.ema_proj.bias)

        # small init helps stability
        nn.init.normal_(self.head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor, cutoff: int | None = None) -> torch.Tensor:
        """x: [B,T] -> logits: [B,chunk,256] for the next chunk."""
        h = self.backbone.forward_hidden(x, cutoff=cutoff)  # [B,T,C]
        last = h[:, -1, :]  # [B,C]

        if self.use_ema:
            # Compute FFT(bytes) over the input window in small chunks and scan EMA.
            # This gives an "infinite memory" summary conditioned on the entire window.
            B, T = x.shape
            L = self.ema_chunk_len
            n_chunks = T // L
            if n_chunks > 0:
                xx = x[:, : n_chunks * L].reshape(B, n_chunks, L).to(torch.float32)
                # normalize bytes to [-1,1]
                xx = (xx / 127.5) - 1.0
                fft_chunks = torch.fft.rfft(xx, dim=-1)  # [B,S,F] complex
                ema_state = self.ema.scan(fft_chunks)    # [B,F] complex
                feat = torch.view_as_real(ema_state).reshape(B, -1)  # [B,2F]
                last = last + self.ema_proj(feat.to(last.dtype))

        flat = self.head(last)  # [B, chunk*256]
        return flat.view(x.size(0), self.chunk, 256)


def vectorized_windows(corpus_u8: torch.Tensor, starts: torch.Tensor, seq_len: int, chunk: int):
    """Gather x:[B,seq_len], y:[B,chunk] from CPU tensor."""
    ar = torch.arange(seq_len + chunk, dtype=torch.long)
    idx = starts[:, None].to(torch.long) + ar[None, :]
    batch = corpus_u8[idx]
    x = batch[:, :seq_len]
    y = batch[:, seq_len:]
    return x.to(torch.long), y.to(torch.long)
