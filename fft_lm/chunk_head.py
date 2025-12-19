"""fft_lm.chunk_head

Chunk-predicting head on top of the spectral backbone.

This enables "piston engine" generation: predict N future bytes at once.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ChunkLM(nn.Module):
    """Backbone + non-autoregressive chunk head."""

    def __init__(self, backbone: nn.Module, chunk: int):
        super().__init__()
        self.backbone = backbone
        self.chunk = int(chunk)
        d_model = backbone.embed.weight.shape[1]
        self.head = nn.Linear(d_model, 256 * self.chunk)

        # small init helps stability
        nn.init.normal_(self.head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor, cutoff: int | None = None) -> torch.Tensor:
        """x: [B,T] -> logits: [B,chunk,256] for the next chunk."""
        h = self.backbone.forward_hidden(x, cutoff=cutoff)  # [B,T,C]
        last = h[:, -1, :]  # [B,C]
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
