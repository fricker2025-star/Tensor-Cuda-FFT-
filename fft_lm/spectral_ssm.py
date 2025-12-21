"""fft_lm.spectral_ssm

Frequency-domain exponential moving average (EMA) a.k.a. a tiny state-space memory.

This implements a *stable* complex-valued recurrence:

  H_t = a ⊙ H_{t-1} + (1 - ρ) ⊙ F_t

where a = ρ ⊙ exp(i θ)

Key points:
  - ρ in (0,1) ensures stability (no exploding resonance)
  - θ allows phase rotation (optional but useful)
  - Designed to be used on FFT(bytes) of small chunks (e.g., 16)

We provide:
  - scan(): compute EMA over a sequence of chunk FFTs (training, no persistent state)
  - update(): update one step given a previous state (generation, persistent state)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class EMAConfig:
    n_freqs: int
    rho_init: float = 0.95
    theta_init: float = 0.0
    mode: str = "aligned"  # "aligned" | "polar"


class SpectralEMA(nn.Module):
    def __init__(self, cfg: EMAConfig):
        super().__init__()
        self.n_freqs = int(cfg.n_freqs)
        self.mode = str(cfg.mode)

        # rho in (0,1) via sigmoid(rho_logit)
        rho_init = float(cfg.rho_init)
        rho_init = min(max(rho_init, 1e-4), 1 - 1e-4)
        rho_logit = math.log(rho_init / (1 - rho_init))
        self.rho_logit = nn.Parameter(torch.full((self.n_freqs,), rho_logit, dtype=torch.float32))

        # theta in [-pi, pi] via pi*tanh(theta_raw)
        self.theta_raw = nn.Parameter(torch.full((self.n_freqs,), float(cfg.theta_init), dtype=torch.float32))

    def decay_params(self, device=None, dtype=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rho = torch.sigmoid(self.rho_logit)
        theta = math.pi * torch.tanh(self.theta_raw)
        if device is not None:
            rho = rho.to(device=device)
            theta = theta.to(device=device)
        if dtype is not None:
            rho = rho.to(dtype=dtype)
            theta = theta.to(dtype=dtype)

        a = rho * torch.exp(1j * theta)
        one_minus_rho = (1.0 - rho)
        return a, rho, one_minus_rho

    @torch.no_grad()
    def init_state(self, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros((batch, self.n_freqs), device=device, dtype=torch.complex64 if dtype == torch.float32 else torch.complex64)

    def update(self, state: torch.Tensor, fft_chunk: torch.Tensor) -> torch.Tensor:
        """Update EMA state.

        Args:
            state: [B, F] complex
            fft_chunk: [B, F] complex
        Returns:
            new_state: [B, F] complex
        """
        assert state.shape == fft_chunk.shape
        a, rho, one_minus_rho = self.decay_params(device=fft_chunk.device, dtype=torch.float32)
        a = a.to(torch.complex64)
        one_minus_rho = one_minus_rho.to(torch.float32)

        if self.mode == "polar":
            # Magnitude EMA, phase from current chunk
            m_prev = torch.abs(state).to(torch.float32)
            m_cur = torch.abs(fft_chunk).to(torch.float32)
            m_new = rho.unsqueeze(0) * m_prev + one_minus_rho.unsqueeze(0) * m_cur
            phi = torch.angle(fft_chunk).to(torch.float32)
            return m_new.to(torch.complex64) * torch.exp(1j * phi).to(torch.complex64)

        if self.mode != "aligned":
            raise ValueError(f"Unknown SpectralEMA mode: {self.mode}")

        # Phase-aligned EMA:
        # rotate previous state toward the phase of the new signal before averaging.
        # This prevents destructive interference when phases differ by ~pi.
        prev_ang = torch.angle(state).to(torch.float32)
        cur_ang = torch.angle(fft_chunk).to(torch.float32)
        rot = torch.exp(1j * (cur_ang - prev_ang)).to(torch.complex64)
        state_aligned = state * rot

        # Then do stable complex decay update
        return a.unsqueeze(0) * state_aligned + one_minus_rho.unsqueeze(0).to(torch.complex64) * fft_chunk

    def scan(self, fft_chunks: torch.Tensor, init: torch.Tensor | None = None) -> torch.Tensor:
        """EMA scan over a sequence of chunks.

        Args:
            fft_chunks: [B, S, F] complex
            init: optional [B, F] complex
        Returns:
            final_state: [B, F] complex
        """
        B, S, F = fft_chunks.shape
        assert F == self.n_freqs
        if init is None:
            state = torch.zeros((B, F), device=fft_chunks.device, dtype=torch.complex64)
        else:
            state = init
        # small loop over S (e.g., 64 for 1024/16) is fine.
        for t in range(S):
            state = self.update(state, fft_chunks[:, t, :])
        return state
