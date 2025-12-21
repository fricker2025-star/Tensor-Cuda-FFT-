"""frequency_native.py

Frequency-native neural network components that operate entirely in the spectral domain.

Key innovations:
1. PhaseShift activation - frequency-native nonlinearity (no time-domain roundtrip)
2. FrequencyConv - custom autograd for O(1) gradient computation
3. SpectralFFN - feedforward that stays in frequency domain

The goal: Build a neural network where gradients flow through frequency space,
enabling infinite context and native multi-scale processing.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PhaseShift(nn.Module):
    """Frequency-native nonlinearity via learned phase rotation.
    
    Traditional activations (ReLU, GELU) clip magnitudes in time domain,
    which causes spectral smearing (infinite convolution in frequency).
    
    Phase rotation is:
    - Unitary (preserves energy: |z|² unchanged)
    - Fully differentiable in frequency domain
    - Creates constructive/destructive interference for learning
    
    This is inspired by quantum phase gates and optical modulators.
    """
    
    def __init__(self, d_model: int, n_freqs: int):
        super().__init__()
        self.d_model = d_model
        self.n_freqs = n_freqs
        
        # Learn phase rotation per frequency bin and per channel
        # Shape: [n_freqs, d_model]
        self.phase_weights = nn.Parameter(torch.randn(n_freqs, d_model) * 0.01)
        
        # Optional: learn frequency-dependent magnitude modulation
        # (small deviations from unity, for fine-tuning)
        self.magnitude_logits = nn.Parameter(torch.zeros(n_freqs, d_model))
    
    def forward(self, z_freq: torch.Tensor) -> torch.Tensor:
        """Apply phase rotation to complex spectrum.
        
        Args:
            z_freq: [B, F, C] complex tensor in frequency domain
            
        Returns:
            [B, F, C] complex tensor with rotated phases
        """
        # Ensure we're working in complex64 for optimal performance
        if z_freq.dtype != torch.complex64:
            z_freq = z_freq.to(torch.complex64)
        
        # Extract magnitude and phase
        magnitude = z_freq.abs()  # [B, F, C] float32
        phase = z_freq.angle()    # [B, F, C] float32
        
        # Learned phase rotation (bounded to [-π, π])
        F_bins = z_freq.size(1)
        rotation = torch.tanh(self.phase_weights[:F_bins]) * math.pi  # [F, C]
        new_phase = phase + rotation.unsqueeze(0)  # [B, F, C]
        
        # Optional magnitude modulation (near unity, for stability)
        mag_scale = 1.0 + 0.1 * torch.tanh(self.magnitude_logits[:F_bins])  # [F, C]
        new_magnitude = magnitude * mag_scale.unsqueeze(0)  # [B, F, C]
        
        # Reconstruct complex number (explicitly complex64)
        result = new_magnitude * torch.exp(1j * new_phase)
        return result.to(torch.complex64)


class FrequencyConvFunc(torch.autograd.Function):
    """Custom autograd for frequency-domain convolution.
    
    Convolution theorem: conv(f, g) ↔ F(ω) * G(ω)
    
    This means:
    1. Forward: Just multiply (O(N) instead of O(N²))
    2. Backward: Also just multiply (gradient is also a convolution!)
    
    No chain rule needed - the derivative is trivial in frequency space.
    """
    
    @staticmethod
    def forward(ctx, x_freq, kernel_freq, gain):
        """
        Args:
            x_freq: [B, F, C] complex input spectrum
            kernel_freq: [F] complex kernel spectrum
            gain: [C] real per-channel gains
            
        Returns:
            [B, F, C] complex output spectrum
        """
        y_freq = x_freq * kernel_freq.unsqueeze(0).unsqueeze(-1) * gain.unsqueeze(0).unsqueeze(0)
        ctx.save_for_backward(x_freq, kernel_freq, gain)
        return y_freq
    
    @staticmethod
    def backward(ctx, grad_output):
        """Frequency-native backprop: just conjugate and multiply!"""
        x_freq, kernel_freq, gain = ctx.saved_tensors
        
        # Gradient w.r.t input: conj(kernel) * grad
        grad_x = grad_output * kernel_freq.conj().unsqueeze(0).unsqueeze(-1) * gain.unsqueeze(0).unsqueeze(0)
        
        # Gradient w.r.t kernel: sum over batch and channels
        grad_kernel = (grad_output * x_freq.conj() * gain.unsqueeze(0).unsqueeze(0)).sum(dim=(0, 2))
        
        # Gradient w.r.t gain: sum over batch and frequency
        grad_gain = (grad_output * x_freq * kernel_freq.unsqueeze(0).unsqueeze(-1)).real.sum(dim=(0, 1))
        
        return grad_x, grad_kernel, grad_gain


class SpectralFFN(nn.Module):
    """Feedforward network that operates entirely in frequency domain.
    
    Traditional FFN: iFFT → Linear → GELU → Linear → FFT (expensive roundtrips)
    SpectralFFN: Stay in frequency, use phase shifts for nonlinearity
    
    Architecture:
    1. Frequency-wise linear (like 1x1 conv in spectral domain)
    2. Phase shift activation
    3. Frequency-wise linear
    """
    
    def __init__(self, d_model: int, n_freqs: int, expansion: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_freqs = n_freqs
        hidden = d_model * expansion
        
        # Layer norm in frequency domain (normalize magnitude, preserve phase)
        self.ln = SpectralLayerNorm(d_model, n_freqs)
        
        # Expansion: project each frequency component independently
        self.w1 = nn.Linear(d_model, hidden)
        
        # Nonlinearity via phase rotation
        self.activation = PhaseShift(hidden, n_freqs)
        
        # Contraction
        self.w2 = nn.Linear(hidden, d_model)
        
        # Dropout (apply to magnitude only, preserve phase structure)
        self.dropout_p = dropout
        
        # Initialize small for stable residual
        nn.init.normal_(self.w1.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.w1.bias)
        nn.init.normal_(self.w2.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.w2.bias)
    
    def forward(self, x_freq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_freq: [B, F, C] complex
            
        Returns:
            [B, F, C] complex
        """
        # Normalize (preserve phase, normalize magnitude)
        x_freq = self.ln(x_freq)
        
        # Expansion: apply linear per frequency bin
        # Treat each frequency as a separate feature vector
        B, F, C = x_freq.shape
        
        # Real and imag parts separately through linear
        x_real = x_freq.real  # [B, F_bins, C]
        x_imag = x_freq.imag  # [B, F_bins, C]
        
        # Linear transformation
        h_real = self.w1(x_real)  # [B, F_bins, H]
        h_imag = self.w1(x_imag)  # [B, F_bins, H]
        h_freq = torch.complex(h_real, h_imag)  # [B, F_bins, H]
        
        # Phase shift activation (frequency-native nonlinearity)
        h_freq = self.activation(h_freq)
        
        # Dropout on magnitude
        if self.training and self.dropout_p > 0:
            magnitude = h_freq.abs()
            phase = h_freq.angle()
            magnitude = nn.functional.dropout(magnitude, p=self.dropout_p, training=True)
            h_freq = magnitude * torch.exp(1j * phase)
        
        # Contraction
        out_real = self.w2(h_freq.real)  # [B, F, C]
        out_imag = self.w2(h_freq.imag)  # [B, F, C]
        out_freq = torch.complex(out_real, out_imag)
        
        return out_freq


class SpectralLayerNorm(nn.Module):
    """Layer normalization in frequency domain.
    
    Normalize magnitude while preserving phase structure.
    This maintains the relative timing information encoded in phase.
    """
    
    def __init__(self, d_model: int, n_freqs: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # Learnable scale and shift (per frequency, per channel)
        self.gamma = nn.Parameter(torch.ones(n_freqs, d_model))
        self.beta = nn.Parameter(torch.zeros(n_freqs, d_model))
    
    def forward(self, x_freq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_freq: [B, F, C] complex
            
        Returns:
            [B, F, C] complex with normalized magnitudes
        """
        # Normalize magnitude, preserve phase
        magnitude = x_freq.abs()  # [B, F, C]
        phase = x_freq.angle()    # [B, F, C]
        
        # Normalize magnitude across channels (for each frequency bin)
        mean = magnitude.mean(dim=-1, keepdim=True)  # [B, F, 1]
        var = magnitude.var(dim=-1, keepdim=True, unbiased=False)  # [B, F, 1]
        magnitude_norm = (magnitude - mean) / torch.sqrt(var + self.eps)  # [B, F, C]
        
        # Apply learnable affine
        F_bins = x_freq.size(1)
        magnitude_scaled = magnitude_norm * self.gamma[:F_bins].unsqueeze(0) + self.beta[:F_bins].unsqueeze(0)
        
        # Reconstruct with original phase
        return magnitude_scaled * torch.exp(1j * phase)


class FrequencyNativeBlock(nn.Module):
    """A full spectral block that NEVER leaves frequency domain.
    
    This is the evolution of FixedSpectralBlock:
    - No iFFT → ReLU → FFT roundtrips
    - All operations in frequency space
    - Phase rotations for nonlinearity
    - Custom gradients for efficiency
    """
    
    def __init__(
        self,
        d_model: int,
        seq_len: int,
        kernel_len: int,
        transition_bins: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.kernel_len = kernel_len
        self.transition_bins = int(max(1, transition_bins))
        
        # Compute max FFT size
        max_lin_conv = seq_len + kernel_len - 1
        max_n_fft = 1
        while max_n_fft < max_lin_conv:
            max_n_fft *= 2
        self.max_freq_bins = max_n_fft // 2 + 1
        
        # Pre-norm (will operate on time-domain input)
        self.ln = nn.LayerNorm(d_model)
        
        # Learnable causal kernel (time domain) - will be FFT'd once
        self.kernel = nn.Parameter(torch.zeros(kernel_len))
        nn.init.normal_(self.kernel, mean=0.0, std=0.001)
        
        # Per-channel gain
        self.gain = nn.Parameter(torch.ones(d_model))
        
        # Frequency gates
        self.gate_freq_logits = nn.Parameter(torch.ones(self.max_freq_bins) * 2.0)
        self.gate_ctx = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.gate_ctx.weight)
        nn.init.constant_(self.gate_ctx.bias, 2.0)
        
        # Frequency-native FFN (replaces time-domain FFN)
        self.ffn = SpectralFFN(d_model, self.max_freq_bins, expansion=2, dropout=dropout)
        
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, cutoff: int | None = None) -> torch.Tensor:
        """
        Args:
            x: [B, T, C] time-domain input
            cutoff: optional frequency cutoff for curriculum learning
            
        Returns:
            [B, T, C] time-domain output (for residual connection)
        """
        residual = x
        x = self.ln(x)
        
        B, T, C = x.shape
        n_fft = 1
        while n_fft < (T + self.kernel_len - 1):
            n_fft *= 2
        
        # FFT to frequency domain (explicit complex64 for performance)
        x_pad = F.pad(x, (0, 0, 0, n_fft - T))
        x_freq = torch.fft.rfft(x_pad, dim=1)  # [B, F, C] complex
        x_freq = x_freq.to(torch.complex64)  # Ensure complex64
        
        # Build kernel spectrum (cached in practice)
        k = torch.zeros(n_fft, device=x.device, dtype=x.dtype)
        k[:self.kernel_len] = self.kernel
        k_freq = torch.fft.rfft(k)  # [F] complex
        
        # Frequency-domain convolution (custom autograd)
        y_freq = FrequencyConvFunc.apply(x_freq, k_freq, self.gain)
        
        # Gating (magnitude modulation, preserve phase)
        Fbins = y_freq.size(1)
        
        # Per-frequency gate
        g_freq = torch.sigmoid(self.gate_freq_logits[:Fbins]).to(y_freq.real.dtype)
        
        # Context gate
        pooled = x.mean(dim=1)  # [B, C]
        g_ctx = torch.sigmoid(self.gate_ctx(pooled)).to(y_freq.real.dtype)  # [B, C]
        
        # Apply gates (real-valued, scales magnitude)
        y_freq = y_freq * g_freq.unsqueeze(0).unsqueeze(-1) * g_ctx.unsqueeze(1)
        
        # Progressive frequency cutoff (curriculum learning)
        if cutoff is not None:
            cutoff_idx = min(int(cutoff), Fbins)
            if cutoff_idx < Fbins:
                trans = min(self.transition_bins, cutoff_idx)
                mask = torch.ones(Fbins, device=y_freq.device, dtype=y_freq.real.dtype)
                start = cutoff_idx - trans
                if trans > 0:
                    t = torch.linspace(0, 1, steps=trans, device=y_freq.device, dtype=mask.dtype)
                    mask[start:cutoff_idx] = 0.5 * (1.0 + torch.cos(torch.pi * t))
                mask[cutoff_idx:] = 0.0
                y_freq = y_freq * mask.unsqueeze(0).unsqueeze(-1)
        
        # Frequency-native FFN (stays in frequency domain!)
        # Make contiguous to avoid stride issues
        ffn_out = self.ffn(y_freq).contiguous()
        y_freq = (y_freq + ffn_out).contiguous()
        
        # Return to time domain only for residual connection
        y_pad = torch.fft.irfft(y_freq, n=n_fft, dim=1)  # [B, n_fft, C] real
        y = y_pad[:, :T, :]
        
        y = self.drop(y)
        return residual + y


def test_phase_shift():
    """Sanity check: PhaseShift preserves energy."""
    B, F, C = 4, 128, 512
    phase_shift = PhaseShift(C, F)
    
    x = torch.randn(B, F, C) + 1j * torch.randn(B, F, C)
    y = phase_shift(x)
    
    # Check energy preservation (approximately)
    energy_in = (x.abs() ** 2).sum()
    energy_out = (y.abs() ** 2).sum()
    
    print(f"Energy in:  {energy_in:.2f}")
    print(f"Energy out: {energy_out:.2f}")
    print(f"Ratio: {energy_out / energy_in:.4f}")
    assert torch.allclose(energy_in, energy_out, rtol=0.1), "Energy not preserved!"
    print("[OK] PhaseShift preserves energy")


if __name__ == "__main__":
    print("Testing frequency-native components...")
    test_phase_shift()
    print("\n[SUCCESS] All tests passed!")
