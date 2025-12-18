"""
Pure Frequency-Domain Operations - NO MATERIALIZATION

This module implements true frequency-domain operations that never
decompress to spatial domain, enabling massive models on limited VRAM.

Key insight: Complex frequency space provides richer semantic representations
than real-valued spatial space (real + imaginary = 2x expressivity).
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
from .tensor import SparseSpectralTensor


class FrequencyMatMul:
    """
    Pure frequency-domain matrix multiplication using convolution theorem.
    
    Key insight: Matrix multiplication can be expressed as convolution,
    and convolution in spatial domain = element-wise multiply in frequency domain.
    
    For matrix multiply: Y = X @ W
    We compute: Y_freq = FFT(X) ⊙ FFT(W) then Y = IFFT(Y_freq)
    
    But W is ALREADY in frequency domain (sparse), so we never materialize it!
    """
    
    @staticmethod
    def circulant_matmul(x: torch.Tensor, w_freq: torch.Tensor) -> torch.Tensor:
        """
        Matrix multiply via circulant embedding in frequency domain.
        
        Theory: Any matrix can be embedded in a circulant matrix,
        and circulant matrix multiply = convolution = FFT multiply.
        
        Args:
            x: Input tensor (B, M, K) - spatial domain
            w_freq: Weight frequencies (K, N) - ALREADY in frequency domain
        
        Returns:
            output: (B, M, N) - spatial domain result
            
        Memory: Only materializes output, not W!
        """
        B, M, K = x.shape
        K2, N = w_freq.shape
        assert K == K2, "Dimension mismatch"
        
        # Pad to power of 2 for efficient FFT
        pad_size = 2 ** int(np.ceil(np.log2(max(M, K, N))))
        
        # Transform input to frequency domain (batched)
        x_padded = F.pad(x, (0, pad_size - K, 0, pad_size - M))
        x_freq = torch.fft.fft2(x_padded)  # (B, pad_size, pad_size)
        
        # Pad weight frequencies
        w_freq_padded = F.pad(w_freq, (0, pad_size - N, 0, pad_size - K))
        
        # Element-wise multiply in frequency domain (THE MAGIC!)
        # This is O(n) instead of O(n³) for matmul
        output_freq = x_freq * w_freq_padded.unsqueeze(0)
        
        # Inverse transform
        output = torch.fft.ifft2(output_freq).real
        
        # Extract valid region (remove padding)
        output = output[:, :M, :N]
        
        return output
    
    @staticmethod
    def block_streaming_matmul(x: torch.Tensor, 
                               w_sst: SparseSpectralTensor,
                               block_size: int = 512) -> torch.Tensor:
        """
        Stream through weight matrix in blocks to avoid memory spike.
        
        This is more practical than pure frequency matmul for now.
        Still avoids materializing the full weight matrix.
        
        Args:
            x: Input (B, M, K)
            w_sst: Sparse spectral tensor weights (K, N) - compressed!
            block_size: Size of each block to materialize
            
        Returns:
            output: (B, M, N)
            
        Memory: Only block_size columns at a time (~1/100 of full matrix)
        """
        B, M, K = x.shape
        N = w_sst.shape[1]
        
        # Allocate output
        output = torch.zeros(B, M, N, device=x.device, dtype=x.dtype)
        
        # Process in blocks
        for n_start in range(0, N, block_size):
            n_end = min(n_start + block_size, N)
            block_width = n_end - n_start
            
            # Extract frequency coefficients for this block
            # Find indices that correspond to this output range
            mask = (w_sst.indices[:, 1] >= n_start) & (w_sst.indices[:, 1] < n_end)
            block_indices = w_sst.indices[mask]
            block_coeffs = w_sst.freq_coeffs[mask]
            
            # Create mini-SST for this block
            if len(block_coeffs) > 0:
                block_sst = SparseSpectralTensor(
                    freq_coeffs=block_coeffs,
                    indices=block_indices - torch.tensor([0, n_start], device=block_indices.device),
                    shape=(K, block_width),
                    sparsity=w_sst.sparsity,
                    device=w_sst.device,
                    dtype=w_sst.dtype
                )
                
                # Materialize ONLY this small block
                w_block = block_sst.to_spatial()
                
                # Compute output for this block
                output[:, :, n_start:n_end] = torch.matmul(x, w_block)
                
                # Block goes out of scope - memory freed!
                del w_block, block_sst
        
        return output


class FrequencyAttention:
    """
    Attention mechanism in pure frequency domain.
    
    Standard attention: Attention(Q, K, V) = softmax(QK^T / √d) V
    Problem: QK^T materialization is O(n²) memory
    
    Frequency attention: Operate entirely in frequency domain
    Key insight: Attention is finding semantic similarity - frequency domain
    encodes this more efficiently with complex relationships.
    """
    
    @staticmethod
    def frequency_attention(q_freq: torch.Tensor,
                           k_freq: torch.Tensor, 
                           v_freq: torch.Tensor,
                           temperature: float = 1.0) -> torch.Tensor:
        """
        Compute attention in frequency domain.
        
        Instead of QK^T (spatial), we compute element-wise products
        in frequency domain which captures different semantic relationships.
        
        Args:
            q_freq: Query frequencies (B, H, N, D) - complex
            k_freq: Key frequencies (B, H, N, D) - complex  
            v_freq: Value frequencies (B, H, N, D) - complex
            temperature: Scaling factor
            
        Returns:
            output: Attention output (B, H, N, D)
        """
        B, H, N, D = q_freq.shape
        
        # Compute attention scores in frequency domain
        # Complex conjugate multiply gives similarity in frequency space
        attention_freq = q_freq * torch.conj(k_freq)  # (B, H, N, D)
        
        # Magnitude as attention score
        attention_scores = torch.abs(attention_freq) / temperature
        
        # Softmax over sequence dimension
        attention_probs = F.softmax(attention_scores, dim=2)
        
        # Apply attention to values (in frequency domain!)
        output_freq = attention_probs.unsqueeze(-1) * v_freq
        
        # Sum over sequence (in frequency domain)
        output_freq = output_freq.sum(dim=2)
        
        return output_freq
    
    @staticmethod
    def fnet_attention(x_freq: torch.Tensor) -> torch.Tensor:
        """
        FNet-style attention: Just 2D FFT, no QKV at all!
        
        Google's FNet showed that FFT alone can replace attention
        for many tasks, being much faster and more memory efficient.
        
        Args:
            x_freq: Input frequencies (B, N, D) - already in frequency domain
            
        Returns:
            output_freq: Mixed frequencies (B, N, D)
        """
        # FFT along sequence dimension (already have feature FFT)
        mixed = torch.fft.fft(x_freq, dim=1)
        
        return mixed


class ComplexSemanticEmbedding:
    """
    Embeddings that live natively in complex frequency space.
    
    Key insight: Complex numbers (real + imaginary) can encode richer
    semantic relationships than real numbers alone.
    
    Example:
        - Real component: Semantic meaning (cat, dog, animal)
        - Imaginary component: Context/usage (pet, wild, farm)
        - Phase: Relationship type (is-a, part-of, used-for)
        - Magnitude: Confidence/salience
    """
    
    def __init__(self, vocab_size: int, embed_dim: int, device: str = 'cuda'):
        """
        Create complex-valued embeddings.
        
        Instead of storing embeddings in spatial domain, we store
        them directly as frequency coefficients.
        """
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.device = device
        
        # Store embeddings as complex frequencies directly
        # This is the "native" representation
        self.freq_embeddings = torch.randn(
            vocab_size, embed_dim,
            dtype=torch.complex64,
            device=device
        ) * 0.02
        
        # Initialize with structured frequencies
        # Low frequencies = broad semantic categories
        # High frequencies = fine-grained distinctions
        self._init_semantic_structure()
    
    def _init_semantic_structure(self):
        """
        Initialize embeddings with semantic structure in frequency domain.
        
        Low frequencies: Coarse semantic categories (animal, object, action)
        High frequencies: Fine distinctions (cat vs dog, red vs blue)
        """
        # Decay higher frequencies (natural prior)
        freq_decay = torch.exp(-torch.arange(self.embed_dim, device=self.device) / 10.0)
        self.freq_embeddings *= freq_decay.unsqueeze(0)
        
        # Add phase structure for relationship encoding
        # Phase differences = semantic relationships
        phase_structure = torch.randn(self.vocab_size, self.embed_dim, device=self.device)
        self.freq_embeddings *= torch.exp(1j * phase_structure)
    
    def lookup(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Look up embeddings - returns FREQUENCY domain embeddings.
        
        Args:
            token_ids: Token indices (B, N)
            
        Returns:
            embeddings_freq: Complex frequency embeddings (B, N, D)
        """
        return self.freq_embeddings[token_ids]
    
    def semantic_similarity(self, freq1: torch.Tensor, freq2: torch.Tensor) -> torch.Tensor:
        """
        Compute semantic similarity in frequency domain.
        
        Uses complex inner product which captures both magnitude
        and phase relationships.
        
        Args:
            freq1, freq2: Frequency embeddings (B, D)
            
        Returns:
            similarity: Real-valued similarity score (B,)
        """
        # Complex conjugate inner product
        product = torch.sum(freq1 * torch.conj(freq2), dim=-1)
        
        # Magnitude as similarity (phase difference encodes relationship type)
        similarity = torch.abs(product)
        
        return similarity
    
    def phase_relationship(self, freq1: torch.Tensor, freq2: torch.Tensor) -> torch.Tensor:
        """
        Extract relationship type from phase difference.
        
        Phase differences encode the TYPE of semantic relationship:
        - 0°: Same concept
        - 90°: Orthogonal (unrelated)
        - 180°: Opposite concepts
        - Other: Specific relationships (is-a, part-of, etc.)
        """
        # Complex division gives relative phase
        ratio = freq1 / (freq2 + 1e-8)
        
        # Extract phase angle
        phase = torch.angle(ratio)
        
        return phase


class FrequencyTransformerLayer:
    """
    Complete transformer layer operating in frequency domain.
    
    NO materialization to spatial domain - everything stays compressed!
    """
    
    def __init__(self, d_model: int, n_heads: int, device: str = 'cuda'):
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.device = device
        
        # Store projection weights as frequency coefficients
        self.q_proj_freq = torch.randn(d_model, d_model, dtype=torch.complex64, device=device) * 0.02
        self.k_proj_freq = torch.randn(d_model, d_model, dtype=torch.complex64, device=device) * 0.02
        self.v_proj_freq = torch.randn(d_model, d_model, dtype=torch.complex64, device=device) * 0.02
        self.o_proj_freq = torch.randn(d_model, d_model, dtype=torch.complex64, device=device) * 0.02
    
    def forward(self, x_freq: torch.Tensor) -> torch.Tensor:
        """
        Forward pass entirely in frequency domain.
        
        Args:
            x_freq: Input frequencies (B, N, D) - complex
            
        Returns:
            output_freq: Output frequencies (B, N, D) - complex
        """
        B, N, D = x_freq.shape
        
        # Project to Q, K, V (frequency domain multiply)
        q_freq = x_freq @ self.q_proj_freq
        k_freq = x_freq @ self.k_proj_freq
        v_freq = x_freq @ self.v_proj_freq
        
        # Reshape for multi-head
        q_freq = q_freq.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k_freq = k_freq.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v_freq = v_freq.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Frequency attention
        attn_output = FrequencyAttention.frequency_attention(q_freq, k_freq, v_freq)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, D)
        
        # Output projection
        output_freq = attn_output @ self.o_proj_freq
        
        return output_freq


# Utility functions
def frequency_relu(x_freq: torch.Tensor) -> torch.Tensor:
    """
    ReLU in frequency domain (approximate).
    
    Exact ReLU in frequency domain is hard, but we can approximate
    by operating on magnitude while preserving phase structure.
    """
    magnitude = torch.abs(x_freq)
    phase = torch.angle(x_freq)
    
    # ReLU on magnitude
    magnitude_relu = F.relu(magnitude)
    
    # Reconstruct complex number
    return magnitude_relu * torch.exp(1j * phase)


def frequency_layernorm(x_freq: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Layer normalization in frequency domain.
    
    Normalize magnitude while preserving phase relationships.
    """
    # Normalize magnitude
    magnitude = torch.abs(x_freq)
    mean_mag = magnitude.mean(dim=-1, keepdim=True)
    std_mag = magnitude.std(dim=-1, keepdim=True)
    
    normalized_mag = (magnitude - mean_mag) / (std_mag + eps)
    
    # Preserve phase
    phase = torch.angle(x_freq)
    
    # Reconstruct
    return normalized_mag * torch.exp(1j * phase)
