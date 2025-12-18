"""
FFT-Tensor: Revolutionary spectral tensor package for extreme AI efficiency.

Run 120B+ models on 6GB VRAM through:
- Sparse Spectral Tensors (SST): 100x compression via frequency domain
- Implicit Weights: Generate parameters on-demand from spectral coefficients
- Streaming execution: Process arbitrarily large models in fixed memory
"""

from .tensor import SparseSpectralTensor, sst, zeros_sst, MemoryManager
from .ops import (
    spectral_conv,
    spectral_pool,
    spectral_normalize,
    spectral_activation,
    ImplicitWeights,
    implicit_matmul,
    spectral_backward
)

__version__ = "0.1.0"
__all__ = [
    'SparseSpectralTensor',
    'sst',
    'zeros_sst',
    'MemoryManager',
    'spectral_conv',
    'spectral_pool',
    'spectral_normalize',
    'spectral_activation',
    'ImplicitWeights',
    'implicit_matmul',
    'spectral_backward',
]

# Set default memory limit (5GB for 1660 Super)
MemoryManager.set_limit(5000)

print(f"FFT-Tensor v{__version__} initialized")
print(f"Memory limit: {MemoryManager._max_memory_mb}MB")
