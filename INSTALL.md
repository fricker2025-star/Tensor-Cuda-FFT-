# Installation Guide

## Prerequisites

### Hardware
- NVIDIA GPU with compute capability 7.5+ (Turing or newer)
  - Tested on: GTX 1660 Super, RTX 2060, RTX 3060, RTX 4090
- Minimum 6GB VRAM recommended

### Software
- Python 3.8 or higher
- CUDA Toolkit 11.0 or higher
- C++14 compatible compiler
  - Linux: GCC 7.0+
  - Windows: Visual Studio 2019+

## Installation Steps

### Option 1: Install from Source (Recommended)

```bash
# Clone repository
git clone https://github.com/yourusername/fft-tensor.git
cd fft-tensor

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install FFT-Tensor (this will compile CUDA extensions)
pip install -e .
```

The installation will automatically:
1. Detect your CUDA version
2. Compile CUDA kernels for your GPU architecture
3. Link against cuFFT
4. Install Python bindings

### Option 2: Install from PyPI (when available)

```bash
pip install fft-tensor
```

## Verify Installation

```python
import torch
from fft_tensor import sst, MemoryManager

# Check CUDA availability
assert torch.cuda.is_available(), "CUDA not available"

# Test basic functionality
data = torch.randn(256, 256, device='cuda')
tensor = sst(data, sparsity=0.05)

print(f"✓ FFT-Tensor installed successfully!")
print(f"  Compression: {tensor.compress_ratio():.1f}x")
print(f"  Memory: {tensor.memory_mb():.2f}MB")

# Check CUDA backend
try:
    import fft_tensor_cuda
    print(f"✓ CUDA backend loaded")
except ImportError:
    print(f"⚠ CUDA backend not available (using PyTorch fallback)")
```

## Troubleshooting

### CUDA Extension Build Fails

**Error:** `nvcc not found`
- **Solution:** Ensure CUDA Toolkit is installed and `nvcc` is in PATH
- Linux: `export PATH=/usr/local/cuda/bin:$PATH`
- Windows: Add `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin` to PATH

**Error:** `torch/extension.h not found`
- **Solution:** Reinstall PyTorch with CUDA support
  ```bash
  pip uninstall torch
  pip install torch --index-url https://download.pytorch.org/whl/cu118
  ```

**Error:** `compute capability X.Y not supported`
- **Solution:** Your GPU is too old. FFT-Tensor requires compute capability 7.5+
- Check your GPU: https://developer.nvidia.com/cuda-gpus

### Runtime Errors

**Error:** `CUDA out of memory`
- **Solution:** Reduce batch size or increase sparsity
  ```python
  MemoryManager.set_limit(4000)  # Lower memory limit
  tensor = sst(data, sparsity=0.02)  # Higher sparsity = more compression
  ```

**Error:** `cuFFT error`
- **Solution:** Update NVIDIA drivers
- Check: `nvidia-smi` and `nvcc --version` match

### Performance Issues

If FFT-Tensor is slower than expected:
1. Ensure CUDA backend is loading (check `import fft_tensor_cuda`)
2. Use GPU with compute capability 7.5+ for optimal performance
3. Enable tensor cores on Ampere+ GPUs
4. Profile with: `nvprof python your_script.py`

## Platform-Specific Notes

### Linux
- Install CUDA Toolkit from NVIDIA website or package manager
- May need to set `LD_LIBRARY_PATH`:
  ```bash
  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
  ```

### Windows
- Install Visual Studio 2019 or newer (Community Edition works)
- Install CUDA Toolkit from NVIDIA website
- Use x64 Native Tools Command Prompt for building

### WSL2
- Full CUDA support available in WSL2
- Follow Linux instructions
- Ensure Windows NVIDIA drivers are up to date

## Building from Source (Advanced)

For development or custom builds:

```bash
# Clone with submodules (if any)
git clone --recursive https://github.com/yourusername/fft-tensor.git

# Build with specific CUDA architecture
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6"
python setup.py build_ext --inplace

# Run tests
pytest tests/ -v
```

## Docker (Recommended for Reproducibility)

```dockerfile
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y python3-pip
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118

COPY . /workspace/fft-tensor
WORKDIR /workspace/fft-tensor
RUN pip3 install -e .

CMD ["python3", "examples/basic_usage.py"]
```

Build and run:
```bash
docker build -t fft-tensor .
docker run --gpus all fft-tensor
```

## Next Steps

- Read the [README](README.md) for usage examples
- Try [basic_usage.py](examples/basic_usage.py)
- Explore [neural_network.py](examples/neural_network.py)
- Check [API documentation](docs/)

## Getting Help

- **Issues:** https://github.com/yourusername/fft-tensor/issues
- **Discussions:** https://github.com/yourusername/fft-tensor/discussions
- **Email:** your.email@example.com
