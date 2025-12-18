"""
FFT-Tensor Setup
Production build system for CUDA extensions
"""

import os
import sys
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

# Check CUDA availability
if not torch.cuda.is_available():
    print("WARNING: CUDA not available. This package requires CUDA.")
    sys.exit(1)

# Get CUDA arch for GTX 1660 Super (Turing, compute capability 7.5)
cuda_arch_list = os.environ.get('TORCH_CUDA_ARCH_LIST', '7.5;8.0;8.6')

# Define CUDA extension
cuda_extension = CUDAExtension(
    name='fft_tensor_cuda',
    sources=[
        'fft-tensor/cuda/fft_ops.cu',
        'fft-tensor/cuda/kernels.cu',
    ],
    include_dirs=[
        'fft-tensor/cuda',
    ],
    extra_compile_args={
        'cxx': [
            '-O3',
            '-std=c++14',
        ],
        'nvcc': [
            '-O3',
            f'-gencode=arch=compute_75,code=sm_75',  # GTX 1660 Super
            '-gencode=arch=compute_80,code=sm_80',    # Ampere
            '-gencode=arch=compute_86,code=sm_86',    # RTX 30xx
            '--use_fast_math',
            '-std=c++14',
            '--expt-relaxed-constexpr',
            '-Xcompiler', '-fPIC',
            # Optimization flags
            '-lineinfo',  # For profiling
            '--ptxas-options=-v',  # Verbose register usage
        ]
    },
    libraries=['cufft'],  # Link cuFFT
)

# Read README
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='fft-tensor',
    version='0.1.0',
    author='Aaron',
    description='Revolutionary sparse spectral tensor package for extreme AI efficiency',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/fft-tensor',
    packages=find_packages(exclude=['tests', 'examples', 'docs']),
    ext_modules=[cuda_extension],
    cmdclass={'build_ext': BuildExtension},
    install_requires=requirements,
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: C++',
        'Programming Language :: CUDA',
    ],
    keywords='fft tensor cuda deep-learning neural-networks optimization',
    project_urls={
        'Bug Reports': 'https://github.com/yourusername/fft-tensor/issues',
        'Source': 'https://github.com/yourusername/fft-tensor',
        'Documentation': 'https://fft-tensor.readthedocs.io',
    },
)
