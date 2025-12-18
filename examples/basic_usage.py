"""
Basic usage examples for FFT-Tensor
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from fft_tensor import sst, zeros_sst, randn_sst, MemoryManager


def example_1_basic_creation():
    """Example 1: Create and manipulate SSTs."""
    print("\n" + "="*60)
    print("Example 1: Basic SST Creation")
    print("="*60)
    
    # Create from spatial data
    spatial = torch.randn(512, 512, device='cuda')
    tensor = sst(spatial, sparsity=0.05)
    
    print(f"Original shape: {tensor.shape}")
    print(f"Compression ratio: {tensor.compress_ratio():.1f}x")
    print(f"Memory usage: {tensor.memory_mb():.2f}MB")
    print(f"Num coefficients: {len(tensor.freq_coeffs)}")
    
    # Reconstruct
    reconstructed = tensor.to_spatial()
    error = torch.norm(reconstructed - spatial) / torch.norm(spatial)
    print(f"Reconstruction error: {error:.4f}")


def example_2_arithmetic():
    """Example 2: Arithmetic operations."""
    print("\n" + "="*60)
    print("Example 2: Arithmetic Operations")
    print("="*60)
    
    a = sst(torch.randn(256, 256, device='cuda'), sparsity=0.05)
    b = sst(torch.randn(256, 256, device='cuda'), sparsity=0.05)
    
    # Addition
    c = a + b
    print(f"Addition: {a.shape} + {b.shape} = {c.shape}")
    
    # Scalar multiplication
    d = a * 2.5
    print(f"Scalar mult: {a.shape} * 2.5 = {d.shape}")
    
    # Matrix multiplication
    e = sst(torch.randn(256, 128, device='cuda'), sparsity=0.05)
    f = a.matmul(e)
    print(f"Matmul: {a.shape} @ {e.shape} = {f.shape}")


def example_3_memory_management():
    """Example 3: Memory management."""
    print("\n" + "="*60)
    print("Example 3: Memory Management")
    print("="*60)
    
    # Set memory limit
    MemoryManager.set_limit(1000)  # 1GB limit
    print(f"Memory limit: {MemoryManager._max_memory_mb}MB")
    
    # Create several tensors
    tensors = []
    for i in range(10):
        t = sst(torch.randn(256, 256, device='cuda'), sparsity=0.05)
        tensors.append(t)
    
    # Check stats
    stats = MemoryManager.get_stats()
    print(f"Tensors created: {stats['n_tensors']}")
    print(f"Total memory: {stats['total_memory_mb']:.2f}MB")
    print(f"Utilization: {stats['utilization']*100:.1f}%")
    
    # Cleanup
    MemoryManager.clear_all()
    print("After clear_all():", MemoryManager.get_stats())


def example_4_compression_levels():
    """Example 4: Different sparsity levels."""
    print("\n" + "="*60)
    print("Example 4: Compression vs Quality")
    print("="*60)
    
    spatial = torch.randn(512, 512, device='cuda')
    
    sparsities = [0.01, 0.02, 0.05, 0.1, 0.2]
    
    print(f"{'Sparsity':<12} {'Compression':<15} {'Error':<12} {'Memory (MB)'}")
    print("-" * 55)
    
    for sparsity in sparsities:
        tensor = sst(spatial, sparsity=sparsity)
        reconstructed = tensor.to_spatial()
        error = torch.norm(reconstructed - spatial) / torch.norm(spatial)
        
        print(f"{sparsity:<12.2f} {tensor.compress_ratio():<15.1f} "
              f"{error:<12.4f} {tensor.memory_mb():.2f}")


def example_5_nd_tensors():
    """Example 5: Multi-dimensional tensors."""
    print("\n" + "="*60)
    print("Example 5: N-Dimensional Tensors")
    print("="*60)
    
    shapes = [
        (1024,),           # 1D: Audio signal
        (512, 512),        # 2D: Image
        (64, 64, 64),      # 3D: Video frame or voxel data
        (8, 16, 32, 32),   # 4D: Batch x Channels x H x W
    ]
    
    print(f"{'Shape':<25} {'Compression':<15} {'Memory (MB)'}")
    print("-" * 55)
    
    for shape in shapes:
        spatial = torch.randn(shape, device='cuda')
        tensor = sst(spatial, sparsity=0.05)
        
        print(f"{str(shape):<25} {tensor.compress_ratio():<15.1f} "
              f"{tensor.memory_mb():.2f}")


def example_6_large_matrix():
    """Example 6: Large matrix that wouldn't fit densely."""
    print("\n" + "="*60)
    print("Example 6: Large Matrix Compression")
    print("="*60)
    
    # 8K x 8K matrix would be 256MB in FP32
    # But as SST with 1% sparsity: only ~2.5MB!
    
    shape = (8192, 8192)
    dense_mb = (shape[0] * shape[1] * 4) / (1024**2)
    
    print(f"Creating {shape} matrix...")
    print(f"Dense size: {dense_mb:.1f}MB")
    
    # Create in chunks to avoid OOM
    tensor = sst(torch.randn(shape, device='cuda'), sparsity=0.01)
    
    print(f"SST size: {tensor.memory_mb():.2f}MB")
    print(f"Compression: {tensor.compress_ratio():.1f}x")
    print(f"Memory saved: {dense_mb - tensor.memory_mb():.1f}MB")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("FFT-Tensor Examples")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Examples require GPU.")
        exit(1)
    
    example_1_basic_creation()
    example_2_arithmetic()
    example_3_memory_management()
    example_4_compression_levels()
    example_5_nd_tensors()
    example_6_large_matrix()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60 + "\n")
