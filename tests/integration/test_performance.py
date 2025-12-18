"""
Performance and integration tests for FFT-Tensor
"""
import torch
import pytest
import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fft_tensor.tensor import SparseSpectralTensor, sst, MemoryManager


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestPerformance:
    """Performance benchmarks."""
    
    def test_fft_performance(self):
        """Benchmark FFT performance vs PyTorch."""
        sizes = [(256, 256), (512, 512), (1024, 1024)]
        
        for size in sizes:
            spatial = torch.randn(size, device='cuda')
            
            # Time SST creation (includes FFT + sparsification)
            torch.cuda.synchronize()
            start = time.time()
            tensor = sst(spatial, sparsity=0.05)
            torch.cuda.synchronize()
            sst_time = time.time() - start
            
            # Time PyTorch FFT only
            torch.cuda.synchronize()
            start = time.time()
            _ = torch.fft.fftn(spatial)
            torch.cuda.synchronize()
            torch_time = time.time() - start
            
            print(f"\n{size}: SST={sst_time*1000:.1f}ms, PyTorch={torch_time*1000:.1f}ms")
            
            # SST should be comparable or faster with CUDA backend
            # Allow 3x overhead for sparsification
            assert sst_time < torch_time * 3
    
    def test_memory_efficiency(self):
        """Test memory efficiency vs dense tensors."""
        shape = (1024, 1024)
        
        # Dense tensor memory
        dense = torch.randn(shape, device='cuda', dtype=torch.float32)
        dense_mb = dense.element_size() * dense.numel() / (1024**2)
        
        # SST memory
        tensor = sst(dense, sparsity=0.05)
        sst_mb = tensor.memory_mb()
        
        print(f"\nDense: {dense_mb:.1f}MB, SST: {sst_mb:.1f}MB, "
              f"Ratio: {dense_mb/sst_mb:.1f}x")
        
        # Should be much smaller
        assert sst_mb < dense_mb / 5
    
    def test_large_model_simulation(self):
        """Simulate running a large model layer."""
        # Simulate 120B model layer: (12288, 12288) weights
        # This would be 576MB in FP32, but only ~29MB at 5% sparsity
        
        weight_shape = (4096, 4096)  # Smaller for testing
        input_shape = (32, 4096)     # Batch of 32
        
        # Create implicit weights as SST
        weights_sst = sst(torch.randn(weight_shape, device='cuda'), sparsity=0.05)
        input_sst = sst(torch.randn(input_shape, device='cuda'), sparsity=0.05)
        
        # Matrix multiply
        torch.cuda.synchronize()
        start = time.time()
        output = input_sst.matmul(weights_sst)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        print(f"\nLarge matmul ({input_shape} @ {weight_shape}): {elapsed*1000:.1f}ms")
        print(f"Output memory: {output.memory_mb():.1f}MB")
        
        assert output.shape == (32, 4096)
    
    def test_streaming_memory_usage(self):
        """Test memory stays bounded with many operations."""
        MemoryManager.clear_all()
        initial_stats = MemoryManager.get_stats()
        
        # Perform many operations
        device = 'cuda'
        for i in range(50):
            a = sst(torch.randn(256, 256, device=device), sparsity=0.05)
            b = sst(torch.randn(256, 256, device=device), sparsity=0.05)
            c = a + b
            _ = c.to_spatial()
            
            # Explicitly cleanup every 10 iterations
            if i % 10 == 0:
                del a, b, c
                MemoryManager.clear_all()
        
        final_stats = MemoryManager.get_stats()
        print(f"\nMemory utilization: {final_stats['utilization']*100:.1f}%")
        
        # Should not have grown unbounded
        assert final_stats['utilization'] < 0.8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCUDAIntegration:
    """Test CUDA backend integration."""
    
    def test_cuda_backend_available(self):
        """Check if CUDA backend loaded."""
        try:
            import fft_tensor_cuda
            assert True
        except ImportError:
            pytest.skip("CUDA backend not compiled")
    
    def test_cuda_vs_pytorch_equivalence(self):
        """Test CUDA and PyTorch backends give same results."""
        spatial = torch.randn(64, 64, device='cuda')
        
        # With CUDA backend
        tensor_cuda = SparseSpectralTensor(data=spatial.clone(), sparsity=0.1, use_cuda_backend=True)
        
        # With PyTorch fallback
        tensor_torch = SparseSpectralTensor(data=spatial.clone(), sparsity=0.1, use_cuda_backend=False)
        
        # Reconstruct
        recon_cuda = tensor_cuda.to_spatial()
        recon_torch = tensor_torch.to_spatial()
        
        # Should be very similar
        diff = torch.norm(recon_cuda - recon_torch) / torch.norm(recon_cuda)
        assert diff < 0.01, f"CUDA vs PyTorch diff: {diff:.3f}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestScalability:
    """Test scalability to large sizes."""
    
    def test_incremental_sizes(self):
        """Test with incrementally larger tensors."""
        sizes = [128, 256, 512, 1024, 2048]
        
        for size in sizes:
            try:
                spatial = torch.randn(size, size, device='cuda')
                tensor = sst(spatial, sparsity=0.05)
                
                print(f"\n{size}x{size}: {tensor.memory_mb():.1f}MB, "
                      f"{tensor.compress_ratio():.1f}x compression")
                
                # Cleanup
                del spatial, tensor
                torch.cuda.empty_cache()
                MemoryManager.clear_all()
                
            except RuntimeError as e:
                print(f"\n{size}x{size}: Failed with {e}")
                break
    
    def test_3d_tensors(self):
        """Test 3D tensors (for video/3D data)."""
        shape = (64, 64, 64)
        
        spatial = torch.randn(shape, device='cuda')
        tensor = sst(spatial, sparsity=0.05)
        
        print(f"\n3D {shape}: {tensor.memory_mb():.1f}MB, "
              f"{tensor.compress_ratio():.1f}x compression")
        
        assert tensor.shape == shape
    
    def test_4d_tensors(self):
        """Test 4D tensors (for batch x channel x H x W)."""
        shape = (4, 16, 32, 32)
        
        spatial = torch.randn(shape, device='cuda')
        tensor = sst(spatial, sparsity=0.05)
        
        print(f"\n4D {shape}: {tensor.memory_mb():.1f}MB, "
              f"{tensor.compress_ratio():.1f}x compression")
        
        assert tensor.shape == shape


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
