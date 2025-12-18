"""
Unit tests for SparseSpectralTensor
"""
import torch
import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fft_tensor.tensor import SparseSpectralTensor, sst, zeros_sst, randn_sst, MemoryManager


class TestSparseSpectralTensor:
    """Test SST core functionality."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        MemoryManager.clear_all()
        yield
        MemoryManager.clear_all()
    
    def test_creation_from_spatial(self):
        """Test SST creation from spatial data."""
        spatial = torch.randn(64, 64, device='cuda' if torch.cuda.is_available() else 'cpu')
        tensor = sst(spatial, sparsity=0.05)
        
        assert tensor.shape == (64, 64)
        assert tensor.compress_ratio() > 1.0
        assert len(tensor.freq_coeffs) < spatial.numel()
    
    def test_to_spatial_reconstruction(self):
        """Test spatial reconstruction accuracy."""
        spatial = torch.randn(32, 32, device='cuda' if torch.cuda.is_available() else 'cpu')
        tensor = sst(spatial, sparsity=0.1)  # Higher sparsity for better reconstruction
        
        reconstructed = tensor.to_spatial()
        
        # Check shape
        assert reconstructed.shape == spatial.shape
        
        # Check reconstruction error (should be small with 10% sparsity)
        error = torch.norm(reconstructed - spatial) / torch.norm(spatial)
        assert error < 0.5, f"Reconstruction error too high: {error:.3f}"
    
    def test_addition(self):
        """Test SST addition in frequency domain."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        a = sst(torch.randn(32, 32, device=device), sparsity=0.05)
        b = sst(torch.randn(32, 32, device=device), sparsity=0.05)
        
        c = a + b
        
        assert c.shape == a.shape
        assert isinstance(c, SparseSpectralTensor)
    
    def test_scalar_multiplication(self):
        """Test SST scalar multiplication."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        a = sst(torch.randn(32, 32, device=device), sparsity=0.05)
        
        b = a * 2.0
        c = 3.0 * a
        
        assert b.shape == a.shape
        assert c.shape == a.shape
        assert isinstance(b, SparseSpectralTensor)
        assert isinstance(c, SparseSpectralTensor)
    
    def test_matmul(self):
        """Test matrix multiplication."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        a = sst(torch.randn(32, 64, device=device), sparsity=0.05)
        b = sst(torch.randn(64, 16, device=device), sparsity=0.05)
        
        c = a.matmul(b)
        
        assert c.shape == (32, 16)
        assert isinstance(c, SparseSpectralTensor)
    
    def test_compression_ratio(self):
        """Test compression ratio calculation."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        tensor = sst(torch.randn(100, 100, device=device), sparsity=0.05)
        
        ratio = tensor.compress_ratio()
        
        assert ratio > 10.0, f"Compression ratio too low: {ratio:.1f}x"
        assert ratio < 200.0, f"Compression ratio too high: {ratio:.1f}x"
    
    def test_memory_tracking(self):
        """Test memory manager tracking."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        initial_count = len(MemoryManager._tensors)
        
        tensor1 = sst(torch.randn(64, 64, device=device), sparsity=0.05)
        assert len(MemoryManager._tensors) == initial_count + 1
        
        tensor2 = sst(torch.randn(64, 64, device=device), sparsity=0.05)
        assert len(MemoryManager._tensors) == initial_count + 2
        
        del tensor1
        # Note: __del__ may not be called immediately
        MemoryManager._tensors = [t for t in MemoryManager._tensors if t is not None]
    
    def test_zeros_creation(self):
        """Test zero SST creation."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        tensor = zeros_sst((64, 64), device=device)
        
        assert tensor.shape == (64, 64)
        spatial = tensor.to_spatial()
        assert torch.allclose(spatial, torch.zeros_like(spatial), atol=1e-3)
    
    def test_randn_creation(self):
        """Test random SST creation."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        tensor = randn_sst((64, 64), device=device)
        
        assert tensor.shape == (64, 64)
        spatial = tensor.to_spatial()
        # Should not be all zeros
        assert not torch.allclose(spatial, torch.zeros_like(spatial))
    
    def test_different_sparsities(self):
        """Test different sparsity levels."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        spatial = torch.randn(64, 64, device=device)
        
        for sparsity in [0.01, 0.05, 0.1, 0.2]:
            tensor = sst(spatial, sparsity=sparsity)
            ratio = tensor.compress_ratio()
            
            expected_ratio = 1.0 / sparsity
            assert 0.5 * expected_ratio < ratio < 2.0 * expected_ratio
    
    def test_nd_tensors(self):
        """Test 1D, 2D, 3D, 4D tensors."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        for shape in [(64,), (32, 32), (16, 16, 16), (8, 8, 8, 8)]:
            spatial = torch.randn(shape, device=device)
            tensor = sst(spatial, sparsity=0.05)
            
            assert tensor.shape == shape
            reconstructed = tensor.to_spatial()
            assert reconstructed.shape == shape


class TestMemoryManager:
    """Test memory management."""
    
    def test_set_limit(self):
        """Test setting memory limit."""
        MemoryManager.set_limit(1000)
        assert MemoryManager._max_memory_mb == 1000
        
        # Reset
        MemoryManager.set_limit(5000)
    
    def test_clear_all(self):
        """Test clearing all tensors."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        _ = sst(torch.randn(64, 64, device=device))
        _ = sst(torch.randn(64, 64, device=device))
        
        MemoryManager.clear_all()
        assert len(MemoryManager._tensors) == 0
    
    def test_get_stats(self):
        """Test getting memory statistics."""
        stats = MemoryManager.get_stats()
        
        assert 'n_tensors' in stats
        assert 'total_memory_mb' in stats
        assert 'limit_mb' in stats
        assert 'utilization' in stats
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_memory_limit_enforcement(self):
        """Test that memory limit is enforced."""
        MemoryManager.clear_all()
        MemoryManager.set_limit(1)  # 1MB limit
        
        with pytest.raises(MemoryError):
            # Try to create a tensor that exceeds limit
            _ = sst(torch.randn(1000, 1000, device='cuda'), sparsity=0.1)
        
        # Reset
        MemoryManager.clear_all()
        MemoryManager.set_limit(5000)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
