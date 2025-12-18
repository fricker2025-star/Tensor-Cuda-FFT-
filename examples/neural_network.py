"""
Example: Building neural networks with FFT-Tensor
Demonstrates how to train models using sparse spectral tensors
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from fft_tensor import sst, MemoryManager
from fft_tensor.ops import ImplicitWeights, implicit_matmul


class SpectralLinear(nn.Module):
    """Linear layer using sparse spectral tensors for weights."""
    
    def __init__(self, in_features, out_features, sparsity=0.05, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sparsity = sparsity
        
        # Initialize weights as SST
        weight_data = torch.randn(out_features, in_features) * 0.01
        self.weight_sst = sst(weight_data, sparsity=sparsity, device='cuda')
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x):
        # Materialize weights for now
        # TODO: Implement pure spectral matmul
        weight = self.weight_sst.to_spatial()
        output = F.linear(x, weight, self.bias)
        return output
    
    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, '\
               f'sparsity={self.sparsity}, compression={self.weight_sst.compress_ratio():.1f}x'


class ImplicitLinear(nn.Module):
    """Linear layer with implicit weights (extreme compression)."""
    
    def __init__(self, in_features, out_features, rank=256, sparsity=0.01):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.implicit_weights = ImplicitWeights(
            shape=(out_features, in_features),
            rank=rank,
            sparsity=sparsity,
            device='cuda'
        )
        self.bias = nn.Parameter(torch.zeros(out_features))
    
    def forward(self, x):
        # Convert input to SST
        x_sst = sst(x, sparsity=0.05, device='cuda')
        
        # Spectral matmul with streaming
        output_sst = implicit_matmul(x_sst, self.implicit_weights, streaming=True)
        output = output_sst.to_spatial()
        
        return output + self.bias
    
    def extra_repr(self):
        compression = self.implicit_weights.compression_ratio()
        return f'in_features={self.in_features}, out_features={self.out_features}, '\
               f'compression={compression:.1f}x'


class SpectralMLP(nn.Module):
    """MLP using spectral linear layers."""
    
    def __init__(self, input_dim, hidden_dim, output_dim, sparsity=0.05):
        super().__init__()
        
        self.fc1 = SpectralLinear(input_dim, hidden_dim, sparsity=sparsity)
        self.fc2 = SpectralLinear(hidden_dim, hidden_dim, sparsity=sparsity)
        self.fc3 = SpectralLinear(hidden_dim, output_dim, sparsity=sparsity)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MassiveModel(nn.Module):
    """
    Massive model that wouldn't fit on consumer GPU normally.
    Uses implicit weights for extreme compression.
    """
    
    def __init__(self, dim=4096, n_layers=8):
        super().__init__()
        
        self.layers = nn.ModuleList([
            ImplicitLinear(dim, dim, rank=256, sparsity=0.01)
            for _ in range(n_layers)
        ])
        
        self.output = nn.Linear(dim, 1000)  # Classification head
    
    def forward(self, x):
        for layer in self.layers:
            x = F.gelu(layer(x))
        return self.output(x)


def train_example():
    """Example training loop with spectral tensors."""
    print("\n" + "="*60)
    print("Training Example: Spectral MLP")
    print("="*60)
    
    # Create model
    model = SpectralMLP(
        input_dim=784,    # MNIST: 28x28
        hidden_dim=2048,
        output_dim=10,
        sparsity=0.05
    ).cuda()
    
    print("\nModel architecture:")
    print(model)
    
    # Print memory usage
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Memory stats: {MemoryManager.get_stats()}")
    
    # Create dummy data
    x = torch.randn(32, 784, device='cuda')
    y = torch.randint(0, 10, (32,), device='cuda')
    
    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    print("\nTraining for 10 steps...")
    for step in range(10):
        optimizer.zero_grad()
        
        output = model(x)
        loss = criterion(output, y)
        
        loss.backward()
        optimizer.step()
        
        if step % 2 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}")
    
    print("\nTraining completed!")
    print(f"Final memory: {MemoryManager.get_stats()}")


def massive_model_example():
    """Example: Fit a massive model on 6GB GPU."""
    print("\n" + "="*60)
    print("Massive Model Example")
    print("="*60)
    
    # This model has 8 layers of (4096, 4096) = ~512MB normally
    # With implicit weights at 1% sparsity: ~5MB total!
    
    print("\nCreating massive model...")
    model = MassiveModel(dim=4096, n_layers=8).cuda()
    
    print("\nModel created successfully!")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Memory stats
    stats = MemoryManager.get_stats()
    print(f"\nMemory usage:")
    print(f"  SST tensors: {stats['total_memory_mb']:.1f}MB")
    print(f"  Utilization: {stats['utilization']*100:.1f}%")
    
    # Test forward pass
    print("\nRunning forward pass...")
    x = torch.randn(4, 4096, device='cuda')
    
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Memory after forward: {MemoryManager.get_stats()['total_memory_mb']:.1f}MB")
    
    print("\nSuccess! Massive model runs on 6GB GPU!")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("FFT-Tensor Neural Network Examples")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Examples require GPU.")
        exit(1)
    
    # Set memory limit
    MemoryManager.set_limit(5000)  # 5GB limit
    
    train_example()
    
    # Clear before massive model
    MemoryManager.clear_all()
    
    massive_model_example()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60 + "\n")
