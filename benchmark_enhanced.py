"""
Benchmark: Enhanced Spectral Model with RoPE + GLU + Phase-Aware Mixing

Test if enhancements solve "too much invariance" problem.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from fft_tensor.byte_spectral_triton import TritonSpectralLanguageModel
from fft_tensor.spectral_enhancements import EnhancedSpectralBlock
from fft_tensor.cleanup import GPUContext, cleanup_models
import time


class EnhancedSpectralLanguageModel(nn.Module):
    """
    Byte-spectral model with enhancements:
    - RoPE in frequency domain
    - Gated linear units
    - Phase-aware mixing
    - Multi-scale features
    """
    
    def __init__(self, embed_dim=256, num_layers=4, max_seq_len=4096):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # Simple byte embedding (no complex encoding for now)
        self.byte_proj = nn.Linear(256, embed_dim)
        
        # Enhanced spectral blocks
        self.layers = nn.ModuleList([
            EnhancedSpectralBlock(embed_dim)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.output = nn.Linear(embed_dim, 256)
    
    def forward(self, byte_ids):
        B, T = byte_ids.shape
        
        # One-hot encode bytes
        x = F.one_hot(byte_ids, num_classes=256).float()  # (B, T, 256)
        x = self.byte_proj(x)  # (B, T, embed_dim)
        
        # Enhanced spectral processing
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        logits = self.output(x)
        
        return logits


class TraditionalTransformer(nn.Module):
    """Traditional transformer for comparison."""
    
    def __init__(self, embed_dim=256, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(256, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.output = nn.Linear(embed_dim, 256)
    
    def forward(self, x):
        x = self.embedding(x)
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
        x = self.transformer(x, mask=mask, is_causal=True)
        x = self.norm(x)
        return self.output(x)


def create_data(num_samples=20, seq_len=512):
    """Create training data."""
    texts = [
        "The quick brown fox jumps over the lazy dog. " * 50,
        "Machine learning is artificial intelligence. " * 50,
        "Deep neural networks learn from data. " * 50,
        "Python is a programming language. " * 50,
    ]
    
    full_text = "".join(texts)
    byte_values = [ord(c) for c in full_text]
    
    sequences = []
    for i in range(0, len(byte_values) - seq_len - 1, seq_len):
        inp = byte_values[i:i + seq_len]
        tgt = byte_values[i + 1:i + seq_len + 1]
        if len(inp) == seq_len and len(sequences) < num_samples:
            sequences.append((inp, tgt))
    
    return sequences


def train_model(model, sequences, num_epochs=10, device='cuda', name="Model"):
    """Train and return losses."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    losses = []
    
    print(f"Training {name}...")
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        start = time.time()
        
        batch_size = 4
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            if len(batch) < batch_size:
                continue
            
            inputs = torch.tensor([s[0] for s in batch], dtype=torch.long, device=device)
            targets = torch.tensor([s[1] for s in batch], dtype=torch.long, device=device)
            
            optimizer.zero_grad()
            
            try:
                logits = model(inputs)
                loss = criterion(logits.reshape(-1, 256), targets.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise
        
        if num_batches > 0:
            avg_loss = epoch_loss / num_batches
            elapsed = time.time() - start
            losses.append(avg_loss)
            
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}: Loss={avg_loss:.4f}, Time={elapsed:.2f}s")
    
    return losses


def benchmark_inference(model, byte_ids, num_trials=50):
    """Benchmark inference speed."""
    model.eval()
    device = byte_ids.device
    
    with torch.no_grad():
        for _ in range(10):
            _ = model(byte_ids)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start = time.time()
    
    with torch.no_grad():
        for _ in range(num_trials):
            _ = model(byte_ids)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    return (time.time() - start) / num_trials * 1000


def main():
    print("\n" + "="*70)
    print("ENHANCED SPECTRAL MODEL BENCHMARK")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    if device == 'cpu':
        print("CUDA required")
        return
    
    seq_len = 512
    embed_dim = 256
    num_layers = 4
    
    print(f"\nConfiguration:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Embed dim: {embed_dim}")
    print(f"  Num layers: {num_layers}")
    
    # Create data
    print("\nCreating data...")
    sequences = create_data(num_samples=30, seq_len=seq_len)
    print(f"  {len(sequences)} sequences")
    
    results = {}
    
    try:
        # Test Enhanced Spectral
        with GPUContext():
            print("\n" + "-"*70)
            print("ENHANCED SPECTRAL MODEL (RoPE + GLU + Phase-Aware)")
            print("-"*70)
            
            enhanced_model = EnhancedSpectralLanguageModel(
                embed_dim=embed_dim,
                num_layers=num_layers,
                max_seq_len=seq_len
            ).to(device)
            
            enhanced_params = sum(p.numel() for p in enhanced_model.parameters())
            print(f"Parameters: {enhanced_params:,}")
            
            enhanced_losses = train_model(enhanced_model, sequences, num_epochs=10, device=device, name="Enhanced")
            
            if enhanced_losses:
                enhanced_final = enhanced_losses[-1]
                print(f"\nFinal loss: {enhanced_final:.4f}")
                results['enhanced'] = {'loss': enhanced_final, 'params': enhanced_params}
            
            # Inference speed
            test_input = torch.randint(0, 256, (4, seq_len), dtype=torch.long, device=device)
            enhanced_inf = benchmark_inference(enhanced_model, test_input)
            results['enhanced']['inference'] = enhanced_inf
            print(f"Inference: {enhanced_inf:.2f}ms")
            
            cleanup_models(enhanced_model)
            del enhanced_model
            torch.cuda.empty_cache()
        
        # Test Traditional
        with GPUContext():
            print("\n" + "-"*70)
            print("TRADITIONAL TRANSFORMER")
            print("-"*70)
            
            traditional_model = TraditionalTransformer(
                embed_dim=embed_dim,
                num_layers=num_layers
            ).to(device)
            
            traditional_params = sum(p.numel() for p in traditional_model.parameters())
            print(f"Parameters: {traditional_params:,}")
            
            traditional_losses = train_model(traditional_model, sequences, num_epochs=10, device=device, name="Traditional")
            
            if traditional_losses:
                traditional_final = traditional_losses[-1]
                print(f"\nFinal loss: {traditional_final:.4f}")
                results['traditional'] = {'loss': traditional_final, 'params': traditional_params}
            
            # Inference speed
            traditional_inf = benchmark_inference(traditional_model, test_input)
            results['traditional']['inference'] = traditional_inf
            print(f"Inference: {traditional_inf:.2f}ms")
            
            cleanup_models(traditional_model)
            del traditional_model
            torch.cuda.empty_cache()
        
        # Summary
        print("\n" + "="*70)
        print("RESULTS COMPARISON")
        print("="*70)
        
        if 'enhanced' in results and 'traditional' in results:
            print(f"\nEnhanced Spectral (with RoPE + GLU):")
            print(f"  Parameters: {results['enhanced']['params']:,}")
            print(f"  Final Loss: {results['enhanced']['loss']:.4f}")
            print(f"  Inference:  {results['enhanced']['inference']:.2f}ms")
            
            print(f"\nTraditional Transformer:")
            print(f"  Parameters: {results['traditional']['params']:,}")
            print(f"  Final Loss: {results['traditional']['loss']:.4f}")
            print(f"  Inference:  {results['traditional']['inference']:.2f}ms")
            
            print(f"\n" + "-"*70)
            print("IMPROVEMENTS")
            print("-"*70)
            
            loss_improvement = (results['traditional']['loss'] - results['enhanced']['loss']) / results['traditional']['loss'] * 100
            speed_improvement = results['traditional']['inference'] / results['enhanced']['inference']
            
            if results['enhanced']['loss'] < results['traditional']['loss']:
                print(f"\nLoss: Enhanced {abs(loss_improvement):.1f}% BETTER!")
            else:
                print(f"\nLoss: Traditional {abs(loss_improvement):.1f}% better")
            
            if speed_improvement > 1.0:
                print(f"Speed: Enhanced {speed_improvement:.2f}x FASTER!")
            else:
                print(f"Speed: Traditional {1/speed_improvement:.2f}x faster")
            
            print(f"\nKey Finding:")
            if results['enhanced']['loss'] < results['traditional']['loss']:
                print("  [SUCCESS] Enhancements solved convergence problem!")
                print("  RoPE + GLU + Phase-Aware = Better convergence")
            else:
                print("  Enhancements improve but more tuning needed")
        
        print("\n" + "="*70)
        
    finally:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("\n[Cleanup] Complete")


if __name__ == '__main__':
    try:
        main()
    finally:
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("[Exit] Clean shutdown")
