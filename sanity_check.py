"""
SANITY CHECK - Can the model overfit ONE sentence?

If it can't, the gradient is BROKEN.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProgressiveSpectralModel(nn.Module):
    def __init__(self, vocab_size=256, embed_dim=512, max_freq=2048, num_layers=6):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_freq = max_freq
        
        self.byte_embed = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(num_layers)
        ])
        
        self.register_buffer('freq_cutoff', torch.tensor(2048))  # Full spectrum
        
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
    
    def set_frequency_cutoff(self, cutoff):
        self.freq_cutoff = torch.tensor(min(cutoff, self.max_freq))
    
    def forward(self, x):
        x = self.byte_embed(x)
        
        for layer in self.layers:
            residual = x
            
            # FFT
            x_freq = torch.fft.rfft(x, dim=1)
            
            # Check gradient flow
            if x_freq.requires_grad:
                pass  # Good
            else:
                print("[ERROR] Gradient detached at FFT!")
            
            x_freq = x_freq * 0.95
            
            # iFFT
            x = torch.fft.irfft(x_freq, n=residual.size(1), dim=1)
            
            # Normalize
            x = self.norm(x)
            
            # Residual
            x = residual + self.dropout(layer(x))
        
        x = self.norm(x)
        logits = torch.matmul(x, self.byte_embed.weight.t())
        
        return logits


def overfit_test():
    """Can we overfit ONE sentence?"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*70)
    print("SANITY CHECK: One Sentence Overfitting Test")
    print("="*70)
    
    # ONE SENTENCE
    text = "The quick brown fox jumps over the lazy dog."
    
    print(f"\nTarget: '{text}'")
    print(f"Length: {len(text)} characters\n")
    
    # Prepare
    byte_vals = [ord(c) for c in text]
    seq_len = len(byte_vals) - 1
    
    inp = byte_vals[:seq_len]
    tgt = byte_vals[1:]
    
    inp_tensor = torch.tensor([inp], dtype=torch.long, device=device)
    tgt_tensor = torch.tensor([tgt], dtype=torch.long, device=device)
    
    # Model
    model = ProgressiveSpectralModel(
        vocab_size=256,
        embed_dim=512,
        max_freq=2048,
        num_layers=6
    ).to(device)
    
    # Check gradients are enabled
    print("Checking gradient flow...")
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"  [ERROR] {name} has requires_grad=False!")
    
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    
    print("\nTraining to overfit ONE sentence...")
    print("Target: Loss < 0.1 (proves gradient works)")
    print("-" * 70)
    
    for epoch in range(500):
        model.train()
        
        opt.zero_grad()
        
        # Forward
        logits = model(inp_tensor)
        loss = crit(logits.reshape(-1, 256), tgt_tensor.reshape(-1))
        
        # Check for NaN
        if torch.isnan(loss):
            print(f"\n[FATAL] NaN at epoch {epoch}!")
            print("Gradient is BROKEN")
            return False
        
        # Backward
        loss.backward()
        
        # Check gradients exist
        grad_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.norm().item()
        
        if grad_norm == 0:
            print(f"\n[FATAL] Zero gradients at epoch {epoch}!")
            print("Gradient is BROKEN")
            return False
        
        opt.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1:3d}: Loss={loss.item():.6f}, GradNorm={grad_norm:.2f}")
        
        # Success condition
        if loss.item() < 0.1:
            print(f"\n[SUCCESS] Overfitted at epoch {epoch+1}!")
            print(f"Final loss: {loss.item():.6f}")
            print("\nGradient is WORKING. Model can learn.")
            
            # Test generation
            model.eval()
            with torch.no_grad():
                out = model(inp_tensor)
                predicted = out.argmax(dim=-1)[0].cpu().numpy()
                predicted_text = ''.join(chr(b) for b in predicted if 32 <= b <= 126)
                
                print(f"\nTarget:    '{text[1:]}'")
                print(f"Predicted: '{predicted_text}'")
            
            return True
    
    # Failed to overfit
    print(f"\n[FAIL] Could not overfit after 500 epochs")
    print(f"Final loss: {loss.item():.6f}")
    print("\nThis proves the gradient is BROKEN.")
    print("No amount of data/epochs will help.")
    
    return False


if __name__ == '__main__':
    success = overfit_test()
    
    print("\n" + "="*70)
    if success:
        print("VERDICT: Code is GOOD")
        print("  - Gradient flows correctly")
        print("  - Model can learn")
        print("  - Problem was: Model too small for big dataset")
        print("\nSolution: Scale to 5M+ parameters")
    else:
        print("VERDICT: Code is BROKEN")
        print("  - Gradient is detached or broken")
        print("  - Check: FFT/iFFT operations")
        print("  - Check: Complex number handling")
        print("  - Check: LayerNorm placement")
        print("\nFix the gradient before scaling up!")
    print("="*70)
