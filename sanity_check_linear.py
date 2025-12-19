"""
SANITY CHECK - Pure Linear (NO FFT)

If this works, FFT is the culprit.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PureLinearModel(nn.Module):
    """NO FFT - Pure linear layers only."""
    
    def __init__(self, vocab_size=256, embed_dim=512, num_layers=6):
        super().__init__()
        self.vocab_size = vocab_size
        
        self.byte_embed = nn.Embedding(vocab_size, embed_dim)
        
        # Pure linear layers (NO FFT)
        self.layers = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = self.byte_embed(x)
        
        for layer in self.layers:
            residual = x
            
            # PURE LINEAR - NO FFT
            x = layer(x)
            x = F.gelu(x)
            x = self.norm(x)
            
            # Residual
            x = residual + self.dropout(x)
        
        x = self.norm(x)
        logits = torch.matmul(x, self.byte_embed.weight.t())
        
        return logits


def overfit_test():
    """Can pure linear model overfit ONE sentence?"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*70)
    print("SANITY CHECK: Pure Linear Model (NO FFT)")
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
    
    # Pure linear model
    model = PureLinearModel(
        vocab_size=256,
        embed_dim=512,
        num_layers=6
    ).to(device)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}\n")
    
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    
    print("Training to overfit ONE sentence...")
    print("Target: Loss < 0.1 within 200 epochs")
    print("-" * 70)
    
    for epoch in range(200):
        model.train()
        
        opt.zero_grad()
        
        # Forward
        logits = model(inp_tensor)
        loss = crit(logits.reshape(-1, 256), tgt_tensor.reshape(-1))
        
        # Check for NaN
        if torch.isnan(loss):
            print(f"\n[FATAL] NaN at epoch {epoch}!")
            return False
        
        # Backward
        loss.backward()
        
        # Check gradients exist
        grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        
        if grad_norm == 0:
            print(f"\n[FATAL] Zero gradients at epoch {epoch}!")
            return False
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}: Loss={loss.item():.6f}")
        
        # Success condition
        if loss.item() < 0.1:
            print(f"\n[SUCCESS] Overfitted at epoch {epoch+1}!")
            print(f"Final loss: {loss.item():.6f}")
            
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
    print(f"\n[FAIL] Could not overfit after 200 epochs")
    print(f"Final loss: {loss.item():.6f}")
    
    return False


if __name__ == '__main__':
    linear_success = overfit_test()
    
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)
    
    if linear_success:
        print("\n[CONFIRMED] Pure Linear Model CAN overfit")
        print("\nThis PROVES:")
        print("  - Gradients work with linear layers")
        print("  - FFT operations are BREAKING gradients")
        print("  - torch.fft.rfft/irfft are the culprit")
        print("\nThe spectral architecture needs:")
        print("  1. Fix FFT gradient flow")
        print("  2. OR use DCT (real-valued)")
        print("  3. OR use learnable conv (not FFT)")
    else:
        print("\n[UNEXPECTED] Even pure linear failed")
        print("Something else is wrong (unlikely)")
    
    print("="*70)
