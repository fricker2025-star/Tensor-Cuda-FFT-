"""(moved to experiments/)
Test with IDENTITY INITIALIZATION

FFT layer starts as pass-through, gradually learns to mix.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentityInitSpectralModel(nn.Module):
    """Spectral model with identity initialization."""
    
    def __init__(self, vocab_size=256, embed_dim=128, num_layers=3):
        super().__init__()
        
        self.byte_embed = nn.Embedding(vocab_size, embed_dim)
        
        # Spectral mixing layers with SMALL init
        self.spectral_mix = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(num_layers)
        ])
        
        # CRITICAL: Initialize with tiny weights (start as identity)
        for layer in self.spectral_mix:
            nn.init.normal_(layer.weight, mean=0.0, std=0.001)  # VERY SMALL!
            nn.init.zeros_(layer.bias)
        
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        x = self.byte_embed(x)
        
        for layer in self.spectral_mix:
            residual = x
            
            # FFT
            x_freq = torch.fft.rfft(x, dim=1)
            
            # iFFT
            x_time = torch.fft.irfft(x_freq, n=residual.size(1), dim=1)
            
            # Mix in time domain with small weights
            x_mixed = layer(x_time)
            
            # CRITICAL: RESIDUAL CONNECTION
            # At init, layer(x) ≈ 0, so x ≈ residual (identity!)
            x = residual + x_mixed  # Not just x = x_mixed!
        
        x = self.norm(x)
        logits = torch.matmul(x, self.byte_embed.weight.t())
        
        return logits


# Test on sentence
text = "The quick brown fox"
byte_vals = [ord(c) for c in text]

X = torch.tensor([byte_vals[:-1]]).long().cuda()
Y = torch.tensor([byte_vals[1:]]).long().cuda()

print("="*70)
print("IDENTITY INITIALIZATION TEST")
print("="*70)
print(f"\nSentence: '{text}'")
print(f"Length: {len(text)} chars\n")

model = IdentityInitSpectralModel(256, 128, 3).cuda()

# Check initial weights are small
first_layer_std = model.spectral_mix[0].weight.std().item()
print(f"Initial weight std: {first_layer_std:.6f} (should be ~0.001)")
print()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

print("Training with identity init...")
print("-" * 70)

for i in range(200):
    model.train()
    
    logits = model(X)
    loss = loss_fn(logits.reshape(-1, 256), Y.reshape(-1))
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    if i % 20 == 0:
        print(f"Step {i:3d}: Loss {loss.item():.4f}")

print("-" * 70)
print(f"\nFinal loss: {loss.item():.6f}")

if loss.item() < 0.1:
    print("[SUCCESS] Overfitted with identity init!")
    
    model.eval()
    with torch.no_grad():
        pred = model(X).argmax(dim=-1)[0].cpu().numpy()
        pred_text = ''.join(chr(b) for b in pred if 32 <= b <= 126)
        print(f"\nTarget:    '{text[1:]}'")
        print(f"Predicted: '{pred_text}'")
else:
    print(f"[PARTIAL] Loss improved to {loss.item():.4f}")
    print("(Better than 2.33, but still needs work)")

print("="*70)
