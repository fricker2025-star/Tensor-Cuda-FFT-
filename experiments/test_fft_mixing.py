"""(moved to experiments/)
Test if FFT actually mixes information across SEQUENCE

The key: FFT must operate on sequence dimension, not feature dimension!
"""
import torch
import torch.nn as nn


class SequenceMixingSpectralModel(nn.Module):
    """FFT mixes across SEQUENCE dimension."""
    
    def __init__(self, vocab_size=256, embed_dim=128):
        super().__init__()
        
        self.byte_embed = nn.Embedding(vocab_size, embed_dim)
        
        # Learnable frequency filter (applied to each feature independently)
        self.freq_filter = nn.Parameter(torch.ones(embed_dim) * 0.01)  # Small init
        
        self.norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, x):
        # x: [batch, seq_len]
        x = self.byte_embed(x)  # [batch, seq_len, embed_dim]
        
        residual = x
        
        # FFT on SEQUENCE dimension (dim=1)
        # This mixes information ACROSS positions!
        x_freq = torch.fft.rfft(x, dim=1)  # [batch, freq_bins, embed_dim]
        
        # Apply learnable filter (small at init, so acts as identity)
        x_freq = x_freq * self.freq_filter.unsqueeze(0).unsqueeze(0)
        
        # iFFT back to sequence
        x = torch.fft.irfft(x_freq, n=residual.size(1), dim=1)
        
        # RESIDUAL (critical!)
        x = residual + x
        
        x = self.norm(x)
        logits = self.output_proj(x)
        
        return logits


# Test
text = "The quick brown fox"
byte_vals = [ord(c) for c in text]

X = torch.tensor([byte_vals[:-1]]).long().cuda()
Y = torch.tensor([byte_vals[1:]]).long().cuda()

print("="*70)
print("SEQUENCE-MIXING FFT TEST")
print("="*70)
print(f"\nSentence: '{text}'")
print(f"FFT operates on SEQUENCE dimension (should mix context)\n")

model = SequenceMixingSpectralModel(256, 128).cuda()

print(f"Initial filter values: {model.freq_filter.mean().item():.4f}")
print()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

print("Training...")
print("-" * 70)

for i in range(300):
    model.train()
    
    logits = model(X)
    loss = loss_fn(logits.reshape(-1, 256), Y.reshape(-1))
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    if i % 30 == 0:
        print(f"Step {i:3d}: Loss {loss.item():.4f}")

print("-" * 70)
print(f"\nFinal loss: {loss.item():.6f}")

if loss.item() < 0.05:
    print("[SUCCESS] FFT sequence mixing works!")
    
    model.eval()
    with torch.no_grad():
        pred = model(X).argmax(dim=-1)[0].cpu().numpy()
        pred_text = ''.join(chr(b) for b in pred if 32 <= b <= 126)
        print(f"\nTarget:    '{text[1:]}'")
        print(f"Predicted: '{pred_text}'")
elif loss.item() < 0.2:
    print("[PROGRESS] Learning! Keep training...")
else:
    print(f"[STUCK] Still at {loss.item():.4f}")

print("="*70)
