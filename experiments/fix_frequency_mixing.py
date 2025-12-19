"""(moved to experiments/)
FIX: Apply mixing IN FREQUENCY DOMAIN

Not after converting back to time!
"""
import torch
import torch.nn as nn


class FixedSpectralModel(nn.Module):
    """Mix in FREQUENCY domain, not time domain."""
    
    def __init__(self, vocab_size=256, embed_dim=128):
        super().__init__()
        
        self.byte_embed = nn.Embedding(vocab_size, embed_dim)
        
        # Learnable COMPLEX filter in frequency domain
        # Start small for stability
        freq_bins = 10  # For seq_len=18, rfft gives 10 bins
        self.freq_filter_real = nn.Parameter(torch.ones(freq_bins, embed_dim) * 0.01)
        self.freq_filter_imag = nn.Parameter(torch.zeros(freq_bins, embed_dim))
        
        self.norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, x):
        x = self.byte_embed(x)
        
        residual = x
        
        # FFT on SEQUENCE
        x_freq = torch.fft.rfft(x, dim=1)  # [batch, freq_bins, embed_dim] COMPLEX
        
        # MIX IN FREQUENCY DOMAIN! (The fix!)
        freq_filter = torch.complex(self.freq_filter_real, self.freq_filter_imag)
        
        # Ensure shapes match
        if freq_filter.size(0) != x_freq.size(1):
            # Adjust filter size dynamically
            actual_bins = x_freq.size(1)
            if actual_bins < freq_filter.size(0):
                freq_filter = freq_filter[:actual_bins]
            else:
                # Pad with ones
                padding = torch.ones(actual_bins - freq_filter.size(0), freq_filter.size(1), 
                                   dtype=freq_filter.dtype, device=freq_filter.device)
                freq_filter = torch.cat([freq_filter, padding], dim=0)
        
        # Apply filter (complex multiplication)
        x_freq_mixed = x_freq * freq_filter.unsqueeze(0)  # Broadcast over batch
        
        # NOW convert back
        x_mixed = torch.fft.irfft(x_freq_mixed, n=residual.size(1), dim=1)
        
        # Residual
        x = residual + x_mixed
        
        x = self.norm(x)
        logits = self.output_proj(x)
        
        return logits


text = "The quick brown fox"
byte_vals = [ord(c) for c in text]

X = torch.tensor([byte_vals[:-1]]).long().cuda()
Y = torch.tensor([byte_vals[1:]]).long().cuda()

print("="*70)
print("FIXED: Mixing in FREQUENCY domain")
print("="*70)
print(f"\nTarget: '{text}'\n")

model = FixedSpectralModel(256, 128).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

print("Training with REAL spectral mixing...")
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
    print("[SUCCESS] Spectral mixing works!")
    
    model.eval()
    with torch.no_grad():
        pred = model(X).argmax(dim=-1)[0].cpu().numpy()
        pred_text = ''.join(chr(b) for b in pred if 32 <= b <= 126)
        print(f"\nTarget:    '{text[1:]}'")
        print(f"Predicted: '{pred_text}'")
elif loss.item() < 0.2:
    print("[PROGRESS] Better than before!")
else:
    print(f"[STUCK] Still at {loss.item():.4f}")

print("="*70)
