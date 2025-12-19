"""(moved to experiments/)
Test with NORMAL init but STRONG residual

Maybe tiny init is TOO conservative. Try normal init with residual.
"""
import torch
import torch.nn as nn


class ResidualSpectralModel(nn.Module):
    """FFT with normal init but strong residual."""
    
    def __init__(self, vocab_size=256, embed_dim=128):
        super().__init__()
        
        self.byte_embed = nn.Embedding(vocab_size, embed_dim)
        
        # Normal initialization (not tiny!)
        self.mix_layer = nn.Linear(embed_dim, embed_dim)
        
        self.norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, x):
        x = self.byte_embed(x)
        
        residual = x
        
        # FFT on sequence
        x_freq = torch.fft.rfft(x, dim=1)
        
        # Back to time
        x_time = torch.fft.irfft(x_freq, n=residual.size(1), dim=1)
        
        # Mix
        x_mixed = self.mix_layer(x_time)
        
        # STRONG RESIDUAL
        x = residual + 0.1 * x_mixed  # Scale down FFT contribution
        
        x = self.norm(x)
        logits = self.output_proj(x)
        
        return logits


text = "The quick brown fox"
byte_vals = [ord(c) for c in text]

X = torch.tensor([byte_vals[:-1]]).long().cuda()
Y = torch.tensor([byte_vals[1:]]).long().cuda()

print("="*70)
print("NORMAL INIT + SCALED RESIDUAL")
print("="*70)
print(f"\nSentence: '{text}'\n")

model = ResidualSpectralModel(256, 128).cuda()

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
    print("[SUCCESS]!")
    model.eval()
    with torch.no_grad():
        pred = model(X).argmax(dim=-1)[0].cpu().numpy()
        pred_text = ''.join(chr(b) for b in pred if 32 <= b <= 126)
        print(f"\nTarget:    '{text[1:]}'")
        print(f"Predicted: '{pred_text}'")
elif loss.item() < 0.2:
    print("[PROGRESS]")
else:
    print(f"[STUCK] at {loss.item():.4f}")

print("="*70)
