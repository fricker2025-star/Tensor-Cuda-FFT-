"""(moved to experiments/)
DEBUG: Check if FFT is actually doing anything

Print magnitudes to see if context is flowing
"""
import torch
import torch.nn as nn


class DebugSpectralModel(nn.Module):
    """FFT with debug prints."""
    
    def __init__(self, vocab_size=256, embed_dim=128):
        super().__init__()
        
        self.byte_embed = nn.Embedding(vocab_size, embed_dim)
        self.mix_layer = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, vocab_size)
        
        self.step = 0
    
    def forward(self, x):
        x = self.byte_embed(x)
        
        residual = x
        
        # FFT on SEQUENCE dimension
        x_freq = torch.fft.rfft(x, dim=1)  # EXPLICITLY dim=1!
        
        # Back to time
        x_time = torch.fft.irfft(x_freq, n=residual.size(1), dim=1)
        
        # Mix
        fft_update = self.mix_layer(x_time)
        
        # DEBUG PRINTS
        if self.step % 50 == 0:
            print(f"\nStep {self.step}:")
            print(f"  Input shape: {x.shape}")
            print(f"  FFT freq shape: {x_freq.shape}")
            print(f"  Residual magnitude: {residual.abs().mean().item():.6f}")
            print(f"  FFT update magnitude: {fft_update.abs().mean().item():.6f}")
            print(f"  Ratio (update/residual): {(fft_update.abs().mean() / residual.abs().mean()).item():.6f}")
            
            # Check if FFT changed anything
            diff = (x_time - residual).abs().mean().item()
            print(f"  FFT changed signal by: {diff:.6f}")
        
        self.step += 1
        
        # Residual
        x = residual + fft_update
        
        x = self.norm(x)
        logits = self.output_proj(x)
        
        return logits


text = "The quick brown fox"
byte_vals = [ord(c) for c in text]

X = torch.tensor([byte_vals[:-1]]).long().cuda()
Y = torch.tensor([byte_vals[1:]]).long().cuda()

print("="*70)
print("FFT DEBUG: Checking axis and magnitude")
print("="*70)
print(f"\nTarget: '{text}'")
print(f"Input shape: {X.shape} (batch=1, seq_len={X.size(1)})")
print("\nFFT should operate on dim=1 (sequence) to mix context")
print("="*70)

model = DebugSpectralModel(256, 128).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

for i in range(200):
    model.train()
    
    logits = model(X)
    loss = loss_fn(logits.reshape(-1, 256), Y.reshape(-1))
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    if i % 50 == 0:
        print(f"\n>>> Loss at step {i}: {loss.item():.4f}")

print("\n" + "="*70)
print("ANALYSIS")
print("="*70)

if model.step > 0:
    print("\nIf FFT update magnitude stays near 0:")
    print("  -> Initialization too small OR gradient broken")
    print("\nIf FFT changed signal by 0:")
    print("  -> FFT is doing nothing (maybe wrong axis?)")
    print("\nIf ratio is very small (<0.01):")
    print("  -> FFT contribution is drowned out by residual")
    
print("="*70)
