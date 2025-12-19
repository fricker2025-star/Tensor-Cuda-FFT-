"""(moved to experiments/)
Test OUR actual model with golden test data
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class OurActualModel(nn.Module):
    """Copy of our actual model from train_tinystories.py"""
    
    def __init__(self, vocab_size=256, embed_dim=64, num_layers=2):
        super().__init__()
        
        self.byte_embed = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = self.byte_embed(x)
        
        for layer in self.layers:
            residual = x
            
            # FFT (REMOVED for this test)
            # Just use linear
            x = layer(x)
            x = F.gelu(x)
            x = self.norm(x)
            
            # Residual
            x = residual + self.dropout(x)
        
        x = self.norm(x)
        
        # Output projection
        logits = torch.matmul(x, self.byte_embed.weight.t())
        
        return logits


# Golden test data
X = torch.tensor([[1, 2, 3, 4]]).long().cuda()
Y = torch.tensor([[2, 3, 4, 5]]).long().cuda()

model = OurActualModel(256, 64, 2).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
loss_fn = nn.CrossEntropyLoss()

print("="*70)
print("Testing OUR ACTUAL model architecture")
print("="*70)

for i in range(100):
    model.train()
    
    logits = model(X)
    loss = loss_fn(logits.reshape(-1, 256), Y.reshape(-1))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if i % 10 == 0:
        print(f"Step {i:2d}: Loss {loss.item():.4f}")

print("-" * 70)

if loss.item() < 0.01:
    print(f"\n[SUCCESS] Our model works!")
    with torch.no_grad():
        pred = model(X).argmax(dim=-1)
        print(f"Predicted: {pred[0].tolist()}")
        print(f"Target:    {Y[0].tolist()}")
else:
    print(f"\n[FAIL] Our model architecture is broken!")
    print(f"Final loss: {loss.item():.4f}")

print("="*70)
