"""
Test OUR exact loss pattern vs the golden pattern
"""
import torch
import torch.nn as nn

X = torch.tensor([[1, 2, 3, 4]]).long().cuda()
Y = torch.tensor([[2, 3, 4, 5]]).long().cuda()

emb = nn.Embedding(256, 64).cuda()
head = nn.Linear(64, 256).cuda()
optimizer = torch.optim.Adam(list(emb.parameters()) + list(head.parameters()), lr=0.1)
loss_fn = nn.CrossEntropyLoss()

print("="*70)  # (moved to experiments/)
print("Testing OUR loss pattern: reshape(-1, 256)")
print("="*70)

for i in range(50):
    h = emb(X)          # [1, 4, 64]
    logits = head(h)    # [1, 4, 256]
    
    # OUR PATTERN: reshape instead of transpose
    loss = loss_fn(logits.reshape(-1, 256), Y.reshape(-1))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if i % 10 == 0:
        print(f"Step {i:2d}: Loss {loss.item():.4f}")

print("-" * 70)

if loss.item() < 0.01:
    print(f"\n[SUCCESS] Our loss pattern works!")
    print(f"Final loss: {loss.item():.6f}")
else:
    print(f"\n[FAIL] Our loss pattern is broken!")
    print(f"Final loss: {loss.item():.4f}")

print("="*70)
