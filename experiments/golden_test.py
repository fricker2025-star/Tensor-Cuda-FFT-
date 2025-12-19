"""
GOLDEN TEST - Proves PyTorch/CUDA work

If this fails, PyTorch is broken.
If this works, OUR code is buggy.
"""
import torch
import torch.nn as nn

# Deterministic Data (Predict Next Number)
# Input:  1, 2, 3, 4
# Target: 2, 3, 4, 5
X = torch.tensor([[1, 2, 3, 4]]).long().cuda()
Y = torch.tensor([[2, 3, 4, 5]]).long().cuda()

# Linear Brain
emb = nn.Embedding(256, 64).cuda()  # Vocab=256, Dim=64
head = nn.Linear(64, 256).cuda()    # Map back to Vocab

# Train
optimizer = torch.optim.Adam(list(emb.parameters()) + list(head.parameters()), lr=0.1)
loss_fn = nn.CrossEntropyLoss()

print("="*70)  # (moved to experiments/)
print("GOLDEN TEST - Minimal PyTorch Sanity Check")
print("="*70)
print("\nInput:  [1, 2, 3, 4]")
print("Target: [2, 3, 4, 5]\n")
print("Starting training...")
print("-" * 70)

for i in range(50):
    # Forward
    h = emb(X)          # [1, 4, 64]
    logits = head(h)    # [1, 4, 256]
    
    # CRITICAL: Transpose for Loss [B, V, T]
    loss = loss_fn(logits.transpose(1, 2), Y)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if i % 10 == 0:
        print(f"Step {i:2d}: Loss {loss.item():.4f}")

print("-" * 70)

if loss.item() < 0.01:
    print(f"\n[SUCCESS] Final loss: {loss.item():.6f}")
    print("PyTorch/CUDA work correctly!")
    
    # Test prediction
    with torch.no_grad():
        pred = logits.argmax(dim=-1)
        print(f"\nPredicted: {pred[0].tolist()}")
        print(f"Target:    {Y[0].tolist()}")
else:
    print(f"\n[FAIL] Final loss: {loss.item():.4f}")
    print("Something is fundamentally broken!")

print("="*70)
