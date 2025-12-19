"""(moved to experiments/)
Test minimal model on ACTUAL sentence
"""
import torch
import torch.nn as nn

# Simple sentence
text = "The quick brown fox"  # Shorter than before
byte_vals = [ord(c) for c in text]

X = torch.tensor([byte_vals[:-1]]).long().cuda()
Y = torch.tensor([byte_vals[1:]]).long().cuda()

print(f"Sentence: '{text}'")
print(f"Length: {len(text)} chars")
print(f"X shape: {X.shape}")
print(f"Y shape: {Y.shape}\n")

# Minimal model
emb = nn.Embedding(256, 128).cuda()
head = nn.Linear(128, 256).cuda()
optimizer = torch.optim.Adam(list(emb.parameters()) + list(head.parameters()), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

print("Training...")
for i in range(200):
    h = emb(X)
    logits = head(h)
    loss = loss_fn(logits.reshape(-1, 256), Y.reshape(-1))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if i % 20 == 0:
        print(f"Step {i:3d}: Loss {loss.item():.4f}")

print(f"\nFinal loss: {loss.item():.6f}")

if loss.item() < 0.1:
    print("[SUCCESS] Overfitted!")
    with torch.no_grad():
        pred = logits.argmax(dim=-1)[0].cpu().numpy()
        pred_text = ''.join(chr(b) for b in pred)
        print(f"\nTarget:    '{text[1:]}'")
        print(f"Predicted: '{pred_text}'")
else:
    print(f"[FAIL] Stuck at {loss.item():.4f}")
