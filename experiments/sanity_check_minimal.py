"""(moved to experiments/)
MINIMAL MODEL - Absolute simplest possible

NO residuals, NO dropout, NO normalization tricks
"""
import torch
import torch.nn as nn


class MinimalModel(nn.Module):
    """Simplest possible model."""
    
    def __init__(self, vocab_size=256, embed_dim=128):
        super().__init__()
        
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.hidden = nn.Linear(embed_dim, embed_dim)
        self.output = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, x):
        x = self.embed(x)
        x = torch.relu(self.hidden(x))
        x = self.output(x)
        return x


def test():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*70)
    print("MINIMAL MODEL TEST")
    print("="*70)
    
    text = "The quick brown fox jumps over the lazy dog."
    print(f"\nTarget: '{text}'\n")
    
    byte_vals = [ord(c) for c in text]
    inp = torch.tensor([byte_vals[:-1]], dtype=torch.long, device=device)
    tgt = torch.tensor([byte_vals[1:]], dtype=torch.long, device=device)
    
    model = MinimalModel(256, 128).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    
    print("Training minimal model...")
    print("-" * 70)
    
    for epoch in range(200):
        opt.zero_grad()
        out = model(inp)
        loss = crit(out.reshape(-1, 256), tgt.reshape(-1))
        loss.backward()
        opt.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}: Loss={loss.item():.6f}")
        
        if loss.item() < 0.01:
            print(f"\n[SUCCESS] Overfitted at epoch {epoch+1}!")
            
            # Test
            with torch.no_grad():
                pred = model(inp).argmax(dim=-1)[0].cpu().numpy()
                pred_text = ''.join(chr(b) for b in pred if 32 <= b <= 126)
                print(f"\nTarget:    '{text[1:]}'")
                print(f"Predicted: '{pred_text}'")
            
            return True
    
    print(f"\n[FAIL] Loss: {loss.item():.6f}")
    return False


if __name__ == '__main__':
    success = test()
    
    print("\n" + "="*70)
    if success:
        print("MINIMAL MODEL WORKS!")
        print("Problem is in the complex architecture")
    else:
        print("EVEN MINIMAL MODEL FAILS!")
        print("Problem is fundamental - maybe:")
        print("  - Sequence too short?")
        print("  - Cross-entropy setup?")
        print("  - PyTorch version bug?")
    print("="*70)
