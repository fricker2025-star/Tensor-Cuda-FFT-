"""
RETRAIN with FIXED spectral mixing (moved to experiments/)

Now that we mix in FREQUENCY domain (not time), let's see how smart it gets!
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from fft_tensor.cleanup import GPUContext


class FixedSpectralModel(nn.Module):
    """Spectral model with CORRECT frequency domain mixing."""
    
    def __init__(self, vocab_size=256, embed_dim=512, num_layers=6, max_seq_len=512):
        super().__init__()
        
        self.byte_embed = nn.Embedding(vocab_size, embed_dim)
        
        # Learnable COMPLEX filters for each layer
        # Max freq bins for seq_len
        max_freq_bins = max_seq_len // 2 + 1
        
        self.freq_filters_real = nn.ParameterList([
            nn.Parameter(torch.randn(max_freq_bins, embed_dim) * 0.02)
            for _ in range(num_layers)
        ])
        self.freq_filters_imag = nn.ParameterList([
            nn.Parameter(torch.randn(max_freq_bins, embed_dim) * 0.02)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = self.byte_embed(x)
        
        for filter_real, filter_imag in zip(self.freq_filters_real, self.freq_filters_imag):
            residual = x
            
            # FFT on SEQUENCE
            x_freq = torch.fft.rfft(x, dim=1)
            
            # MIX IN FREQUENCY DOMAIN!
            freq_filter = torch.complex(filter_real, filter_imag)
            
            # Adjust filter size if needed
            actual_bins = x_freq.size(1)
            if actual_bins < freq_filter.size(0):
                freq_filter = freq_filter[:actual_bins]
            
            # Apply complex filter
            x_freq = x_freq * freq_filter.unsqueeze(0)
            
            # iFFT back
            x = torch.fft.irfft(x_freq, n=residual.size(1), dim=1)
            
            # Residual + dropout
            x = residual + self.dropout(x)
        
        x = self.norm(x)
        
        # Output
        logits = torch.matmul(x, self.byte_embed.weight.t())
        
        return logits


def train_fixed(model, text, epochs=200, device='cuda'):
    """Train with FIXED architecture."""
    byte_vals = [ord(c) if ord(c) < 256 else ord('?') for c in text]
    
    seqs = []
    seq_len = 256
    for i in range(0, len(byte_vals) - seq_len - 1, seq_len // 2):
        if len(seqs) >= 1000:
            break
        seqs.append((byte_vals[i:i+seq_len], byte_vals[i+1:i+seq_len+1]))
    
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    
    print(f"Training FIXED model: {epochs} epochs on {len(seqs)} sequences")
    print("-" * 70)
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        count = 0
        
        for i in range(0, len(seqs), 4):
            batch = seqs[i:i+4]
            if len(batch) < 4:
                continue
            
            inp = torch.tensor([s[0] for s in batch], dtype=torch.long, device=device)
            tgt = torch.tensor([s[1] for s in batch], dtype=torch.long, device=device)
            
            opt.zero_grad()
            out = model(inp)
            loss = crit(out.reshape(-1, 256), tgt.reshape(-1))
            
            if not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                
                total_loss += loss.item()
                count += 1
        
        if count == 0:
            continue
        
        avg_loss = total_loss / count
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            marker = " [BEST]"
        else:
            marker = ""
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}: Loss={avg_loss:.4f}, Best={best_loss:.4f}{marker}")
    
    return best_loss


def generate_story(model, prompt, max_new=300, device='cuda'):
    model.eval()
    
    generated = [ord(c) for c in prompt]
    
    with torch.no_grad():
        for _ in range(max_new):
            ctx = generated[-256:]
            inp = torch.tensor([ctx], dtype=torch.long, device=device)
            
            logits = model(inp)
            next_logits = logits[0, -1, :]
            
            # Repetition penalty
            for b in set(generated[-10:]):
                next_logits[b] /= 1.2
            
            # Sample
            next_logits = next_logits / 0.7
            v, _ = torch.topk(next_logits, 40)
            next_logits[next_logits < v[-1]] = float('-inf')
            
            probs = F.softmax(next_logits, dim=-1)
            next_byte = torch.multinomial(probs, 1).item()
            
            if 32 <= next_byte <= 126 or next_byte == 10:
                generated.append(next_byte)
    
    return ''.join(chr(b) if 32 <= b <= 126 or b == 10 else '?' for b in generated)


def main():
    print("\n" + "="*70)
    print("RETRAIN WITH FIXED SPECTRAL MIXING")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    if device == 'cpu':
        return
    
    # Load TinyStories
    try:
        with open("tinystories_train.txt", "r", encoding='utf-8') as f:
            text = f.read()
        print(f"Loaded TinyStories: {len(text):,} characters\n")
    except FileNotFoundError:
        print("ERROR: tinystories_train.txt not found")
        return
    
    try:
        with GPUContext():
            # Fixed model
            model = FixedSpectralModel(
                vocab_size=256,
                embed_dim=512,
                num_layers=6,
                max_seq_len=512
            ).to(device)
            
            params = sum(p.numel() for p in model.parameters())
            print(f"Parameters: {params:,} (~{params/1e6:.1f}M)\n")
            
            # Train with FIXED architecture
            best_loss = train_fixed(model, text, epochs=200, device=device)
            
            print("\n" + "="*70)
            print(f"FINAL LOSS: {best_loss:.4f}")
            print("="*70)
            
            # Generate stories
            print("\n" + "="*70)
            print("STORY GENERATION")
            print("="*70)
            
            prompts = [
                "Once upon a time",
                "There was a little girl",
                "One day, a boy",
            ]
            
            for prompt in prompts:
                print(f"\n{'='*70}")
                print(f"Prompt: '{prompt}'")
                print("="*70)
                
                story = generate_story(model, prompt, 300, device)
                print(story)
            
            print("\n" + "="*70)
            print("THE FIX THAT CHANGED EVERYTHING")
            print("="*70)
            print("""
OLD (BROKEN):
  x_freq = fft(x)
  x_time = ifft(x_freq)  ← Identity! No mixing!
  x = linear(x_time)
  
NEW (FIXED):
  x_freq = fft(x)
  x_freq = x_freq * learnable_filter  ← Mix in FREQUENCY!
  x_time = ifft(x_freq)
  
Result: Context actually flows through the model!
            """)
            print("="*70 + "\n")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    try:
        main()
    finally:
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[Exit]")
