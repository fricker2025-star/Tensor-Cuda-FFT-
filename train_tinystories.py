"""
Train on TinyStories - The "Goldilocks" Dataset

Simple vocabulary + Perfect grammar + Narrative structure
Perfect for 2M parameter spectral models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from fft_tensor.cleanup import GPUContext
import os


class ProgressiveSpectralModel(nn.Module):
    """Progressive frequency model - scaled up."""
    
    def __init__(self, vocab_size=256, embed_dim=512, max_freq=2048, num_layers=6):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_freq = max_freq
        
        self.byte_embed = nn.Embedding(vocab_size, embed_dim)
        
        # More layers for better capacity
        self.layers = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(num_layers)
        ])
        
        self.register_buffer('freq_cutoff', torch.tensor(128))
        
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
    
    def set_frequency_cutoff(self, cutoff):
        self.freq_cutoff = torch.tensor(min(cutoff, self.max_freq))
    
    def forward(self, x):
        x = self.byte_embed(x)
        
        for layer in self.layers:
            residual = x
            
            # FFT
            x_freq = torch.fft.rfft(x, dim=1)
            
            # Progressive masking
            freq_bins = x_freq.size(1)
            cutoff_idx = min(self.freq_cutoff.item(), freq_bins)
            if cutoff_idx < freq_bins:
                x_freq[:, cutoff_idx:, :] = 0
            
            x_freq = x_freq * 0.95
            
            # iFFT
            x = torch.fft.irfft(x_freq, n=residual.size(1), dim=1)
            x = residual + self.dropout(layer(x))
        
        x = self.norm(x)
        logits = torch.matmul(x, self.byte_embed.weight.t())
        
        return logits


def download_tinystories():
    """Download TinyStories dataset."""
    print("\n" + "="*70)
    print("DOWNLOADING TINYSTORIES")
    print("="*70)
    
    try:
        from datasets import load_dataset
        
        print("Loading TinyStories from HuggingFace...")
        print("This may take a few minutes on first run...\n")
        
        # Load dataset
        dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        
        # Take subset (50,000 stories ~ 50MB)
        print("Extracting 50,000 stories...")
        
        text_data = []
        count = 0
        
        for item in dataset:
            if count >= 50000:
                break
            text_data.append(item['text'])
            count += 1
            
            if count % 5000 == 0:
                print(f"  Loaded {count} stories...")
        
        # Combine
        full_text = "\n\n".join(text_data)
        
        # Save
        with open("tinystories_train.txt", "w", encoding='utf-8') as f:
            f.write(full_text)
        
        print(f"\nSaved to: tinystories_train.txt")
        print(f"Size: {len(full_text):,} characters")
        print(f"Stories: {count:,}")
        
        return full_text
    
    except ImportError:
        print("\nERROR: 'datasets' package not installed")
        print("Install with: pip install datasets")
        return None
    except Exception as e:
        print(f"\nError downloading: {e}")
        print("\nUsing fallback demo data...")
        return create_demo_stories()


def create_demo_stories():
    """Fallback: Create demo stories in TinyStories style."""
    stories = [
        """Once upon a time, there was a little girl named Lily. She had a red ball that she loved to play with. One day, she went to the park with her mom.

At the park, Lily saw a big dog. The dog was very friendly. Lily threw her ball and the dog ran to get it. They played together for a long time.

When it was time to go home, Lily was happy. She had made a new friend. The dog wagged its tail and Lily waved goodbye. It was a good day.""",

        """Tom was a small boy who liked to help his dad. One day, his dad was fixing the car. Tom wanted to help too.

Tom gave his dad the tools. He was very careful. His dad said thank you. Tom felt proud.

After they finished, they went inside. Mom made cookies for them. Tom ate three cookies. They were delicious. Tom was tired but happy.""",

        """There was a cat named Whiskers. Whiskers was very curious. She liked to explore the house.

One morning, Whiskers found a box. The box was big and brown. Whiskers jumped inside. It was dark but cozy.

Whiskers stayed in the box for a while. Then she heard her owner calling. Whiskers jumped out and ran to get her food. She was hungry after her adventure.""",

        """Anna and Ben were best friends. They liked to play in the sandbox. One day, they decided to build a castle.

They worked together. Anna made the walls. Ben made the towers. The castle was beautiful.

A bird came and sat on top of their castle. Anna and Ben laughed. The bird flew away. They were proud of their castle. It was the best one they ever made.""",

        """Max had a new toy truck. It was blue and shiny. He loved his truck very much.

Max took his truck outside. He pushed it on the sidewalk. The truck went fast. Max was excited.

Then the truck rolled down the hill. Max ran after it. He caught it at the bottom. Max hugged his truck. He was glad he didn't lose it.""",
    ]
    
    # Repeat stories to get more data
    full_text = "\n\n".join(stories * 200)
    
    print("\nUsing demo stories (simplified TinyStories style)")
    print(f"Size: {len(full_text):,} characters")
    
    return full_text


def get_frequency_schedule(epoch):
    """Progressive schedule."""
    if epoch < 15:
        return 128
    elif epoch < 30:
        return 512
    elif epoch < 45:
        return 1024
    else:
        return 2048


def train_progressive(model, text, epochs=60, device='cuda'):
    """Train with progressive frequencies."""
    byte_vals = [ord(c) if ord(c) < 256 else ord('?') for c in text]
    
    # More sequences from larger dataset
    seqs = []
    seq_len = 256
    for i in range(0, len(byte_vals) - seq_len - 1, seq_len // 2):
        if len(seqs) >= 1000:  # More data
            break
        seqs.append((byte_vals[i:i+seq_len], byte_vals[i+1:i+seq_len+1]))
    
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    
    print(f"\nProgressive Training: {epochs} epochs on {len(seqs)} sequences")
    print("Schedule: 128 -> 512 -> 1024 -> 2048 frequencies")
    print("-" * 70)
    
    for epoch in range(epochs):
        freq_cutoff = get_frequency_schedule(epoch)
        model.set_frequency_cutoff(freq_cutoff)
        
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
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
            total_loss += loss.item()
            count += 1
        
        if (epoch + 1) % 5 == 0 and count > 0:
            avg_loss = total_loss / count
            print(f"  Epoch {epoch+1:3d}: Loss={avg_loss:.4f}, Freq={freq_cutoff}")


def generate_story(model, prompt, max_new=300, temperature=0.7, device='cuda'):
    """Generate a story."""
    model.eval()
    model.set_frequency_cutoff(model.max_freq)
    
    generated = [ord(c) for c in prompt]
    
    with torch.no_grad():
        for _ in range(max_new):
            ctx = generated[-256:]
            inp = torch.tensor([ctx], dtype=torch.long, device=device)
            
            logits = model(inp)
            next_logits = logits[0, -1, :]
            
            # Repetition penalty
            for byte_val in set(generated[-10:]):
                next_logits[byte_val] /= 1.2
            
            # Temperature + Top-K
            next_logits = next_logits / temperature
            v, _ = torch.topk(next_logits, 40)
            next_logits[next_logits < v[-1]] = float('-inf')
            
            probs = F.softmax(next_logits, dim=-1)
            next_byte = torch.multinomial(probs, num_samples=1).item()
            
            if 32 <= next_byte <= 126 or next_byte == 10:
                generated.append(next_byte)
    
    return ''.join(chr(b) if 32 <= b <= 126 or b == 10 else '?' for b in generated)


def main():
    print("\n" + "="*70)
    print("TRAINING ON TINYSTORIES")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    if device == 'cpu':
        print("CUDA required")
        return
    
    # Download or load TinyStories
    if os.path.exists("tinystories_train.txt"):
        print("\nLoading cached TinyStories...")
        with open("tinystories_train.txt", "r", encoding='utf-8') as f:
            text = f.read()
        print(f"Loaded: {len(text):,} characters")
    else:
        text = download_tinystories()
        if text is None:
            return
    
    print(f"\nTraining data: {len(text):,} characters")
    
    try:
        with GPUContext():
            # Scaled up model (2M params)
            model = ProgressiveSpectralModel(
                vocab_size=256,
                embed_dim=512,  # Increased
                max_freq=2048,
                num_layers=6    # More layers
            ).to(device)
            
            params = sum(p.numel() for p in model.parameters())
            print(f"Parameters: {params:,} (~{params/1e6:.1f}M)\n")
            
            # Train EXTREME
            print("\nEXTREME MODE: 10x LR, 200 epochs...")
            train_progressive(model, text, epochs=200, device=device)
            
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
                
                story = generate_story(model, prompt, 300, 0.7, device)
                print(story)
            
            print("\n" + "="*70)
            print("TINYSTORIES TRAINING COMPLETE")
            print("="*70)
            print("\nWhy TinyStories:")
            print("  - Simple vocabulary (3-year-old level)")
            print("  - Perfect grammar")
            print("  - Narrative structure (beginning -> end)")
            print("  - Teaches long-range causality")
            print("  - Ideal for 2M parameter models")
            print("="*70 + "\n")
    
    except Exception as e:
        print(f"\nError: {e}")
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
            torch.cuda.synchronize()
        print("[Exit] Clean")
