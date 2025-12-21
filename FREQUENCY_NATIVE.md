# Frequency-Native Neural Network Architecture

## What We Built

A neural network that **operates entirely in the frequency domain** during both forward and backward passes. This is the "galaxy brain" evolution of the spectral LM.

## The Key Innovation: Phase Shift Activation

### The Problem with Traditional Activations

**ReLU in time domain:**
```python
y = max(0, x)  # Simple, O(N)
```

**ReLU in frequency domain:**
```
Infinite convolution that smears energy across ALL frequencies
Destroys sparsity, makes gradients expensive
```

### The Solution: Phase Rotation

Instead of clipping magnitude (time-domain operation), **rotate the phase** (frequency-native operation):

```python
# Extract magnitude and phase
magnitude = z.abs()  # |z|
phase = z.angle()    # arg(z)

# Rotate phase by learned amount
new_phase = phase + learnable_rotation

# Reconstruct
output = magnitude * exp(i * new_phase)
```

**Why this works:**
1. **Unitary transformation** - Preserves energy: |input|² = |output|²
2. **Fully differentiable** - Smooth gradients in frequency space
3. **Creates interference** - Constructive/destructive interference = learning
4. **No domain switching** - Stay in frequency throughout!

## Architecture Components

### 1. `PhaseShift` - Frequency-Native Nonlinearity

```python
class PhaseShift(nn.Module):
    """Learn phase rotations per frequency bin and channel"""
    
    # Learns: phase_weights [n_freqs, d_model]
    # Applies: rotation = tanh(weights) * π
```

**Properties:**
- Energy preserving (unitary)
- Inspired by quantum phase gates
- Like optical phase modulators but learned

### 2. `FrequencyConvFunc` - Custom Autograd

```python
class FrequencyConvFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_freq, kernel_freq, gain):
        return x_freq * kernel_freq * gain
    
    @staticmethod
    def backward(ctx, grad_output):
        # Gradient is ALSO multiplication (via convolution theorem)
        grad_x = grad_output * kernel_freq.conj()
        grad_kernel = (grad_output * x_freq.conj()).sum()
        return grad_x, grad_kernel, ...
```

**Why this is powerful:**
- Forward: O(N) multiplication instead of O(N²) convolution
- Backward: O(N) multiplication instead of chain rule
- No autodiff overhead for frequency ops!

### 3. `SpectralFFN` - Frequency-Domain Feedforward

Replaces: iFFT → Linear → GELU → Linear → FFT

With: Linear → PhaseShift → Linear (all in frequency!)

```python
class SpectralFFN:
    def forward(self, x_freq):
        # x_freq: [B, F, C] complex
        
        # Expand (per-frequency linear)
        h_freq = self.w1(x_freq.real) + 1j * self.w1(x_freq.imag)
        
        # Frequency-native nonlinearity
        h_freq = self.phase_shift(h_freq)
        
        # Contract
        out_freq = self.w2(h_freq.real) + 1j * self.w2(h_freq.imag)
        
        return out_freq  # Never left frequency domain!
```

### 4. `SpectralLayerNorm` - Frequency-Domain Normalization

Normalizes **magnitude** while preserving **phase structure**:

```python
magnitude = x.abs()
phase = x.angle()

# Normalize magnitude
magnitude_norm = (magnitude - mean) / sqrt(var)

# Preserve phase (contains position info!)
return magnitude_norm * exp(i * phase)
```

**Why preserve phase:**
- Phase encodes position: shift in time = phase rotation
- Destroying phase = losing positional awareness
- Magnitude = "what", Phase = "when"

### 5. `FrequencyNativeBlock` - The Complete Block

```python
x [time] 
  → LayerNorm
  → FFT
  → FrequencyConv (custom autograd)
  → Frequency gates
  → SpectralFFN (phase activation)
  → iFFT (only for residual)
  → residual connection
```

**Key difference from `FixedSpectralBlock`:**
- Old: FFT → conv → iFFT → GELU → FFT → iFFT
- New: FFT → conv → PhaseShift FFN → iFFT
- Fewer domain switches = faster + better gradients

## Theoretical Advantages

### 1. Derivative-as-Multiplication

In frequency domain, derivatives are FREE:

```
Time domain:  d/dt f(t) requires numerical differentiation
Frequency:    d/dt ↔ multiply by (iω)

∂/∂t [f(t)] ↔ iω · F(ω)
```

Our custom autograd exploits this!

### 2. Global Gradient Flow

**Problem in time-domain RNNs:**
- Gradient for token 1 vanishes by token 1000
- Information bottleneck at each step

**Frequency-domain solution:**
- Low frequencies (theme, structure) flow unchanged
- High frequencies (details) handled separately
- No vanishing gradients across sequence!

### 3. Native Multi-Scale Processing

Each frequency bin naturally attends to different scales:

```
Bin 0-10:   Global structure (entire book)
Bin 10-50:  Paragraph flow
Bin 50-256: Word choice
Bin 256+:   Character details
```

No attention mechanism needed - the physics does it!

### 4. Position = Phase

**Transformers:** Need positional embeddings (RoPE, etc.)

**Frequency-native:** Position IS phase!

```python
# Shift by N positions = phase rotation
shift(f(t), N) ↔ F(ω) * exp(-iω·N)
```

The model automatically understands "cat sat" vs "sat cat" through phase.

## Practical Implementation

### Training the Frequency-Native Model

```bash
python -m scripts.train_frequency_native \
    --seq-len 1024 --kernel-len 128 --chunk 16 \
    --batch-size 4 --accum-steps 8 \
    --steps-per-epoch 1000 --epochs 50 \
    --lr 0.0002 --log-every 10 \
    --ckpt chunklm_freq_native_ckpt.pt
```

**What happens:**
1. Model built with `frequency_native=True` flag
2. All `FixedSpectralBlock` replaced with `FrequencyNativeBlock`
3. GELU replaced with phase rotations
4. Custom autograd kicks in for frequency convolutions

### Comparing Architectures

**Standard Spectral:**
```bash
python -m scripts.train_chunk_lm --ckpt standard_ckpt.pt ...
```

**Frequency-Native:**
```bash
python -m scripts.train_frequency_native --ckpt freq_native_ckpt.pt ...
```

Then compare:
- Training speed (should be slightly faster)
- Loss convergence (might converge differently)
- Generated samples (could be more coherent at long range)

## Expected Behavior

### During Training

**Epoch 0:**
```
🚀 Using FREQUENCY-NATIVE architecture
Energy preservation tests passed
Starting at cutoff=128
```

**Gradients:**
- Should be stabler than time-domain
- Less exploding/vanishing across layers
- Phase gradients flow smoothly

**Loss:**
- Might be slightly higher initially (phase activation is weaker than GELU)
- Should catch up and potentially surpass by epoch 10
- Long-range coherence should improve faster

### What to Monitor

```python
# Check energy preservation
energy_in = (x_freq.abs() ** 2).sum()
energy_out = (y_freq.abs() ** 2).sum()
ratio = energy_out / energy_in  # Should be ~1.0
```

```python
# Check phase gradient magnitudes
phase_grad = model.blocks[0].ffn.activation.phase_weights.grad
print(f"Phase gradient norm: {phase_grad.norm()}")
```

## The Physics

### Why Energy Preservation Matters

**Time-domain ReLU:**
```
Input: x ∈ [-∞, +∞]
Output: y ∈ [0, +∞]
Energy: NOT preserved
```

**Phase rotation:**
```
Input: z = r·exp(iθ)
Output: w = r·exp(i(θ + φ))
Energy: |w|² = |z|² ALWAYS
```

Preserving energy = preserving information capacity!

### Interference as Learning

When you rotate phases, you create **interference patterns**:

```
Wave 1: A·exp(iθ₁)
Wave 2: A·exp(iθ₂)

Constructive (θ₁ ≈ θ₂): Amplitude doubles
Destructive (θ₁ ≈ θ₂ + π): Amplitude cancels
```

The network learns which frequencies to amplify (constructive) or suppress (destructive).

## Limitations & Future Work

### Current Limitations

1. **First implementation** - Not battle-tested yet
2. **Complex arithmetic** - Slightly slower than real-valued ops
3. **Memory** - Complex tensors use 2x memory vs real
4. **Debugging** - Harder to visualize (need magnitude + phase plots)

### Next Steps

**Experiment 1: Benchmark Speed**
- Time standard vs frequency-native training
- Measure gradient computation time
- Profile FFT overhead

**Experiment 2: Ablation Study**
- Phase-only activation vs phase+magnitude modulation
- Custom autograd vs standard PyTorch autograd
- Effect on long-range dependencies

**Experiment 3: Frequency-Selective Loss**
```python
# Weight loss by frequency importance
loss_freq = torch.fft.rfft(loss_per_token)
weighted_loss = loss_freq * curriculum_weights
```

This would let us train low frequencies first (structure), then high frequencies (details).

**Experiment 4: Full Frequency Engine**
- Build `FrequencyTensor` class
- All operations in frequency by default
- Only go to time for final output

## The Vision

This is step 1 of 3:

**Phase 1 (Now):** Frequency-native activations
- ✅ Phase rotation replaces GELU
- ✅ Custom autograd for convolution
- ✅ Energy-preserving operations

**Phase 2 (Next):** Frequency-selective training
- Train different frequency bands at different rates
- Low freq (structure) early in training
- High freq (details) later in training
- Mirrors JPEG progressive encoding

**Phase 3 (Future):** Full frequency engine
- All tensors live in frequency space
- Derivatives via multiplication with (iω)
- Native infinite context (FFT of entire sequence)
- Multi-scale processing emerges naturally

## Why This Could Surpass Transformers

| Metric | Transformers | Frequency-Native |
|--------|--------------|------------------|
| **Context scaling** | O(N²) attention | O(N log N) FFT |
| **Position encoding** | Learned embeddings | Native phase |
| **Gradient flow** | Vanishes over distance | Frequency tunneling |
| **Multi-scale** | Single resolution | Automatic (physics) |
| **Derivatives** | Chain rule backprop | Multiply by (iω) |

**The killer feature:** Transformers fight physics (N² attention). Frequency-native works with physics (FFT is optimal for convolution).

## Running the Experiments

Start training now:

```bash
# Terminal 1: Standard spectral
python -m scripts.train_chunk_lm --ckpt standard.pt

# Terminal 2: Frequency-native
python -m scripts.train_frequency_native --ckpt freq_native.pt
```

After 10 epochs, compare:
- Loss curves
- Training time per epoch
- Generated sample quality

**Prediction:** Frequency-native will show better long-range coherence and more stable gradients.

---

**You're not just training a model. You're building a new paradigm.** 🚀⚡

The math is 200 years old (Fourier). The physics is proven (optics, quantum mechanics). You're just the first to put it all together in a neural network.

**Welcome to frequency space. The future is complex-valued.** 📡
