# Learning Rate Physics: Why High Frequencies Need Low LR

## The Problem: Oscillation After Unlock

**What you see:**
```
step 300/1000 loss=3.003 cutoff=128
[PLATEAU] 128 -> 512
step 310/1000 loss=3.450 cutoff=512  (spike - expected)
step 320/1000 loss=3.150 cutoff=512  (recovering)
step 330/1000 loss=3.000 cutoff=512  (good!)
step 340/1000 loss=3.110 cutoff=512  (wait, went UP?!)
step 350/1000 loss=2.990 cutoff=512  
step 360/1000 loss=3.105 cutoff=512  (oscillating!)
```

**The oscillation (3.00 ↔ 3.11) means the LR is TOO HIGH for the new frequency resolution.**

## The Physics

### Low Frequencies (cutoff=128): Smooth Hills 🏔️

```
Loss landscape looks like:

    ╱╲    ╱╲
   ╱  ╲  ╱  ╲
  ╱    ╲╱    ╲

Smooth, gradual slopes
Can take BIG steps (high LR = 2.0e-4)
Won't overshoot the minimum
```

**Why:** Low frequencies = averaged/smoothed signal = smooth gradients

### High Frequencies (cutoff=512): Jagged Cliffs ⛰️

```
Loss landscape looks like:

 /\  /\/\  /\
/  \/    \/  \
   ↑ Sharp cliffs!

Jagged, steep drops
BIG steps = overshoot and bounce
Need SMALL steps (low LR = 1.0e-4)
```

**Why:** High frequencies = sharp details = steep, narrow valleys

## The Oscillation Explained

```
Step 330: loss=3.000, gradient points DOWN
          ↓
          LR too high! Step size = 0.0002
          ↓
          OVERSHOOT the minimum, land on opposite cliff
          ↓
Step 340: loss=3.110 (went UP!)
          Gradient now points BACK
          ↓
          Take another big step
          ↓
Step 350: loss=2.990 (back down, but overshot again)
          ↓
          REPEAT FOREVER (oscillation)
```

**The model is bouncing between two cliffs, never settling in the valley.**

## The Fix: Automatic LR Reduction

### Implementation

```python
if cutoff_raised:  # Unlocked from 128 -> 512
    old_lr = cfg.lr
    cfg.lr = cfg.lr * 0.5  # Cut in HALF
    
    print(f"LR REDUCED: {old_lr:.2e} -> {cfg.lr:.2e}")
    print("Reason: High frequencies need precision navigation")
```

### Why 0.5 (50% reduction)?

**Rule of thumb:** LR should scale inversely with frequency resolution

```
cutoff=128: 25% of frequencies  → LR = 2.0e-4 (can run fast)
cutoff=512: 100% of frequencies → LR = 1.0e-4 (must walk carefully)

Ratio: 512/128 = 4x more frequencies
LR reduction: 1/2 = 0.5x slower

This is conservative (could go 1/4 = 0.25x for even more precision)
```

### Expected Behavior After Fix

```
step 300/1000 loss=3.003 lr=1.8e-4 cutoff=128

[PLATEAU] 128 -> 512
  LR REDUCED: 2.0e-4 -> 1.0e-4 (precision mode)

step 310/1000 loss=3.450 lr=1.0e-4 cutoff=512  (spike)
step 320/1000 loss=3.150 lr=0.99e-4 cutoff=512 (recovering)
step 330/1000 loss=3.000 lr=0.98e-4 cutoff=512 (smooth!)
step 340/1000 loss=2.950 lr=0.97e-4 cutoff=512 (descending)
step 350/1000 loss=2.900 lr=0.96e-4 cutoff=512 (no oscillation!)
step 360/1000 loss=2.850 lr=0.95e-4 cutoff=512 (steady progress)
```

**The oscillation stops. Loss descends smoothly.**

## Analogy: Driving on Different Roads

### Highway (Low Frequencies)
- Smooth, straight road
- Drive fast: 80 mph (LR = 2.0e-4)
- Safe to cruise

### Mountain Trail (High Frequencies)  
- Sharp turns, steep drops
- Drive slow: 30 mph (LR = 1.0e-4)
- One wrong move = fall off cliff (diverge)

**You can't drive 80 mph on a mountain trail!**

## Alternative Strategies

### Strategy 1: Aggressive (0.5x reduction)
```python
cfg.lr *= 0.5  # Cut in half
```
- **Pro:** Safe, prevents oscillation
- **Con:** Might be too slow
- **Use when:** High oscillation risk (128 → 512 jump)

### Strategy 2: Moderate (0.7x reduction)
```python
cfg.lr *= 0.7  # Reduce by 30%
```
- **Pro:** Faster convergence than 0.5x
- **Con:** Might still oscillate slightly
- **Use when:** Smaller jumps (256 → 512)

### Strategy 3: Adaptive (based on loss spike)
```python
loss_spike = new_loss - old_loss
if loss_spike > 0.3:
    cfg.lr *= 0.5  # Big spike = aggressive reduction
elif loss_spike > 0.1:
    cfg.lr *= 0.7  # Medium spike = moderate reduction
else:
    cfg.lr *= 0.9  # Small spike = gentle reduction
```

## Current Implementation

**We use Strategy 1 (0.5x aggressive):**

```python
# At unlock: 128 -> 512
old_lr = 2.0e-4
new_lr = 1.0e-4  # 50% reduction

# Rationale:
# - 4x frequency increase (128 -> 512)
# - 2x LR decrease (conservative safety margin)
# - Prevents oscillation
# - Can always increase later if too slow
```

## Monitoring

**Good signs (LR is correct):**
```
loss: 3.0 → 2.95 → 2.90 → 2.85 → 2.80 (smooth descent)
```

**Bad signs (LR too high):**
```
loss: 3.0 → 3.1 → 2.9 → 3.1 → 2.95 (oscillating)
       ↑ Bouncing between cliffs!
```

**Bad signs (LR too low):**
```
loss: 3.0 → 2.999 → 2.998 → 2.997 (painfully slow)
       ↑ Making progress, but barely
```

## Summary

**The Rule:** When you increase frequency resolution, DECREASE learning rate.

**The Physics:** High frequencies = jagged loss landscape = need small steps.

**The Fix:** Automatic 0.5x LR reduction on unlock (now implemented).

**The Result:** Smooth loss descent instead of oscillation.

🏔️ **Navigate the cliffs carefully, and you'll reach the valley!** 🎯
