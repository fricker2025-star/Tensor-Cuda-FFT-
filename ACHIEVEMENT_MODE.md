# Achievement Mode: Beat Your Own Score

## The New Unlocking System

Instead of fixed thresholds, each frequency band unlocks when you **beat your previous score**!

### How It Works

```
Level 1: cutoff=128
├─ Goal: Loss < 3.2 (initial threshold)
├─ You achieve: 3.15
└─ 🏆 UNLOCK cutoff=256, set bar at 3.15

Level 2: cutoff=256  
├─ Goal: Loss < 3.15 (your previous score)
├─ Loss spikes to 3.4 (new frequencies are hard!)
├─ Loss drops to 3.08
└─ 🏆 UNLOCK cutoff=512, set bar at 3.08

Level 3: cutoff=512
├─ Goal: Loss < 3.08 (your previous score)
├─ You achieve: 3.02
└─ 🏆 UNLOCK full resolution, set bar at 3.02

Level 4: Full resolution
├─ Goal: Loss < 3.02 (your previous score)
└─ Keep improving until convergence!
```

## Why This Is Better

### Old System (Fixed Thresholds)
```
128 → 256: Need loss < 3.0
256 → 512: Need loss < 2.5  
512 → full: Need loss < 2.0

Problem: Too conservative! You waste time at each level.
```

### New System (Adaptive Thresholds)
```
128 → 256: Need loss < 3.2  (more aggressive!)
256 → 512: Need loss < [whatever you got at 256]
512 → full: Need loss < [whatever you got at 512]

Advantage: Unlocks as soon as you're ready, each level is a fair challenge!
```

## Expected Training Timeline

```
Steps 0-400:
  loss: 4.5 → 3.2 → 3.15
  cutoff: 128
  [UNLOCK] 128 -> 256 (loss=3.150)
  Next unlock target: loss < 3.150

Steps 400-800:
  loss: 3.4 → 3.2 → 3.08  (spike then drop)
  cutoff: 256
  [UNLOCK] 256 -> 512 (loss=3.080, beat prev 3.150)
  Next unlock target: loss < 3.080

Steps 800-1500:
  loss: 3.15 → 3.0 → 2.95
  cutoff: 512
  [UNLOCK] 512 -> 513 (loss=2.950, beat prev 3.080)
  Next unlock target: loss < 2.950

Steps 1500+:
  loss: 3.0 → 2.5 → 2.0 → 1.5
  cutoff: full (513)
  Keep training to convergence!
```

## What You'll See in Logs

```
Epoch 0:
  step 10/1000 loss=4.250 lr=2.0e-4 cutoff=128
  step 20/1000 loss=3.850 lr=2.0e-4 cutoff=128
  step 30/1000 loss=3.450 lr=1.99e-4 cutoff=128
  step 40/1000 loss=3.180 lr=1.99e-4 cutoff=128
  [UNLOCK] 128 -> 256 (loss=3.178)
    Next unlock target: loss < 3.178
  step 50/1000 loss=3.420 lr=2.0e-4 cutoff=256  ← LR restarted, loss spikes
  step 60/1000 loss=3.250 lr=1.99e-4 cutoff=256
  step 70/1000 loss=3.150 lr=1.99e-4 cutoff=256
  [UNLOCK] 256 -> 512 (loss=3.145, beat prev 3.178)
    Next unlock target: loss < 3.145
  Hemisphere: Freq=52.0% Time=48.0%
```

## Why The Loss Spikes

**Expected behavior:**
```
At 128: loss=3.15 → Unlock 256
At 256: loss=3.40 (spike!) → Drops to 3.10

Why spike? New frequencies = new information = temporary confusion
Why drop? The time path still knows spelling, helps stabilize
```

**The bicameral advantage:**
- **Frequency path**: Struggles with new bands (spike)
- **Time path**: Still sharp (no cutoff, always full detail)
- **Together**: Spike is smaller than frequency-only would be!

## Tuning the Initial Threshold

Default: `unlock_threshold=3.2`

**Too aggressive** (unlocking too early):
```python
unlock_threshold=3.5  # Will unlock sooner, but might be unstable
```

**Too conservative** (unlocking too late):
```python
unlock_threshold=2.8  # Will unlock later, more stable but slower
```

**Recommended:** Keep default 3.2 and let the adaptive system handle the rest!

## The "Beat Your Score" Philosophy

This creates a consistent challenge throughout training:

1. **First unlock**: Achieve 3.2 (fixed target, moderate difficulty)
2. **Subsequent unlocks**: Beat your previous best (adaptive, always fair)

**Result:** The model is always challenged at the right level - not too easy, not too hard.

Like a good video game, each level is tough but achievable! 🎮🏆

## Command

```bash
C:\Users\Aaron\AppData\Local\Programs\Python\Python312\python.exe -m scripts.train_bicameral \
    --seq-len 1024 --kernel-len 128 --chunk 16 \
    --batch-size 4 --accum-steps 8 --steps-per-epoch 1000 --epochs 50 \
    --lr 0.0002 --log-every 10 \
    --ckpt bicameral_ckpt.pt \
    --achievement-mode
```

Watch the unlocks happen in real-time! Each one means your model leveled up. 🚀⚡
