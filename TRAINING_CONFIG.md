# Aligned Training Configuration

## The Winner's Setup: Plateau Rule + Stage-Aligned LR

This configuration ensures cutoff raises are synchronized with LR restarts for maximum learning efficiency.

## Current Configuration (Fixed!)

### Frequency Cutoff Progression
```
Epoch 0-1:  cutoff = 128  (basic syntax/characters)
Epoch 2+:   cutoff = 256  (complex words, variable names)
Epoch 5+:   cutoff = 512  (fine details, spelling precision)
Epoch 10+:  cutoff = full (max detail)
```

### LR Schedule Stages (ALIGNED!)
```python
stage1_epochs = 2      # Epochs 0-1
stage1_lr_mult = 1.0   # Start at full LR
stage1_min_mult = 0.1  # Decay to 10%

stage2_epochs = 3      # Epochs 2-4
stage2_lr_mult = 1.0   # RESTART to full LR at epoch 2
stage2_min_mult = 0.1  # Decay to 10%

stage3_epochs = rest   # Epochs 5+
stage3_lr_mult = 1.0   # RESTART to full LR at epoch 5
stage3_min_mult = 0.05 # Decay to 5%
```

### The Magic: What Happens at Epoch 2

**Three things happen simultaneously:**

1. **Stage transition**: Enter stage 2 → LR resets to peak (base_lr * 1.0)
2. **Plateau detection**: Loss has plateaued at 128 → raise cutoff to 256
3. **LR restart flag**: `cutoff_raised=True` → forces LR to peak (safety net)

**Result:** New frequency bands (256) + Maximum plasticity (full LR) = Rapid learning ⚡

## Training Commands

### Start Fresh Training
```bash
python -m scripts.train_chunk_lm \
    --seq-len 1024 --kernel-len 128 --chunk 16 \
    --batch-size 4 --accum-steps 8 \
    --steps-per-epoch 1000 --epochs 50 \
    --lr 0.0002 --log-every 10 \
    --ckpt chunklm_ckpt.pt
```

### Resume Training
```bash
python -m scripts.train_chunk_lm \
    --seq-len 1024 --kernel-len 128 --chunk 16 \
    --batch-size 4 --accum-steps 8 \
    --steps-per-epoch 1000 --epochs 50 \
    --lr 0.0002 --log-every 10 \
    --ckpt chunklm_ckpt.pt --resume
```

The script auto-detects:
- Current cutoff from epoch number (no regression!)
- EMA configuration from checkpoint
- Proper LR restart points

## What to Watch For

### ✅ Good Signs (Training Working)

```
Epoch 1 avg_loss=2.86 cutoff=128/513
Epoch 2 🚀 CUTOFF RAISED: 128 -> 256 (loss plateaued, opening frequency gates)
  step 10/1000 loss=3.45 lr=2.0e-4 cutoff=256  ← SPIKE expected!
  step 50/1000 loss=3.12 lr=1.9e-4 cutoff=256  ← Recovering
  step 100/1000 loss=2.78 lr=1.7e-4 cutoff=256 ← Learning new details!
Epoch 2 avg_loss=2.65 cutoff=256/513
```

**Pattern:**
- 🚀 emoji signals cutoff raise
- Loss spikes temporarily (model adjusting to new detail)
- High LR (2.0e-4) at start
- Loss drops below previous plateau within epoch

### ⚠️ Warning Signs (Configuration Problem)

```
Epoch 2 🚀 CUTOFF RAISED: 128 -> 256
  step 10/1000 loss=3.45 lr=2.0e-5 cutoff=256  ← LR TOO LOW!
  step 50/1000 loss=3.44 lr=2.0e-5 cutoff=256  ← Not learning!
  step 100/1000 loss=3.43 lr=1.9e-5 cutoff=256 ← STUCK!
```

**Problem:** LR is 2.0e-5 instead of 2.0e-4 (10x too low)
**Cause:** Stage alignment broken or base LR set wrong
**Fix:** Check stage_epochs configuration

## Plateau Detection Parameters

```python
min_epoch_before_raise = 2      # Never raise before epoch 2
plateau_window = 50             # Look at last 50 optimizer steps
plateau_threshold = 0.005       # 0.5% improvement required
```

**Tuning:**
- Raising too fast? Increase `plateau_threshold` to 0.01 (1%)
- Raising too slow? Decrease to 0.003 (0.3%)
- Model needs more time at each level? Increase `min_epoch_before_raise`

## The Philosophy

**Before (Fixed Schedule):**
```
Epoch 20: Cutoff raises
Problem: Maybe loss already plateaued at epoch 5 → wasted 15 epochs
         OR maybe model not ready → introduces noise too early
```

**After (Plateau Rule):**
```
Epoch X: Loss plateaus → Cutoff raises → LR restarts
Result: Always optimal timing, no wasted compute
        Model gets new detail exactly when ready to learn it
```

**Stage Alignment Bonus:**
- LR schedule naturally aligns with expected cutoff raises
- Stage transitions give LR restarts at typical plateau points
- Cutoff_raised flag is safety net for dynamic timing

## Summary

✅ **Start at 128** - Skip 64 (too blurry)
✅ **Stages aligned** - Epochs 0-1, 2-4, 5+ match expected plateaus  
✅ **LR restarts sync** - High plasticity when opening new frequency gates
✅ **Smart resuming** - Never regress to lower cutoff
✅ **Plateau driven** - Adapt to actual training dynamics

**Result:** Maximum learning efficiency, no wasted compute, no manual schedule tuning.

🐎 **Expect the mule to kick when you open the floodgates!**
