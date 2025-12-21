# The Plateau Rule: Dynamic Frequency Cutoff with LR Synchronization

## Overview

This training strategy implements **adaptive frequency curriculum learning** tied to **learning rate restarts**. Instead of blindly raising the frequency cutoff on a fixed schedule, we watch the loss and only increase the cutoff when the model has plateaued.

## The Problem

**Fixed schedules waste compute:**
- If loss hits 2.85 at epoch 3 and stays there, waiting until epoch 20 to raise the cutoff is like forcing a college student to sit through kindergarten
- If you raise too early, you introduce high-frequency noise before the structure is solid → model confusion
- If you raise too late, you waste electricity teaching what's already learned

## The Solution: Performance-Based Curriculum

### 1. Plateau Detection

The system monitors a sliding window of recent losses (default: last 50 optimizer steps):

```python
# Compare first half vs second half of window
first_half_avg = mean(losses[0:25])
second_half_avg = mean(losses[25:50])

# Calculate relative improvement
rel_improvement = (first_half_avg - second_half_avg) / first_half_avg

# If improvement < 0.5%, it's a plateau
if rel_improvement < 0.005:
    # RAISE THE CUTOFF
```

### 2. Progressive Cutoff Schedule

When a plateau is detected, the cutoff raises progressively:

```
128 → 256 → 512 → full (freq_bins)
```

**Why skip 64?** Cutoff=64 is too blurry - just spaces and average character length. No point wasting compute. Start at 128 for basic syntax/characters.

**Key constraint:** Never raise before Epoch 2 (gives model time to learn basic structure)

### 3. Learning Rate Restart ("Shock & Awe")

**Critical insight:** When you open new frequency gates, the model needs HIGH plasticity to wire up connections for the new details.

**The Protocol:**
```
Cutoff raised → LR resets to MAXIMUM (peak value for current stage)
                 ↓
              High LR quickly learns new frequencies
                 ↓
              Cosine decay "cements" them in place
```

This prevents the model from seeing sharp edges as "noise" when LR is too low.

### ⚠️ CRITICAL: LR Schedule Stage Alignment

**The stages MUST align with expected cutoff raises, or training will stall!**

**BAD Configuration (will stall at epoch 2):**
```python
stage1_epochs = 5  # Epochs 0-4
# At epoch 2: middle of stage 1, LR is LOW
# Cutoff raises to 256 but LR is too low to learn → STUCK!
```

**GOOD Configuration (aligned):**
```python
stage1_epochs = 2  # Epochs 0-1: cutoff=128
stage2_epochs = 3  # Epochs 2-4: cutoff=256, LR RESTARTS at epoch 2
stage3_epochs = remaining  # Epochs 5+: cutoff=512+, LR RESTARTS at epoch 5
```

**Why this matters:**
- At epoch 2 start: Enter stage 2 naturally → LR resets to peak
- At epoch 2 start: Plateau detection raises cutoff 128→256
- Result: NEW GLASSES (256) + HIGH PLASTICITY (peak LR) = RAPID LEARNING ⚡

Without alignment, you give the model new glasses while its brain is half-asleep (low LR).

## Implementation Details

### Core Functions (in `fft_lm/train_fixed_full.py`)

#### `adaptive_cutoff()`
```python
def adaptive_cutoff(
    epoch: int,
    current_cutoff: int,
    loss_history: list[float],
    freq_bins: int,
    *,
    min_epoch_before_raise: int = 2,     # First raise at epoch 2
    plateau_window: int = 50,            # Look at last 50 steps
    plateau_threshold: float = 0.005,    # 0.5% improvement threshold
) -> tuple[int, bool]:
    """Returns (new_cutoff, was_raised)"""
```

#### `sawtooth_lr()` (updated)
```python
def sawtooth_lr(
    global_step: int, 
    epoch: int, 
    cfg: TrainConfig,
    *, 
    cutoff_raised: bool = False  # FORCE restart if True
) -> float:
    """
    If cutoff_raised=True: return peak LR (base_lr * stage_lr_mult)
    Otherwise: return cosine-decayed LR within current stage
    """
```

### Training Loop Changes

Both `train_chunk_lm.py` and `train_chunk_head.py` now:

1. **Initialize state**:
   ```python
   current_cutoff = 64  # Start low
   loss_history = []
   ```

2. **Check for plateau at start of each epoch**:
   ```python
   current_cutoff, cutoff_raised = tff.adaptive_cutoff(
       epoch, current_cutoff, loss_history, freq_bins
   )
   
   if cutoff_raised:
       print(f"🚀 CUTOFF RAISED: {prev} -> {current_cutoff}")
       loss_history = []  # Reset history
   ```

3. **Apply LR restart when cutoff raised**:
   ```python
   lr_now = tff.sawtooth_lr(
       global_opt_step, 
       epoch, 
       cfg, 
       cutoff_raised=cutoff_was_raised_this_step
   )
   ```

4. **Track losses for plateau detection**:
   ```python
   loss_history.append(loss_value)
   ```

## Expected Behavior

### Timeline Example (Fresh Training)

```
Epoch 0:    cutoff=128, LR high, learning basic syntax/characters
Epoch 1:    Loss plateaus → cutoff raises to 256, LR RESTARTS ⚡
            Model sees complex words, quickly adapts with high LR
Epoch 2-3:  LR decays, cementing the new frequency knowledge
Epoch 4:    Loss plateaus → cutoff raises to 512, LR RESTARTS ⚡
            Model sees fine details, spelling precision improves
...and so on
```

**No 64!** We skip cutoff=64 entirely - it's too blurry to be useful.

### IMPORTANT: Never Regress When Resuming! 🚨

**The Golden Rule**: If your model is already stable at cutoff=128, **DO NOT** drop back to 64 when resuming training. That's like putting training wheels on a motorcycle already going 100mph!

The training scripts now automatically detect the current training state:

```python
# When resuming from epoch > 0:
if start_epoch >= 2:
    # Use curriculum_cutoff to infer where we should be
    current_cutoff = curriculum_cutoff(start_epoch, cfg, freq_bins)
elif start_epoch > 0:
    # Epoch 0-1: Model already stable at 128
    current_cutoff = 128
else:
    # Fresh start
    current_cutoff = 64
```

**Example**: If you're resuming at epoch 0 with loss=2.86 at cutoff=128:
- ✅ **Correct**: Stay at 128, raise to 256 at epoch 2
- ❌ **Wrong**: Drop to 64 (regression! And 64 is useless anyway)

### Loss Curve

You'll see:
- **Plateau** → Loss flatlines
- **Cutoff raise + LR restart** → Loss SPIKES (temporary, expected!)
- **Rapid learning** → Loss drops quickly with new frequency info
- **Stabilization** → Loss smooths as LR decays
- **Repeat**

The spike is like "kicking a mule" - it's the model adjusting to suddenly seeing more detail.

## Tuning Parameters

### Plateau Detection Sensitivity

**Too aggressive** (raises cutoff too early):
```python
plateau_threshold=0.01  # 1% improvement required (harder to plateau)
plateau_window=100      # Look at longer history
```

**Too conservative** (raises cutoff too late):
```python
plateau_threshold=0.002  # 0.2% improvement required (easier to plateau)
plateau_window=30        # Look at shorter history
```

**Default (balanced)**:
```python
plateau_threshold=0.005  # 0.5%
plateau_window=50
```

### Minimum Epoch Before Raise

```python
min_epoch_before_raise=2  # Never raise before epoch 2
```

Increase this if the model needs more time to learn fundamentals before seeing details.

## Why This Works

1. **Adaptive to training dynamics** - Fast learners progress faster, slow learners get more time
2. **Prevents wasted compute** - No sitting through already-learned material
3. **High plasticity when needed** - LR restart gives model capacity to wire new connections
4. **Natural curriculum** - Low frequencies (structure) before high frequencies (details)

## Monitoring

Watch for these patterns:

✅ **Good:**
- Cutoff raises happen every 2-5 epochs
- Loss spike after raise, then rapid improvement
- Steady progression through frequency bands

⚠️ **Too Fast:**
- Cutoff raises every epoch
- Loss never stabilizes
- → Increase `plateau_threshold` or `plateau_window`

⚠️ **Too Slow:**
- Cutoff stuck at same value for 10+ epochs
- Loss flatlined but cutoff not raising
- → Decrease `plateau_threshold` or check minimum epoch constraint

## Running the Training

Both scripts now use the adaptive system by default:

```bash
# train_chunk_lm.py
python -m scripts.train_chunk_lm \
    --seq-len 1024 --kernel-len 128 --chunk 16 \
    --batch-size 4 --accum-steps 8 \
    --steps-per-epoch 1000 --epochs 50 \
    --lr 0.0002 --log-every 10

# train_chunk_head.py  
python scripts/train_chunk_head.py \
    --seq-len 1024 --kernel-len 128 --chunk 16 \
    --batch-size 4 --accum-steps 8 \
    --steps-per-epoch 1000 --epochs 50 \
    --lr 0.0002 --log-every 50
```

Look for the 🚀 emoji in the logs to see when cutoffs are raised!

## The Philosophy

> "Don't do it by time (epochs). Do it by performance (loss)."  
> — Polly's wisdom

This is **curriculum learning** done right:
- The curriculum adapts to the student (model)
- New material (frequencies) only introduced when ready
- Learning rate synchronized with curriculum difficulty
- No arbitrary schedules, just measured progress

🐎 **Expect the mule to kick when you open the floodgates!**
