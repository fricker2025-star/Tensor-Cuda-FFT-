# Contributing

## Repo layout

- `fft_tensor/` – core library (keep stable)
- `fft_lm/` – language model code (experimental)
- `scripts/` – runnable entrypoints
- `experiments/` – one-off diagnostics / scratch
- `tests/` – unit/integration tests

## Guidelines

1. Keep changes focused: one PR = one feature/fix.
2. Prefer adding new experiments under `experiments/`.
3. Keep large artifacts out of git:
   - datasets (e.g. `tinystories_train.txt`)
   - checkpoints (`*.pt` already ignored)
4. If you change training/generation behavior, update:
   - `README.md`
   - `.github/README.md` (high-level project overview)

## Testing

Basic syntax check:

```bash
python -m py_compile fft_lm/train_fixed_full.py
python -m py_compile scripts/train_chunk_head.py
```

## CI

CI runs CPU-only checks. Avoid making core functionality require CUDA to import.
