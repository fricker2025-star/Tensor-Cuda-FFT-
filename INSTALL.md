# Installation

This repository contains two layers:

1. **`fft_tensor/`** – core spectral tensor utilities.
2. **`fft_lm/`** + **`scripts/`** – the experimental byte-level LM training / generation pipeline.

## Python environment

- Python 3.10+ recommended (3.9–3.12 supported)
- PyTorch 2.x

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Editable install (recommended)

From the repo root:

```bash
python -m pip install -e .
```

If you don’t want to install editable, you can run scripts from the repo root with:

```bash
python -m scripts.train_chunk_head ...
```

## CUDA / Windows notes

- Training and fast generation are GPU-oriented.
- Some optional CUDA extension builds are defined in `setup.py`. If you hit build issues,
  you can still use PyTorch fallbacks.
