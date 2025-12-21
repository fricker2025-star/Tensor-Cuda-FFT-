"""fft_lm.ckpt_io

Security / integrity helpers for checkpoints.

Why this exists:
  - `torch.load()` uses Python pickle and can execute arbitrary code if you load
    an untrusted checkpoint.
  - This project frequently loads checkpoints from disk while experimenting.

Mitigation:
  - When saving a checkpoint, we also write a `.sha256` file next to it.
  - When loading, if a `.sha256` is present, we verify integrity before loading.

This does NOT make pickle "safe", but it prevents accidental loading of
modified/mismatched checkpoints and provides a lightweight trust boundary.
Only load checkpoints you created.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any

import torch


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha_path(path: str) -> str:
    return path + ".sha256"


def save_checkpoint(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(obj, path)
    digest = _sha256_file(path)
    with open(_sha_path(path), "w", encoding="utf-8") as f:
        f.write(digest + "\n")


def verify_checkpoint(path: str) -> bool:
    sha = _sha_path(path)
    if not os.path.exists(sha):
        return False
    expected = open(sha, "r", encoding="utf-8").read().strip().splitlines()[0]
    actual = _sha256_file(path)
    if expected != actual:
        raise RuntimeError(
            f"Checkpoint integrity check failed for {path}. "
            f"Expected sha256={expected}, got sha256={actual}."
        )
    return True


def load_checkpoint(path: str, *, map_location: str | None = "cpu") -> Any:
    # Verify if signature present
    verify_checkpoint(path)
    # Still uses pickle internally; only load trusted files.
    return torch.load(path, map_location=map_location)
