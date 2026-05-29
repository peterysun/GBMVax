"""
device.py — hardware selection and reproducibility helpers.

The pipeline targets three back-ends:
    * CUDA — Colab GPU, lab GPU box. Training and large-scale inference.
    * MPS — Mac M-series Metal Performance Shaders. Dev box.
    * CPU — fallback; tests and CI.

torch.device("auto") doesn't exist — we have to detect. We also seed all
RNGs here so experiments are reproducible across runs and devices.
"""

from __future__ import annotations

import logging                     # Standard logging — we wrap to set a consistent format
import os
import random                      # Python RNG — used by data shuffling utilities
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch


# ----------------------------------------------------------------------------
# Device selection.
# ----------------------------------------------------------------------------
def get_device(preference: str = "auto") -> torch.device:
    """
    Return the best available torch.device.

    Priority: explicit > CUDA > MPS > CPU.
    MPS is preferred over CPU on Mac M-series — roughly 5–10x speedup for
    transformer ops at this size. We also check is_built() because MPS can
    be available on older macOS where it crashes on certain ops.
    """
    if preference != "auto":
        # User pinned a device — respect it, but warn if unavailable.
        dev = torch.device(preference)
        if preference == "cuda" and not torch.cuda.is_available():
            logging.warning("CUDA requested but unavailable; falling back to CPU.")
            return torch.device("cpu")
        return dev

    if torch.cuda.is_available():
        return torch.device("cuda")

    # MPS is the Metal back-end on macOS. Both checks needed: is_available is
    # runtime, is_built is build-time.
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")

    return torch.device("cpu")


# ----------------------------------------------------------------------------
# Reproducibility. We seed Python, NumPy, and PyTorch (CPU + CUDA).
# Note: full determinism requires torch.use_deterministic_algorithms(True),
# but that disables some fast kernels and slows training ~2x. We opt for
# "seeded but fast" — runs are reproducible within a device, not across.
# ----------------------------------------------------------------------------
def seed_everything(seed: int) -> None:
    """Seed all RNGs the pipeline touches."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # CUDA has separate RNG states per device.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Hash seed affects dict iteration order in some Python ops.
    os.environ["PYTHONHASHSEED"] = str(seed)


# ----------------------------------------------------------------------------
# Logging. One root logger for the package; submodules call get_logger(__name__).
# Format includes module so we can tell where messages originate during a
# multi-stage pipeline run.
# ----------------------------------------------------------------------------
_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def setup_logging(level: str = "INFO", log_file: Optional[Path] = None) -> None:
    """Configure the root logger. Call once at the start of every script."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=_LOG_FORMAT,
        datefmt=_DATE_FORMAT,
        handlers=handlers,
        force=True,                # Overwrite any default handler set by a notebook
    )


def get_logger(name: str) -> logging.Logger:
    """Return a logger scoped to the calling module."""
    return logging.getLogger(name)
