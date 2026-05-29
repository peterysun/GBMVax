"""
config.py — load and resolve the YAML config.

Why this exists as its own module:
    Hardcoding paths inside data loaders is the #1 cause of "works on my
    machine" issues in research code. Every path in the pipeline flows
    through here. If you want to run on a different box, change config.yaml
    or set GBMVAX_ROOT — never touch module code.

The loader does three things:
    1. Reads the YAML.
    2. Expands ~ and resolves all data paths to absolute paths under root.
    3. Lets GBMVAX_ROOT env var override the root in the YAML (useful for
       Colab where the repo is at /content/GBMVax instead of ~/GBMVax).
"""

from __future__ import annotations

import os                                  # Environment variables and path joining
from pathlib import Path                   # Modern path handling — beats os.path
from typing import Any                     # For the nested-dict return type

import yaml                                # Config file format


# ----------------------------------------------------------------------------
# Path keys we want resolved to absolute paths. Every leaf in `paths:` that
# is a string gets joined to `root` unless it is already absolute. We list
# them explicitly so a typo in the YAML (extra key) doesn't silently break.
# ----------------------------------------------------------------------------
_RESOLVE_SECTIONS = ("iedb", "hla", "proteome", "tcga", "validation")


def _resolve_path(value: str, root: Path) -> str:
    """
    Resolve a single path string relative to `root`.

    If the value is absolute (starts with '/' or '~'), expand and return.
    Otherwise treat it as relative to root.
    """
    # ~ refers to the user's home dir; absolute paths bypass root entirely.
    p = Path(value).expanduser()

    if p.is_absolute():
        return str(p)

    # Relative paths are joined to root. We don't .resolve() because the
    # file may not exist yet (e.g. checkpoints dir before first training run).
    return str(root / p)


def load_config(path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    """
    Load the YAML config and resolve all data paths.

    Args:
        path: Path to the YAML config. Relative to CWD if not absolute.

    Returns:
        A nested dict mirroring the YAML structure, with all paths resolved
        to absolute strings.
    """
    # Load YAML — safe_load avoids arbitrary code execution from !!python tags.
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    # Determine root. Environment variable wins, then YAML, then CWD.
    root_str = os.environ.get("GBMVAX_ROOT") or cfg["paths"].get("root", ".")
    root = Path(root_str).expanduser().resolve()
    cfg["paths"]["root"] = str(root)

    # Resolve every data section. We iterate explicitly so the YAML structure
    # is documentation: if you add a new path, you also add it to _RESOLVE_SECTIONS.
    for section in _RESOLVE_SECTIONS:
        if section not in cfg["paths"]:
            continue
        sec = cfg["paths"][section]
        if isinstance(sec, dict):
            # Nested section like `tcga.gbm_2013` — resolve each leaf.
            for k, v in sec.items():
                if isinstance(v, str):
                    sec[k] = _resolve_path(v, root)
        elif isinstance(sec, str):
            cfg["paths"][section] = _resolve_path(sec, root)

    # Resolve output directories — these are created on demand, not required to exist.
    for key in ("checkpoints", "results", "logs"):
        if key in cfg["paths"]:
            cfg["paths"][key] = _resolve_path(cfg["paths"][key], root)

    # NetMHCpan binary — single string, resolve directly.
    if "netmhcpan" in cfg["paths"]:
        cfg["paths"]["netmhcpan"] = _resolve_path(cfg["paths"]["netmhcpan"], root)

    return cfg


def ensure_output_dirs(cfg: dict[str, Any]) -> None:
    """Create checkpoints/results/logs directories if they don't exist."""
    for key in ("checkpoints", "results", "logs"):
        if key in cfg["paths"]:
            Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)
