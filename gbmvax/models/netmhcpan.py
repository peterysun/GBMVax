"""
netmhcpan.py — wrapper around the NetMHCpan 4.2c binary.

NetMHCpan is the field-standard baseline for peptide–HLA binding
prediction. We invoke it as a subprocess and parse its tabular output to
produce the same (peptide, allele) -> log10(IC50 nM) interface as our own
transformer. This lets `scripts/evaluate.py` run head-to-head comparisons
on identical test sets.

NetMHCpan input format (peptide file):
    one peptide per line, no header. We write to a temp file.

Invocation:
    netMHCpan -p <peptide_file> -a <allele> -BA -l <length>
        -p  : peptide-list input mode
        -a  : allele (single or comma-separated)
        -BA : also produce binding affinity prediction (nM)
        -l  : peptide length (8, 9, 10, 11)

Output is a fixed-width table with columns including 'Peptide', 'MHC',
'Score_EL', 'Aff(nM)'. We parse Aff(nM) and convert to log10.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from gbmvax.utils.device import get_logger

logger = get_logger(__name__)


# Header line in NetMHCpan output starts with "Pos" and contains "Aff(nM)"
_NETMHCPAN_HEADER_PATTERN = re.compile(r"^\s*Pos\s+MHC\s+Peptide")


def _convert_allele_netmhcpan(allele: str) -> str:
    """
    Convert canonical 'HLA-A*02:01' to NetMHCpan's expected 'HLA-A02:01'.
    NetMHCpan dropped the '*' separator in 4.x; we strip it.
    """
    return allele.replace("*", "")


def run_netmhcpan(
    peptides: list[str],
    allele: str,
    binary_path: Path | str,
    length: int | None = None,
) -> pd.DataFrame:
    """
    Run NetMHCpan on a list of peptides for a single allele.

    Returns DataFrame with columns: peptide, allele, ic50_nM, log_ic50, score_el.

    For mixed-length peptide lists, call once per length.
    """
    # If lengths are mixed, dispatch per length.
    lengths = sorted(set(len(p) for p in peptides))
    if length is None and len(lengths) > 1:
        parts = []
        for L in lengths:
            sub = [p for p in peptides if len(p) == L]
            parts.append(run_netmhcpan(sub, allele, binary_path, length=L))
        return pd.concat(parts, ignore_index=True)

    length = length or lengths[0]

    # Write peptides to a temp file.
    with tempfile.NamedTemporaryFile("w", suffix=".pep", delete=False) as f:
        for p in peptides:
            f.write(p + "\n")
        pep_path = f.name

    nm_allele = _convert_allele_netmhcpan(allele)
    cmd = [
        str(binary_path),
        "-p", pep_path,
        "-a", nm_allele,
        "-BA",
        "-l", str(length),
    ]
    logger.debug(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except FileNotFoundError:
        raise RuntimeError(f"NetMHCpan binary not found at {binary_path}")
    finally:
        Path(pep_path).unlink(missing_ok=True)

    if result.returncode != 0:
        logger.error(f"NetMHCpan failed:\nSTDERR:\n{result.stderr[:2000]}")
        raise RuntimeError("NetMHCpan invocation failed")

    return _parse_netmhcpan_output(result.stdout, allele)


def _parse_netmhcpan_output(stdout: str, allele: str) -> pd.DataFrame:
    """
    Parse NetMHCpan stdout. The relevant section is between two long lines
    of dashes, with a header line starting "Pos MHC Peptide ...".
    """
    rows = []
    in_table = False
    columns: list[str] = []

    for line in stdout.splitlines():
        # Detect header.
        if _NETMHCPAN_HEADER_PATTERN.match(line):
            in_table = True
            columns = line.split()
            continue
        if not in_table:
            continue
        # End-of-table heuristic: blank line or another '---' separator.
        if line.strip().startswith("-----") or not line.strip():
            if rows:
                in_table = False
            continue

        parts = line.split()
        if len(parts) < len(columns):
            continue
        # Some columns (e.g. 'BindLevel') only appear for binders, padding
        # the row. We index by the columns we need: 'Peptide', 'Aff(nM)', 'Score_EL'.
        try:
            pep_idx = columns.index("Peptide")
            aff_idx = columns.index("Aff(nM)") if "Aff(nM)" in columns else None
            el_idx = columns.index("Score_EL") if "Score_EL" in columns else None
        except ValueError:
            continue

        pep = parts[pep_idx]
        aff = float(parts[aff_idx]) if aff_idx is not None and aff_idx < len(parts) else np.nan
        el = float(parts[el_idx]) if el_idx is not None and el_idx < len(parts) else np.nan
        rows.append({"peptide": pep, "allele": allele, "ic50_nM": aff, "score_el": el})

    df = pd.DataFrame(rows)
    if not df.empty:
        df["log_ic50"] = np.log10(df["ic50_nM"].clip(lower=0.001))
    return df


def run_netmhcpan_batched(
    pairs: list[tuple[str, str]],
    binary_path: Path | str,
) -> pd.DataFrame:
    """
    Run NetMHCpan on (peptide, allele) pairs. Groups by allele for efficiency.
    """
    by_allele: dict[str, list[str]] = {}
    for pep, allele in pairs:
        by_allele.setdefault(allele, []).append(pep)

    parts = []
    for allele, peps in by_allele.items():
        try:
            parts.append(run_netmhcpan(peps, allele, binary_path))
        except RuntimeError as e:
            logger.warning(f"Skipping allele {allele}: {e}")
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
