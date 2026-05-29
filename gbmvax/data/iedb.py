"""
iedb.py — load the IEDB MHC ligand dataset and prepare training tables.

The Immune Epitope Database (IEDB) is the canonical public source for
peptide-MHC binding data. The 'mhc_ligand_full.csv' download contains
millions of entries spanning:
    * Binding affinity assays (quantitative IC50/Kd in nM)
    * Eluted ligand mass spec (qualitative: was this peptide presented?)
    * T cell assays (also in tcell_full_v3, used for the immunogenicity model)

We need to split entries into two training streams:
    1. AFFINITY:    real-valued log10(IC50). For the binding head.
    2. PRESENTATION: binary {presented, not_presented}. For the presentation head.

Why both: a peptide can bind tightly in vitro (low IC50) but never make it
to the cell surface. Eluted-ligand data captures the joint event of binding
+ processing + loading + presentation. Training the multi-task model on
both teaches it that presentation is a stricter criterion than affinity
alone.

CRITICAL DATA QUIRKS WE HANDLE:
    * IEDB CSV has TWO header rows. pandas needs header=[0, 1].
    * Quantitative measurements come in IC50, EC50, Kd — we normalize all
      to IC50-equivalent in nM, dropping ambiguous units.
    * Many entries have qualitative measurement ('Positive'/'Negative') with
      no numeric — those go to the presentation stream.
    * Alleles need normalization (HLA-A*02:01 vs A0201 etc.).
    * Class II entries (HLA-DR, DP, DQ) are EXCLUDED. We are building a
      class I pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from gbmvax.utils.device import get_logger
from gbmvax.utils.hla import normalize_allele
from gbmvax.utils.sequences import AMINO_ACIDS

logger = get_logger(__name__)


# ----------------------------------------------------------------------------
# Constants describing the IEDB schema. These column names come from the
# second header row of mhc_ligand_full.csv (the first row is grouping).
# If IEDB changes their schema, only these constants need updating.
# ----------------------------------------------------------------------------
# We use flexible column selection because IEDB has reformatted the CSV
# multiple times. The loader tries each candidate in order and picks the
# first that exists.
COL_CANDIDATES = {
    "peptide": ["Description", "Epitope - Description", "Linear Sequence"],
    "allele": ["Allele Name", "MHC - Allele Name", "Restricting MHC Allele"],
    "mhc_class": ["MHC allele class", "MHC - Class"],
    "assay_type": ["Method/Technique", "Assay - Method/Technique"],
    "measurement_value": ["Quantitative measurement", "Assay - Quantitative measurement"],
    "measurement_inequality": ["Measurement Inequality", "Assay - Measurement Inequality"],
    "units": ["Units", "Assay - Units"],
    "qualitative": ["Qualitative Measure", "Assay - Qualitative Measure"],
    "host": ["Host - Name", "Host Organism Name"],
}


def _pick_column(df_cols: pd.Index, candidates: list[str]) -> Optional[str]:
    """Return the first candidate that exists in df_cols, else None."""
    for c in candidates:
        if c in df_cols:
            return c
    return None


# ----------------------------------------------------------------------------
# Identifiers for elution / mass spec assays. Anything containing these
# substrings is treated as PRESENTATION (positive label = ligand observed).
# Everything else with a quantitative IC50 is AFFINITY.
# ----------------------------------------------------------------------------
ELUTION_ASSAY_KEYWORDS = (
    "mass spectrometry",
    "ms ligand",
    "ligand presentation",
    "eluted",
)


def load_iedb(
    csv_path: Path | str,
    peptide_lengths: tuple[int, ...] = (8, 9, 10, 11),
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load and clean the IEDB MHC ligand CSV.

    Returns a tidy DataFrame with columns:
        peptide          str       Cleaned amino acid sequence
        allele           str       Normalized HLA allele 'HLA-A*02:01'
        length           int
        affinity_nM      float     IC50 in nM, NaN if assay was qualitative
        log_affinity     float     log10(affinity_nM), NaN if no quant value
        presented        int       1 if elution-positive, 0 if elution-negative, -1 if no elution data
        is_affinity      bool      True if this row contributes to the affinity head
        is_presentation  bool      True if this row contributes to the presentation head
        source_assay     str       Original assay method (for debugging)
    """
    logger.info(f"Loading IEDB CSV from {csv_path} (this can take 1–2 minutes for 8GB file)...")

    # IEDB has a two-row header. We use header=[0, 1] then flatten by taking
    # the second level — the first level is a grouping ('Epitope', 'Assay', ...)
    # which is redundant with the more specific second-level names.
    df_raw = pd.read_csv(
        csv_path,
        header=[0, 1],
        low_memory=False,
        nrows=max_rows,
    )

    # Flatten the MultiIndex: take the second-level (more specific) names.
    df_raw.columns = [col[1] if isinstance(col, tuple) else col for col in df_raw.columns]

    # Map our logical column names to actual columns present in this file version.
    col_map: dict[str, str] = {}
    for logical, candidates in COL_CANDIDATES.items():
        actual = _pick_column(df_raw.columns, candidates)
        if actual is None and logical in ("peptide", "allele"):
            raise KeyError(f"Required column {logical!r} not found in IEDB CSV. Schema may have changed.")
        if actual is not None:
            col_map[logical] = actual

    logger.info(f"IEDB raw rows: {len(df_raw):,}")

    # Build the tidy frame column by column.
    out = pd.DataFrame()
    out["peptide"] = df_raw[col_map["peptide"]].astype(str).str.strip().str.upper()
    out["allele_raw"] = df_raw[col_map["allele"]].astype(str)

    # MHC class filter — we want class I only (length 8-11 is class I anyway,
    # but filtering by class is more reliable).
    if "mhc_class" in col_map:
        cls = df_raw[col_map["mhc_class"]].astype(str).str.upper()
        out = out[cls.str.contains("I", na=False) & ~cls.str.contains("II", na=False)]

    # Drop rows whose peptide contains non-canonical residues (X, B, Z, U,
    # *, space, lowercase). These are sequencing artifacts or modifications
    # the model can't represent.
    valid_aa = set(AMINO_ACIDS)
    mask = out["peptide"].apply(lambda s: len(s) > 0 and all(c in valid_aa for c in s))
    out = out[mask]

    # Peptide length filter — 8–11 captures >99% of class I ligands.
    out["length"] = out["peptide"].str.len()
    out = out[out["length"].isin(peptide_lengths)]

    # Normalize allele names. Drop any row whose allele we can't parse —
    # this catches mouse alleles (H-2-Kb), chimp alleles, etc.
    def safe_norm(a: str) -> Optional[str]:
        try:
            return normalize_allele(a)
        except ValueError:
            return None

    out["allele"] = out["allele_raw"].apply(safe_norm)
    out = out.dropna(subset=["allele"])
    out = out.drop(columns=["allele_raw"])

    # ------------------------------------------------------------------
    # Affinity stream: quantitative measurement in nM, IC50 or Kd.
    # We accept measurements in nM directly; EC50 / activity values are
    # not directly comparable so we drop them. Inequality measurements
    # (>50000 nM = "doesn't bind") are kept but capped at 50000 — they
    # are informative as weak-binder anchors.
    # ------------------------------------------------------------------
    if "measurement_value" in col_map:
        # Re-index df_raw to align with our filtered frame.
        df_raw_aligned = df_raw.loc[out.index]
        meas = pd.to_numeric(df_raw_aligned[col_map["measurement_value"]], errors="coerce")
        units = (
            df_raw_aligned[col_map["units"]].astype(str).str.lower()
            if "units" in col_map else pd.Series("nm", index=out.index)
        )

        # Convert non-nM units. IEDB has uM, pM, M — convert each.
        affinity_nm = meas.copy()
        affinity_nm = affinity_nm.where(~units.str.contains("um", na=False), meas * 1000.0)
        affinity_nm = affinity_nm.where(~units.str.contains("pm", na=False), meas / 1000.0)

        # Cap at 50,000 nM — the standard "non-binder" ceiling used by NetMHCpan.
        affinity_nm = affinity_nm.clip(upper=50000.0, lower=0.001)
        out["affinity_nM"] = affinity_nm
        out["log_affinity"] = np.log10(affinity_nm.replace(0, np.nan))
    else:
        out["affinity_nM"] = np.nan
        out["log_affinity"] = np.nan

    # ------------------------------------------------------------------
    # Presentation stream: assays whose method is mass-spec elution.
    # ------------------------------------------------------------------
    if "assay_type" in col_map:
        assay_str = df_raw.loc[out.index, col_map["assay_type"]].astype(str).str.lower()
        is_elution = assay_str.apply(lambda s: any(kw in s for kw in ELUTION_ASSAY_KEYWORDS))
        out["source_assay"] = assay_str

        # Qualitative measure: 'Positive', 'Positive-High', 'Negative', etc.
        if "qualitative" in col_map:
            qual = df_raw.loc[out.index, col_map["qualitative"]].astype(str).str.lower()
            # Most eluted-ligand mass spec rows are positives by construction
            # (something had to be eluted for it to appear). Negatives are rare
            # and come from decoy panels. Treat anything starting with 'positive'
            # as 1, anything starting with 'negative' as 0, else -1 (unknown).
            presented = pd.Series(-1, index=out.index, dtype=np.int8)
            presented = presented.where(~qual.str.startswith("positive", na=False), 1)
            presented = presented.where(~qual.str.startswith("negative", na=False), 0)
        else:
            presented = pd.Series(1, index=out.index, dtype=np.int8)

        # Only count as presentation evidence if this row is from an elution assay.
        out["presented"] = np.where(is_elution, presented, -1).astype(np.int8)
        out["is_presentation"] = is_elution
    else:
        out["presented"] = -1
        out["is_presentation"] = False
        out["source_assay"] = "unknown"

    # An affinity row needs a numeric measurement and must NOT be an elution row
    # (elution rows go to presentation only, even if they happen to have a number).
    out["is_affinity"] = (~out["is_presentation"]) & out["affinity_nM"].notna()

    # Deduplicate identical (peptide, allele) entries — IEDB has many redundant
    # records from re-curation. Keep the lowest IC50 (most informative).
    out = out.sort_values(["peptide", "allele", "affinity_nM"])
    out = out.drop_duplicates(subset=["peptide", "allele"], keep="first")

    logger.info(
        f"IEDB processed: {len(out):,} rows | "
        f"affinity: {out['is_affinity'].sum():,} | "
        f"presentation: {out['is_presentation'].sum():,}"
    )
    return out.reset_index(drop=True)


def stratified_train_val_test_split(
    df: pd.DataFrame,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split IEDB by ALLELE such that train/val/test contain disjoint sets of
    (peptide, allele) pairs but every allele appears in every split (as long
    as it has enough peptides).

    Why stratified by allele: a random row split would let the model memorize
    specific peptides. We want to test generalization to UNSEEN peptides for
    every allele the model knows. Test peptides are held out per allele.
    """
    rng = np.random.default_rng(seed)
    train_parts, val_parts, test_parts = [], [], []

    for allele, group in df.groupby("allele"):
        idx = group.index.to_numpy()
        rng.shuffle(idx)
        n = len(idx)
        n_test = int(n * test_frac)
        n_val = int(n * val_frac)
        test_parts.append(group.loc[idx[:n_test]])
        val_parts.append(group.loc[idx[n_test:n_test + n_val]])
        train_parts.append(group.loc[idx[n_test + n_val:]])

    train = pd.concat(train_parts).reset_index(drop=True)
    val = pd.concat(val_parts).reset_index(drop=True)
    test = pd.concat(test_parts).reset_index(drop=True)
    logger.info(f"Split sizes — train: {len(train):,} | val: {len(val):,} | test: {len(test):,}")
    return train, val, test
