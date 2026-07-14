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
# Constants describing the IEDB schema. IEDB exports have changed names over
# time, and many useful labels are duplicated across the two header rows
# (for example several groups contain a column named "Name"). We therefore
# select columns by the full (group, field) pair when a two-row header exists.
# ----------------------------------------------------------------------------
ColumnCandidate = tuple[str, str] | str

COL_CANDIDATES: dict[str, list[ColumnCandidate]] = {
    "peptide": [
        ("Epitope", "Description"),
        ("Epitope", "Name"),
        ("Epitope", "Linear Sequence"),
        "Description",
        "Epitope - Description",
        "Linear Sequence",
    ],
    "allele": [
        ("MHC Restriction", "Name"),
        "Allele Name",
        "MHC - Allele Name",
        "Restricting MHC Allele",
    ],
    "mhc_class": [
        ("MHC Restriction", "Class"),
        "MHC allele class",
        "MHC - Class",
    ],
    "assay_type": [
        ("Assay", "Method"),
        "Method/Technique",
        "Assay - Method/Technique",
    ],
    "measurement_value": [
        ("Assay", "Quantitative measurement"),
        "Quantitative measurement",
        "Assay - Quantitative measurement",
    ],
    "measurement_inequality": [
        ("Assay", "Measurement Inequality"),
        "Measurement Inequality",
        "Assay - Measurement Inequality",
    ],
    "units": [
        ("Assay", "Units"),
        "Units",
        "Assay - Units",
    ],
    "qualitative": [
        ("Assay", "Qualitative Measurement"),
        "Qualitative Measure",
        "Qualitative Measurement",
        "Assay - Qualitative Measure",
    ],
    "host": [
        ("Host", "Name"),
        "Host - Name",
        "Host Organism Name",
    ],
}


def _norm_header(value) -> str:
    return str(value).strip().lower()


def _pick_column_index(df_cols: pd.Index, candidates: list[ColumnCandidate]) -> Optional[int]:
    """Return the integer column position for the first matching candidate."""
    for cand in candidates:
        if isinstance(cand, tuple):
            want = tuple(_norm_header(x) for x in cand)
            for i, col in enumerate(df_cols):
                if isinstance(col, tuple) and tuple(_norm_header(x) for x in col[:2]) == want:
                    return i
        else:
            want = _norm_header(cand)
            for i, col in enumerate(df_cols):
                if isinstance(col, tuple):
                    # Match either the specific field name or a previously
                    # flattened "Group - Field" style export.
                    group = _norm_header(col[0])
                    field = _norm_header(col[1])
                    if field == want or f"{group} - {field}" == want:
                        return i
                elif _norm_header(col) == want:
                    return i
    return None


def _series(df: pd.DataFrame, col_idx: int) -> pd.Series:
    """Return a column by position, avoiding duplicate-name ambiguity."""
    return df.iloc[:, col_idx]


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
    exclude_peptides: set[str] | None = None,
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

    # Map our logical column names to integer positions. Position-based access
    # is important because IEDB has duplicate second-level names like "Name".
    col_map: dict[str, int] = {}
    for logical, candidates in COL_CANDIDATES.items():
        actual = _pick_column_index(df_raw.columns, candidates)
        if actual is None and logical in ("peptide", "allele"):
            raise KeyError(f"Required column {logical!r} not found in IEDB CSV. Schema may have changed.")
        if actual is not None:
            col_map[logical] = actual

    logger.info(f"IEDB raw rows: {len(df_raw):,}")

    # Build the tidy frame column by column.
    out = pd.DataFrame()
    out["peptide"] = _series(df_raw, col_map["peptide"]).astype(str).str.strip().str.upper()
    out["allele_raw"] = _series(df_raw, col_map["allele"]).astype(str)

    # MHC class filter — we want class I only (length 8-11 is class I anyway,
    # but filtering by class is more reliable).
    if "mhc_class" in col_map:
        cls = _series(df_raw, col_map["mhc_class"]).astype(str).str.upper()
        out = out[cls.str.contains("I", na=False) & ~cls.str.contains("II", na=False)]

    # Drop rows whose peptide contains non-canonical residues (X, B, Z, U,
    # *, space, lowercase). These are sequencing artifacts or modifications
    # the model can't represent.
    valid_aa = set(AMINO_ACIDS)
    mask = out["peptide"].apply(lambda s: len(s) > 0 and all(c in valid_aa for c in s))
    out = out[mask]

    if exclude_peptides:
        excluded = {str(p).strip().upper() for p in exclude_peptides}
        before = len(out)
        out = out[~out["peptide"].isin(excluded)]
        logger.info(f"Excluded {before - len(out):,} rows matching held-out validation peptides")

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
        meas = pd.to_numeric(_series(df_raw_aligned, col_map["measurement_value"]), errors="coerce")
        units = (
            _series(df_raw_aligned, col_map["units"]).astype(str).str.lower()
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
        assay_str = _series(df_raw.loc[out.index], col_map["assay_type"]).astype(str).str.lower()
        is_elution = assay_str.apply(lambda s: any(kw in s for kw in ELUTION_ASSAY_KEYWORDS))
        out["source_assay"] = assay_str

        # Qualitative measure: 'Positive', 'Positive-High', 'Negative', etc.
        if "qualitative" in col_map:
            qual = _series(df_raw.loc[out.index], col_map["qualitative"]).astype(str).str.lower()
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
