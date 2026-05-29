"""
mutations.py — load TCGA / CPTAC / Columbia MAF files and produce a unified
per-mutation table ready for neoantigen generation.

Input format (cBioPortal MAF, tab-separated):
    Hugo_Symbol  Entrez_Gene_Id  Center  ...  Variant_Classification
    Variant_Type Reference_Allele  Tumor_Seq_Allele1  Tumor_Seq_Allele2
    HGVSp_Short  Protein_position  Amino_acids  Codons
    t_alt_count  t_ref_count  t_vaf  Tumor_Sample_Barcode  ...

We care about:
    * Hugo_Symbol           — gene name (for downstream annotation)
    * Tumor_Sample_Barcode  — patient ID
    * Variant_Classification — keep only missense; drop silent/synonymous,
                              splice, nonsense (those don't produce a stable
                              mutant peptide bound to HLA), and indels (v1
                              scope; v2 will add frameshift neoantigens).
    * HGVSp_Short           — protein change in HGVS form: 'p.R132H'
    * VAF                   — variant allele frequency (clonal weighting)

We do NOT have full protein sequences in the MAF, so we cannot generate
the full peptide window from the MAF alone. The pipeline uses HGVSp_Short
+ an Ensembl/UniProt protein database to fetch the flanking residues.
For v1 we ship a minimal in-memory protein fetcher that uses the UniProt
human_proteome.fasta already loaded.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd

from gbmvax.utils.device import get_logger

logger = get_logger(__name__)


# Variants that produce a defined point mutation in a translated protein.
MISSENSE_CLASSES = {
    "Missense_Mutation",
    "Missense",
}

# Pattern to parse 'p.R132H', 'p.A1234V', 'p.*123V' etc. We capture:
#   group(1): WT amino acid (one letter, optionally '*')
#   group(2): position (1-indexed in protein)
#   group(3): MT amino acid
HGVS_PATTERN = re.compile(r"^p\.([A-Z\*])(\d+)([A-Z\*])$")


def _detect_vaf_column(columns: pd.Index, candidates: list[str]) -> Optional[str]:
    """Return the first VAF-like column present, or None."""
    for c in candidates:
        if c in columns:
            return c
    return None


def load_maf(
    path: Path | str,
    vaf_columns: list[str] = None,
    min_vaf: float = 0.05,
) -> pd.DataFrame:
    """
    Load a single MAF file and return a cleaned per-mutation DataFrame.

    Returns columns:
        patient_id     str    Tumor_Sample_Barcode
        gene           str    Hugo_Symbol
        protein_pos    int    1-indexed amino acid position
        wt_aa          str    Single-letter WT residue
        mt_aa          str    Single-letter mutant residue
        vaf            float  Variant allele frequency
        hgvsp          str    Original HGVSp_Short string
        cohort         str    Filename stem — used to track provenance
    """
    vaf_columns = vaf_columns or ["t_vaf", "tumor_f", "vaf", "TUMOR_VAF"]

    path = Path(path)
    logger.info(f"Loading MAF {path.name}...")

    # MAFs sometimes have a '#version' comment at the top.
    df = pd.read_csv(path, sep="\t", comment="#", low_memory=False)

    # Filter to missense only. Variant_Classification is the standard MAF column.
    if "Variant_Classification" not in df.columns:
        raise KeyError(f"{path}: missing Variant_Classification column")
    df = df[df["Variant_Classification"].isin(MISSENSE_CLASSES)].copy()

    # Parse HGVSp_Short. Some MAFs have HGVSp; some have HGVSp_Short. Try both.
    hgvs_col = None
    for c in ("HGVSp_Short", "HGVSp", "Protein_Change", "amino_acid_change"):
        if c in df.columns:
            hgvs_col = c
            break
    if hgvs_col is None:
        raise KeyError(f"{path}: no HGVSp column found")

    # Parse the protein-change string. Drop any rows that don't match the
    # simple substitution pattern (covers ~95% of missense; the rest are
    # complex variants we exclude for v1).
    parsed = df[hgvs_col].astype(str).str.extract(HGVS_PATTERN)
    parsed.columns = ["wt_aa", "protein_pos", "mt_aa"]
    df = df.join(parsed)
    df = df.dropna(subset=["wt_aa", "protein_pos", "mt_aa"])
    df["protein_pos"] = df["protein_pos"].astype(int)

    # Drop nonsense substitutions (mt = '*') — those don't yield a translatable peptide.
    df = df[df["mt_aa"] != "*"]
    df = df[df["wt_aa"] != "*"]

    # VAF column detection.
    vaf_col = _detect_vaf_column(df.columns, vaf_columns)
    if vaf_col is None:
        # Try to compute from t_alt_count / (t_alt_count + t_ref_count).
        if "t_alt_count" in df.columns and "t_ref_count" in df.columns:
            df["vaf"] = df["t_alt_count"] / (df["t_alt_count"] + df["t_ref_count"])
            vaf_col = "vaf"
        else:
            logger.warning(f"{path}: no VAF column found; defaulting to 0.5 (assume heterozygous)")
            df["vaf"] = 0.5
            vaf_col = "vaf"
    df["vaf"] = pd.to_numeric(df[vaf_col], errors="coerce")
    df = df.dropna(subset=["vaf"])

    # Filter low-VAF variants — these are dominated by sequencing noise.
    df = df[df["vaf"] >= min_vaf]

    out = pd.DataFrame({
        "patient_id": df["Tumor_Sample_Barcode"].astype(str) if "Tumor_Sample_Barcode" in df.columns else "unknown",
        "gene": df["Hugo_Symbol"].astype(str),
        "protein_pos": df["protein_pos"].values,
        "wt_aa": df["wt_aa"].values,
        "mt_aa": df["mt_aa"].values,
        "vaf": df["vaf"].values,
        "hgvsp": df[hgvs_col].astype(str).values,
        "cohort": path.parent.name,
    })

    logger.info(f"{path.name}: {len(out):,} missense mutations from {out['patient_id'].nunique()} patients")
    return out


def load_all_training_cohorts(cfg: dict) -> pd.DataFrame:
    """
    Load TCGA 2013 + TCGA 2008 + CPTAC 2021 (training cohorts).
    Columbia 2019 is held out for validation — load it separately via
    load_validation_cohort().
    """
    paths = cfg["paths"]["tcga"]
    vaf_cols = cfg["clonal"]["vaf_column_candidates"]
    min_vaf = cfg["clonal"]["min_vaf"]

    dfs = []
    for key in ("gbm_2013", "gbm_2008", "cptac_2021"):
        if key in paths:
            dfs.append(load_maf(paths[key], vaf_columns=vaf_cols, min_vaf=min_vaf))
    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Training cohorts combined: {len(combined):,} mutations from {combined['patient_id'].nunique()} patients")
    return combined


def load_validation_cohort(cfg: dict) -> pd.DataFrame:
    """Load Columbia 2019 — held-out validation cohort, never seen during training."""
    paths = cfg["paths"]["tcga"]
    vaf_cols = cfg["clonal"]["vaf_column_candidates"]
    min_vaf = cfg["clonal"]["min_vaf"]
    return load_maf(paths["columbia_2019"], vaf_columns=vaf_cols, min_vaf=min_vaf)
