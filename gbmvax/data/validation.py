"""
validation.py — explicit Keskin 2019 and Hilf 2019 clinical-validation loaders.

The previous heuristic loader silently parsed the wrong Hilf column as a
positive-only cohort and missed Keskin Table S5 entirely. These loaders are
purposefully table-specific: the validation claim depends on exact columns.

Returned schema:
    patient_id    str
    peptide       str
    hla_allele    str
    response      int     1 for Keskin immunizing peptide, 0 for Hilf mutant background
    cohort        str     'keskin_2019' or 'hilf_2019_mutant_background'
    source_file   str
    source_sheet  str
    source_row    int     1-based spreadsheet row for auditability
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from gbmvax.utils.device import get_logger
from gbmvax.utils.hla import normalize_allele
from gbmvax.utils.sequences import AMINO_ACIDS

logger = get_logger(__name__)

_VALID_AA = set(AMINO_ACIDS)
KESKIN_TABLE_S5 = "41586_2018_792_MOESM5_ESM.xlsx"
HILF_TABLE = "41586_2018_810_MOESM3_ESM.xlsx"


def _is_peptide(value) -> bool:
    s = str(value).strip().upper()
    return 8 <= len(s) <= 15 and all(c in _VALID_AA for c in s)


def _clean_peptide(value) -> str:
    return str(value).strip().upper()


def _patient_id(value, width: int | None = None) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        out = str(int(value))
    else:
        out = str(value).strip()
    return out.zfill(width) if width else out


def _safe_normalize_allele(value) -> str:
    if pd.isna(value):
        return ""
    raw = str(value).strip()
    if not raw:
        return ""
    try:
        return normalize_allele(raw)
    except ValueError:
        return ""


def _validation_dir(cfg: dict, key: str) -> Path:
    return Path(cfg["paths"]["validation"][key])


def load_keskin(cfg: dict, holdout_patient: str | None = None) -> pd.DataFrame:
    """
    Load Keskin 2019 Table S5 immunizing mutated class-I peptides.

    Table S5 columns are fixed in the Nature supplement:
      A Patient ID, F HLA allele, G mutated peptide sequence, H affinity.
    These rows are positive examples for response/ranking validation.
    """
    path = _validation_dir(cfg, "keskin_2019") / KESKIN_TABLE_S5
    raw = pd.read_excel(path, sheet_name="Table S5", header=None)
    rows: list[dict] = []
    holdout = _patient_id(holdout_patient) if holdout_patient is not None else None

    for idx, r in raw.iloc[4:].iterrows():
        patient = _patient_id(r.iloc[0])
        peptide = _clean_peptide(r.iloc[6])
        if not patient or not _is_peptide(peptide):
            continue
        if holdout is not None and patient != holdout:
            continue
        rows.append({
            "patient_id": patient,
            "peptide": peptide,
            "hla_allele": _safe_normalize_allele(r.iloc[5]),
            "response": 1,
            "cohort": "keskin_2019",
            "source_file": path.name,
            "source_sheet": "Table S5",
            "source_row": int(idx + 1),
        })

    out = pd.DataFrame(rows)
    logger.info(
        f"keskin_2019: loaded {len(out)} Table S5 rows "
        f"({out['patient_id'].nunique() if len(out) else 0} patients)"
    )
    return out


def load_hilf(cfg: dict, exclude_peptides: set[str] | None = None) -> pd.DataFrame:
    """
    Load Hilf 2019 mutant/background epitope rows as response=0.

    In the Hilf supplement, row 2 contains the specific subheaders:
      A Patient, I best allele, K mutant epitope, L wild-type epitope.
    The historical Keskin-vs-Hilf validation uses the mutant epitope
    column as negatives against Keskin immunizing positives.
    """
    path = _validation_dir(cfg, "hilf_2019") / HILF_TABLE
    raw = pd.read_excel(path, sheet_name="Suppl. Table. 3_FINAL", header=None)
    excluded = {p.strip().upper() for p in exclude_peptides or set()}
    rows: list[dict] = []

    for idx, r in raw.iloc[2:].iterrows():
        patient = _patient_id(r.iloc[0], width=2)
        peptide = _clean_peptide(r.iloc[10])
        if not patient or not _is_peptide(peptide):
            continue
        if peptide in excluded:
            continue
        rows.append({
            "patient_id": patient,
            "peptide": peptide,
            "hla_allele": _safe_normalize_allele(r.iloc[8]),
            "response": 0,
            "cohort": "hilf_2019_mutant_background",
            "source_file": path.name,
            "source_sheet": "Suppl. Table. 3_FINAL",
            "source_row": int(idx + 1),
        })

    out = pd.DataFrame(rows)
    logger.info(
        f"hilf_2019_mutant_background: loaded {len(out)} mutant/background rows "
        f"({out['patient_id'].nunique() if len(out) else 0} patients)"
    )
    return out


def load_all_validation(
    cfg: dict,
    holdout_patient: str | None = None,
    exclude_hilf_peptides: set[str] | None = None,
) -> pd.DataFrame:
    """Return Keskin positives plus Hilf mutant/background negatives."""
    return pd.concat([
        load_keskin(cfg, holdout_patient=holdout_patient),
        load_hilf(cfg, exclude_peptides=exclude_hilf_peptides),
    ], ignore_index=True)


def load_keskin_as_iedb(
    cfg: dict,
    holdout_patient: str | None = None,
    upsample: int = 10,
) -> pd.DataFrame:
    """
    Convert Keskin Table S5 positives to the cleaned IEDB training schema.

    `holdout_patient` is excluded from the returned rows. This is the core
    leakage guard for leave-one-patient-out fine-tuning.
    """
    all_rows = load_keskin(cfg)
    if holdout_patient is not None:
        holdout = _patient_id(holdout_patient)
        heldout_peptides = set(all_rows.loc[all_rows["patient_id"] == holdout, "peptide"])
        all_rows = all_rows[
            (all_rows["patient_id"] != holdout)
            & ~all_rows["peptide"].isin(heldout_peptides)
        ].reset_index(drop=True)

    train_rows = all_rows[all_rows["hla_allele"].str.startswith("HLA-")].copy()
    out = pd.DataFrame({
        "peptide": train_rows["peptide"],
        "allele": train_rows["hla_allele"],
        "length": train_rows["peptide"].str.len(),
        "affinity_nM": pd.NA,
        "log_affinity": pd.NA,
        "presented": 1,
        "is_affinity": False,
        "is_presentation": True,
        "source_assay": "keskin_2019_table_s5_lopo_finetune",
        "source_patient_id": train_rows["patient_id"],
        "source_row": train_rows["source_row"],
    })
    out["affinity_nM"] = pd.to_numeric(out["affinity_nM"], errors="coerce")
    out["log_affinity"] = pd.to_numeric(out["log_affinity"], errors="coerce")

    if upsample > 1 and len(out):
        out = pd.concat([out] * upsample, ignore_index=True)
    return out.reset_index(drop=True)


def external_validation_peptides(cfg: dict) -> set[str]:
    """All peptide sequences used by the external clinical validation set."""
    df = load_all_validation(cfg)
    return set(df["peptide"].astype(str).str.strip().str.upper())
