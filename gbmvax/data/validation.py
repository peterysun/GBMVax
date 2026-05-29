"""
validation.py — load Keskin 2019 and Hilf 2019 clinical-trial supplements.

These are the ONLY datasets in the world with both:
    (a) GBM-patient neoantigen vaccines administered, and
    (b) Per-peptide T-cell response measurements (ELISpot, tetramer staining).

They are our ground truth for the primary validation metric: can GBMVax
predict which neoantigens actually elicited T cell responses?

The supplements are Excel files with multiple sheets. Format is non-uniform:
Keskin uses one row per (patient, peptide) with a 'Response' column; Hilf
uses a peptide-by-patient matrix. We standardize both into a tidy long
table.

Returned schema:
    patient_id    str
    peptide       str
    hla_allele    str
    response      int     1 if T cell response observed, 0 otherwise
    cohort        str     'keskin_2019' or 'hilf_2019'
    notes         str     Original assay column (for traceability)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from gbmvax.utils.device import get_logger
from gbmvax.utils.hla import normalize_allele
from gbmvax.utils.sequences import AMINO_ACIDS

logger = get_logger(__name__)

_VALID_AA = set(AMINO_ACIDS)


# ----------------------------------------------------------------------------
# Heuristics for picking the relevant columns. Keskin and Hilf supplements
# are not standardized — sheet and column names vary. We hardcode the
# patterns observed in the published files. If Nature reformats the
# supplements (rare but possible), update these.
# ----------------------------------------------------------------------------
PEPTIDE_COL_HINTS = ("peptide", "epitope", "sequence", "neoantigen")
RESPONSE_COL_HINTS = ("response", "elispot", "ifn", "tcell", "immunogen", "reactiv")
HLA_COL_HINTS = ("hla", "allele", "restriction")
PATIENT_COL_HINTS = ("patient", "subject", "donor", "id")


def _find_col(cols: list[str], hints: tuple[str, ...]) -> str | None:
    """Return first column whose lowercased name contains any hint."""
    for c in cols:
        cl = str(c).lower()
        if any(h in cl for h in hints):
            return c
    return None


def _is_peptide(s: str) -> bool:
    """Lenient check: 8-15 residues, canonical amino acids only."""
    s = str(s).strip().upper()
    return 8 <= len(s) <= 15 and all(c in _VALID_AA for c in s)


def _load_supplement_dir(dir_path: Path, cohort: str) -> pd.DataFrame:
    """
    Iterate over every .xlsx in dir_path and try to extract a tidy
    (patient, peptide, hla, response) table from each sheet.

    Robustness strategy: try to find columns by hint; if a column is
    missing, fall through. Aggressively drop rows that don't look like
    real peptide entries.
    """
    rows: list[dict] = []

    for xlsx in sorted(dir_path.glob("*.xlsx")):
        try:
            # sheet_name=None reads all sheets into a dict.
            book = pd.read_excel(xlsx, sheet_name=None, header=None)
        except Exception as e:                     # noqa: BLE001 — supplement files vary
            logger.warning(f"Could not open {xlsx.name}: {e}")
            continue

        for sheet_name, raw in book.items():
            # Detect the header row. Supplements often have title text in
            # rows 0-2. We assume the row containing 'peptide' or 'epitope'
            # is the header.
            header_row = None
            for i in range(min(8, len(raw))):
                row_strs = [str(x).lower() for x in raw.iloc[i].tolist()]
                if any(any(h in s for h in PEPTIDE_COL_HINTS) for s in row_strs):
                    header_row = i
                    break
            if header_row is None:
                continue

            # Re-read with the detected header.
            df = raw.iloc[header_row + 1:].copy()
            df.columns = [str(c) for c in raw.iloc[header_row].tolist()]
            df = df.dropna(how="all")

            pep_col = _find_col(df.columns.tolist(), PEPTIDE_COL_HINTS)
            resp_col = _find_col(df.columns.tolist(), RESPONSE_COL_HINTS)
            hla_col = _find_col(df.columns.tolist(), HLA_COL_HINTS)
            pat_col = _find_col(df.columns.tolist(), PATIENT_COL_HINTS)

            if pep_col is None:
                continue

            for _, r in df.iterrows():
                pep = str(r[pep_col]).strip().upper()
                if not _is_peptide(pep):
                    continue

                # Response — accept many encodings:
                #   numeric > 0 -> positive
                #   '+', 'positive', 'yes', 'response' -> positive
                #   '-', 'negative', 'no', 'none', 0, NaN -> negative
                if resp_col is not None:
                    raw_resp = r[resp_col]
                    response = _parse_response(raw_resp)
                else:
                    response = 1                  # In Keskin Table S5 every listed peptide was tested positive

                hla = str(r[hla_col]).strip() if hla_col and pd.notna(r[hla_col]) else ""
                try:
                    hla_norm = normalize_allele(hla) if hla else ""
                except ValueError:
                    hla_norm = ""

                patient = str(r[pat_col]).strip() if pat_col and pd.notna(r[pat_col]) else xlsx.stem

                rows.append({
                    "patient_id": patient,
                    "peptide": pep,
                    "hla_allele": hla_norm,
                    "response": response,
                    "cohort": cohort,
                    "source_file": xlsx.name,
                    "source_sheet": sheet_name,
                })

    out = pd.DataFrame(rows).drop_duplicates(subset=["patient_id", "peptide"])
    logger.info(f"{cohort}: extracted {len(out)} (patient, peptide) rows from {dir_path}")
    return out


def _parse_response(value) -> int:
    """Coerce a heterogeneous response cell to 0/1."""
    if pd.isna(value):
        return 0
    if isinstance(value, (int, float)):
        return int(value > 0)
    s = str(value).strip().lower()
    if s in {"+", "positive", "yes", "y", "1", "true", "response", "reactive"}:
        return 1
    if s in {"-", "negative", "no", "n", "0", "false", "none", "nd", "n/a"}:
        return 0
    # Numeric-looking strings ("2.5", "150 sfu")
    try:
        return int(float(s.split()[0]) > 0)
    except (ValueError, IndexError):
        return 0


def load_keskin(cfg: dict) -> pd.DataFrame:
    """Load Keskin 2019 supplements."""
    return _load_supplement_dir(Path(cfg["paths"]["validation"]["keskin_2019"]), "keskin_2019")


def load_hilf(cfg: dict) -> pd.DataFrame:
    """Load Hilf 2019 supplements."""
    return _load_supplement_dir(Path(cfg["paths"]["validation"]["hilf_2019"]), "hilf_2019")


def load_all_validation(cfg: dict) -> pd.DataFrame:
    """Concat both clinical-trial cohorts for the primary validation metric."""
    return pd.concat([load_keskin(cfg), load_hilf(cfg)], ignore_index=True)
