"""
clonal.py — clonal architecture weighting.

A tumor is a population of cells with shared (truncal) and divergent
(subclonal) mutations. A vaccine targeting a SUBCLONAL mutation only kills
the subclone — the truncal majority keeps growing and the tumor recurs.
A vaccine targeting a TRUNCAL mutation hits every cancer cell.

We measure clonality via Variant Allele Frequency (VAF): the fraction of
sequencing reads at a locus that carry the mutation. For a heterozygous
truncal mutation in a diploid pure tumor:
    VAF = 0.5 * tumor_purity
For a subclonal mutation:
    VAF = 0.5 * tumor_purity * subclone_fraction

v1: raw VAF -> piecewise score.
v2 (stub): call PyClone-VI for proper Cancer Cell Fraction (CCF) estimation
        that corrects for tumor purity and local copy number.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from gbmvax.utils.device import get_logger

logger = get_logger(__name__)


def vaf_to_clonal_score(vaf: float, clonal_threshold: float = 0.3) -> float:
    """
    Convert a VAF into a clonality score in [0, 1].

    Scoring:
        VAF >= clonal_threshold     -> 1.0  (highly clonal, prioritize)
        VAF < clonal_threshold      -> linear ramp from 0.0 at VAF=0 to 1.0 at threshold
        VAF >= 0.45                 -> capped at 1.0 (heterozygous truncal in 90%+ purity tumor)

    The piecewise design gives a flat "definitely clonal" plateau rather
    than continuing to reward higher VAFs (which often reflect LOH or
    copy-number gain, not more clonal). This prevents copy-number outliers
    from dominating the ranking.
    """
    vaf = max(0.0, float(vaf))
    if vaf >= clonal_threshold:
        return 1.0
    return vaf / clonal_threshold


def score_mutations_clonal(
    mutations: pd.DataFrame,
    clonal_threshold: float = 0.3,
) -> pd.DataFrame:
    """
    Add a `clonal_score` column to a mutations DataFrame.

    Operates on the per-mutation table from data/mutations.py.
    """
    out = mutations.copy()
    out["clonal_score"] = out["vaf"].apply(lambda v: vaf_to_clonal_score(v, clonal_threshold))
    return out


# ----------------------------------------------------------------------------
# v2 hook — PyClone-VI integration.
# Stubbed but documented for the next iteration. PyClone-VI takes per-mutation
# read counts plus copy-number segments and outputs CCF (Cancer Cell Fraction).
# CCF is the proper measure: VAF=0.25 in a 50%-pure tumor means CCF=1.0
# (every cancer cell carries it), but VAF=0.25 in a 100%-pure diploid tumor
# means CCF=0.5 (half the cancer cells carry it).
# ----------------------------------------------------------------------------
def estimate_ccf_pyclone(
    mutations: pd.DataFrame,
    tumor_purity: Optional[float] = None,
    copy_number: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    [v2 STUB] Estimate Cancer Cell Fraction using PyClone-VI.

    Requirements (not satisfied in v1):
        * tumor_purity from ABSOLUTE or ASCAT
        * per-segment copy-number calls
        * PyClone-VI installed (`pip install pyclone-vi`)

    For v1 we just emit a warning and fall back to VAF. Implementing this
    properly is the highest-impact methodological upgrade for the paper.
    """
    if tumor_purity is None or copy_number is None:
        logger.warning("estimate_ccf_pyclone called without purity/CN — returning VAF unchanged. Implement before v2.")
        out = mutations.copy()
        out["ccf"] = out["vaf"] * 2                   # Naive estimate
        return out

    # Real implementation goes here. Outline:
    #   1. Format mutations as PyClone-VI input TSV
    #   2. Run pyclone-vi fit + assign
    #   3. Parse output, attach 'ccf' column
    raise NotImplementedError("Full PyClone-VI integration is a v2 deliverable")
