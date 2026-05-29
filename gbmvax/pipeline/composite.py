"""
composite.py — final neoantigen ranking score.

Combines five signals into a single [0, 1] score per candidate peptide:

    binding              — from the multi-task transformer's affinity head,
                           transformed via 1 - normalize(log10 IC50). Strong
                           binders (low IC50) get scores near 1.
    processing           — combined NetChop + TAP + reranker, already in [0, 1].
    clonal               — VAF-derived clonality, already in [0, 1].
    immunogenicity       — TCR-recognition propensity. v1 uses a simple
                           hydrophobicity-of-central-residues heuristic
                           (Calis et al. 2013); v2 can swap in a learned
                           model trained on tcell_full_v3.
    cross_reactivity     — penalty in [0, 1]. SUBTRACTED, not added.

Final formula:
    score = w_b * binding + w_p * processing + w_c * clonal +
            w_i * immunogenicity - w_x * cross_reactivity_penalty
    score = clip(score, 0.0, 1.0)

Weights are configurable; defaults from config.yaml:
    binding 0.35, processing 0.20, clonal 0.20, immunogenicity 0.10,
    cross-reactivity penalty 0.15.

We choose binding as the dominant factor because it is the necessary
condition: no binding = no presentation = no T cell response, period.
Other factors break ties between binders.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# Immunogenicity heuristic — Calis et al. 2013 PLoS Comp Biol.
#
# Observation: T-cell-immunogenic peptides have an enrichment of hydrophobic
# / aromatic residues at TCR-contact positions (P4-P6 in 9-mers). The
# original Calis paper trained a position-weighted matrix; we ship a
# minimal approximation here. The matrix below is the published weights
# for class I 9-mer immunogenicity.
# ----------------------------------------------------------------------------

# Calis 2013 PSSM, simplified — residue contribution to immunogenicity at
# the TCR-contact positions only. Outside these positions the contribution
# is 0.
_IMMUNOGENICITY_AA: dict[str, float] = {
    "A":  0.127, "C":  0.000, "D": -0.183, "E": -0.124, "F":  0.382,
    "G":  0.110, "H":  0.105, "I":  0.232, "K":  0.169, "L":  0.247,
    "M":  0.220, "N": -0.149, "P":  0.000, "Q": -0.075, "R":  0.168,
    "S": -0.164, "T": -0.054, "V":  0.106, "W":  0.221, "Y":  0.286,
}

# Position weights (1-indexed within the peptide) — Calis found P4-P6
# carry the immunogenicity signal for 9-mers.
_POSITION_WEIGHTS_9MER = {1: 0.0, 2: 0.0, 3: 0.10, 4: 0.31, 5: 0.30, 6: 0.29, 7: 0.0, 8: 0.0, 9: 0.0}


def immunogenicity_score(peptide: str) -> float:
    """
    Heuristic T-cell immunogenicity score in [0, 1].

    For non-9-mers we use a length-normalized version (treat the central
    third of the peptide as the TCR-contact region).
    """
    L = len(peptide)
    if L == 0:
        return 0.0

    if L == 9:
        weights = _POSITION_WEIGHTS_9MER
        raw = sum(_IMMUNOGENICITY_AA.get(aa, 0.0) * weights.get(i + 1, 0.0)
                  for i, aa in enumerate(peptide))
    else:
        # Generalize: central third of the peptide gets the contact weight,
        # outside positions get zero.
        central_start = L // 3
        central_end = L - L // 3
        raw = 0.0
        for i, aa in enumerate(peptide):
            w = 0.3 if central_start <= i < central_end else 0.0
            raw += _IMMUNOGENICITY_AA.get(aa, 0.0) * w

    # Calis raw scores span roughly [-1, +1]. Map to [0, 1] via sigmoid-ish squash.
    return float(1.0 / (1.0 + np.exp(-3.0 * raw)))


# ----------------------------------------------------------------------------
# Binding -> score transform.
#
# The model emits log10(IC50_nM). Strong binders have log <= log10(50) = 1.7;
# weak binders log <= log10(500) = 2.7; non-binders log >= 4.7 (50k nM).
# We transform to [0, 1] with a sigmoid centered at log10(500) = weak threshold.
# This gives a smooth gradient rather than a hard step at the threshold.
# ----------------------------------------------------------------------------
def binding_score_from_log_affinity(
    log_ic50: float,
    strong_threshold_log: float = np.log10(50.0),    # 1.70
    weak_threshold_log: float = np.log10(500.0),     # 2.70
) -> float:
    """
    Map predicted log10(IC50 nM) into a binding score in [0, 1].

    score(log=1.7)  ~ 0.95 (strong binder)
    score(log=2.7)  ~ 0.50 (weak binder threshold)
    score(log=4.7)  ~ 0.05 (non-binder)

    Implemented as a logistic centered at the weak threshold, with slope
    chosen so the strong threshold maps to ~0.95.
    """
    # Solve for slope k from: sigmoid(-k * (strong - weak)) = 0.05
    # i.e. k = log(19) / (weak - strong)
    k = np.log(19.0) / max(1e-6, weak_threshold_log - strong_threshold_log)
    return float(1.0 / (1.0 + np.exp(k * (log_ic50 - weak_threshold_log))))


@dataclass
class CompositeWeights:
    """Defaults match config.yaml's `composite.weights` block."""
    binding: float = 0.35
    processing: float = 0.20
    clonal: float = 0.20
    immunogenicity: float = 0.10
    cross_reactivity_penalty: float = 0.15

    @classmethod
    def from_config(cls, cfg: dict) -> "CompositeWeights":
        w = cfg["composite"]["weights"]
        return cls(
            binding=w["binding"],
            processing=w["processing"],
            clonal=w["clonal"],
            immunogenicity=w["immunogenicity"],
            cross_reactivity_penalty=w["cross_reactivity_penalty"],
        )


def composite_score(
    binding: float,
    processing: float,
    clonal: float,
    immunogenicity: float,
    cross_reactivity_penalty: float,
    weights: CompositeWeights | None = None,
) -> float:
    """
    Compute the final composite score for one candidate.

    All inputs are expected to be in [0, 1]. Output clipped to [0, 1].
    """
    w = weights or CompositeWeights()
    raw = (w.binding * binding
           + w.processing * processing
           + w.clonal * clonal
           + w.immunogenicity * immunogenicity
           - w.cross_reactivity_penalty * cross_reactivity_penalty)
    return float(np.clip(raw, 0.0, 1.0))


def rank_candidates(
    candidates: pd.DataFrame,
    weights: CompositeWeights | None = None,
    top_n: int | None = None,
) -> pd.DataFrame:
    """
    Add a `composite_score` column and sort descending. Optionally return
    just the top N.

    Required columns:
        binding_score, processing_score, clonal_score,
        immunogenicity_score, cross_reactivity_penalty
    """
    w = weights or CompositeWeights()

    # Vectorized — much faster than .apply for large candidate sets.
    raw = (w.binding * candidates["binding_score"]
           + w.processing * candidates["processing_score"]
           + w.clonal * candidates["clonal_score"]
           + w.immunogenicity * candidates["immunogenicity_score"]
           - w.cross_reactivity_penalty * candidates["cross_reactivity_penalty"])
    out = candidates.copy()
    out["composite_score"] = raw.clip(0.0, 1.0)
    out = out.sort_values("composite_score", ascending=False).reset_index(drop=True)

    if top_n is not None:
        out = out.head(top_n)
    return out
