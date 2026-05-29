"""
cross_reactivity.py — autoimmune safety filter.

A neoantigen vaccine ELICITS a T cell population whose TCRs recognize the
mutant peptide on HLA. If those same TCRs also recognize a self-peptide
that happens to be displayed on healthy tissue, the patient develops
autoimmunity. This filter rejects candidates whose nearest self-match is
too similar.

Algorithm:
    for each candidate (peptide, length L):
        candidates_in_proteome = k-mer prefilter -> small set of windows
        for each window:
            score = normalized BLOSUM62 with anchor weighting
        return max(scores)
    if max > threshold: reject candidate

Why BLOSUM62 instead of identity / hamming distance:
    Identity would miss biochemically equivalent substitutions
    (e.g. I -> L, K -> R) that TCRs can still cross-react against.
    BLOSUM62 captures the gradient of substitution tolerance.

Why anchor weighting:
    The TCR contacts the central residues of the peptide most heavily
    (P3-P5 for 9-mers). Anchor residues (P2, P-omega) point INTO the HLA
    groove and the TCR rarely sees them. A self-match that differs only
    at anchors is essentially identical from the TCR's perspective and
    SHOULD be flagged. We invert this: we give anchors HIGHER weight in
    similarity scoring so that anchor-matched self-peptides hit threshold
    more easily — i.e. we are more conservative about rejecting them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gbmvax.data.proteome import Proteome
from gbmvax.utils.sequences import normalized_similarity


@dataclass
class CrossReactivityHit:
    """A single self-protein match for a candidate peptide."""
    protein_id: str
    start: int
    self_sequence: str
    similarity: float


def nearest_self_match(
    peptide: str,
    proteome: Proteome,
    anchor_weight: float = 1.5,
    use_prefilter: bool = True,
    early_exit_threshold: float | None = None,
) -> CrossReactivityHit | None:
    """
    Find the closest self-peptide match in the proteome.

    Args:
        peptide: candidate neoantigen sequence.
        proteome: Indexed Proteome instance.
        anchor_weight: weight on P2 and P-omega positions for similarity scoring.
        use_prefilter: if True, only score windows sharing a k-mer with the
            peptide. ~100x speedup with negligible recall loss.
        early_exit_threshold: if a similarity exceeds this, return immediately
            without scoring the rest. Useful for the filter step where we
            only care that *some* match crosses threshold.

    Returns:
        The best (highest-similarity) hit, or None if no candidate windows
        existed (extremely rare — most short peptides have at least one
        k-mer hit somewhere in the proteome).
    """
    L = len(peptide)
    if L < 5:
        return None

    best: CrossReactivityHit | None = None

    # Choose iterator: prefilter (~thousands of windows) vs exhaustive (millions).
    if use_prefilter:
        window_iter = proteome.candidate_matches(peptide)
    else:
        # Exhaustive — iterate every position in every protein. Only used
        # when the prefilter would miss (e.g. for full-proteome benchmarks).
        def _exhaustive():
            for pid, seq in proteome.proteins.items():
                for start in range(len(seq) - L + 1):
                    yield pid, start, start + L
        window_iter = _exhaustive()

    for pid, start, end in window_iter:
        window = proteome.proteins[pid][start:end]
        if len(window) != L:
            continue
        sim = normalized_similarity(peptide, window, anchor_weight=anchor_weight)
        if best is None or sim > best.similarity:
            best = CrossReactivityHit(
                protein_id=pid, start=start, self_sequence=window, similarity=sim,
            )
            if early_exit_threshold is not None and sim >= early_exit_threshold:
                return best

    return best


def cross_reactivity_penalty(
    peptide: str,
    proteome: Proteome,
    similarity_threshold: float = 0.8,
    anchor_weight: float = 1.5,
) -> float:
    """
    Return a penalty in [0, 1] for a candidate peptide.

    0.0 = no significant self-match (safe to include in vaccine)
    1.0 = identical to a self-peptide somewhere in the proteome

    The penalty grows linearly above the threshold:
        penalty = max(0, (similarity - threshold) / (1 - threshold))
    Below threshold the penalty is zero. This rewards candidates with no
    plausible self-match and progressively punishes ones that approach
    identity to self.
    """
    hit = nearest_self_match(
        peptide, proteome, anchor_weight=anchor_weight,
        early_exit_threshold=None,
    )
    if hit is None:
        return 0.0
    if hit.similarity < similarity_threshold:
        return 0.0
    return float((hit.similarity - similarity_threshold) / max(1e-6, 1.0 - similarity_threshold))


def passes_cross_reactivity_filter(
    peptide: str,
    proteome: Proteome,
    similarity_threshold: float = 0.8,
    anchor_weight: float = 1.5,
) -> bool:
    """
    Hard filter version: True if peptide is safe to include.

    Uses early-exit: as soon as any self-match exceeds threshold, returns False.
    """
    hit = nearest_self_match(
        peptide, proteome,
        anchor_weight=anchor_weight,
        early_exit_threshold=similarity_threshold,
    )
    return hit is None or hit.similarity < similarity_threshold
