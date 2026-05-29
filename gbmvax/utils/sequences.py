"""
sequences.py — amino acid encoding, BLOSUM62, and HLA pseudosequence helpers.

Every model in this pipeline operates on amino acid sequences. We need:
    * Integer encoding for transformer input (peptide -> [int])
    * BLOSUM62 for similarity scoring (cross-reactivity)
    * The NetMHCpan 34-residue HLA pseudosequence for binding prediction

Biology:
    HLA class I molecules are ~365 residues long but only ~34 residues line
    the peptide-binding groove. NetMHCpan and successors all use these 34
    positions as the "pseudosequence" — the only part of the HLA that
    actually contacts the peptide. Using the full HLA would force the model
    to learn that 90%+ of the input is irrelevant.

    BLOSUM62 is a substitution matrix derived from blocks of conserved
    protein regions. Positive scores indicate residues that often replace
    each other in evolution — and, by extension, are biochemically similar
    enough that a TCR might confuse them. Hence its use in cross-reactivity
    scoring.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from Bio.Align import substitution_matrices    # Source of BLOSUM62


# ----------------------------------------------------------------------------
# Amino acid vocabulary.
# Index 0 reserved for padding so we can pad short peptides up to max_len.
# ----------------------------------------------------------------------------
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"            # Canonical 20, in IUPAC order
PAD_TOKEN = "-"
VOCAB = PAD_TOKEN + AMINO_ACIDS                  # 21 symbols total
AA_TO_IDX: dict[str, int] = {aa: i for i, aa in enumerate(VOCAB)}
IDX_TO_AA: dict[int, str] = {i: aa for aa, i in AA_TO_IDX.items()}
VOCAB_SIZE = len(VOCAB)


def encode_peptide(seq: str, max_len: int) -> np.ndarray:
    """
    Convert an amino acid string to a left-padded int array of length max_len.

    We left-pad (pad at the C-terminus, end of the array) because most
    MHC-I peptides are anchored at positions P2 and P-omega. Keeping the
    N-terminus aligned at index 0 means the model sees P2 at index 1
    regardless of peptide length — useful for the attention pattern to
    learn anchor preferences.

    Non-canonical residues (X, U, B, Z, *, etc.) map to PAD. This is a
    deliberate filtering choice — peptides containing them are usually
    sequencing artifacts or selenocysteine, and we exclude them from
    training upstream. The mapping here is a safety net.
    """
    if len(seq) > max_len:
        raise ValueError(f"Peptide length {len(seq)} exceeds max_len {max_len}")

    # Pre-fill with pad index, then overwrite the actual residues.
    out = np.zeros(max_len, dtype=np.int64)
    for i, aa in enumerate(seq):
        out[i] = AA_TO_IDX.get(aa, 0)            # Unknown -> pad
    return out


def decode_peptide(indices: Iterable[int]) -> str:
    """Inverse of encode_peptide — strips trailing pads."""
    chars = [IDX_TO_AA[int(i)] for i in indices]
    # Strip pad tokens from the right; internal pads would indicate a bug.
    return "".join(chars).rstrip(PAD_TOKEN)


# ----------------------------------------------------------------------------
# BLOSUM62.
# Loaded once at import — it's a 24x24 dense matrix, negligible memory.
# We expose a dict-of-dicts wrapper for fast lookups and a NumPy version
# for vectorized scoring.
# ----------------------------------------------------------------------------
_BLOSUM62 = substitution_matrices.load("BLOSUM62")


def blosum_score(a: str, b: str) -> int:
    """
    BLOSUM62 score for a residue pair. Symmetric.

    Score interpretation:
        +ve: residues are evolutionarily/biochemically similar (e.g. I-L: +2)
        0  : neutral substitution (e.g. A-S: 0)
        -ve: dissimilar (e.g. W-D: -4)
    """
    # Biopython's matrix supports both orderings; .get is forgiving.
    try:
        return int(_BLOSUM62[a, b])
    except (KeyError, IndexError):
        return -4                                # Penalize unknowns harshly


# Build a NumPy lookup matrix once — keyed by AMINO_ACIDS index order — so
# downstream code can do `mat[i, j]` instead of dict lookups in tight loops.
_BLOSUM_MATRIX = np.full((len(AMINO_ACIDS), len(AMINO_ACIDS)), -4, dtype=np.int8)
for i, a in enumerate(AMINO_ACIDS):
    for j, b in enumerate(AMINO_ACIDS):
        _BLOSUM_MATRIX[i, j] = blosum_score(a, b)


def blosum_matrix() -> np.ndarray:
    """Return the 20x20 BLOSUM62 matrix indexed by AMINO_ACIDS order."""
    return _BLOSUM_MATRIX


def pairwise_blosum_score(
    pep_a: str, pep_b: str, anchor_weight: float = 1.0
) -> float:
    """
    Sum-of-pairs BLOSUM62 score between two equal-length peptides.

    Anchor positions (P2, second-from-last) get `anchor_weight` multiplier
    because these residues sit in deep HLA pockets (B and F) and dominate
    both HLA binding and TCR recognition. A peptide that matches self only
    at non-anchor positions is much less likely to cross-react.
    """
    if len(pep_a) != len(pep_b):
        raise ValueError("Peptides must be same length for sum-of-pairs scoring")

    total = 0.0
    L = len(pep_a)
    for i, (a, b) in enumerate(zip(pep_a, pep_b)):
        # Position 1 (0-indexed) is P2; position L-1 is P-omega.
        weight = anchor_weight if (i == 1 or i == L - 1) else 1.0
        total += weight * blosum_score(a, b)
    return total


def normalized_similarity(pep_a: str, pep_b: str, anchor_weight: float = 1.0) -> float:
    """
    Normalize sum-of-pairs BLOSUM62 to [0, 1] where 1.0 = identical.

    Normalization scheme:
        score_norm = (score - score_min) / (score_max - score_min)
    where score_max is the self-self score for pep_a (perfect match) and
    score_min is the worst plausible score (all -4 substitutions, weighted).
    This makes the threshold (config: 0.8) interpretable across peptide
    lengths 8–11.
    """
    if len(pep_a) != len(pep_b):
        return 0.0

    raw = pairwise_blosum_score(pep_a, pep_b, anchor_weight)
    self_score = pairwise_blosum_score(pep_a, pep_a, anchor_weight)
    # Worst case: every position scores -4. Total weight = (L-2) + 2*anchor_weight.
    L = len(pep_a)
    total_weight = (L - 2) + 2 * anchor_weight
    min_score = -4.0 * total_weight

    if self_score == min_score:                  # Degenerate; shouldn't happen for real peptides
        return 0.0
    return float((raw - min_score) / (self_score - min_score))
