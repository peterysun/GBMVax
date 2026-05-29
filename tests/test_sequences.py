"""Tests for gbmvax.utils.sequences."""

import numpy as np
import pytest

from gbmvax.utils.sequences import (
    AA_TO_IDX,
    AMINO_ACIDS,
    VOCAB_SIZE,
    blosum_score,
    decode_peptide,
    encode_peptide,
    normalized_similarity,
    pairwise_blosum_score,
)


def test_encode_decode_roundtrip():
    pep = "SIINFEKL"
    enc = encode_peptide(pep, max_len=11)
    assert enc.shape == (11,)
    # First 8 positions are the peptide; positions 8-10 are pad (0).
    assert decode_peptide(enc) == pep
    assert enc[8] == 0 and enc[9] == 0 and enc[10] == 0


def test_encode_unknown_residue_maps_to_pad():
    enc = encode_peptide("ACDX", max_len=11)
    # X is non-canonical and maps to pad index 0.
    assert enc[3] == 0


def test_encode_overflow_raises():
    with pytest.raises(ValueError):
        encode_peptide("ACDEFGHIKLMNP", max_len=11)


def test_vocab_size():
    # 20 canonical AAs + 1 pad token.
    assert VOCAB_SIZE == 21


def test_blosum_identity_positive():
    # Self-self BLOSUM62 is always positive.
    for aa in AMINO_ACIDS:
        assert blosum_score(aa, aa) > 0


def test_blosum_known_pair():
    # I and L are biochemically similar — BLOSUM62 puts I-L at +2.
    assert blosum_score("I", "L") == 2


def test_normalized_similarity_identity():
    sim = normalized_similarity("SIINFEKL", "SIINFEKL", anchor_weight=1.0)
    assert sim == pytest.approx(1.0)


def test_normalized_similarity_anchor_weight():
    # Mismatch at P2 weighs more with higher anchor_weight.
    sim_low = normalized_similarity("SIINFEKL", "SXINFEKL", anchor_weight=1.0)
    sim_high = normalized_similarity("SIINFEKL", "SXINFEKL", anchor_weight=3.0)
    assert sim_high < sim_low


def test_pairwise_blosum_symmetric():
    a, b = "SIINFEKL", "SLLNFAKL"
    assert pairwise_blosum_score(a, b) == pairwise_blosum_score(b, a)
