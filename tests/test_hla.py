"""Tests for gbmvax.utils.hla."""

import pytest

from gbmvax.utils.hla import (
    PSEUDOSEQUENCE_LENGTH,
    extract_pseudosequence,
    normalize_allele,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("HLA-A*02:01", "HLA-A*02:01"),
        ("A*02:01", "HLA-A*02:01"),
        ("HLA-A0201", "HLA-A*02:01"),
        ("A0201", "HLA-A*02:01"),
        ("hla-a*02:01:01:02", "HLA-A*02:01"),
        ("B*07:02", "HLA-B*07:02"),
        ("C*05:01", "HLA-C*05:01"),
    ],
)
def test_normalize_allele(raw, expected):
    assert normalize_allele(raw) == expected


def test_normalize_allele_invalid():
    with pytest.raises(ValueError):
        normalize_allele("not_an_hla")
    with pytest.raises(ValueError):
        normalize_allele("H-2-Kb")              # Mouse, should fail


def test_pseudosequence_length():
    # Build a fake HLA sequence long enough to extract from.
    seq = "M" * 24 + "G" + "S" + "H" + "S" + "M" + "R" + "Y" + "A" * 400
    pseudo = extract_pseudosequence(seq)
    assert len(pseudo) == PSEUDOSEQUENCE_LENGTH == 34


def test_pseudosequence_truncated_sequence_pads_with_X():
    # If the sequence is too short, missing positions become 'X'.
    pseudo = extract_pseudosequence("M" * 30)
    assert "X" in pseudo
