"""Tests for composite scoring, clonal weighting, and processing scores."""

import numpy as np
import pandas as pd
import pytest

from gbmvax.models.processing import (
    combined_processing_score,
    netchop_cterm_score,
    tap_score,
)
from gbmvax.pipeline.clonal import vaf_to_clonal_score
from gbmvax.pipeline.composite import (
    CompositeWeights,
    binding_score_from_log_affinity,
    composite_score,
    immunogenicity_score,
    rank_candidates,
)


# ---------------------------------------------------------------------------
# Clonal
# ---------------------------------------------------------------------------
def test_clonal_zero_vaf_zero_score():
    assert vaf_to_clonal_score(0.0) == 0.0


def test_clonal_above_threshold_caps_at_one():
    assert vaf_to_clonal_score(0.4, clonal_threshold=0.3) == 1.0
    assert vaf_to_clonal_score(0.9, clonal_threshold=0.3) == 1.0


def test_clonal_linear_below_threshold():
    assert vaf_to_clonal_score(0.15, clonal_threshold=0.3) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Binding score transform
# ---------------------------------------------------------------------------
def test_binding_score_monotone_decreasing():
    """Higher predicted IC50 -> lower binding score."""
    s_strong = binding_score_from_log_affinity(np.log10(50.0))
    s_weak = binding_score_from_log_affinity(np.log10(500.0))
    s_none = binding_score_from_log_affinity(np.log10(50000.0))
    assert s_strong > s_weak > s_none


def test_binding_score_strong_threshold_near_one():
    s = binding_score_from_log_affinity(np.log10(50.0))
    assert s > 0.9


def test_binding_score_at_weak_threshold():
    s = binding_score_from_log_affinity(np.log10(500.0))
    assert 0.45 < s < 0.55


# ---------------------------------------------------------------------------
# Immunogenicity
# ---------------------------------------------------------------------------
def test_immunogenicity_in_range():
    s = immunogenicity_score("SIINFEKL")
    assert 0.0 <= s <= 1.0


def test_immunogenicity_hydrophobic_higher():
    """Hydrophobic central residues should score higher than acidic ones."""
    s_hydrophobic = immunogenicity_score("AAFLLYAAA")
    s_acidic = immunogenicity_score("AADDEAAA")
    assert s_hydrophobic > s_acidic


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------
def test_tap_score_hydrophobic_cterm_higher():
    high = tap_score("AAAAAAAAL")
    low = tap_score("AAAAAAAAD")
    assert high > low


def test_netchop_acidic_cterm_low():
    """Acidic C-terminus should give a low cleavage score."""
    low = netchop_cterm_score("SIINFEKD", "AAAAAAAAAAA", "AAAAAAAAAAA")
    high = netchop_cterm_score("SIINFEKL", "AAAAAAAAAAA", "AAAAAAAAAAA")
    assert low < high


def test_combined_processing_in_range():
    s = combined_processing_score(0.7, 0.5, tap_weight=0.3)
    assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------
def test_composite_score_clipped():
    """Composite must remain in [0, 1] even with extreme inputs."""
    # With default weights (binding 0.35 + processing 0.20 + clonal 0.20 +
    # immunogenicity 0.10) the maximum positive contribution is 0.85.
    # Cross-reactivity is subtracted, so the actual top of the scale is 0.85.
    w = CompositeWeights()
    max_positive = w.binding + w.processing + w.clonal + w.immunogenicity
    s = composite_score(1.0, 1.0, 1.0, 1.0, 0.0)
    assert s == pytest.approx(max_positive)
    # Maximum penalty with zero positives -> clipped to 0.
    s = composite_score(0.0, 0.0, 0.0, 0.0, 1.0)
    assert s == 0.0


def test_rank_candidates_sorts_descending():
    df = pd.DataFrame({
        "binding_score": [0.9, 0.1, 0.5],
        "processing_score": [0.5, 0.5, 0.5],
        "clonal_score": [0.5, 0.5, 0.5],
        "immunogenicity_score": [0.5, 0.5, 0.5],
        "cross_reactivity_penalty": [0.0, 0.0, 0.0],
    })
    ranked = rank_candidates(df)
    # Row 0 has highest binding -> should be at top.
    assert ranked.iloc[0]["binding_score"] == 0.9
    assert ranked["composite_score"].is_monotonic_decreasing
