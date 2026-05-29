"""
dataset.py — PyTorch Dataset for IEDB peptide–HLA binding training.

The dataset wraps the cleaned IEDB DataFrame produced by data/iedb.py and
the pseudosequence map produced by utils/hla.py. Each item yields the
tensors the multi-task transformer needs:

    peptide_tokens     [P]   int       padded peptide
    hla_tokens         [H]   int       HLA pseudosequence
    log_affinity       []    float     log10(IC50 nM); 0.0 if missing (masked)
    affinity_mask      []    float     1.0 if affinity label present, else 0.0
    presentation       []    float     0 or 1; 0.0 if missing (masked)
    presentation_mask  []    float     1.0 if presentation label present, else 0.0
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from gbmvax.utils.sequences import encode_peptide


class IEDBDataset(Dataset):
    """Wraps a cleaned IEDB DataFrame for the multi-task binding model."""

    def __init__(
        self,
        df: pd.DataFrame,
        pseudoseq_map: dict[str, str],
        max_peptide_len: int = 11,
        pseudoseq_len: int = 34,
    ):
        self.max_peptide_len = max_peptide_len
        self.pseudoseq_len = pseudoseq_len

        # Drop rows whose allele has no pseudosequence — the model can't
        # represent them. This typically removes ~1-2% of rows (alleles
        # missing from IMGT or with malformed pseudosequences).
        df = df[df["allele"].isin(pseudoseq_map)].reset_index(drop=True)

        # Pre-encode everything to NumPy arrays. The Dataset only does
        # indexing + tensor conversion, no per-item work — keeps the
        # DataLoader workers fast.
        self.peptides = df["peptide"].tolist()
        self.alleles = df["allele"].tolist()

        # Vectorized peptide encoding into one large [N, max_peptide_len] array.
        N = len(df)
        self.pep_tokens = np.zeros((N, max_peptide_len), dtype=np.int64)
        for i, pep in enumerate(self.peptides):
            self.pep_tokens[i] = encode_peptide(pep, max_peptide_len)

        # HLA tokens — same allele appears many times, so cache once.
        allele_to_tokens: dict[str, np.ndarray] = {}
        for allele, pseudo in pseudoseq_map.items():
            allele_to_tokens[allele] = encode_peptide(pseudo, pseudoseq_len)
        self.hla_tokens = np.stack([allele_to_tokens[a] for a in self.alleles])

        # Affinity labels. NaN -> 0 with mask 0.
        log_aff = df["log_affinity"].to_numpy(dtype=np.float32)
        aff_mask = np.isfinite(log_aff).astype(np.float32)
        log_aff = np.nan_to_num(log_aff, nan=0.0)
        self.log_affinity = log_aff
        self.affinity_mask = aff_mask

        # Presentation labels. -1 -> 0 with mask 0; 0/1 -> mask 1.
        pres = df["presented"].to_numpy(dtype=np.int8)
        pres_mask = (pres >= 0).astype(np.float32)
        pres = np.where(pres >= 0, pres, 0).astype(np.float32)
        self.presentation = pres
        self.presentation_mask = pres_mask

    def __len__(self) -> int:
        return len(self.peptides)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "peptide_tokens": torch.from_numpy(self.pep_tokens[idx]),
            "hla_tokens": torch.from_numpy(self.hla_tokens[idx]),
            "log_affinity": torch.tensor(self.log_affinity[idx], dtype=torch.float32),
            "affinity_mask": torch.tensor(self.affinity_mask[idx], dtype=torch.float32),
            "presentation": torch.tensor(self.presentation[idx], dtype=torch.float32),
            "presentation_mask": torch.tensor(self.presentation_mask[idx], dtype=torch.float32),
        }


class InferenceDataset(Dataset):
    """
    Dataset for prediction-only use (no labels).
    Used by the pipeline to score patient-specific candidate peptides.
    """

    def __init__(
        self,
        peptides: list[str],
        alleles: list[str],
        pseudoseq_map: dict[str, str],
        max_peptide_len: int = 11,
        pseudoseq_len: int = 34,
    ):
        assert len(peptides) == len(alleles), "peptides and alleles must align"
        N = len(peptides)

        self.pep_tokens = np.zeros((N, max_peptide_len), dtype=np.int64)
        for i, pep in enumerate(peptides):
            self.pep_tokens[i] = encode_peptide(pep, max_peptide_len)

        self.hla_tokens = np.zeros((N, pseudoseq_len), dtype=np.int64)
        for i, allele in enumerate(alleles):
            pseudo = pseudoseq_map.get(allele, "X" * pseudoseq_len)
            self.hla_tokens[i] = encode_peptide(
                pseudo.replace("X", "-"),               # X -> pad in our encoding
                pseudoseq_len,
            )

    def __len__(self) -> int:
        return len(self.pep_tokens)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "peptide_tokens": torch.from_numpy(self.pep_tokens[idx]),
            "hla_tokens": torch.from_numpy(self.hla_tokens[idx]),
        }
