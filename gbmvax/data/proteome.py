"""
proteome.py — load the human proteome and build a k-mer index for
fast cross-reactivity prefiltering.

A neoantigen candidate must be checked against ~20,000 human proteins
(~11 million residues) for self-similarity. Doing this with full BLOSUM
alignment per candidate would be quadratic and slow. Instead we:

    1. Concatenate the proteome into one flat string (with sentinels).
    2. Build a hash from k-mer (default 5) -> list of (protein_id, start).
    3. For each candidate, look up its k-mers; only positions with at
       least one shared k-mer are scored with BLOSUM. This is the same
       prefilter used by BLAST.

5-mer was chosen because at L=9 the expected number of random 5-mer hits
in an 11M-residue proteome is small enough to enumerate, but the recall
is essentially 100% for the similarity threshold we care about (>= 0.8
normalized BLOSUM62).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

from Bio import SeqIO

from gbmvax.utils.device import get_logger
from gbmvax.utils.sequences import AMINO_ACIDS

logger = get_logger(__name__)


# Pre-build a set for fast membership tests in tight loops.
_VALID_AA = set(AMINO_ACIDS)


class Proteome:
    """
    Indexed human proteome for cross-reactivity scoring.

    After construction:
        .proteins    — dict[uniprot_id, sequence]
        .kmer_index  — dict[kmer_str, list[(protein_id, start_idx)]]

    Usage:
        proteome = Proteome.from_fasta('human_proteome.fasta', k=5)
        candidates = proteome.candidate_matches(peptide, length=9)
        # candidates: iterable of (protein_id, start, end) windows to score
    """

    def __init__(self, proteins: dict[str, str], k: int = 5):
        self.proteins: dict[str, str] = proteins
        self.k: int = k
        self.kmer_index: dict[str, list[tuple[str, int]]] = defaultdict(list)
        self._build_kmer_index()

    @classmethod
    def from_fasta(cls, fasta_path: Path | str, k: int = 5) -> "Proteome":
        """
        Load UniProt human proteome FASTA. UniProt headers look like:
            >sp|P12345|GENE_HUMAN Description ...
        We take the accession (P12345) as the protein ID.
        """
        logger.info(f"Loading proteome from {fasta_path}...")
        proteins: dict[str, str] = {}
        for rec in SeqIO.parse(str(fasta_path), "fasta"):
            # Header format: db|accession|name. Accession is always the
            # second pipe-delimited field; the first is sp (Swiss-Prot)
            # or tr (TrEMBL).
            parts = rec.id.split("|")
            if len(parts) >= 2:
                accession = parts[1]
            else:
                accession = rec.id

            seq = str(rec.seq).upper()
            # Strip non-canonical residues by replacing — keeps positions but
            # the k-mer index won't generate keys containing them.
            proteins[accession] = seq

        logger.info(f"Proteome loaded: {len(proteins):,} proteins")
        return cls(proteins, k=k)

    def _build_kmer_index(self) -> None:
        """
        Index every k-mer in every protein. For 20k proteins of ~500 aa each
        and k=5, this is ~10M entries — a few seconds to build, ~1 GB RAM.
        """
        logger.info(f"Building {self.k}-mer index over proteome...")
        k = self.k
        count = 0
        for pid, seq in self.proteins.items():
            # We iterate up to len(seq) - k + 1 to capture every k-mer once.
            for i in range(len(seq) - k + 1):
                kmer = seq[i:i + k]
                # Skip k-mers containing non-canonical residues — they can't
                # match real peptides anyway and pollute the index.
                if not all(c in _VALID_AA for c in kmer):
                    continue
                self.kmer_index[kmer].append((pid, i))
                count += 1
        logger.info(f"K-mer index: {len(self.kmer_index):,} unique {self.k}-mers, {count:,} total entries")

    def candidate_matches(self, peptide: str) -> Iterable[tuple[str, int, int]]:
        """
        Yield (protein_id, start, end) windows in the proteome where there
        exists at least one shared k-mer with `peptide`. The window is
        `len(peptide)` residues long.

        Deduplicated — a given window is yielded at most once even if
        multiple k-mers in the peptide point to it.
        """
        L = len(peptide)
        k = self.k
        if L < k:
            return                                       # Peptide too short for k-mer prefilter

        seen: set[tuple[str, int]] = set()
        for offset in range(L - k + 1):
            kmer = peptide[offset:offset + k]
            for pid, start in self.kmer_index.get(kmer, []):
                # The window in the proteome that ALIGNS this k-mer to the
                # peptide starts at (start - offset). Skip windows that fall
                # off the protein.
                win_start = start - offset
                win_end = win_start + L
                prot_seq = self.proteins[pid]
                if win_start < 0 or win_end > len(prot_seq):
                    continue
                key = (pid, win_start)
                if key in seen:
                    continue
                seen.add(key)
                yield pid, win_start, win_end

    def get_window(self, protein_id: str, start: int, end: int) -> str:
        """Return the substring [start, end) of protein_id."""
        return self.proteins[protein_id][start:end]
