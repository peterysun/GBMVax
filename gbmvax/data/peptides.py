"""
peptides.py — generate mutant peptide windows from missense mutations.

For a missense mutation at protein position P that changes WT residue X to
MT residue Y, a class I HLA can present any 8/9/10/11-mer window containing
position P. For an L-mer, the mutated residue can sit at any of L positions,
so we generate (L choices) * (4 lengths) = up to 38 peptides per mutation.

Both the WT and MT versions of each window are generated:
    * The MT peptide is what we ultimately want to predict binding for.
    * The WT peptide is needed for two downstream filters:
        - Differential agretopicity (does mutation IMPROVE binding?)
        - Self-tolerance check (the WT was present during T cell development;
          if WT and MT differ only outside the TCR contact face, the
          response will be tolerized.)

We require a protein sequence database to extract flanking context. v1
uses the UniProt human proteome FASTA already loaded — we match by gene
symbol via a lazy lookup table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
from Bio import SeqIO

from gbmvax.utils.device import get_logger
from gbmvax.utils.sequences import AMINO_ACIDS

logger = get_logger(__name__)

_VALID_AA = set(AMINO_ACIDS)


@dataclass
class PeptideWindow:
    """A single mutant peptide candidate."""
    mt_peptide: str          # Mutant 8/9/10/11-mer
    wt_peptide: str          # Same window but with WT residue restored
    length: int              # Peptide length
    mutation_offset: int     # 0-indexed position of the mutation within the peptide
    gene: str
    protein_id: str
    protein_pos: int         # 1-indexed position of mutation in the parent protein
    wt_aa: str
    mt_aa: str
    vaf: float               # Inherited from the source mutation
    patient_id: str
    flank_left: str          # 11 residues N-terminal of the peptide (for processing prediction)
    flank_right: str         # 11 residues C-terminal of the peptide


# ----------------------------------------------------------------------------
# Gene-symbol-to-sequence lookup. UniProt FASTA contains GN=<symbol> in the
# description for most entries. We build a {gene_symbol: (uniprot_id, seq)}
# table. When a gene has multiple isoforms, we keep the longest (canonical).
# ----------------------------------------------------------------------------
def build_gene_to_protein(fasta_path: str) -> dict[str, tuple[str, str]]:
    """
    Parse UniProt human_proteome.fasta into {gene_symbol: (uniprot_id, sequence)}.

    Header format example:
        >sp|P04637|P53_HUMAN Cellular tumor antigen p53 OS=Homo sapiens OX=9606 GN=TP53 PE=1 SV=4
    The GN= field is the gene symbol; we extract it via regex.
    """
    import re
    gn_pattern = re.compile(r"GN=([^\s]+)")
    out: dict[str, tuple[str, str]] = {}

    for rec in SeqIO.parse(fasta_path, "fasta"):
        m = gn_pattern.search(rec.description)
        if not m:
            continue
        gene = m.group(1)
        # UniProt accession is the second pipe field.
        parts = rec.id.split("|")
        accession = parts[1] if len(parts) >= 2 else rec.id
        seq = str(rec.seq).upper()

        # Keep the LONGEST isoform per gene — typically the canonical form.
        if gene not in out or len(seq) > len(out[gene][1]):
            out[gene] = (accession, seq)

    logger.info(f"Gene -> protein lookup built: {len(out):,} genes")
    return out


# ----------------------------------------------------------------------------
# Window generation.
# ----------------------------------------------------------------------------
def generate_windows_for_mutation(
    gene: str,
    protein_pos: int,
    wt_aa: str,
    mt_aa: str,
    gene_to_prot: dict[str, tuple[str, str]],
    vaf: float,
    patient_id: str,
    peptide_lengths: tuple[int, ...] = (8, 9, 10, 11),
    flank_length: int = 11,
) -> list[PeptideWindow]:
    """
    Return all valid (mt, wt) peptide pairs covering the mutated position.

    Skips silently and returns [] if:
        * The gene is not in the UniProt lookup (rare; mostly pseudogenes
          and non-coding loci that shouldn't have HGVSp anyway).
        * The protein sequence is too short to support the position.
        * The WT residue at protein_pos in the reference doesn't match the
          wt_aa in the MAF (indicates a stale transcript reference — we
          err on the side of skipping rather than producing a wrong peptide).
    """
    if gene not in gene_to_prot:
        return []
    protein_id, seq = gene_to_prot[gene]

    # Convert to 0-indexed.
    pos0 = protein_pos - 1
    if pos0 < 0 or pos0 >= len(seq):
        return []

    # Sanity check: the WT residue in the MAF must match what UniProt has.
    # Mismatches happen with alternative isoforms or stale references.
    # We allow mismatches at the C-terminus (sometimes off by one) by also
    # accepting positions pos0-1 and pos0+1 — but the simpler approach is
    # to just skip. We skip.
    if seq[pos0] != wt_aa:
        return []

    # Build the mutant sequence by replacing one residue. We don't mutate
    # the stored protein, just the local window.
    windows: list[PeptideWindow] = []

    for L in peptide_lengths:
        # The mutated residue can sit at any offset i in [0, L-1] within the peptide.
        # The peptide starts at protein index (pos0 - i) and ends at (pos0 - i + L).
        for i in range(L):
            start = pos0 - i
            end = start + L

            # Skip if the window runs off either end of the protein.
            if start < 0 or end > len(seq):
                continue

            wt_window = seq[start:end]

            # Quality check: peptide must contain only canonical residues
            # (no X, U, B in the parent protein).
            if not all(c in _VALID_AA for c in wt_window):
                continue

            # Construct mutant peptide.
            mt_window = wt_window[:i] + mt_aa + wt_window[i + 1:]
            if mt_aa not in _VALID_AA:
                continue

            # Flanks for processing prediction. Pad with 'X' if we hit the
            # protein termini — NetChop tolerates Xs.
            fl_start = max(0, start - flank_length)
            fl_left = "X" * (flank_length - (start - fl_start)) + seq[fl_start:start]
            fr_end = min(len(seq), end + flank_length)
            fl_right = seq[end:fr_end] + "X" * (flank_length - (fr_end - end))

            windows.append(PeptideWindow(
                mt_peptide=mt_window,
                wt_peptide=wt_window,
                length=L,
                mutation_offset=i,
                gene=gene,
                protein_id=protein_id,
                protein_pos=protein_pos,
                wt_aa=wt_aa,
                mt_aa=mt_aa,
                vaf=vaf,
                patient_id=patient_id,
                flank_left=fl_left,
                flank_right=fl_right,
            ))

    return windows


def generate_windows_for_patient(
    mutations: pd.DataFrame,
    gene_to_prot: dict[str, tuple[str, str]],
    peptide_lengths: tuple[int, ...] = (8, 9, 10, 11),
    flank_length: int = 11,
) -> list[PeptideWindow]:
    """
    Run generate_windows_for_mutation across every row in a per-patient
    mutation table. Returns the flattened list of PeptideWindow objects.
    """
    all_windows: list[PeptideWindow] = []
    for _, m in mutations.iterrows():
        all_windows.extend(generate_windows_for_mutation(
            gene=m["gene"],
            protein_pos=int(m["protein_pos"]),
            wt_aa=m["wt_aa"],
            mt_aa=m["mt_aa"],
            gene_to_prot=gene_to_prot,
            vaf=float(m["vaf"]),
            patient_id=str(m["patient_id"]),
            peptide_lengths=peptide_lengths,
            flank_length=flank_length,
        ))
    return all_windows


def windows_to_dataframe(windows: list[PeptideWindow]) -> pd.DataFrame:
    """Convert a list of PeptideWindow dataclasses to a flat DataFrame."""
    return pd.DataFrame([w.__dict__ for w in windows])
