"""
hla.py — HLA allele utilities.

Three jobs:
    1. Normalize allele names (HLA-A*02:01, HLA-A02:01, A*0201, etc. all
       refer to the same molecule but appear differently across datasets).
    2. Extract the 34-residue pseudosequence used by NetMHCpan-style models.
    3. Load the full HLA protein FASTA into a dict keyed by normalized name.

The 34 pseudosequence positions are NOT arbitrary — they are the residues
within 8 Å of bound peptide in HLA crystal structures, identified by
Nielsen et al. (2007). Using these 34 lets the model generalize to alleles
it has never seen during training, because new alleles share the same
positional grammar.
"""

from __future__ import annotations

import re
from pathlib import Path

from Bio import SeqIO                          # FASTA parsing


# ----------------------------------------------------------------------------
# Canonical pseudosequence positions (NetMHCpan convention, 1-indexed
# residues in the mature HLA class I heavy chain). 34 positions total.
# Source: Nielsen 2007 / NetMHCpan documentation.
# ----------------------------------------------------------------------------
PSEUDOSEQUENCE_POSITIONS_1IDX: tuple[int, ...] = (
    7, 9, 24, 45, 59, 62, 63, 66, 67, 69, 70, 73, 74, 76, 77, 80, 81,
    84, 95, 97, 99, 114, 116, 118, 143, 147, 150, 152, 156, 158, 159,
    163, 167, 171,
)
PSEUDOSEQUENCE_LENGTH = len(PSEUDOSEQUENCE_POSITIONS_1IDX)   # 34


def normalize_allele(allele: str) -> str:
    """
    Convert any common HLA notation to canonical form 'HLA-A*02:01'.

    Handles inputs like:
        'A*02:01'           -> 'HLA-A*02:01'
        'HLA-A0201'         -> 'HLA-A*02:01'
        'A0201'             -> 'HLA-A*02:01'
        'HLA-A*02:01'       -> 'HLA-A*02:01'   (no-op)
        'hla-a*02:01:01:02' -> 'HLA-A*02:01'   (truncate to 4-digit)

    We truncate to 4-digit resolution (2-field) because expression-level
    differences below that almost never affect peptide binding.
    """
    s = allele.strip().upper().replace(" ", "").replace("_", "")

    # Drop any HLA prefix temporarily — we re-add it canonically.
    # Supplements often use variants like "HLA B*5801" or "HLA-B*58:01".
    if s.startswith("HLA-"):
        s = s[4:]
    elif s.startswith("HLA"):
        s = s[3:]

    # Pattern 1: A*02:01 or A*02:01:01:02
    m = re.match(r"^([A-CEG])\*?(\d{2}):?(\d{2,3})", s)
    if m:
        gene, group, protein = m.group(1), m.group(2), m.group(3)
        # Truncate protein-field to 2 digits (4-digit resolution overall).
        protein = protein[:2] if len(protein) >= 2 else protein.zfill(2)
        return f"HLA-{gene}*{group}:{protein}"

    raise ValueError(f"Unrecognized HLA allele: {allele!r}")


def load_hla_sequences(fasta_path: Path | str) -> dict[str, str]:
    """
    Parse the IMGT/HLA hla_prot.fasta into {normalized_allele: full_sequence}.

    The IMGT FASTA headers look like:
        >HLA:HLA00001 A*01:01:01:01 365 bp
    The second whitespace-delimited token is the allele name.
    """
    out: dict[str, str] = {}
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        # Header format: ID then allele then length. The allele is the
        # second token in .description after splitting on whitespace.
        parts = record.description.split()
        if len(parts) < 2:
            continue
        raw_allele = parts[1]
        try:
            allele = normalize_allele(raw_allele)
        except ValueError:
            # IMGT includes some non-classical and class II — skip silently.
            continue
        seq = str(record.seq)
        # If we already have this 4-digit allele, keep the first (canonical) entry.
        if allele not in out:
            out[allele] = seq
    return out


def extract_pseudosequence(full_seq: str) -> str:
    """
    Extract the 34-residue NetMHCpan pseudosequence from a full HLA sequence.

    The positions are 1-indexed positions in the MATURE protein (signal
    peptide already removed). IMGT FASTA sequences include the signal
    peptide (~24 residues for HLA-A); we have to detect and strip it.

    Heuristic: the mature HLA class I chain starts with 'GSHSMRY' for
    HLA-A, 'GSHSMRY' for HLA-B, 'CSHSMRY' for HLA-C (approximately).
    We look for the conserved 'SHSMRY' motif and use its position to set
    the offset. Falls back to the IMGT convention of 24 residues if not
    found.
    """
    # Locate the mature-chain start using the conserved early motif.
    motif_idx = full_seq.find("SHSMR")
    if motif_idx > 0:
        # 'SHSMR' starts at mature position 3 (G-S-H-S-M-R-Y...), so
        # the mature chain begins at motif_idx - 2.
        offset = motif_idx - 2
    else:
        offset = 24                              # IMGT default signal peptide length

    # Extract residues at the canonical positions. 1-indexed -> 0-indexed.
    # If the sequence is shorter than expected (truncated entry), pad with X.
    chars = []
    for pos in PSEUDOSEQUENCE_POSITIONS_1IDX:
        idx = offset + pos - 1
        if 0 <= idx < len(full_seq):
            chars.append(full_seq[idx])
        else:
            chars.append("X")                    # Will be filtered downstream
    return "".join(chars)


def build_pseudosequence_map(fasta_path: Path | str) -> dict[str, str]:
    """
    One-shot: load HLA FASTA and return {allele: 34-residue pseudosequence}.
    This is what the binding model actually consumes.
    """
    full_map = load_hla_sequences(fasta_path)
    return {allele: extract_pseudosequence(seq) for allele, seq in full_map.items()}
