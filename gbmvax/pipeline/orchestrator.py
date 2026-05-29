"""
orchestrator.py — end-to-end GBMVax neoantigen prediction pipeline.

Wires together every module:

    mutations + HLA type
        |
        v
    peptide window generation
        |
        v
    HLA binding prediction (trained transformer)
        |
        v
    antigen processing scoring (NetChop + TAP + reranker)
        |
        v
    cross-reactivity filtering (BLOSUM62 vs human proteome)
        |
        v
    clonal weighting (VAF -> clonal_score)
        |
        v
    immunogenicity heuristic (Calis 2013)
        |
        v
    composite scoring + ranking
        |
        v
    top N ranked neoantigen candidates

The class loads all heavy resources (proteome index, model weights,
gene-to-protein map) once at construction. Then predict() can be called
repeatedly across patients without reloading.

Typical usage:
    pipe = GBMVaxPipeline.from_config(cfg)
    pipe.load_model('checkpoints/hla_binding_best.pt')
    ranked = pipe.predict(mutations_df, hla_alleles=['HLA-A*02:01', 'HLA-B*07:02'])
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from gbmvax.data.peptides import (
    PeptideWindow,
    build_gene_to_protein,
    generate_windows_for_patient,
    windows_to_dataframe,
)
from gbmvax.data.proteome import Proteome
from gbmvax.models.dataset import InferenceDataset
from gbmvax.models.hla_binding import HLABindingTransformer
from gbmvax.models.processing import (
    combined_processing_score,
    netchop_cterm_score,
    tap_score,
)
from gbmvax.pipeline.clonal import score_mutations_clonal
from gbmvax.pipeline.composite import (
    CompositeWeights,
    binding_score_from_log_affinity,
    immunogenicity_score,
    rank_candidates,
)
from gbmvax.pipeline.cross_reactivity import cross_reactivity_penalty
from gbmvax.utils.device import get_device, get_logger
from gbmvax.utils.hla import build_pseudosequence_map, normalize_allele

logger = get_logger(__name__)


@dataclass
class PipelineResources:
    """Heavy objects loaded once at startup."""
    pseudoseq_map: dict[str, str]
    proteome: Proteome
    gene_to_protein: dict[str, tuple[str, str]]


class GBMVaxPipeline:
    """
    Orchestrates the full GBMVax pipeline. Stateful: load once, predict many.
    """

    def __init__(
        self,
        cfg: dict,
        resources: PipelineResources,
        model: Optional[HLABindingTransformer] = None,
        device: Optional[torch.device] = None,
    ):
        self.cfg = cfg
        self.resources = resources
        self.model = model
        self.device = device or get_device(cfg["hardware"]["device"])
        self.weights = CompositeWeights.from_config(cfg)

    # ------------------------------------------------------------------
    # Construction.
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: dict) -> "GBMVaxPipeline":
        """
        Load all heavy resources from disk paths in cfg. Does NOT load the
        binding model — call `load_model()` separately. This split lets
        tests construct the pipeline without trained weights.
        """
        logger.info("Building pipeline resources (this takes ~1 minute)...")
        pseudoseq_map = build_pseudosequence_map(cfg["paths"]["hla"]["sequences"])
        logger.info(f"  HLA pseudosequences: {len(pseudoseq_map)} alleles")

        proteome = Proteome.from_fasta(
            cfg["paths"]["proteome"]["human"],
            k=cfg["cross_reactivity"]["kmer_prefilter_k"],
        )

        gene_to_protein = build_gene_to_protein(cfg["paths"]["proteome"]["human"])

        resources = PipelineResources(
            pseudoseq_map=pseudoseq_map,
            proteome=proteome,
            gene_to_protein=gene_to_protein,
        )
        return cls(cfg=cfg, resources=resources)

    def load_model(self, checkpoint_path: Path | str) -> None:
        """Load trained binding-model weights from a .pt checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        hp = self.cfg["hla_model"]
        model = HLABindingTransformer(
            embed_dim=hp["embed_dim"],
            num_heads=hp["num_heads"],
            num_layers=hp["num_layers"],
            ff_hidden=hp["ff_hidden"],
            dropout=hp["dropout"],
            max_peptide_len=hp["max_peptide_len"],
            pseudoseq_len=hp["hla_pseudosequence_length"],
        )
        model.load_state_dict(ckpt["model_state"])
        model.to(self.device)
        model.eval()
        self.model = model
        logger.info(f"Loaded model from {checkpoint_path}")

    # ------------------------------------------------------------------
    # Prediction.
    # ------------------------------------------------------------------
    def predict(
        self,
        mutations: pd.DataFrame,
        hla_alleles: list[str],
        top_n: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Full pipeline: mutations + HLA alleles -> ranked neoantigen table.

        Args:
            mutations: DataFrame with columns
                patient_id, gene, protein_pos, wt_aa, mt_aa, vaf
            hla_alleles: list of patient HLA class I alleles.
            top_n: keep only this many top candidates (None = config default).

        Returns:
            DataFrame sorted by composite_score descending, with all
            intermediate scores preserved for downstream analysis.
        """
        if self.model is None:
            raise RuntimeError("No binding model loaded. Call load_model() first.")

        top_n = top_n if top_n is not None else self.cfg["composite"]["top_n"]
        cfg = self.cfg

        # --- Step 1: peptide window generation -----------------------------
        logger.info("Step 1/7: generating peptide windows...")
        windows = generate_windows_for_patient(
            mutations=mutations,
            gene_to_prot=self.resources.gene_to_protein,
            peptide_lengths=tuple(cfg["peptide"]["lengths"]),
            flank_length=cfg["peptide"]["flank_length"],
        )
        if not windows:
            logger.warning("No peptide windows generated — check mutation data and gene symbols.")
            return pd.DataFrame()
        cand = windows_to_dataframe(windows)
        logger.info(f"  generated {len(cand):,} candidate (peptide, length, position) entries")

        # --- Step 2: cross with HLA alleles --------------------------------
        # Each window is evaluated against each of the patient's HLAs.
        # The pipeline assigns the BEST (lowest-IC50) HLA per peptide later.
        logger.info("Step 2/7: pairing peptides with HLA alleles...")
        alleles_norm = []
        for a in hla_alleles:
            try:
                alleles_norm.append(normalize_allele(a))
            except ValueError:
                logger.warning(f"  unrecognized HLA allele {a!r} — skipping")
        if not alleles_norm:
            raise ValueError("No valid HLA alleles provided")

        # Keep only alleles for which we have a pseudosequence.
        alleles_norm = [a for a in alleles_norm if a in self.resources.pseudoseq_map]
        if not alleles_norm:
            raise ValueError("None of the provided HLA alleles are in the HLA database")

        # Cartesian product candidate-peptide x patient-allele.
        cand = cand.merge(pd.DataFrame({"allele": alleles_norm}), how="cross")
        logger.info(f"  {len(cand):,} (peptide, allele) pairs to score")

        # --- Step 3: HLA binding prediction --------------------------------
        logger.info("Step 3/7: predicting HLA binding (transformer inference)...")
        log_ic50 = self._predict_binding(
            peptides=cand["mt_peptide"].tolist(),
            alleles=cand["allele"].tolist(),
        )
        cand["pred_log_ic50"] = log_ic50
        cand["pred_ic50_nM"] = np.power(10.0, log_ic50)
        cand["binding_score"] = [binding_score_from_log_affinity(x) for x in log_ic50]

        # Pre-filter to plausible binders before the expensive cross-reactivity
        # step. Weak threshold (500 nM) -> log10(500) = 2.7. We keep anything
        # under that, which is the standard NetMHCpan cutoff for "binder".
        cand = cand[cand["pred_log_ic50"] <= np.log10(cfg["hla_model"]["weak_binder_threshold"])]
        logger.info(f"  after binding filter (<= {cfg['hla_model']['weak_binder_threshold']} nM): {len(cand):,}")

        if len(cand) == 0:
            logger.warning("No binders found — returning empty table")
            return cand

        # --- Step 4: antigen processing ------------------------------------
        logger.info("Step 4/7: scoring antigen processing (NetChop + TAP)...")
        nc = [netchop_cterm_score(p, l, r) for p, l, r in zip(
            cand["mt_peptide"], cand["flank_left"], cand["flank_right"]
        )]
        tp = [tap_score(p) for p in cand["mt_peptide"]]
        cand["netchop_score"] = nc
        cand["tap_score"] = tp
        cand["processing_score"] = [
            combined_processing_score(n, t, tap_weight=cfg["processing"]["tap_affinity_weight"])
            for n, t in zip(nc, tp)
        ]

        # --- Step 5: cross-reactivity --------------------------------------
        logger.info("Step 5/7: cross-reactivity filtering against human proteome...")
        xr = [
            cross_reactivity_penalty(
                p, self.resources.proteome,
                similarity_threshold=cfg["cross_reactivity"]["similarity_threshold"],
                anchor_weight=cfg["cross_reactivity"]["anchor_weight"],
            )
            for p in cand["mt_peptide"]
        ]
        cand["cross_reactivity_penalty"] = xr

        # --- Step 6: clonal + immunogenicity -------------------------------
        logger.info("Step 6/7: scoring clonal + immunogenicity...")
        cand["clonal_score"] = [
            self._vaf_to_clonal(v) for v in cand["vaf"]
        ]
        cand["immunogenicity_score"] = [immunogenicity_score(p) for p in cand["mt_peptide"]]

        # --- Step 7: composite + per-mutation best-HLA selection -----------
        logger.info("Step 7/7: composite scoring + ranking...")
        ranked = rank_candidates(cand, weights=self.weights, top_n=None)

        # For each unique (patient, peptide), keep only the best HLA match.
        # This is the row the patient's immune system would actually use.
        ranked = ranked.sort_values("composite_score", ascending=False)
        ranked = ranked.drop_duplicates(subset=["patient_id", "mt_peptide"], keep="first")

        # Final top-N.
        ranked = ranked.head(top_n).reset_index(drop=True)
        logger.info(f"Done. Returning top {len(ranked)} ranked candidates.")
        return ranked

    # ------------------------------------------------------------------
    # Internals.
    # ------------------------------------------------------------------
    def _vaf_to_clonal(self, vaf: float) -> float:
        from gbmvax.pipeline.clonal import vaf_to_clonal_score
        return vaf_to_clonal_score(vaf, self.cfg["clonal"]["clonal_vaf_threshold"])

    @torch.no_grad()
    def _predict_binding(self, peptides: list[str], alleles: list[str]) -> np.ndarray:
        """
        Run the transformer on (peptide, allele) pairs in batches.

        Returns predicted log10(IC50 nM) as a NumPy array of shape [N].
        """
        ds = InferenceDataset(
            peptides=peptides,
            alleles=alleles,
            pseudoseq_map=self.resources.pseudoseq_map,
            max_peptide_len=self.cfg["hla_model"]["max_peptide_len"],
            pseudoseq_len=self.cfg["hla_model"]["hla_pseudosequence_length"],
        )
        loader = DataLoader(
            ds,
            batch_size=self.cfg["hla_model"]["batch_size"],
            shuffle=False,
            num_workers=0,                    # 0 = single-threaded; safe on MPS
        )

        outs = []
        for batch in loader:
            pep = batch["peptide_tokens"].to(self.device)
            hla = batch["hla_tokens"].to(self.device)
            log_aff, _ = self.model(pep, hla)
            outs.append(log_aff.cpu().numpy())

        return np.concatenate(outs) if outs else np.array([])
