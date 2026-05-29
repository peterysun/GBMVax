"""
predict.py — run the full GBMVax pipeline on a single patient.

Usage:
    python scripts/predict.py \
        --config configs/config.yaml \
        --mutations path/to/patient_mutations.tsv \
        --hla HLA-A*02:01 HLA-A*03:01 HLA-B*07:02 HLA-B*44:02 HLA-C*05:01 HLA-C*07:02 \
        --output results/patient_X_neoantigens.tsv \
        --top-n 20

Input mutation TSV must have columns:
    patient_id, gene, protein_pos, wt_aa, mt_aa, vaf
(matching the schema produced by data/mutations.py)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from gbmvax.pipeline.orchestrator import GBMVaxPipeline
from gbmvax.utils.config import ensure_output_dirs, load_config
from gbmvax.utils.device import get_logger, seed_everything, setup_logging


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    ap.add_argument("--mutations", type=Path, required=True,
                    help="TSV with columns: patient_id, gene, protein_pos, wt_aa, mt_aa, vaf")
    ap.add_argument("--hla", nargs="+", required=True,
                    help="Patient HLA class I alleles, e.g. HLA-A*02:01 HLA-B*07:02")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="Override checkpoint path. Default: checkpoints/hla_binding_best.pt")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--top-n", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_output_dirs(cfg)
    setup_logging(level="INFO", log_file=Path(cfg["paths"]["logs"]) / "predict.log")
    logger = get_logger("predict")
    seed_everything(cfg["hardware"]["seed"])

    # --- Load mutations ----------------------------------------------
    mutations = pd.read_csv(args.mutations, sep="\t")
    required = {"patient_id", "gene", "protein_pos", "wt_aa", "mt_aa", "vaf"}
    missing = required - set(mutations.columns)
    if missing:
        raise ValueError(f"Mutation file is missing columns: {missing}")
    logger.info(f"Loaded {len(mutations)} mutations for {mutations['patient_id'].nunique()} patient(s)")

    # --- Build pipeline ----------------------------------------------
    pipe = GBMVaxPipeline.from_config(cfg)
    ckpt = args.checkpoint or (Path(cfg["paths"]["checkpoints"]) / "hla_binding_best.pt")
    pipe.load_model(ckpt)

    # --- Predict ------------------------------------------------------
    ranked = pipe.predict(mutations, hla_alleles=args.hla, top_n=args.top_n)

    # --- Write --------------------------------------------------------
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(args.output, sep="\t", index=False)
    logger.info(f"Wrote {len(ranked)} ranked candidates to {args.output}")

    # Print top 5 to console.
    print("\nTop 5 candidates:")
    cols = ["mt_peptide", "allele", "gene", "pred_ic50_nM", "composite_score", "vaf"]
    print(ranked[cols].head().to_string(index=False))


if __name__ == "__main__":
    main()
