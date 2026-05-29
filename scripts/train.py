"""
train.py — train the GBMVax multi-task HLA binding transformer on IEDB.

Usage:
    python scripts/train.py --config configs/config.yaml
    python scripts/train.py --config configs/config.yaml --debug   # small subset for fast iteration

The training run produces:
    checkpoints/hla_binding_best.pt    — best-val checkpoint (auto-saved)
    logs/train.log                     — full text log
    results/train_history.json         — per-epoch metrics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gbmvax.data.iedb import load_iedb, stratified_train_val_test_split
from gbmvax.models.dataset import IEDBDataset
from gbmvax.models.trainer import train_hla_binding
from gbmvax.utils.config import ensure_output_dirs, load_config
from gbmvax.utils.device import get_logger, seed_everything, setup_logging
from gbmvax.utils.hla import build_pseudosequence_map


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    ap.add_argument("--debug", action="store_true",
                    help="Load only 100k IEDB rows for fast iteration.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_output_dirs(cfg)

    log_path = Path(cfg["paths"]["logs"]) / "train.log"
    setup_logging(level="INFO", log_file=log_path)
    logger = get_logger("train")
    seed_everything(cfg["hardware"]["seed"])

    logger.info("=" * 60)
    logger.info("GBMVax — HLA binding model training")
    logger.info("=" * 60)

    # --- Load IEDB ----------------------------------------------------
    iedb = load_iedb(
        cfg["paths"]["iedb"]["mhc_ligand"],
        peptide_lengths=tuple(cfg["peptide"]["lengths"]),
        max_rows=100_000 if args.debug else None,
    )

    # --- Load HLA pseudosequences ------------------------------------
    pseudoseq_map = build_pseudosequence_map(cfg["paths"]["hla"]["sequences"])
    logger.info(f"HLA pseudosequences loaded: {len(pseudoseq_map)} alleles")

    # --- Split --------------------------------------------------------
    train_df, val_df, test_df = stratified_train_val_test_split(
        iedb,
        val_frac=cfg["validation"]["val_fraction"],
        test_frac=cfg["validation"]["test_fraction"],
        seed=cfg["hardware"]["seed"],
    )

    # Persist the test split so evaluate.py reuses it exactly.
    test_path = Path(cfg["paths"]["results"]) / "iedb_test_split.parquet"
    test_df.to_parquet(test_path)
    logger.info(f"Test split saved to {test_path}")

    # --- Build datasets ----------------------------------------------
    max_pep = cfg["hla_model"]["max_peptide_len"]
    pseudo_len = cfg["hla_model"]["hla_pseudosequence_length"]
    train_ds = IEDBDataset(train_df, pseudoseq_map, max_pep, pseudo_len)
    val_ds = IEDBDataset(val_df, pseudoseq_map, max_pep, pseudo_len)
    logger.info(f"Datasets: train={len(train_ds):,} val={len(val_ds):,}")

    # --- Train --------------------------------------------------------
    model, history = train_hla_binding(
        train_ds=train_ds,
        val_ds=val_ds,
        cfg=cfg,
        checkpoint_dir=Path(cfg["paths"]["checkpoints"]),
        log_dir=Path(cfg["paths"]["logs"]),
    )

    # --- Persist history ---------------------------------------------
    hist_path = Path(cfg["paths"]["results"]) / "train_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history saved to {hist_path}")


if __name__ == "__main__":
    main()
