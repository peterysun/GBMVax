"""
finetune_gbm.py — GBM-specific fine-tuning with Keskin LOPO leakage guard.

This script fine-tunes an existing IEDB-trained HLA binding transformer on
Keskin Table S5 presentation-positive peptides. For a defensible clinical
estimate, pass --holdout-patient so that patient's peptides are excluded
from fine-tuning and used only by scripts/validate.py.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pandas as pd
import torch

from gbmvax.data.iedb import load_iedb, stratified_train_val_test_split
from gbmvax.data.validation import external_validation_peptides, load_keskin_as_iedb
from gbmvax.models.dataset import IEDBDataset
from gbmvax.models.trainer import train_hla_binding
from gbmvax.utils.config import ensure_output_dirs, load_config
from gbmvax.utils.device import get_logger, seed_everything, setup_logging
from gbmvax.utils.hla import build_pseudosequence_map


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    ap.add_argument("--base-checkpoint", type=Path, default=None,
                    help="IEDB-pretrained checkpoint to initialize from.")
    ap.add_argument("--holdout-patient", type=str, default=None,
                    help="Keskin patient ID to exclude from fine-tuning for LOPO validation.")
    ap.add_argument("--upsample", type=int, default=10,
                    help="Repeat non-held-out Keskin rows this many times in fine-tuning data.")
    ap.add_argument("--epochs", type=int, default=8,
                    help="Fine-tuning epochs; overrides hla_model.num_epochs.")
    ap.add_argument("--lr", type=float, default=2.0e-5,
                    help="Fine-tuning learning rate; overrides hla_model.lr.")
    ap.add_argument("--max-iedb-rows", type=int, default=None,
                    help="Optional cap for smoke tests.")
    ap.add_argument("--include-validation-peptides-in-iedb", action="store_true",
                    help="Disable the external-validation peptide exclusion guard. Do not use for paper runs.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_output_dirs(cfg)
    setup_logging(level="INFO", log_file=Path(cfg["paths"]["logs"]) / "finetune_gbm.log")
    logger = get_logger("finetune_gbm")
    seed_everything(cfg["hardware"]["seed"])

    base_ckpt = args.base_checkpoint or (Path(cfg["paths"]["checkpoints"]) / "hla_binding_best.pt")
    ft_cfg = copy.deepcopy(cfg)
    if base_ckpt.exists():
        ckpt = torch.load(base_ckpt, map_location="cpu")
        ckpt_hla = ckpt.get("config", {}).get("hla_model")
        if isinstance(ckpt_hla, dict):
            ft_cfg["hla_model"].update(ckpt_hla)
            logger.info(f"Using hla_model architecture from {base_ckpt}")
    ft_cfg["hla_model"]["num_epochs"] = args.epochs
    ft_cfg["hla_model"]["lr"] = args.lr

    excluded = None if args.include_validation_peptides_in_iedb else external_validation_peptides(cfg)
    iedb = load_iedb(
        cfg["paths"]["iedb"]["mhc_ligand"],
        peptide_lengths=tuple(cfg["peptide"]["lengths"]),
        max_rows=args.max_iedb_rows,
        exclude_peptides=excluded,
    )
    keskin_df = load_keskin_as_iedb(
        cfg,
        holdout_patient=args.holdout_patient,
        upsample=args.upsample,
    )
    logger.info(
        f"Fine-tuning rows: IEDB={len(iedb):,} Keskin_upweighted={len(keskin_df):,} "
        f"holdout_patient={args.holdout_patient or 'none'}"
    )

    finetune_df = pd.concat([iedb, keskin_df], ignore_index=True)
    train_df, val_df, test_df = stratified_train_val_test_split(
        finetune_df,
        val_frac=cfg["validation"]["val_fraction"],
        test_frac=cfg["validation"]["test_fraction"],
        seed=cfg["hardware"]["seed"],
    )

    fold = f"pt{args.holdout_patient}" if args.holdout_patient else "all_keskin"
    results_dir = Path(cfg["paths"]["results"])
    test_path = results_dir / f"iedb_gbm_finetune_test_split_{fold}.parquet"
    test_df.to_parquet(test_path)
    logger.info(f"Fine-tune test split saved to {test_path}")

    pseudoseq_map = build_pseudosequence_map(cfg["paths"]["hla"]["sequences"])
    hp = ft_cfg["hla_model"]
    train_ds = IEDBDataset(train_df, pseudoseq_map, hp["max_peptide_len"], hp["hla_pseudosequence_length"])
    val_ds = IEDBDataset(val_df, pseudoseq_map, hp["max_peptide_len"], hp["hla_pseudosequence_length"])

    ckpt_name = f"hla_binding_gbm_finetuned_{fold}.pt"
    model, history = train_hla_binding(
        train_ds=train_ds,
        val_ds=val_ds,
        cfg=ft_cfg,
        checkpoint_dir=Path(cfg["paths"]["checkpoints"]),
        log_dir=Path(cfg["paths"]["logs"]),
        init_checkpoint=base_ckpt,
        checkpoint_name=ckpt_name,
    )

    hist_path = results_dir / f"finetune_history_{fold}.json"
    with open(hist_path, "w") as f:
        json.dump({
            "holdout_patient": args.holdout_patient,
            "upsample": args.upsample,
            "base_checkpoint": str(base_ckpt),
            "checkpoint": str(Path(cfg["paths"]["checkpoints"]) / ckpt_name),
            "n_iedb_rows": int(len(iedb)),
            "n_keskin_rows_after_upsample": int(len(keskin_df)),
            "history": history,
        }, f, indent=2)
    logger.info(f"Fine-tuning history saved to {hist_path}")


if __name__ == "__main__":
    main()
