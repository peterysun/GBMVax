"""
evaluate.py — secondary IEDB holdout evaluation + NetMHCpan baseline.

Two evaluations:
    (1) Spearman rho on the IEDB test split's affinity rows. Verifies the
        binding model learned the underlying biology, independent of GBM
        clinical outcomes.
    (2) Head-to-head vs NetMHCpan 4.2c on the same test peptides. If
        GBMVax beats NetMHCpan on GBM-relevant alleles, that's the second
        paper headline (after Keskin/Hilf primary validation).

The IEDB test split is the file produced and saved by train.py
(results/iedb_test_split.parquet) — we reload it here so the comparison
is exact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from gbmvax.models.dataset import IEDBDataset
from gbmvax.models.hla_binding import HLABindingTransformer
from gbmvax.models.netmhcpan import run_netmhcpan_batched
from gbmvax.utils.config import ensure_output_dirs, load_config
from gbmvax.utils.device import get_device, get_logger, seed_everything, setup_logging
from gbmvax.utils.hla import build_pseudosequence_map


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--skip-netmhcpan", action="store_true",
                    help="Skip NetMHCpan baseline (useful when binary not installed).")
    ap.add_argument("--max-netmhcpan-rows", type=int, default=10_000,
                    help="Cap NetMHCpan runs to this many rows (it is slow).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_output_dirs(cfg)
    setup_logging(level="INFO", log_file=Path(cfg["paths"]["logs"]) / "evaluate.log")
    logger = get_logger("evaluate")
    seed_everything(cfg["hardware"]["seed"])

    # --- Reload test split -------------------------------------------
    test_path = Path(cfg["paths"]["results"]) / "iedb_test_split.parquet"
    if not test_path.exists():
        raise FileNotFoundError(f"Test split not found at {test_path}. Run scripts/train.py first.")
    test_df = pd.read_parquet(test_path)
    logger.info(f"Loaded test split: {len(test_df):,} rows")

    # --- Load model ---------------------------------------------------
    ckpt_path = args.checkpoint or (Path(cfg["paths"]["checkpoints"]) / "hla_binding_best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    hp = cfg["hla_model"]
    model = HLABindingTransformer(
        embed_dim=hp["embed_dim"], num_heads=hp["num_heads"], num_layers=hp["num_layers"],
        ff_hidden=hp["ff_hidden"], dropout=hp["dropout"],
        max_peptide_len=hp["max_peptide_len"],
        pseudoseq_len=hp["hla_pseudosequence_length"],
    )
    model.load_state_dict(ckpt["model_state"])
    device = get_device(cfg["hardware"]["device"])
    model.to(device).eval()

    # --- Score test rows with GBMVax ---------------------------------
    pseudoseq_map = build_pseudosequence_map(cfg["paths"]["hla"]["sequences"])
    ds = IEDBDataset(test_df, pseudoseq_map, hp["max_peptide_len"], hp["hla_pseudosequence_length"])
    loader = DataLoader(ds, batch_size=hp["batch_size"], shuffle=False, num_workers=0)

    aff_pred, pres_pred = [], []
    with torch.no_grad():
        for batch in loader:
            pep = batch["peptide_tokens"].to(device)
            hla = batch["hla_tokens"].to(device)
            la, pr = model(pep, hla)
            aff_pred.append(la.cpu().numpy())
            pres_pred.append(torch.sigmoid(pr).cpu().numpy())
    aff_pred = np.concatenate(aff_pred)
    pres_pred = np.concatenate(pres_pred)

    # Align with the dataset (the dataset filtered to alleles in pseudoseq_map)
    test_aligned = test_df[test_df["allele"].isin(pseudoseq_map)].reset_index(drop=True)
    test_aligned["gbmvax_log_ic50"] = aff_pred
    test_aligned["gbmvax_pres"] = pres_pred

    # --- Affinity Spearman ------------------------------------------
    aff_rows = test_aligned[test_aligned["is_affinity"]]
    rho_gbmvax = float("nan")
    if len(aff_rows) >= 10:
        rho, _ = spearmanr(aff_rows["gbmvax_log_ic50"], aff_rows["log_affinity"])
        rho_gbmvax = float(rho)
    logger.info(f"GBMVax Spearman on IEDB affinity holdout: {rho_gbmvax:.4f} (n={len(aff_rows)})")

    # --- Presentation AUC -------------------------------------------
    pres_rows = test_aligned[test_aligned["is_presentation"] & (test_aligned["presented"] >= 0)]
    auc_gbmvax = float("nan")
    if len(pres_rows) >= 10 and pres_rows["presented"].nunique() > 1:
        auc_gbmvax = float(roc_auc_score(pres_rows["presented"], pres_rows["gbmvax_pres"]))
    logger.info(f"GBMVax presentation AUC on IEDB holdout: {auc_gbmvax:.4f} (n={len(pres_rows)})")

    metrics: dict = {
        "gbmvax_affinity_spearman": rho_gbmvax,
        "gbmvax_presentation_auc": auc_gbmvax,
        "n_affinity_rows": int(len(aff_rows)),
        "n_presentation_rows": int(len(pres_rows)),
    }

    # --- NetMHCpan baseline -----------------------------------------
    if not args.skip_netmhcpan:
        logger.info("Running NetMHCpan baseline on the same test rows (this is slow)...")
        net_path = cfg["paths"]["netmhcpan"]

        sample = aff_rows.head(args.max_netmhcpan_rows)
        pairs = list(zip(sample["peptide"], sample["allele"]))
        try:
            nm_df = run_netmhcpan_batched(pairs, binary_path=net_path)
            merged = sample.merge(
                nm_df.rename(columns={"log_ic50": "netmhcpan_log_ic50"}),
                on=["peptide", "allele"], how="left",
            ).dropna(subset=["netmhcpan_log_ic50"])
            if len(merged) >= 10:
                rho_nm, _ = spearmanr(merged["netmhcpan_log_ic50"], merged["log_affinity"])
                metrics["netmhcpan_affinity_spearman"] = float(rho_nm)
                metrics["n_netmhcpan_rows"] = int(len(merged))
                # Same rows, same labels — apples-to-apples GBMVax vs NetMHCpan.
                rho_gb_same, _ = spearmanr(merged["gbmvax_log_ic50"], merged["log_affinity"])
                metrics["gbmvax_spearman_on_netmhcpan_rows"] = float(rho_gb_same)
                logger.info(
                    f"Head-to-head (n={len(merged)}): "
                    f"GBMVax rho={rho_gb_same:.4f}, NetMHCpan rho={rho_nm:.4f}"
                )
        except Exception as e:
            logger.warning(f"NetMHCpan baseline failed: {e}")

    out_path = Path(cfg["paths"]["results"]) / "iedb_evaluation.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics written to {out_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
