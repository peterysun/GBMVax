"""
run_lopo.py — run Keskin leave-one-patient-out fine-tuning and validation.

For each Keskin Table S5 patient, this driver:
  1. fine-tunes with that patient's peptides excluded, and
  2. validates the resulting checkpoint on that held-out patient against Hilf
     mutant/background peptides.

Use --dry-run first to print the exact commands without launching training.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from gbmvax.data.validation import load_keskin
from gbmvax.utils.config import ensure_output_dirs, load_config


def _patient_sort_key(value: str) -> tuple[int, str]:
    try:
        return (int(value), value)
    except ValueError:
        return (10**9, value)


def _auc_or_none(labels: pd.Series, scores: pd.Series) -> float | None:
    if len(labels) < 2 or labels.nunique() < 2:
        return None
    return float(roc_auc_score(labels, scores))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    ap.add_argument("--patients", nargs="*", default=None,
                    help="Optional subset of Keskin patient IDs. Defaults to all Table S5 patients.")
    ap.add_argument("--base-checkpoint", type=Path, default=None)
    ap.add_argument("--upsample", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2.0e-5)
    ap.add_argument("--max-iedb-rows", type=int, default=None,
                    help="Optional cap for smoke-test folds.")
    ap.add_argument("--exclude-hilf-peptides-file", type=Path, default=None,
                    help="Optional background exclusion list for sensitivity analysis.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print commands without running them.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if not args.dry_run:
        ensure_output_dirs(cfg)
    project_root = Path(cfg["paths"]["root"])
    patients = args.patients
    if not patients:
        patients = sorted(load_keskin(cfg)["patient_id"].unique().tolist(), key=_patient_sort_key)

    results = []
    pooled_predictions = []
    for patient in patients:
        fold = f"pt{patient}"
        checkpoint = Path(cfg["paths"]["checkpoints"]) / f"hla_binding_gbm_finetuned_{fold}.pt"
        finetune_cmd = [
            sys.executable, "scripts/finetune_gbm.py",
            "--config", str(args.config),
            "--holdout-patient", str(patient),
            "--upsample", str(args.upsample),
            "--epochs", str(args.epochs),
            "--lr", str(args.lr),
        ]
        if args.base_checkpoint is not None:
            finetune_cmd.extend(["--base-checkpoint", str(args.base_checkpoint)])
        if args.max_iedb_rows is not None:
            finetune_cmd.extend(["--max-iedb-rows", str(args.max_iedb_rows)])

        validate_cmd = [
            sys.executable, "scripts/validate.py",
            "--config", str(args.config),
            "--checkpoint", str(checkpoint),
            "--holdout-patient", str(patient),
        ]
        if args.exclude_hilf_peptides_file is not None:
            validate_cmd.extend(["--exclude-hilf-peptides-file", str(args.exclude_hilf_peptides_file)])

        print(" ".join(finetune_cmd))
        print(" ".join(validate_cmd))
        if args.dry_run:
            continue

        subprocess.run(finetune_cmd, cwd=project_root, check=True)
        subprocess.run(validate_cmd, cwd=project_root, check=True)

        metrics_path = Path(cfg["paths"]["results"]) / f"clinical_validation_holdout_pt{patient}.json"
        with open(metrics_path) as f:
            metrics = json.load(f)
        results.append(metrics)

        pred_path = Path(cfg["paths"]["results"]) / f"clinical_validation_holdout_pt{patient}.tsv"
        pred_df = pd.read_csv(pred_path, sep="\t")
        pred_df["fold_holdout_patient"] = str(patient)
        pooled_predictions.append(pred_df)

    if not args.dry_run and results:
        pooled = pd.concat(pooled_predictions, ignore_index=True)
        pooled_path = Path(cfg["paths"]["results"]) / "clinical_validation_lopo_pooled_predictions.tsv"
        pooled.to_csv(pooled_path, sep="\t", index=False)

        fold_aucs = [
            m.get("auc_composite")
            for m in results
            if m.get("auc_composite") is not None
        ]
        summary = {
            "patients": patients,
            "pooled_auc_composite": _auc_or_none(pooled["response"], pooled["composite_score"]),
            "pooled_auc_binding_only": _auc_or_none(pooled["response"], pooled["binding_score"]),
            "n_pooled_rows": int(len(pooled)),
            "n_pooled_positive": int(pooled["response"].sum()),
            "n_pooled_negative": int((pooled["response"] == 0).sum()),
            "pooled_predictions": str(pooled_path),
            "per_fold_auc_composite": fold_aucs,
            "per_fold_auc_composite_range": [
                min(fold_aucs) if fold_aucs else None,
                max(fold_aucs) if fold_aucs else None,
            ],
            "folds": results,
        }
        out_path = Path(cfg["paths"]["results"]) / "clinical_validation_lopo_summary.json"
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
