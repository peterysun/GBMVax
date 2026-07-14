"""
trainer.py — training loop for the multi-task HLA binding transformer.

Uses the `accelerate` library so the same code runs on:
    * Mac MPS  (dev box)
    * CUDA     (Colab / lab GPU)
    * CPU      (CI)

Accelerate handles device placement, mixed precision (where supported),
and gradient accumulation. We do NOT use Distributed Data Parallel here —
single-GPU training is fast enough for the model size we're working with.

Key training tricks:
    * Warmup + cosine LR schedule (standard for transformers).
    * Gradient clipping (transformers can explode on outliers like 50,000 nM measurements).
    * Early stopping on val composite metric (Spearman on aff + AUC on pres).
    * Checkpoint saving on every val improvement, not just at the end.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from accelerate import Accelerator
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from gbmvax.models.hla_binding import HLABindingTransformer, multitask_loss
from gbmvax.utils.device import get_logger

logger = get_logger(__name__)


@dataclass
class TrainState:
    """Snapshot used for checkpointing."""
    epoch: int
    global_step: int
    best_val_metric: float
    best_epoch: int


def cosine_with_warmup(step: int, warmup: int, total: int) -> float:
    """LR multiplier: linear warmup -> cosine decay to 0."""
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def train_hla_binding(
    train_ds,
    val_ds,
    cfg: dict,
    checkpoint_dir: Path,
    log_dir: Optional[Path] = None,
    init_checkpoint: Optional[Path] = None,
    checkpoint_name: str = "hla_binding_best.pt",
) -> tuple[HLABindingTransformer, dict]:
    """
    Train the multi-task HLA binding model.

    Returns:
        model     — best-checkpoint model loaded back in
        history   — dict of per-epoch metrics
    """
    hp = cfg["hla_model"]
    train_dl = DataLoader(
        train_ds,
        batch_size=hp["batch_size"],
        shuffle=True,
        num_workers=cfg["hardware"]["num_workers"],
        pin_memory=cfg["hardware"]["pin_memory"],
        drop_last=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=hp["batch_size"],
        shuffle=False,
        num_workers=cfg["hardware"]["num_workers"],
        pin_memory=cfg["hardware"]["pin_memory"],
    )

    # accelerate selects device automatically; we don't need to specify.
    # mixed_precision='no' on MPS (not yet supported), 'fp16' on CUDA.
    accelerator = Accelerator(
        mixed_precision="fp16" if torch.cuda.is_available() else "no",
    )

    model = HLABindingTransformer(
        embed_dim=hp["embed_dim"],
        num_heads=hp["num_heads"],
        num_layers=hp["num_layers"],
        ff_hidden=hp["ff_hidden"],
        dropout=hp["dropout"],
        max_peptide_len=hp["max_peptide_len"],
        pseudoseq_len=hp["hla_pseudosequence_length"],
    )
    if init_checkpoint is not None:
        sd = torch.load(init_checkpoint, map_location="cpu")
        state_dict = sd.get("model_state", sd)
        model.load_state_dict(state_dict)
        logger.info(f"Initialized model from {init_checkpoint}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hp["lr"],
        weight_decay=hp["weight_decay"],
    )

    model, optimizer, train_dl, val_dl = accelerator.prepare(model, optimizer, train_dl, val_dl)

    total_steps = hp["num_epochs"] * len(train_dl)
    warmup_steps = min(hp["warmup_steps"], total_steps // 10)

    state = TrainState(epoch=0, global_step=0, best_val_metric=-float("inf"), best_epoch=-1)
    history: dict[str, list[float]] = {
        "train_loss": [], "val_aff_spearman": [], "val_pres_auc": [], "val_combined": [],
    }
    patience_counter = 0
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = checkpoint_dir / checkpoint_name

    for epoch in range(hp["num_epochs"]):
        # -------- TRAIN --------
        model.train()
        epoch_losses = []
        progress = tqdm(train_dl, desc=f"Epoch {epoch+1}/{hp['num_epochs']}", disable=not accelerator.is_local_main_process)
        for batch in progress:
            # Manual LR schedule.
            lr_mult = cosine_with_warmup(state.global_step, warmup_steps, total_steps)
            for g in optimizer.param_groups:
                g["lr"] = hp["lr"] * lr_mult

            optimizer.zero_grad(set_to_none=True)
            log_aff, pres = model(batch["peptide_tokens"], batch["hla_tokens"])
            loss, parts = multitask_loss(
                log_aff_pred=log_aff,
                log_aff_true=batch["log_affinity"],
                aff_mask=batch["affinity_mask"],
                pres_pred=pres,
                pres_true=batch["presentation"],
                pres_mask=batch["presentation_mask"],
                binding_weight=hp["binding_loss_weight"],
                presentation_weight=hp["presentation_loss_weight"],
            )
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), hp["gradient_clip"])
            optimizer.step()

            epoch_losses.append(parts["loss_total"])
            state.global_step += 1
            progress.set_postfix(loss=f"{parts['loss_total']:.4f}", aff=f"{parts['loss_affinity']:.4f}", pres=f"{parts['loss_presentation']:.4f}")

        train_loss = float(np.mean(epoch_losses))
        history["train_loss"].append(train_loss)

        # -------- VALIDATE --------
        val_metrics = evaluate(model, val_dl, accelerator)
        history["val_aff_spearman"].append(val_metrics["aff_spearman"])
        history["val_pres_auc"].append(val_metrics["pres_auc"])
        # Combined metric: average of the two head-level metrics. We use
        # this for early stopping because either alone can be misleading
        # (e.g. presentation head can hit AUC 0.99 on imbalanced data
        # while affinity head regresses).
        combined = 0.5 * val_metrics["aff_spearman"] + 0.5 * val_metrics["pres_auc"]
        history["val_combined"].append(combined)

        logger.info(
            f"Epoch {epoch+1}: train_loss={train_loss:.4f}  "
            f"val_aff_rho={val_metrics['aff_spearman']:.4f}  "
            f"val_pres_auc={val_metrics['pres_auc']:.4f}  "
            f"combined={combined:.4f}"
        )

        # -------- CHECKPOINT + EARLY STOP --------
        if combined > state.best_val_metric:
            state.best_val_metric = combined
            state.best_epoch = epoch
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped = accelerator.unwrap_model(model)
                torch.save({
                    "model_state": unwrapped.state_dict(),
                    "epoch": epoch,
                    "val_metric": combined,
                    "config": cfg,
                }, best_ckpt_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= hp["early_stopping_patience"]:
                logger.info(f"Early stopping at epoch {epoch+1} (no improvement for {patience_counter} epochs)")
                break

        state.epoch = epoch + 1

    # Reload the best checkpoint before returning.
    if best_ckpt_path.exists():
        sd = torch.load(best_ckpt_path, map_location="cpu")
        accelerator.unwrap_model(model).load_state_dict(sd["model_state"])
        logger.info(f"Loaded best checkpoint from epoch {sd['epoch']+1} (val={sd['val_metric']:.4f})")

    return accelerator.unwrap_model(model), history


@torch.no_grad()
def evaluate(model, dataloader, accelerator) -> dict[str, float]:
    """Compute Spearman on affinity head + AUC on presentation head."""
    model.eval()
    all_aff_pred, all_aff_true, all_aff_mask = [], [], []
    all_pres_pred, all_pres_true, all_pres_mask = [], [], []

    for batch in dataloader:
        log_aff, pres = model(batch["peptide_tokens"], batch["hla_tokens"])
        # Gather across processes if multi-GPU.
        log_aff = accelerator.gather_for_metrics(log_aff)
        pres = accelerator.gather_for_metrics(pres)
        aff_true = accelerator.gather_for_metrics(batch["log_affinity"])
        aff_mask = accelerator.gather_for_metrics(batch["affinity_mask"])
        pres_true = accelerator.gather_for_metrics(batch["presentation"])
        pres_mask = accelerator.gather_for_metrics(batch["presentation_mask"])

        all_aff_pred.append(log_aff.cpu().numpy())
        all_aff_true.append(aff_true.cpu().numpy())
        all_aff_mask.append(aff_mask.cpu().numpy())
        all_pres_pred.append(torch.sigmoid(pres).cpu().numpy())
        all_pres_true.append(pres_true.cpu().numpy())
        all_pres_mask.append(pres_mask.cpu().numpy())

    aff_pred = np.concatenate(all_aff_pred)
    aff_true = np.concatenate(all_aff_true)
    aff_mask = np.concatenate(all_aff_mask).astype(bool)
    pres_pred = np.concatenate(all_pres_pred)
    pres_true = np.concatenate(all_pres_true)
    pres_mask = np.concatenate(all_pres_mask).astype(bool)

    aff_rho = 0.0
    if aff_mask.sum() >= 10:
        rho, _ = spearmanr(aff_pred[aff_mask], aff_true[aff_mask])
        # Spearman returns negative for inverse correlation; we flip sign
        # convention: we predict log(IC50), and higher predicted == higher
        # true. We want correlation positive when model is good.
        aff_rho = float(rho) if not math.isnan(rho) else 0.0

    pres_auc = 0.5
    if pres_mask.sum() >= 10 and len(np.unique(pres_true[pres_mask])) > 1:
        pres_auc = float(roc_auc_score(pres_true[pres_mask], pres_pred[pres_mask]))

    return {"aff_spearman": aff_rho, "pres_auc": pres_auc}
