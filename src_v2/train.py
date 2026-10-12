"""
src_v2/train.py
===============
Complete Study-Level Training, Validation, and Cross-Validation Pipeline for RSNA Knee MRI V2.

Features:
  - 5-Fold GroupKFold using identical fold assignments from outputs_v2/folds.csv.
  - Study-level DataLoaders with max 2 series per plane sampling in train, all series in val.
  - GPU Affine + Photometric + Selective Flip Augmentation.
  - AdamW optimizer with two parameter groups (Backbone: lr 1e-4, wd 1e-2; Transformer/Head: lr 5e-4, wd 5e-2).
  - 1-epoch linear warmup + exact step-level Cosine Annealing decay to 1e-6 over 15 epochs.
  - Exponential Moving Average (EMA) with decay 0.998 evaluated at every validation epoch.
  - Automatic AMP precision (bf16 if supported on GPU, else fp16 with GradScaler).
  - Full checkpoint resumption support.
  - Automatic logging of full config, history CSVs, best/last checkpoints, and OOF predictions.
  - CRASH FIX: Workers return uint8 only; normalization is on GPU. num_workers=3 train / 0 val.
    torch.multiprocessing sharing strategy = 'file_system' to avoid Windows commit-limit errors.
"""

import argparse
import copy
import glob
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

sys.path.insert(0, r"d:\RSNA")

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src_v2.dataset import (
    LABELS,
    GPUAugmentor,
    StudyKneeDataset,
    collate_study_fn,
    gpu_normalize_volumes,
    save_augmentation_check,
)
from src_v2.model import StudyKneeClassifier
from src.metrics import compute_macro_auc

# ── Windows: Use file_system sharing to avoid ERROR_COMMITMENT_LIMIT (1455) ──
# This must be called before any DataLoader is created.
mp.set_sharing_strategy("file_system")


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility Seed
# ─────────────────────────────────────────────────────────────────────────────
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


# ─────────────────────────────────────────────────────────────────────────────
# Append-mode Log Writer (flushes every N steps)
# ─────────────────────────────────────────────────────────────────────────────
class TrainLogger:
    """Append-mode logger that always flushes after every write."""

    def __init__(self, log_path: str) -> None:
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def log(self, msg: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()


# ─────────────────────────────────────────────────────────────────────────────
# Exponential Moving Average (EMA)
# ─────────────────────────────────────────────────────────────────────────────
class ModelEMA:
    """
    Model Exponential Moving Average for PyTorch models.
    """

    def __init__(self, model: nn.Module, decay: float = 0.998) -> None:
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ema_p, model_p in zip(self.module.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1.0 - self.decay)
        for ema_b, model_b in zip(self.module.buffers(), model.buffers()):
            ema_b.data.copy_(model_b.data)


# ─────────────────────────────────────────────────────────────────────────────
# Optimizer & Parameter Group Configuration
# ─────────────────────────────────────────────────────────────────────────────
def build_optimizer(
    model: nn.Module,
    backbone_lr: float = 1e-4,
    backbone_wd: float = 1e-2,
    head_lr: float = 5e-4,
    head_wd: float = 5e-2,
) -> torch.optim.Optimizer:
    """
    Build AdamW with separate parameter groups for backbone vs rest,
    excluding 1D parameters (biases, LayerNorm, BatchNorm) from weight decay.
    """
    backbone_decay, backbone_no_decay = [], []
    head_decay, head_no_decay = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_1d = param.ndim <= 1 or "bias" in name or "norm" in name or "ln" in name or "bn" in name
        if "backbone" in name:
            if is_1d:
                backbone_no_decay.append(param)
            else:
                backbone_decay.append(param)
        else:
            if is_1d:
                head_no_decay.append(param)
            else:
                head_decay.append(param)

    param_groups = [
        {"params": backbone_decay, "lr": backbone_lr, "weight_decay": backbone_wd},
        {"params": backbone_no_decay, "lr": backbone_lr, "weight_decay": 0.0},
        {"params": head_decay, "lr": head_lr, "weight_decay": head_wd},
        {"params": head_no_decay, "lr": head_lr, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(param_groups)


# ─────────────────────────────────────────────────────────────────────────────
# Custom Warmup + Cosine Annealing Learning Rate Scheduler
# ─────────────────────────────────────────────────────────────────────────────
class WarmupCosineScheduler:
    """
    Step-level scheduler with linear warmup and cosine decay to eta_min.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        eta_min: float = 1e-6,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(self.warmup_steps + 1, total_steps)
        self.eta_min = eta_min
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.current_step = 0

    def step(self) -> None:
        self.current_step += 1
        self._update_lrs()

    def set_step(self, step: int) -> None:
        self.current_step = step
        self._update_lrs()

    def _update_lrs(self) -> None:
        step = self.current_step
        if step <= self.warmup_steps:
            factor = step / self.warmup_steps
            for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                group["lr"] = self.eta_min + (base_lr - self.eta_min) * factor
        else:
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(1.0, max(0.0, progress))
            cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
            for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                group["lr"] = self.eta_min + (base_lr - self.eta_min) * cosine_decay

    def get_last_lr(self) -> List[float]:
        return [group["lr"] for group in self.optimizer.param_groups]


# ─────────────────────────────────────────────────────────────────────────────
# Validation Epoch (Study-Level)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_dtype: torch.dtype,
    labels: List[str] = LABELS,
) -> Tuple[float, float, Dict[str, float], pd.DataFrame]:
    """
    Evaluate model at study level. Returns (val_loss, macro_auc, per_class_auc, pred_df).
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_study_uids = []

    for batch in loader:
        vols = batch["volumes"].to(device, non_blocking=True)
        planes = batch["planes"].to(device, non_blocking=True)
        fluid = batch["fluid"].to(device, non_blocking=True)
        fat = batch["fat"].to(device, non_blocking=True)
        mask = batch["series_mask"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
            logits = model(vols, planes, fluid, fat, mask, token_dropout_p=0.0)
            loss = criterion(logits, targets)

        probs = torch.sigmoid(logits).cpu().numpy()
        targets_np = targets.cpu().numpy()

        all_preds.append(probs)
        all_targets.append(targets_np)
        all_study_uids.extend(batch["study_uids"])
        total_loss += loss.item() * len(batch["study_uids"])

    val_loss = total_loss / max(1, len(all_study_uids))
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)

    macro_auc, per_class = compute_macro_auc(y_true, y_pred, labels=labels)

    pred_df = pd.DataFrame({"StudyInstanceUID": all_study_uids})
    for i, l in enumerate(labels):
        pred_df[l] = y_pred[:, i]

    return val_loss, macro_auc, per_class, pred_df


# ─────────────────────────────────────────────────────────────────────────────
# Training Engine for 1 Fold
# ─────────────────────────────────────────────────────────────────────────────
def train_fold(
    fold: int,
    config: Dict,
    studies_df: pd.DataFrame,
    series_df: pd.DataFrame,
    folds_df: pd.DataFrame,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_scaler: bool,
    logger: TrainLogger,
) -> Tuple[float, float, Dict]:
    """
    Train a single fold end-to-end.
    """
    logger.log(f"\n{'='*70}\n  STARTING V2 TRAINING: FOLD {fold} / 5\n{'='*70}")
    outputs_dir = Path(config["outputs_dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dataset splits
    train_uids = set(folds_df[folds_df["fold"] != fold]["StudyInstanceUID"])
    val_uids = set(folds_df[folds_df["fold"] == fold]["StudyInstanceUID"])

    train_studies_df = studies_df[studies_df["StudyInstanceUID"].isin(train_uids)].reset_index(drop=True)
    val_studies_df = studies_df[studies_df["StudyInstanceUID"].isin(val_uids)].reset_index(drop=True)

    logger.log(f"[Data] Train Studies: {len(train_studies_df)} | Val Studies: {len(val_studies_df)}")

    # 2. Pos Weight computation if pos_weight_mode == 'sqrt'
    pos_weight = None
    if config.get("pos_weight_mode") == "sqrt":
        y_train = train_studies_df[LABELS].values
        pos_counts = (y_train >= 0.5).sum(axis=0)
        neg_counts = len(y_train) - pos_counts
        pos_weight = torch.tensor(np.sqrt(neg_counts / np.maximum(pos_counts, 1)), dtype=torch.float32, device=device)
        logger.log(f"[Loss] Using sqrt pos_weight: {pos_weight.cpu().numpy().round(3)}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 3. Datasets
    train_ds = StudyKneeDataset(
        train_studies_df,
        series_df,
        base_dir=config["data_dir"],
        labels=LABELS,
        is_train=True,
        max_series_per_plane=config["max_series_per_plane"],
    )
    val_ds = StudyKneeDataset(
        val_studies_df,
        series_df,
        base_dir=config["data_dir"],
        labels=LABELS,
        is_train=False,
    )

    # ── Training loader: 3 workers, no persistent workers ──────────────────
    # persistent_workers=False: worker pool is torn down after each epoch so
    # it doesn't hold onto shared-memory segments across train+val.
    train_loader = DataLoader(
        train_ds,
        batch_size=config["micro_batch_size"],
        shuffle=True,
        num_workers=config["num_workers_train"],  # = 3
        prefetch_factor=2 if config["num_workers_train"] > 0 else None,
        collate_fn=collate_study_fn,
        pin_memory=True,
        persistent_workers=False,  # MUST be False on Windows
        drop_last=True,
    )

    # 4. Model & EMA
    model = StudyKneeClassifier(
        num_classes=12,
        backbone_name=config["backbone"],
        pretrained=config["pretrained"],
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_encoder_layers=config["num_encoder_layers"],
        dim_feedforward=config["dim_feedforward"],
        dropout=config["dropout"],
        head_dropout=config["head_dropout"],
        fixed_depth=config["fixed_depth"],
        use_grad_checkpointing=config["use_grad_checkpointing"],
        no_metadata=config["no_metadata"],
        mean_pool_instead_of_label_attention=config["mean_pool_instead_of_label_attention"],
    ).to(device)

    ema = ModelEMA(model, decay=config["ema_decay"])
    augmentor = GPUAugmentor().to(device)

    # 5. Optimizer & Scheduler
    accum_steps = config["accum_steps"]
    effective_batch = config["micro_batch_size"] * accum_steps
    steps_per_epoch = len(train_loader) // accum_steps
    total_opt_steps = steps_per_epoch * config["epochs"]
    warmup_steps = steps_per_epoch * config["warmup_epochs"]

    optimizer = build_optimizer(
        model,
        backbone_lr=config["backbone_lr"],
        backbone_wd=config["backbone_wd"],
        head_lr=config["head_lr"],
        head_wd=config["head_wd"],
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_opt_steps,
        eta_min=1e-6,
    )
    scaler = torch.amp.GradScaler(enabled=use_scaler)

    logger.log(f"[Setup] Epochs: {config['epochs']} | Steps/Epoch: {steps_per_epoch} | Total Opt Steps: {total_opt_steps}")
    logger.log(f"[Setup] Micro Batch: {config['micro_batch_size']} | Accum Steps: {accum_steps} | Effective Batch: {effective_batch}")
    logger.log(f"[Setup] Precision: {'bf16' if amp_dtype == torch.bfloat16 else 'fp16 (GradScaler)'}")
    logger.log(f"[Setup] Workers (train/val): {config['num_workers_train']} / {config['num_workers_val']}")

    # Checkpoint paths
    best_ckpt_path = outputs_dir / f"best_model_fold{fold}.pt"
    last_ckpt_path = outputs_dir / f"last_model_fold{fold}.pt"
    hist_csv_path = outputs_dir / f"training_history_fold{fold}.csv"
    oof_best_path = outputs_dir / f"oof_best_fold{fold}.csv"
    oof_last_path = outputs_dir / f"oof_last_fold{fold}.csv"
    emrg_ckpt_path = outputs_dir / f"emergency_checkpoint_fold{fold}.pt"

    start_epoch = 1
    best_macro_auc = -1.0
    best_epoch = 1
    history = []
    best_pred_df = None
    last_pred_df = None
    macro_auc = 0.0

    # ── Checkpoint Resumption ─────────────────────────────────────────────────
    if last_ckpt_path.exists():
        logger.log(f"[Resume] Found checkpoint at {last_ckpt_path}, attempting to resume...")
        try:
            ckpt = torch.load(last_ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            if "ema_state" in ckpt:
                ema.module.load_state_dict(ckpt["ema_state"])
            elif "model_state" in ckpt:
                ema.module.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scheduler.set_step(ckpt["scheduler_step"])
            start_epoch = ckpt["epoch"] + 1
            best_macro_auc = ckpt.get("best_auc", -1.0)
            best_epoch = ckpt.get("best_epoch", 1)
            if hist_csv_path.exists():
                history = pd.read_csv(hist_csv_path).to_dict("records")
            if oof_best_path.exists():
                best_pred_df = pd.read_csv(oof_best_path)
            if oof_last_path.exists():
                last_pred_df = pd.read_csv(oof_last_path)
            logger.log(
                f"[Resume] Successfully resumed fold {fold} at Epoch {start_epoch} "
                f"(Best AUC: {best_macro_auc:.6f} @ ep {best_epoch})"
            )
        except Exception as res_err:
            logger.log(f"[Resume WARNING] Failed to load checkpoint {last_ckpt_path}: {res_err}. Starting from scratch.")

    if start_epoch > config["epochs"]:
        logger.log(f"[Fold {fold} Already Complete] ({start_epoch - 1}/{config['epochs']} epochs finished)")
        return best_macro_auc, ckpt.get("last_auc", best_macro_auc), {"best_epoch": best_epoch, "history": history}

    for epoch in range(start_epoch, config["epochs"] + 1):
        epoch_start_time = time.time()
        try:
            model.train()
            train_loss_accum = 0.0
            optimizer.zero_grad(set_to_none=True)
            train_t0 = time.time()

            for step, batch in enumerate(tqdm(train_loader, desc=f"Ep {epoch:02d}/{config['epochs']:02d} [Train]", leave=False)):
                vols = batch["volumes"].to(device, non_blocking=True)
                planes = batch["planes"].to(device, non_blocking=True)
                fluid = batch["fluid"].to(device, non_blocking=True)
                fat = batch["fat"].to(device, non_blocking=True)
                mask = batch["series_mask"].to(device, non_blocking=True)
                targets = batch["targets"].to(device, non_blocking=True)

                # ── Training path: normalize on GPU first, then augment ────────
                # 1. Normalize uint8 → float32 (per-volume percentile + ImageNet)
                vols_f = gpu_normalize_volumes(vols, mask)   # [B, S, 24, H, W] float32
                # 2. Apply spatial + photometric augmentation on normalized float data
                vols_aug = augmentor(vols_f, planes, mask)   # [B, S, 24, H, W] float32
                # 3. Forward pass (model detects float32 and skips internal normalization)

                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                    logits = model(vols_aug, planes, fluid, fat, mask, token_dropout_p=config["token_dropout_p"])
                    loss = criterion(logits, targets)
                    loss_scaled = loss / accum_steps

                if use_scaler:
                    scaler.scale(loss_scaled).backward()
                else:
                    loss_scaled.backward()

                train_loss_accum += loss.item()

                if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                    if use_scaler:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["grad_clip"])
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["grad_clip"])
                        optimizer.step()

                    ema.update(model)
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                # Flush log every 20 steps
                if (step + 1) % 20 == 0 or (step + 1) == len(train_loader):
                    opt_step = (step + 1) // accum_steps
                    cur_lr = scheduler.get_last_lr()[2]
                    running_loss = train_loss_accum / (step + 1)
                    logger.log(
                        f"  [Fold {fold} | Ep {epoch:02d} | Step {step+1}/{len(train_loader)} (opt {opt_step}/{steps_per_epoch})] "
                        f"train_loss={running_loss:.4f}  lr={cur_lr:.2e}"
                    )

            train_time = time.time() - train_t0
            mean_train_loss = train_loss_accum / len(train_loader)

            # ── Validation ───────────────────────────────────────────────────
            val_t0 = time.time()
            val_workers = config["num_workers_val"]
            val_loader = None
            try:
                val_loader = DataLoader(
                    val_ds,
                    batch_size=config["val_batch_size"],
                    shuffle=False,
                    num_workers=val_workers,
                    prefetch_factor=2 if val_workers > 0 else None,
                    collate_fn=collate_study_fn,
                    pin_memory=True,
                    persistent_workers=False,
                    drop_last=False,
                )
                val_loss, macro_auc, per_class, val_pred_df = validate_epoch(
                    ema.module, val_loader, criterion, device, amp_dtype
                )
            except RuntimeError as val_err:
                if "1455" in str(val_err) or "shared" in str(val_err).lower():
                    logger.log(
                        f"[WARNING] Validation with num_workers={val_workers} hit shared memory error: {val_err}. "
                        f"Falling back to num_workers=0..."
                    )
                    config["num_workers_val"] = 0
                    if val_loader is not None:
                        del val_loader
                    import gc
                    gc.collect()

                    fb_start = time.time()
                    val_loader = DataLoader(
                        val_ds,
                        batch_size=config["val_batch_size"],
                        shuffle=False,
                        num_workers=0,
                        collate_fn=collate_study_fn,
                        pin_memory=True,
                        persistent_workers=False,
                        drop_last=False,
                    )
                    val_loss, macro_auc, per_class, val_pred_df = validate_epoch(
                        ema.module, val_loader, criterion, device, amp_dtype
                    )
                    fb_time = time.time() - fb_start
                    logger.log(f"[Fallback Speed Cost] num_workers=0 validation took {fb_time:.1f}s")
                else:
                    raise val_err
            finally:
                if val_loader is not None:
                    del val_loader
                import gc
                gc.collect()

            val_time = time.time() - val_t0
            val_pred_df["fold"] = fold
            last_pred_df = val_pred_df

            epoch_time = time.time() - epoch_start_time
            current_lr = scheduler.get_last_lr()[2]
            peak_gpu_mem = (torch.cuda.max_memory_allocated() / (1024 ** 3)) if torch.cuda.is_available() else 0.0

            # Track best model
            is_best = macro_auc > best_macro_auc
            if is_best:
                best_macro_auc = macro_auc
                best_epoch = epoch
                best_pred_df = val_pred_df

                torch.save(
                    {
                        "fold": fold,
                        "epoch": epoch,
                        "best_epoch": best_epoch,
                        "best_auc": best_macro_auc,
                        "model_state": model.state_dict(),
                        "ema_state": ema.module.state_dict(),
                        "config": config,
                    },
                    best_ckpt_path,
                )
                best_pred_df.to_csv(oof_best_path, index=False)
                tag = f" <-- BEST [saved -> {best_ckpt_path.name}]"
            else:
                tag = ""

            # Save last checkpoint EVERY epoch so a crash loses at most one epoch
            torch.save(
                {
                    "fold": fold,
                    "epoch": epoch,
                    "best_epoch": best_epoch,
                    "best_auc": best_macro_auc,
                    "last_auc": macro_auc,
                    "model_state": model.state_dict(),
                    "ema_state": ema.module.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_step": scheduler.current_step,
                    "config": config,
                },
                last_ckpt_path,
            )
            last_pred_df.to_csv(oof_last_path, index=False)

            # Per-epoch summary line required by user
            epoch_summary = (
                f"[Fold {fold} | Epoch {epoch:02d}/{config['epochs']:02d}] "
                f"Train Loss: {mean_train_loss:.4f} ({train_time:.1f}s) | "
                f"Val Loss: {val_loss:.4f} ({val_time:.1f}s) | "
                f"Val Macro AUC: {macro_auc:.6f} (best: {best_macro_auc:.6f} @ ep {best_epoch}) | "
                f"Total Time: {epoch_time:.1f}s | Peak GPU Mem: {peak_gpu_mem:.2f} GB | "
                f"LR: {current_lr:.2e}{tag}"
            )
            logger.log(epoch_summary)

            epoch_row = {
                "fold": fold,
                "epoch": epoch,
                "train_loss": mean_train_loss,
                "val_loss": val_loss,
                "macro_auc": macro_auc,
                "train_time_sec": round(train_time, 2),
                "val_time_sec": round(val_time, 2),
                "peak_gpu_mem_gb": round(peak_gpu_mem, 2),
                "lr": current_lr,
                **per_class,
            }
            history.append(epoch_row)

            # Per-epoch CSV save (append mode and flush)
            hist_df = pd.DataFrame([epoch_row])
            write_header = not hist_csv_path.exists()
            hist_df.to_csv(hist_csv_path, mode="a", header=write_header, index=False)

            # Check for verify_only mode
            if config.get("verify_only", False) and epoch == 1:
                logger.log(f"[VERIFY COMPLETE] 1 full epoch of Fold {fold} completed successfully.")
                logger.log(f"  Train Time      : {train_time:.2f} s")
                logger.log(f"  Val Time        : {val_time:.2f} s")
                logger.log(f"  Total Time      : {epoch_time:.2f} s")
                logger.log(f"  Peak GPU Memory : {peak_gpu_mem:.2f} GB")
                logger.log(f"  Val Macro AUC   : {macro_auc:.6f}")
                return best_macro_auc, macro_auc, {"best_epoch": best_epoch, "history": history}

        except Exception as e:
            err_msg = traceback.format_exc()
            logger.log(f"[ERROR] Fold {fold}, Epoch {epoch} crashed: {e}")
            err_path = outputs_dir / f"error_fold{fold}.txt"
            with open(err_path, "a", encoding="utf-8") as ef:
                ef.write(f"\n\n=== Fold {fold}, Epoch {epoch}, {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                ef.write(err_msg)
            try:
                torch.save(
                    {
                        "fold": fold,
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "ema_state": ema.module.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_step": scheduler.current_step,
                        "config": config,
                        "error": str(e),
                    },
                    emrg_ckpt_path,
                )
                logger.log(f"[ERROR] Emergency checkpoint saved to {emrg_ckpt_path}")
            except Exception as ckpt_err:
                logger.log(f"[ERROR] Failed to save emergency checkpoint: {ckpt_err}")
            sys.exit(1)

    last_macro_auc = macro_auc
    logger.log(f"\n[Fold {fold} Finished] Best Macro AUC: {best_macro_auc:.6f} (@ ep {best_epoch}) | Last Macro AUC: {last_macro_auc:.6f}")
    return best_macro_auc, last_macro_auc, {"best_epoch": best_epoch, "history": history}


# ─────────────────────────────────────────────────────────────────────────────
# Sanity Checks (Phase 4)
# ─────────────────────────────────────────────────────────────────────────────
def run_sanity_checks(
    config: Dict,
    studies_df: pd.DataFrame,
    series_df: pd.DataFrame,
    folds_df: pd.DataFrame,
    device: torch.device,
    amp_dtype: torch.dtype,
    logger: TrainLogger,
) -> bool:
    """
    Run the 4 mandatory pre-flight checks:
      1. Disjoint StudyInstanceUID sets across all 5 folds.
      2. End-to-end forward pass shape verification.
      3. 16-study overfit verification test.
      4. GPU memory profiling and speed benchmark.
    """
    logger.log(f"\n{'='*70}\n  RUNNING PHASE 4 PRE-FLIGHT SANITY CHECKS\n{'='*70}")

    # Check 1: Disjointness
    logger.log("\n[Check 1/4] Verifying Study-to-Fold Disjointness ...")
    for i in range(5):
        for j in range(i + 1, 5):
            set_i = set(folds_df[folds_df["fold"] == i]["StudyInstanceUID"])
            set_j = set(folds_df[folds_df["fold"] == j]["StudyInstanceUID"])
            assert len(set_i.intersection(set_j)) == 0, f"Error: Fold {i} and {j} overlap!"
    logger.log("  --> PASS: All 5 validation study sets are 100% mutually disjoint.")

    # Check 2: Forward Pass & Tensor Shapes
    logger.log("\n[Check 2/4] End-to-End Forward Pass Shape Test ...")
    dummy_studies = studies_df.head(4).reset_index(drop=True)
    dummy_ds = StudyKneeDataset(dummy_studies, series_df, base_dir=config["data_dir"], labels=LABELS, is_train=True)
    # Use num_workers=0 for sanity checks to keep things simple
    dummy_loader = DataLoader(dummy_ds, batch_size=2, collate_fn=collate_study_fn, num_workers=0)
    batch = next(iter(dummy_loader))

    vols = batch["volumes"].to(device)
    planes = batch["planes"].to(device)
    fluid = batch["fluid"].to(device)
    fat = batch["fat"].to(device)
    mask = batch["series_mask"].to(device)
    targets = batch["targets"].to(device)

    logger.log(f"  Batch volumes shape     : {vols.shape} (B, S_max, D=24, H=224, W=224)  dtype={vols.dtype}")
    logger.log(f"  Batch planes shape      : {planes.shape}")
    logger.log(f"  Batch fluid shape       : {fluid.shape}")
    logger.log(f"  Batch series_mask shape : {mask.shape}")
    logger.log(f"  Batch targets shape     : {targets.shape}")

    model = StudyKneeClassifier(
        num_classes=12,
        backbone_name=config["backbone"],
        pretrained=False,
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_encoder_layers=config["num_encoder_layers"],
        dim_feedforward=config["dim_feedforward"],
    ).to(device)

    with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
        logits, attn = model(vols, planes, fluid, fat, mask, token_dropout_p=0.1, return_attention=True)

    logger.log(f"  Output logits shape     : {logits.shape} (Expected: [2, 12])")
    logger.log(f"  Output attention shape  : {attn.shape} (Expected: [2, 12, S_max*24])")
    assert logits.shape == (2, 12), f"Logits shape mismatch: {logits.shape}"
    logger.log("  --> PASS: End-to-end model forward pass verified successfully.")

    # Check 3: Overfit Test on 8 Studies
    logger.log("\n[Check 3/4] Overfit Verification on 8 Studies (Target: Loss -> 0) ...")
    overfit_studies = studies_df.head(8).reset_index(drop=True)
    overfit_ds = StudyKneeDataset(overfit_studies, series_df, base_dir=config["data_dir"], labels=LABELS, is_train=False)
    overfit_loader = DataLoader(overfit_ds, batch_size=2, collate_fn=collate_study_fn, shuffle=False, num_workers=0)

    # Preload batches in memory
    cached_batches = [b for b in overfit_loader]

    model_overfit = StudyKneeClassifier(
        num_classes=12,
        backbone_name=config["backbone"],
        pretrained=False,
        d_model=config["d_model"],
    ).to(device)
    opt_overfit = torch.optim.AdamW(model_overfit.parameters(), lr=1e-3, weight_decay=0.0)
    crit_overfit = nn.BCEWithLogitsLoss()

    model_overfit.train()
    initial_loss, min_loss = 0.0, 999.0
    for step in range(30):
        epoch_loss = 0.0
        for b in cached_batches:
            v = b["volumes"].to(device)
            p = b["planes"].to(device)
            fl = b["fluid"].to(device)
            fa = b["fat"].to(device)
            m = b["series_mask"].to(device)

            # Use binarized targets for the overfit test so theoretical loss minimum is 0.0
            t_bin = (b["targets"] >= 0.5).float().to(device)

            opt_overfit.zero_grad()
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                out = model_overfit(v, p, fl, fa, m)
                loss = crit_overfit(out, t_bin)
            loss.backward()
            opt_overfit.step()
            epoch_loss += loss.item()

        avg_l = epoch_loss / len(cached_batches)
        if step == 0:
            initial_loss = avg_l
        min_loss = min(min_loss, avg_l)
        if (step + 1) % 5 == 0 or step == 0:
            logger.log(f"    Overfit Step [{step+1:02d}/30] Loss: {avg_l:.4f} (Min: {min_loss:.4f})")

    logger.log(f"  Overfit initial loss: {initial_loss:.4f} -> Best overfit loss: {min_loss:.4f}")
    assert min_loss < 0.20, f"Overfit test failed! Min loss {min_loss} did not reach below 0.20"
    logger.log("  --> PASS: Model successfully memorized test studies.")

    # Check 4: GPU Memory Profile & Visual Augmentation Check
    logger.log("\n[Check 4/4] Generating Augmentation Verification Grid ...")
    save_augmentation_check(dummy_ds, output_path="outputs_v2/aug_check.png", num_samples=3)

    if torch.cuda.is_available():
        mem_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
        logger.log(f"  Peak GPU Memory Allocated: {mem_allocated:.2f} GB")

    logger.log("\n" + "="*70 + "\n  ALL SANITY CHECKS PASSED!\n" + "="*70 + "\n")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Main Orchestrator
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="RSNA Knee MRI V2 Study-Level Training")
    parser.add_argument("--fold", type=int, default=None, help="Train specific fold (0-4), or None for all 5 folds")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--micro_batch_size", type=int, default=2, help="Micro batch size per step")
    parser.add_argument("--accum_steps", type=int, default=4, help="Gradient accumulation steps (effective batch = 8)")
    parser.add_argument("--skip_sanity_checks", action="store_true", help="Skip pre-flight checks")
    parser.add_argument("--verify_only", action="store_true", help="Run 1 epoch of fold 0 for verification and exit")
    parser.add_argument("--num_workers_train", type=int, default=3, help="Workers for training DataLoader")
    parser.add_argument("--num_workers_val", type=int, default=2, help="Workers for validation DataLoader (<=2)")
    parser.add_argument("--pos_weight_mode", type=str, default="none", choices=["none", "sqrt"])
    parser.add_argument("--no_metadata", action="store_true", help="Ablation: drop metadata embeddings")
    parser.add_argument("--mean_pool", action="store_true", help="Ablation: mean pool instead of label attention")
    parser.add_argument("--backbone", type=str, default="resnet34", choices=["resnet34", "efficientnet_v2_s", "convnext_tiny"])
    parser.add_argument("--skip_gate_check", action="store_true", help="Bypass Fold 0 gate check and train all remaining folds")
    parser.add_argument("--force_rerun", action="store_true", help="Force re-training even if fold checkpoints already exist")
    args = parser.parse_args()

    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Precision setup: use bfloat16 if supported, else fp16 with GradScaler
    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if bf16_supported else torch.float16
    use_scaler = not bf16_supported

    # Central Config
    config = {
        "version": "v2_study_transformer",
        "data_dir": "D:/rsna_knee_dataset",
        "outputs_dir": "outputs_v2",
        "backbone": args.backbone,
        "pretrained": True,
        "d_model": 256,
        "nhead": 8,
        "num_encoder_layers": 2,
        "dim_feedforward": 512,
        "dropout": 0.1,
        "head_dropout": 0.2,
        "fixed_depth": 24,
        "max_series_per_plane": 2,
        "epochs": 1 if args.verify_only else args.epochs,
        "warmup_epochs": 1,
        "micro_batch_size": args.micro_batch_size,
        "accum_steps": args.accum_steps,
        "val_batch_size": 1,       # 1 study per batch for val (max stability)
        "backbone_lr": 1e-4,
        "backbone_wd": 1e-2,
        "head_lr": 5e-4,
        "head_wd": 5e-2,
        "grad_clip": 1.0,
        "ema_decay": 0.998,
        "token_dropout_p": 0.10,
        "num_workers_train": args.num_workers_train,    # 3 workers for training
        "num_workers_val": args.num_workers_val,        # <= 2 workers for val
        "use_grad_checkpointing": False,
        "pos_weight_mode": args.pos_weight_mode,
        "no_metadata": args.no_metadata,
        "mean_pool_instead_of_label_attention": args.mean_pool,
        "seed": 42,
        "sharing_strategy": "file_system",
        "baseline_fold0_auc": 0.880280,   # used for fold 0 gate check
        "fold0_auc_tolerance": 0.010,
        "verify_only": args.verify_only,
    }

    # Save full config to outputs_v2/config.json
    os.makedirs(config["outputs_dir"], exist_ok=True)
    with open(os.path.join(config["outputs_dir"], "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Set up the logger
    log_path = os.path.join(config["outputs_dir"], "train_log.txt")
    logger = TrainLogger(log_path)
    logger.log("=" * 70)
    logger.log(f"RSNA Knee MRI V2 Training — Starting (verify_only={args.verify_only})")
    logger.log(f"torch.multiprocessing sharing strategy: {mp.get_sharing_strategy()}")
    logger.log(f"Device: {device} | AMP dtype: {amp_dtype} | GradScaler: {use_scaler}")
    logger.log(f"Workers: train={config['num_workers_train']} | val={config['num_workers_val']}")
    logger.log(f"Config saved to: {os.path.join(config['outputs_dir'], 'config.json')}")
    logger.log("=" * 70)

    # Load Data & Folds
    studies_df = pd.read_csv(os.path.join(config["data_dir"], "train_targets_soft.csv"))
    series_df = pd.read_csv(os.path.join(config["data_dir"], "train_series.csv"))
    folds_df = pd.read_csv("outputs_v2/folds.csv")

    # Run Sanity Checks
    if not args.skip_sanity_checks and not args.verify_only:
        run_sanity_checks(config, studies_df, series_df, folds_df, device, amp_dtype, logger)

    # Determine folds to train
    if args.verify_only:
        target_folds = [0]
    else:
        target_folds = [args.fold] if args.fold is not None else list(range(5))

    fold_best_aucs = {}
    fold_last_aucs = {}

    for fold in target_folds:
        ckpt_best = os.path.join(config["outputs_dir"], f"best_model_fold{fold}.pt")
        oof_best_f = os.path.join(config["outputs_dir"], f"oof_best_fold{fold}.csv")
        hist_f = os.path.join(config["outputs_dir"], f"training_history_fold{fold}.csv")

        # Skip fold if already fully completed
        if not args.force_rerun and os.path.exists(ckpt_best) and os.path.exists(oof_best_f) and os.path.exists(hist_f):
            try:
                h_df = pd.read_csv(hist_f)
                if len(h_df) >= config["epochs"]:
                    b_auc = float(h_df["macro_auc"].max())
                    l_auc = float(h_df.iloc[-1]["macro_auc"])
                    fold_best_aucs[fold] = b_auc
                    fold_last_aucs[fold] = l_auc
                    logger.log(f"\n[Fold {fold} Already Completed] Best AUC: {b_auc:.6f} | Last AUC: {l_auc:.6f}. Skipping re-training.")
                    continue
                else:
                    logger.log(f"\n[Fold {fold}] Incomplete history ({len(h_df)}/{config['epochs']} epochs completed). Resuming from checkpoint...")
            except Exception as e:
                logger.log(f"[Fold {fold}] Existing artifacts could not be read ({e}), re-training.")

        best_auc, last_auc, _ = train_fold(
            fold=fold,
            config=config,
            studies_df=studies_df,
            series_df=series_df,
            folds_df=folds_df,
            device=device,
            amp_dtype=amp_dtype,
            use_scaler=use_scaler,
            logger=logger,
        )
        fold_best_aucs[fold] = best_auc
        fold_last_aucs[fold] = last_auc

        if args.verify_only:
            logger.log(f"[Verification Finished] Fold 0 verified successfully.")
            return

        # Gate check for fold 0 against baseline
        if fold == 0 and not args.skip_gate_check:
            baseline_auc = config["baseline_fold0_auc"]
            tol = config["fold0_auc_tolerance"]
            delta = best_auc - baseline_auc
            logger.log(
                f"\n[Fold 0 Gate Check]\n"
                f"  V2 Best AUC : {best_auc:.6f}\n"
                f"  Baseline AUC: {baseline_auc:.6f}\n"
                f"  Delta       : {delta:+.6f}\n"
                f"  Tolerance   : -{tol:.3f}"
            )
            if delta < -tol:
                logger.log(
                    f"[STOP] Fold 0 best AUC {best_auc:.6f} is worse than baseline {baseline_auc:.6f} "
                    f"by {abs(delta):.6f} (exceeds tolerance of -{tol:.3f}). STOPPING as requested."
                )
                sys.exit(1)
            else:
                logger.log(f"[Gate Check PASSED] Fold 0 AUC within tolerance (delta: {delta:+.6f}). Proceeding to remaining folds.")
        elif fold == 0 and args.skip_gate_check:
            logger.log(f"[Gate Check BYPASSED] Proceeding to train all remaining folds as requested.")

    # Merge OOF Predictions across all completed folds
    best_oof_files = sorted(glob.glob("outputs_v2/oof_best_fold*.csv"))
    last_oof_files = sorted(glob.glob("outputs_v2/oof_last_fold*.csv"))

    if len(best_oof_files) == 5:
        df_best_all = pd.concat([pd.read_csv(f) for f in best_oof_files], ignore_index=True)
        df_last_all = pd.concat([pd.read_csv(f) for f in last_oof_files], ignore_index=True)
        df_best_all.to_csv("outputs_v2/oof_best.csv", index=False)
        df_last_all.to_csv("outputs_v2/oof_last.csv", index=False)
        logger.log("\nMerged 5-fold OOF predictions into outputs_v2/oof_best.csv and outputs_v2/oof_last.csv")

        # Compute full 5-fold CV AUC
        merged_targets = pd.merge(df_best_all, studies_df, on="StudyInstanceUID", suffixes=("_pred", "_true"))
        y_true_5f = merged_targets[[l + "_true" for l in LABELS]].values
        y_pred_5f = merged_targets[[l + "_pred" for l in LABELS]].values
        cv_auc, cv_per_class = compute_macro_auc(y_true_5f, y_pred_5f, labels=LABELS)
        logger.log(f"\n=======================================================")
        logger.log(f"V2 FULL 5-FOLD CV MACRO AUC: {cv_auc:.6f}")
        for l in LABELS:
            logger.log(f"  {l:<20}: {cv_per_class[l]:.6f}")
        logger.log(f"=======================================================")

    logger.log("\n[All folds complete]")
    logger.log(f"Best AUCs per fold: {fold_best_aucs}")
    logger.log(f"Last AUCs per fold: {fold_last_aucs}")


if __name__ == "__main__":
    main()
