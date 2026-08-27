"""
train_baseline.py
=================
Entry-point for multi-label knee MRI classification training.

Architecture : 2.5D CNN  (see src/model.py for details)
  - Pretrained ResNet-34 backbone applied per-slice via depth-fold trick
  - Mean / max-pool over depth → study embedding
  - Dropout → Linear head → 12 sigmoid-activated logits

Hardware target
---------------
  CPU  : Intel i9-13900K  (8P + 16E cores, 32 threads)
  RAM  : 64 GB DDR5-4000
  GPU  : 24 GB VRAM  (multiple GPUs supported via DataParallel)
  OS   : Windows 11 Pro 64-bit

Usage
-----
    # Train fold 0 with default settings (hardware-tuned):
    python train_baseline.py

    # Custom run:
    python train_baseline.py --epochs 50 --batch_size 8 --fold 2 --accum_steps 2

    --fold        : GroupKFold fold index to train (0-indexed, 0-4). Default=0.
    --accum_steps : Gradient accumulation steps (effective batch = batch*accum).

Project structure
-----------------
    rsna/
    ├── train_baseline.py   ← this file (entry-point + training loop)
    ├── src/
    │   ├── dataset.py      ← KneeMRIDataset
    │   ├── model.py        ← SliceEncoder + KneeClassifier
    │   ├── losses.py       ← compute_bce_loss
    │   └── metrics.py      ← compute_macro_auc
    ├── outputs/            ← checkpoints + training histories (runtime)
    └── requirements.txt

Requirements
------------
    pip install -r requirements.txt
"""

# ─────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────
import argparse
import os
import random
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

# ── Project modules ────────────────────────────────────────────
from src.dataset import KneeMRIDataset
from src.model   import KneeClassifier
from src.losses  import compute_bce_loss
from src.metrics import compute_macro_auc

warnings.filterwarnings("ignore")

# Enable TF32 on Ampere/Ada GPUs for free matmul throughput gains
# (no accuracy loss for this task — fp32 accumulator is preserved)
torch.set_float32_matmul_precision("high")


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
class Config:
    """
    Central configuration object.  All values can be overridden at runtime
    via CLI arguments (see ``parse_args``).

    Hardware defaults are tuned for:
      CPU  i9-13900K (32 threads)  →  NUM_WORKERS = 8
      GPU  24 GB VRAM × N GPUs     →  BATCH_SIZE  = 8
      RAM  64 GB DDR5-4000         →  PREFETCH_FACTOR = 2
    """

    # ── Paths ───────────────────────────────────
    # .npz volumes (Kaggle: barun2104/rsna-knee-mri-processed-3d-volumes)
    BASE_IMAGE_DIR: str = "D:/rsna_knee_dataset"
    # Competition CSVs (kaggle.com/competitions/rsna-knee-abnormality-detection/data)
    TRAIN_CSV:      str = "D:/rsna_knee_dataset/train_targets_soft.csv"
    SERIES_CSV:     str = "D:/rsna_knee_dataset/train_series.csv"
    # Runtime output directory (checkpoints + histories)
    OUTPUTS_DIR:    str = "outputs"

    # ── Labels ──────────────────────────────────
    LABELS: List[str] = [
        "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
        "Medial OA", "Lateral OA", "PF OA", "Effusion",
        "Synovitis", "Baker's", "Contusion", "Fracture",
    ]
    NUM_CLASSES: int = len(LABELS)   # 12

    # ── Volume settings ─────────────────────────
    FIXED_DEPTH: int = 24    # slices per volume (RSNA processed dataset = 24)
    IMG_HEIGHT:  int = 224
    IMG_WIDTH:   int = 224

    # ── Training (hardware-tuned for i9-13900K + 24 GB GPU) ─────
    EPOCHS:          int   = 30
    BATCH_SIZE:      int   = 8     # ↑ from 4: 24 GB VRAM easily fits B=8 in fp16
    LR:              float = 1e-4
    WEIGHT_DECAY:    float = 1e-2
    ACCUM_STEPS:     int   = 1     # gradient accumulation; effective batch = BATCH_SIZE × ACCUM_STEPS
    N_FOLDS:         int   = 5     # GroupKFold cross-validation splits
    FOLD:            int   = 0     # which fold to train (overridden by --fold)
    SEED:            int   = 42
    NUM_WORKERS:     int   = 8     # ↑ from 4: i9-13900K has 32 threads
    PREFETCH_FACTOR: int   = 2     # batches pre-fetched per worker
    MIXED_PREC:      bool  = True  # AMP fp16 via autocast + GradScaler

    # ── Model ───────────────────────────────────
    BACKBONE:   str  = "resnet34"
    PRETRAINED: bool = True
    POOL_MODE:  str  = "mean"      # "mean" | "max"


CFG = Config()


# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────
def seed_everything(seed: int = CFG.SEED) -> None:
    """
    Set all random seeds for reproducibility.

    ``cudnn.benchmark = True`` is intentional: all volumes are resized to
    the same spatial shape (24 × 224 × 224), so cuDNN can auto-select the
    fastest convolution kernel after the first batch and cache it for the
    rest of training (~10-15 % throughput gain).

    ``cudnn.deterministic = False`` is the counterpart required to allow
    cuDNN benchmark mode; this introduces negligible non-determinism in
    conv ops but is acceptable for this task.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark     = True   # auto-tune for fixed shapes
    torch.backends.cudnn.deterministic = False  # required by benchmark mode


# ─────────────────────────────────────────────
# Data Loading & Metadata
# ─────────────────────────────────────────────
def load_full_metadata(
    train_csv:  str       = CFG.TRAIN_CSV,
    series_csv: str       = CFG.SERIES_CSV,
    labels:     List[str] = CFG.LABELS,
) -> pd.DataFrame:
    """
    Load train_targets_soft.csv (soft/continuous probability targets for all
    ~4,407 studies) and join with train_series.csv to obtain SeriesInstanceUIDs.

    No NaN filtering is applied – all studies are kept.  If a 'sample_weight'
    column is present in train_targets_soft.csv it is carried through to the
    DataFrame so the Dataset can expose it for weighted-loss computation.

    Returns
    -------
    df : pd.DataFrame
        Columns: StudyInstanceUID, SeriesInstanceUID, <labels...>
                 [, sample_weight]
    """
    # ── Friendly check for missing CSVs ─────────────────────────
    for csv_path, name in [
        (train_csv,  "train_targets_soft.csv"),
        (series_csv, "train_series.csv"),
    ]:
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(
                f"\n\n[ERROR] Could not find: {csv_path}\n"
                f"  These CSV files come from the COMPETITION page (not the 3D-volumes dataset).\n"
                f"  Download them from:\n"
                f"    https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data\n"
                f"  Then place '{name}' at: {csv_path}\n"
                f"  (or update CFG.TRAIN_CSV / CFG.SERIES_CSV to wherever you saved them)\n"
            )

    train_df  = pd.read_csv(train_csv)
    series_df = pd.read_csv(series_csv)

    print(f"[Data] Total studies in {Path(train_csv).name}: {len(train_df)}")

    # ── Join with series to get SeriesInstanceUIDs ──────────────
    # Keep all columns from train_df so sample_weight (if present) is retained.
    merged = train_df.merge(
        series_df[["StudyInstanceUID", "SeriesInstanceUID"]],
        on="StudyInstanceUID",
        how="inner",
    )

    # Ensure label columns are float32 (soft probabilities in [0, 1])
    merged[labels] = merged[labels].astype(np.float32)

    print(f"[Data] Total study-series rows after join : {len(merged)}")
    print(f"[Data] Unique studies                     : {merged['StudyInstanceUID'].nunique()}")
    print(f"[Data] Unique series                      : {merged['SeriesInstanceUID'].nunique()}")
    has_weights = "sample_weight" in merged.columns
    print(f"[Data] sample_weight column found         : {has_weights}")
    return merged


# ─────────────────────────────────────────────
# 5-Fold GroupKFold Split
# ─────────────────────────────────────────────
def get_fold_split(
    df:      pd.DataFrame,
    fold:    int = CFG.FOLD,
    n_folds: int = CFG.N_FOLDS,
    seed:    int = CFG.SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split df into train/val using GroupKFold on StudyInstanceUID so that
    no series from the same study appears in both splits.

    Returns
    -------
    train_df, val_df : pd.DataFrame
    """
    gkf    = GroupKFold(n_splits=n_folds)
    groups = df["StudyInstanceUID"].values
    X      = np.arange(len(df))   # dummy features (index-only)

    splits = list(gkf.split(X, groups=groups))
    train_idx, val_idx = splits[fold]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df   = df.iloc[val_idx  ].reset_index(drop=True)

    print(f"[Fold {fold}] Train rows: {len(train_df)}  |  Val rows: {len(val_df)}")
    print(f"[Fold {fold}] Train studies: {train_df['StudyInstanceUID'].nunique()}"
          f"  |  Val studies: {val_df['StudyInstanceUID'].nunique()}")
    return train_df, val_df


# ─────────────────────────────────────────────────────────────────────────────
# NOTE: KneeMRIDataset, SliceEncoder, KneeClassifier, compute_macro_auc,
#       and compute_bce_loss have been moved to the src/ package.
#       They are imported at the top of this file via:
#           from src.dataset import KneeMRIDataset
#           from src.model   import KneeClassifier
#           from src.losses  import compute_bce_loss
#           from src.metrics import compute_macro_auc
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────
# Training & Validation Epochs
# ─────────────────────────────────────────────
def train_one_epoch(
    model:       nn.Module,
    loader:      DataLoader,
    optimizer:   torch.optim.Optimizer,
    criterion:   nn.Module,
    scaler:      GradScaler,
    device:      torch.device,
    has_weight:  bool = False,
    accum_steps: int  = 1,
) -> float:
    """
    Run one full training epoch with optional gradient accumulation.

    Gradient accumulation (`accum_steps > 1`) lets you train with an
    effectively larger batch size without increasing VRAM usage:
        effective_batch = CFG.BATCH_SIZE × accum_steps

    Returns
    -------
    float  Mean training loss for the epoch.
    """
    model.train()
    total_loss   = 0.0
    accum_steps  = max(accum_steps, 1)
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(tqdm(loader, desc="  Train", leave=False)):
        if has_weight:
            volumes, labels, weights = batch
            weights = weights.to(device, non_blocking=True)
        else:
            volumes, labels = batch
            weights = None

        volumes = volumes.to(device, non_blocking=True)
        labels  = labels.to(device,  non_blocking=True)

        with autocast(dtype=torch.float16, enabled=CFG.MIXED_PREC):
            logits = model(volumes)                                       # [B, 12]
            # Scale loss by accum_steps so the gradient magnitude stays
            # equivalent to a single large-batch update
            loss   = compute_bce_loss(criterion, logits, labels, weights)
            loss   = loss / accum_steps

        scaler.scale(loss).backward()
        total_loss += loss.item() * accum_steps   # un-scale for logging

        # Update weights only after accumulating `accum_steps` mini-batches
        is_last_batch  = (step + 1) == len(loader)
        should_update  = ((step + 1) % accum_steps == 0) or is_last_batch
        if should_update:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate_one_epoch(
    model:      nn.Module,
    loader:     DataLoader,
    criterion:  nn.Module,
    device:     torch.device,
    has_weight: bool = False,
) -> Tuple[float, float, dict]:
    """Run one validation epoch. Returns (val_loss, macro_auc, per_class_auc)."""
    model.eval()
    total_loss  = 0.0
    all_preds   = []
    all_targets = []

    for batch in tqdm(loader, desc="  Val  ", leave=False):
        if has_weight:
            volumes, labels, weights = batch
            weights = weights.to(device, non_blocking=True)
        else:
            volumes, labels = batch
            weights = None

        volumes = volumes.to(device, non_blocking=True)
        labels  = labels.to(device,  non_blocking=True)

        with autocast(dtype=torch.float16, enabled=CFG.MIXED_PREC):
            logits = model(volumes)
            loss   = compute_bce_loss(criterion, logits, labels, weights)

        probs = torch.sigmoid(logits).cpu().float().numpy()
        all_preds.append(probs)
        all_targets.append(labels.cpu().float().numpy())
        total_loss += loss.item()

    val_loss             = total_loss / max(len(loader), 1)
    y_pred               = np.concatenate(all_preds,   axis=0)
    y_true               = np.concatenate(all_targets, axis=0)
    macro_auc, per_class = compute_macro_auc(y_true, y_pred)

    return val_loss, macro_auc, per_class


# ─────────────────────────────────────────────
# Main Training Loop
# ─────────────────────────────────────────────
def run_training(args: argparse.Namespace) -> None:
    seed_everything(CFG.SEED)

    # ── Override config from CLI args ───────────────────────────
    CFG.EPOCHS       = args.epochs
    CFG.BATCH_SIZE   = args.batch_size
    CFG.SEED         = args.seed
    CFG.FOLD         = args.fold
    CFG.ACCUM_STEPS  = args.accum_steps

    # ── Device & AMP setup ───────────────────────────────────────
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp   = CFG.MIXED_PREC and device.type == "cuda"
    n_gpus    = torch.cuda.device_count() if device.type == "cuda" else 0

    # ── Ensure outputs directory exists ─────────────────────────
    outputs_dir = Path(CFG.OUTPUTS_DIR)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # ── Startup banner ───────────────────────────────────────────
    eff_batch = CFG.BATCH_SIZE * CFG.ACCUM_STEPS * max(n_gpus, 1)
    print(f"\n{'='*65}")
    print(f"  Device              : {device}")
    if device.type == "cuda":
        for i in range(n_gpus):
            vram_gb = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}               : {torch.cuda.get_device_name(i)}  ({vram_gb:.0f} GB)")
    print(f"  GPUs used           : {max(n_gpus, 1)}")
    print(f"  Mixed Precision     : {use_amp}  (fp16 autocast + GradScaler)")
    print(f"  cuDNN benchmark     : {torch.backends.cudnn.benchmark}")
    print(f"  TF32 matmul prec.   : {torch.get_float32_matmul_precision()}")
    print(f"  Epochs              : {CFG.EPOCHS}")
    print(f"  Batch size          : {CFG.BATCH_SIZE}  (per GPU)")
    print(f"  Accum steps         : {CFG.ACCUM_STEPS}")
    print(f"  Effective batch     : {eff_batch}")
    print(f"  Fold                : {CFG.FOLD} / {CFG.N_FOLDS}")
    print(f"  Num workers         : {CFG.NUM_WORKERS}")
    print(f"  Prefetch factor     : {CFG.PREFETCH_FACTOR}")
    print(f"  Outputs dir         : {outputs_dir.resolve()}")
    print(f"{'='*65}\n")

    # ── Load metadata (all ~4,407 studies, no NaN filter) ───────
    df = load_full_metadata(CFG.TRAIN_CSV, CFG.SERIES_CSV, CFG.LABELS)
    has_weight = "sample_weight" in df.columns

    # ── 5-Fold GroupKFold split on StudyInstanceUID ──────────────
    train_df, val_df = get_fold_split(df, fold=CFG.FOLD, n_folds=CFG.N_FOLDS)

    # ── Datasets & Loaders ──────────────────────────────────────
    ds_kwargs = dict(
        base_dir    = CFG.BASE_IMAGE_DIR,
        labels      = CFG.LABELS,
        fixed_depth = CFG.FIXED_DEPTH,
        img_h       = CFG.IMG_HEIGHT,
        img_w       = CFG.IMG_WIDTH,
    )
    train_ds = KneeMRIDataset(train_df, augment=True,  **ds_kwargs)
    val_ds   = KneeMRIDataset(val_df,   augment=False, **ds_kwargs)

    loader_kwargs = dict(
        batch_size            = CFG.BATCH_SIZE,
        num_workers           = CFG.NUM_WORKERS,
        pin_memory            = device.type == "cuda",
        persistent_workers    = CFG.NUM_WORKERS > 0,  # avoid per-epoch worker respawn
        prefetch_factor       = CFG.PREFETCH_FACTOR if CFG.NUM_WORKERS > 0 else None,
        # spawn is required on Windows to avoid CUDA fork issues with multiprocessing
        multiprocessing_context = "spawn" if CFG.NUM_WORKERS > 0 else None,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  drop_last=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, drop_last=False, **loader_kwargs)

    print(f"[Loaders] Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}\n")

    # ── Model ───────────────────────────────────────────────────
    model = KneeClassifier(
        num_classes   = CFG.NUM_CLASSES,
        backbone_name = CFG.BACKBONE,
        pretrained    = CFG.PRETRAINED,
        pool_mode     = CFG.POOL_MODE,
    ).to(device)

    # ── Multi-GPU: DataParallel ──────────────────────────────────
    if n_gpus > 1:
        print(f"[Model] Wrapping with DataParallel across {n_gpus} GPUs")
        model = nn.DataParallel(model)

    # ── torch.compile (PyTorch ≥ 2.0) ───────────────────────────
    # 'reduce-overhead' mode traces the graph once and replays a CUDA graph,
    # eliminating Python overhead in subsequent steps (~20-40 % speedup).
    # The first epoch is slower while compilation runs — this is expected.
    try:
        print("[Model] Compiling with torch.compile(mode='reduce-overhead') ...")
        model = torch.compile(model, mode="reduce-overhead")
        print("[Model] Compilation scheduled  (first epoch will be slower)")
    except Exception as e:
        print(f"[Model] torch.compile unavailable ({e}), running in eager mode")

    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Total params    : {total_params:,}")
    print(f"[Model] Trainable params: {trainable:,}\n")

    # ── Loss / Optimizer / Scheduler / Scaler ───────────────────
    # BCEWithLogitsLoss natively supports continuous targets in [0, 1].
    # reduction='mean' is used when no sample weights; element-wise
    # weighted reduction is handled in compute_bce_loss().
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG.EPOCHS, eta_min=CFG.LR * 0.01
    )
    scaler = GradScaler(enabled=use_amp)

    # ── Training Loop ───────────────────────────────────────────
    best_auc     = -1.0
    best_weights = None
    history      = []
    per_class    = {lbl: float("nan") for lbl in CFG.LABELS}
    ckpt_path    = outputs_dir / f"best_model_fold{CFG.FOLD}.pt"

    for epoch in range(1, CFG.EPOCHS + 1):
        print(f"Epoch [{epoch:03d}/{CFG.EPOCHS:03d}]  Fold {CFG.FOLD}")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device,
            has_weight=has_weight, accum_steps=CFG.ACCUM_STEPS,
        )
        val_loss, macro_auc, per_class = validate_one_epoch(
            model, val_loader, criterion, device, has_weight
        )
        scheduler.step()

        # Track best model by macro AUC
        if macro_auc > best_auc:
            best_auc     = macro_auc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            # Save immediately so training can be interrupted safely
            torch.save(
                {
                    "fold":        CFG.FOLD,
                    "epoch":       epoch,
                    "best_auc":    best_auc,
                    "model_state": best_weights,
                    "config": {
                        "backbone":    CFG.BACKBONE,
                        "pool_mode":   CFG.POOL_MODE,
                        "fixed_depth": CFG.FIXED_DEPTH,
                        "num_classes": CFG.NUM_CLASSES,
                        "labels":      CFG.LABELS,
                    },
                },
                ckpt_path,
            )
            tag = f" <- best  [saved -> {ckpt_path}]"
        else:
            tag = ""

        print(
            f"  train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"macro_AUC={macro_auc:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
            f"{tag}"
        )

        # Per-class AUC (summarised on one line)
        valid_pairs = [(k, v) for k, v in per_class.items() if not np.isnan(v)]
        if valid_pairs:
            auc_str = "  " + "  ".join(f"{k}:{v:.3f}" for k, v in valid_pairs)
            print(auc_str)

        history.append({
            "fold":       CFG.FOLD,
            "epoch":      epoch,
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "macro_auc":  macro_auc,
            **per_class,
        })

    # ── Print final summary ─────────────────────────────────────
    print("\n" + "="*60)
    print(f"  Fold {CFG.FOLD}  |  Best macro ROC-AUC : {best_auc:.4f}")
    print("  Per-class AUC (final val epoch):")
    for lbl, auc in per_class.items():
        marker = "" if not np.isnan(auc) else "  [skipped - single class]"
        val    = f"{auc:.4f}" if not np.isnan(auc) else "N/A"
        print(f"    {lbl:<20} {val}{marker}")
    print("="*60)

    # ── Save training history ────────────────────────────────────
    hist_csv = outputs_dir / f"training_history_fold{CFG.FOLD}.csv"
    hist_df  = pd.DataFrame(history)
    hist_df.to_csv(hist_csv, index=False)
    print(f"\n[History] Saved -> {hist_csv}")


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RSNA Knee MRI — multi-label classification (soft labels + GroupKFold)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs",      type=int,   default=CFG.EPOCHS,
                        help="Number of training epochs")
    parser.add_argument("--batch_size",  type=int,   default=CFG.BATCH_SIZE,
                        help="Per-GPU mini-batch size")
    parser.add_argument("--accum_steps", type=int,   default=CFG.ACCUM_STEPS,
                        help="Gradient accumulation steps (effective_batch = batch_size × accum_steps × n_gpus)")
    parser.add_argument("--seed",        type=int,   default=CFG.SEED,
                        help="Global random seed")
    parser.add_argument("--lr",          type=float, default=CFG.LR,
                        help="Initial learning rate (AdamW)")
    parser.add_argument("--pool_mode",   type=str,   default=CFG.POOL_MODE,
                        choices=["mean", "max"],
                        help="Depth aggregation mode")
    parser.add_argument("--fold",        type=int,   default=CFG.FOLD,
                        help=f"Fold index to train (0-indexed, 0-{CFG.N_FOLDS - 1})")
    return parser.parse_args()


if __name__ == "__main__":
    args          = parse_args()
    CFG.LR        = args.lr
    CFG.POOL_MODE = args.pool_mode
    run_training(args)
