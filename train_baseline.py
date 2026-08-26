"""
train_baseline.py
=================
Modular PyTorch baseline for multi-label knee abnormality classification
using 3D MRI volumes (RSNA dataset).

Architecture : 2.5D CNN
  - Pretrained ResNet-34 backbone applied independently across 24 slices
  - Mean-pooling over the depth dimension → study embedding
  - Single linear head → 12 binary logits

Usage
-----
    python train_baseline.py [--epochs 30] [--batch_size 4] [--seed 42]

Requirements
------------
    pip install torch torchvision scikit-learn pandas numpy tqdm
"""

# ─────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────
import argparse
import os
import random
import warnings
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
class Config:
    # ── Paths ──────────────────────────────────
    BASE_IMAGE_DIR: str  = "D:/rsna_knee_dataset"
    TRAIN_CSV:      str  = "train.csv"
    SERIES_CSV:     str  = "train_series.csv"

    # ── Labels ─────────────────────────────────
    LABELS: List[str] = [
        "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
        "Medial OA", "Lateral OA", "PF OA", "Effusion",
        "Synovitis", "Baker's", "Contusion", "Fracture",
    ]
    NUM_CLASSES: int = len(LABELS)   # 12

    # ── Volume settings ─────────────────────────
    FIXED_DEPTH:   int   = 24        # slices kept per volume
    IMG_HEIGHT:    int   = 224
    IMG_WIDTH:     int   = 224

    # ── Training ────────────────────────────────
    EPOCHS:        int   = 30
    BATCH_SIZE:    int   = 4
    LR:            float = 1e-4
    WEIGHT_DECAY:  float = 1e-2
    VAL_SPLIT:     float = 0.20
    SEED:          int   = 42
    NUM_WORKERS:   int   = 2
    MIXED_PREC:    bool  = True      # AMP / fp16

    # ── Model ───────────────────────────────────
    BACKBONE:      str   = "resnet34"
    PRETRAINED:    bool  = True
    POOL_MODE:     str   = "mean"    # "mean" | "max"


CFG = Config()


# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────
def seed_everything(seed: int = CFG.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────
# Data Loading & Metadata
# ─────────────────────────────────────────────
def load_gold_metadata(
    train_csv: str = CFG.TRAIN_CSV,
    series_csv: str = CFG.SERIES_CSV,
    labels: List[str] = CFG.LABELS,
) -> pd.DataFrame:
    """
    Load train.csv and train_series.csv, filter to the 58
    Gold Ground Truth studies (no NaN in any of the 12 label columns),
    and join with per-series metadata.

    Returns
    -------
    df : pd.DataFrame
        Columns: StudyInstanceUID, SeriesInstanceUID, <labels...>
    """
    train_df  = pd.read_csv(train_csv)
    series_df = pd.read_csv(series_csv)

    # ── Gold filter: keep only rows with all 12 labels present ──
    gold_df = train_df.dropna(subset=labels).reset_index(drop=True)
    print(f"[Data] Gold studies found: {len(gold_df)}")

    # ── Join with series to get SeriesInstanceUIDs ──────────────
    merged = gold_df.merge(
        series_df[["StudyInstanceUID", "SeriesInstanceUID"]],
        on="StudyInstanceUID",
        how="inner",
    )

    # Cast labels to int
    merged[labels] = merged[labels].astype(int)

    print(f"[Data] Total study-series rows after join: {len(merged)}")
    print(f"[Data] Unique studies : {merged['StudyInstanceUID'].nunique()}")
    print(f"[Data] Unique series  : {merged['SeriesInstanceUID'].nunique()}")
    return merged


# ─────────────────────────────────────────────
# Custom PyTorch Dataset
# ─────────────────────────────────────────────
class KneeMRIDataset(Dataset):
    """
    Dataset for multi-label knee MRI classification.

    Each item is one (SeriesInstanceUID, StudyInstanceUID) pair.
    The volume is loaded from  <base_dir>/<SeriesInstanceUID>.npz,
    resampled to FIXED_DEPTH slices, resized spatially, and returned
    as a float32 tensor of shape [1, D, H, W].
    """

    def __init__(
        self,
        df: pd.DataFrame,
        base_dir: str        = CFG.BASE_IMAGE_DIR,
        labels: List[str]    = CFG.LABELS,
        fixed_depth: int     = CFG.FIXED_DEPTH,
        img_h: int           = CFG.IMG_HEIGHT,
        img_w: int           = CFG.IMG_WIDTH,
        augment: bool        = False,
    ) -> None:
        self.df          = df.reset_index(drop=True)
        self.base_dir    = Path(base_dir)
        self.labels      = labels
        self.fixed_depth = fixed_depth
        self.img_h       = img_h
        self.img_w       = img_w
        self.augment     = augment

    # ────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.df)

    # ────────────────────────────────────────
    def _load_volume(self, series_uid: str) -> np.ndarray:
        """Load .npz, return float32 array of shape [D, H, W]."""
        path = self.base_dir / f"{series_uid}.npz"
        with np.load(path) as f:
            # Try common key names; fall back to the first key
            key = "arr_0" if "arr_0" in f else list(f.keys())[0]
            vol = f[key].astype(np.float32)   # [D, H, W]
        return vol

    # ────────────────────────────────────────
    def _normalize(self, vol: np.ndarray) -> np.ndarray:
        """Min-max scale to [0, 1]. Handle flat volumes gracefully."""
        v_min, v_max = vol.min(), vol.max()
        if v_max - v_min < 1e-6:
            return np.zeros_like(vol)
        return (vol - v_min) / (v_max - v_min)

    # ────────────────────────────────────────
    def _sample_depth(self, vol: np.ndarray) -> np.ndarray:
        """
        Uniformly sample / interpolate along depth axis to FIXED_DEPTH.
        vol : [D, H, W]  ->  output : [FIXED_DEPTH, H, W]
        """
        D = vol.shape[0]
        if D == self.fixed_depth:
            return vol
        # Uniform index sampling (works for both up- and down-sampling)
        indices = np.linspace(0, D - 1, self.fixed_depth)
        lo = np.floor(indices).astype(int)
        hi = np.minimum(lo + 1, D - 1)
        w  = (indices - lo).astype(np.float32)          # interpolation weight
        sampled = (
            vol[lo] * (1.0 - w[:, None, None]) +
            vol[hi] *  w[:, None, None]
        )
        return sampled.astype(np.float32)

    # ────────────────────────────────────────
    def _resize_slices(self, vol: np.ndarray) -> torch.Tensor:
        """
        Resize H x W of every slice to (img_h, img_w) using trilinear interp.
        vol  : [D, H, W]  ->  tensor : [D, img_h, img_w]
        """
        t = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0)   # [1,1,D,H,W]
        t = F.interpolate(
            t,
            size=(self.fixed_depth, self.img_h, self.img_w),
            mode="trilinear",
            align_corners=False,
        )
        return t.squeeze(0).squeeze(0)   # [D, img_h, img_w]

    # ────────────────────────────────────────
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row        = self.df.iloc[idx]
        series_uid = str(row["SeriesInstanceUID"])
        label_vec  = row[self.labels].values.astype(np.float32)

        # Load -> normalise -> sample depth -> resize spatial
        vol  = self._load_volume(series_uid)     # [D, H, W]
        vol  = self._normalize(vol)              # [D, H, W]  in [0,1]
        vol  = self._sample_depth(vol)           # [fixed_depth, H, W]
        vol  = self._resize_slices(vol)          # [fixed_depth, img_h, img_w]

        # Optional simple augmentation (horizontal flip)
        if self.augment and random.random() > 0.5:
            vol = vol.flip(dims=[-1])

        # Add channel dim -> [1, D, H, W]
        volume_tensor = vol.unsqueeze(0)
        label_tensor  = torch.from_numpy(label_vec)

        return volume_tensor, label_tensor


# ─────────────────────────────────────────────
# 2.5D Model
# ─────────────────────────────────────────────
class SliceEncoder(nn.Module):
    """
    Thin wrapper around a 2D backbone that adapts its first conv layer
    to accept single-channel (greyscale) MRI slices.
    """

    def __init__(self, backbone_name: str = CFG.BACKBONE, pretrained: bool = CFG.PRETRAINED):
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        backbone = getattr(models, backbone_name)(weights=weights)

        # ── Patch first conv: 3-ch RGB -> 1-ch greyscale ────────
        orig_conv  = backbone.conv1
        self.conv1 = nn.Conv2d(
            1, orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=False,
        )
        # Initialise by averaging the pretrained RGB weights
        if pretrained:
            with torch.no_grad():
                self.conv1.weight.copy_(orig_conv.weight.mean(dim=1, keepdim=True))
        else:
            nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")

        # ── Rest of backbone (bn1, relu, maxpool, layer1-4, avgpool) ──
        self.bn1    = backbone.bn1
        self.relu   = backbone.relu
        self.maxpool= backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool= backbone.avgpool

        self.embed_dim = backbone.fc.in_features   # 512 for resnet34

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : [B, 1, H, W]  ->  features : [B, embed_dim]"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)   # [B, embed_dim]


class KneeClassifier(nn.Module):
    """
    2.5D CNN for multi-label knee abnormality classification.

    Forward pass:
      1. Reshape [B, 1, D, H, W] -> [B*D, 1, H, W]
      2. Encode each slice with SliceEncoder -> [B*D, embed_dim]
      3. Reshape -> [B, D, embed_dim]
      4. Pool over depth -> [B, embed_dim]
      5. Classify -> [B, num_classes]
    """

    def __init__(
        self,
        num_classes: int = CFG.NUM_CLASSES,
        backbone_name: str  = CFG.BACKBONE,
        pretrained: bool    = CFG.PRETRAINED,
        pool_mode: str      = CFG.POOL_MODE,
        dropout: float      = 0.3,
    ) -> None:
        super().__init__()
        self.pool_mode    = pool_mode
        self.encoder      = SliceEncoder(backbone_name, pretrained)
        embed_dim         = self.encoder.embed_dim

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, 1, D, H, W]
        returns logits : [B, num_classes]
        """
        B, C, D, H, W = x.shape

        # Fold depth into batch
        x = x.permute(0, 2, 1, 3, 4)          # [B, D, 1, H, W]
        x = x.reshape(B * D, C, H, W)          # [B*D, 1, H, W]

        feats = self.encoder(x)                 # [B*D, embed_dim]
        feats = feats.view(B, D, -1)            # [B, D, embed_dim]

        # Depth aggregation
        if self.pool_mode == "max":
            feats, _ = feats.max(dim=1)
        else:
            feats = feats.mean(dim=1)           # [B, embed_dim]

        return self.classifier(feats)           # [B, num_classes]


# ─────────────────────────────────────────────
# Evaluation Metric
# ─────────────────────────────────────────────
def compute_macro_auc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str] = CFG.LABELS,
) -> Tuple[float, dict]:
    """
    Compute per-class and macro-averaged ROC-AUC.
    Classes with only one unique label value in y_true are skipped (NaN handled).

    Parameters
    ----------
    y_true : [N, num_classes]  binary ground truth
    y_pred : [N, num_classes]  predicted probabilities (after sigmoid)

    Returns
    -------
    macro_auc : float
    per_class  : dict {label: auc}
    """
    per_class = {}
    valid_aucs = []

    for i, lbl in enumerate(labels):
        col_true = y_true[:, i]
        col_pred = y_pred[:, i]

        if len(np.unique(col_true)) < 2:
            # Only one class present -> AUC undefined
            per_class[lbl] = float("nan")
            continue

        try:
            auc = roc_auc_score(col_true, col_pred)
            per_class[lbl] = auc
            valid_aucs.append(auc)
        except Exception:
            per_class[lbl] = float("nan")

    macro_auc = float(np.mean(valid_aucs)) if valid_aucs else float("nan")
    return macro_auc, per_class


# ─────────────────────────────────────────────
# Training & Validation Epochs
# ─────────────────────────────────────────────
def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler:    GradScaler,
    device:    torch.device,
) -> float:
    """Run one training epoch. Returns mean loss."""
    model.train()
    total_loss = 0.0

    for volumes, labels in tqdm(loader, desc="  Train", leave=False):
        volumes = volumes.to(device, non_blocking=True)
        labels  = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=CFG.MIXED_PREC):
            logits = model(volumes)             # [B, 12]
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> Tuple[float, float, dict]:
    """Run one validation epoch. Returns (val_loss, macro_auc, per_class_auc)."""
    model.eval()
    total_loss  = 0.0
    all_preds   = []
    all_targets = []

    for volumes, labels in tqdm(loader, desc="  Val  ", leave=False):
        volumes = volumes.to(device, non_blocking=True)
        labels  = labels.to(device, non_blocking=True)

        with autocast(enabled=CFG.MIXED_PREC):
            logits = model(volumes)
            loss   = criterion(logits, labels)

        probs = torch.sigmoid(logits).cpu().float().numpy()
        all_preds.append(probs)
        all_targets.append(labels.cpu().float().numpy())
        total_loss += loss.item()

    val_loss    = total_loss / max(len(loader), 1)
    y_pred      = np.concatenate(all_preds,   axis=0)
    y_true      = np.concatenate(all_targets, axis=0)
    macro_auc, per_class = compute_macro_auc(y_true, y_pred)

    return val_loss, macro_auc, per_class


# ─────────────────────────────────────────────
# Main Training Loop
# ─────────────────────────────────────────────
def run_training(args: argparse.Namespace) -> None:
    seed_everything(CFG.SEED)

    # Override config from CLI args
    CFG.EPOCHS     = args.epochs
    CFG.BATCH_SIZE = args.batch_size
    CFG.SEED       = args.seed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Device         : {device}")
    if device.type == "cuda":
        print(f"  GPU            : {torch.cuda.get_device_name(0)}")
    print(f"  Mixed Precision: {CFG.MIXED_PREC and device.type == 'cuda'}")
    print(f"  Epochs         : {CFG.EPOCHS}")
    print(f"  Batch size     : {CFG.BATCH_SIZE}")
    print(f"{'='*60}\n")

    # ── Load metadata ───────────────────────────────────────────
    df = load_gold_metadata(CFG.TRAIN_CSV, CFG.SERIES_CSV, CFG.LABELS)

    # ── Study-level 80/20 split ──────────────────────────────────
    unique_studies = df["StudyInstanceUID"].unique()
    train_studies, val_studies = train_test_split(
        unique_studies,
        test_size=CFG.VAL_SPLIT,
        random_state=CFG.SEED,
    )
    print(f"[Split] Train studies: {len(train_studies)}  |  Val studies: {len(val_studies)}")

    train_df = df[df["StudyInstanceUID"].isin(train_studies)].reset_index(drop=True)
    val_df   = df[df["StudyInstanceUID"].isin(val_studies)  ].reset_index(drop=True)
    print(f"[Split] Train series : {len(train_df)}  |  Val series : {len(val_df)}\n")

    # ── Datasets & Loaders ──────────────────────────────────────
    train_ds = KneeMRIDataset(train_df, augment=True)
    val_ds   = KneeMRIDataset(val_df,   augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=CFG.BATCH_SIZE,
        shuffle=True,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CFG.BATCH_SIZE,
        shuffle=False,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )

    # ── Model ───────────────────────────────────────────────────
    model = KneeClassifier(
        num_classes   = CFG.NUM_CLASSES,
        backbone_name = CFG.BACKBONE,
        pretrained    = CFG.PRETRAINED,
        pool_mode     = CFG.POOL_MODE,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Total params    : {total_params:,}")
    print(f"[Model] Trainable params: {trainable:,}\n")

    # ── Loss / Optimizer / Scheduler / Scaler ───────────────────
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG.EPOCHS, eta_min=CFG.LR * 0.01
    )
    scaler = GradScaler(enabled=CFG.MIXED_PREC and device.type == "cuda")

    # ── Training Loop ───────────────────────────────────────────
    best_auc      = -1.0
    best_weights  = None
    history       = []
    per_class     = {lbl: float("nan") for lbl in CFG.LABELS}

    for epoch in range(1, CFG.EPOCHS + 1):
        print(f"Epoch [{epoch:03d}/{CFG.EPOCHS:03d}]")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device)
        val_loss, macro_auc, per_class = validate_one_epoch(model, val_loader, criterion, device)
        scheduler.step()

        # Track best model
        if macro_auc > best_auc:
            best_auc     = macro_auc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            tag = " <- best"
        else:
            tag = ""

        print(
            f"  train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"macro_AUC={macro_auc:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
            f"{tag}"
        )

        # Per-class AUC (summarised)
        valid_pairs = [(k, v) for k, v in per_class.items() if not np.isnan(v)]
        if valid_pairs:
            auc_str = "  " + "  ".join(f"{k}:{v:.3f}" for k, v in valid_pairs)
            print(auc_str)

        history.append({
            "epoch":      epoch,
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "macro_auc":  macro_auc,
            **per_class,
        })

    # ── Save best checkpoint ────────────────────────────────────
    ckpt_path = "best_knee_model.pth"
    if best_weights is not None:
        torch.save(
            {
                "epoch":       CFG.EPOCHS,
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
        print(f"\n[Checkpoint] Best model saved -> {ckpt_path}  (AUC={best_auc:.4f})")

    # ── Print final summary ─────────────────────────────────────
    print("\n" + "="*60)
    print(f"  Best macro ROC-AUC : {best_auc:.4f}")
    print("  Per-class AUC (final val epoch):")
    for lbl, auc in per_class.items():
        marker = "" if not np.isnan(auc) else "  [skipped - single class]"
        val    = f"{auc:.4f}" if not np.isnan(auc) else "N/A"
        print(f"    {lbl:<20} {val}{marker}")
    print("="*60)

    # ── Save training history ────────────────────────────────────
    hist_df = pd.DataFrame(history)
    hist_df.to_csv("training_history.csv", index=False)
    print("\n[History] Saved -> training_history.csv")


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-label knee MRI classification - PyTorch baseline"
    )
    parser.add_argument("--epochs",     type=int,   default=CFG.EPOCHS,     help="Number of training epochs")
    parser.add_argument("--batch_size", type=int,   default=CFG.BATCH_SIZE, help="Batch size")
    parser.add_argument("--seed",       type=int,   default=CFG.SEED,       help="Random seed")
    parser.add_argument("--lr",         type=float, default=CFG.LR,         help="Learning rate")
    parser.add_argument("--pool_mode",  type=str,   default=CFG.POOL_MODE,  choices=["mean", "max"],
                        help="Depth aggregation: mean or max pooling")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    CFG.LR        = args.lr
    CFG.POOL_MODE = args.pool_mode
    run_training(args)
