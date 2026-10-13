"""
src_v3/dataset.py
=================
Series-level PyTorch Dataset and GPU augmentation pipeline for RSNA Knee MRI v3.

Key design points (Model-3 recipe):
  - One item = ONE SERIES: uint8 volume [24,224,224] from the memmap cache.
  - Training: all series of training studies, one series per __getitem__ call.
  - Validation: all series of validation studies, deterministic order.
  - Workers return ONLY uint8 tensors (float ops deferred to GPU).
  - GPU normalisation: 0.5th–99.5th percentile clip + [0,1] scale + ImageNet
    mean/std – EXACTLY replicating src_v2/dataset.py gpu_normalize_volumes.
  - 2.5D tri-slice construction on GPU: slice d -> channels [d-1, d, d+1].
  - Train-time affine + photometric augmentation on GPU with ONE random
    parameter set shared across ALL 24 slices of a series.
  - Horizontal flip p=0.5 for Coronal/Axial ONLY, never Sagittal.
  - No augmentation at validation.
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# ── Plane / label constants ───────────────────────────────────────────────────
PLANE_TO_ID: Dict[str, int] = {"Sagittal": 0, "Coronal": 1, "Axial": 2}
ID_TO_PLANE: Dict[int, str] = {0: "Sagittal", 1: "Coronal", 2: "Axial"}

LABELS: List[str] = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]


# ─────────────────────────────────────────────────────────────────────────────
# Series-Level Dataset
# ─────────────────────────────────────────────────────────────────────────────
class SeriesDataset(Dataset):
    """
    Each item is ONE series.

    Parameters
    ----------
    series_index : pd.DataFrame
        outputs_v3/series_index.csv columns:
        row_index, SeriesInstanceUID, StudyInstanceUID, plane, Fluid_Sensitive, fold
    targets_df : pd.DataFrame
        train_targets_soft.csv with StudyInstanceUID + 12 label cols.
    memmap_path : str
        Path to D:/rsna_cache/volumes_uint8.npy  [N,24,224,224] uint8 memmap.
    fold_ids : set of int
        Which fold values are included in this split (e.g. {0} for val fold 0).
    is_val_split : bool
        If True this is the validation set (subset matching fold_ids).
        If False this is the training set (all folds NOT in fold_ids).
    labels : list of str
    """

    def __init__(
        self,
        series_index: pd.DataFrame,
        targets_df: pd.DataFrame,
        memmap_path: str,
        fold_ids: set,
        is_val_split: bool,
        labels: Optional[List[str]] = None,
    ) -> None:
        self.labels = labels or LABELS
        self.is_val = is_val_split

        # Filter rows to the correct split
        if is_val_split:
            mask = series_index["fold"].isin(fold_ids)
        else:
            mask = ~series_index["fold"].isin(fold_ids)
        self.rows = series_index[mask].reset_index(drop=True)

        # Build study -> target lookup
        tgt_cols = [c for c in self.labels if c in targets_df.columns]
        self._target_lookup: Dict[str, np.ndarray] = {}
        for _, r in targets_df.iterrows():
            uid = str(r["StudyInstanceUID"])
            self._target_lookup[uid] = r[tgt_cols].values.astype(np.float32)

        # Open memmap read-only (shared across all workers)
        self._memmap_path = memmap_path
        self._memmap: Optional[np.ndarray] = None  # opened lazily per worker

    def _get_memmap(self) -> np.ndarray:
        if self._memmap is None:
            self._memmap = np.load(self._memmap_path, mmap_mode="r")
        return self._memmap

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict:
        row = self.rows.iloc[idx]
        row_index = int(row["row_index"])
        study_uid = str(row["StudyInstanceUID"])
        series_uid = str(row["SeriesInstanceUID"])
        plane_id = int(row["plane"])
        fluid = int(row["Fluid_Sensitive"])

        # Read uint8 volume from memmap — no float conversion in worker!
        vol_np = self._get_memmap()[row_index]          # [24, 224, 224] uint8
        vol_tensor = torch.from_numpy(vol_np.copy())    # copy avoids memmap pitfalls

        target = self._target_lookup.get(study_uid, np.zeros(len(self.labels), np.float32))
        target_tensor = torch.from_numpy(target)

        return {
            "volume": vol_tensor,      # uint8 [24, 224, 224]
            "plane": plane_id,         # int
            "fluid": fluid,            # int
            "target": target_tensor,   # float32 [12]
            "series_uid": series_uid,
            "study_uid": study_uid,
            "row_index": row_index,
        }


def collate_series_fn(batch: List[Dict]) -> Dict:
    """
    Stack a list of series items into a batch. All tensors are kept uint8.
    """
    vols = torch.stack([b["volume"] for b in batch], dim=0)      # [B,24,224,224] uint8
    planes = torch.tensor([b["plane"] for b in batch], dtype=torch.long)
    fluids = torch.tensor([b["fluid"] for b in batch], dtype=torch.long)
    targets = torch.stack([b["target"] for b in batch], dim=0)   # [B,12] float32
    series_uids = [b["series_uid"] for b in batch]
    study_uids = [b["study_uid"] for b in batch]
    row_indices = [b["row_index"] for b in batch]

    return {
        "volume": vols,
        "plane": planes,
        "fluid": fluids,
        "target": targets,
        "series_uid": series_uids,
        "study_uid": study_uids,
        "row_index": row_indices,
    }


# ─────────────────────────────────────────────────────────────────────────────
# GPU Normalisation  (identical logic to src_v2/dataset.py gpu_normalize_volumes)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def gpu_normalize_volume(vols_uint8: torch.Tensor) -> torch.Tensor:
    """
    Convert uint8 volume batch [B,24,H,W] to float32 with per-volume
    0.5th–99.5th percentile clipping + [0,1] scaling + ImageNet mean/std.

    Parameters
    ----------
    vols_uint8 : [B, 24, H, W]  uint8  on GPU

    Returns
    -------
    [B, 24, H, W]  float32  on GPU
    """
    B, D, H, W = vols_uint8.shape
    vols = vols_uint8.float()               # cast once on GPU

    flat = vols.view(B, -1)                 # [B, 24*H*W]
    q = torch.tensor([0.005, 0.995], device=vols.device, dtype=torch.float32)
    pcts = torch.quantile(flat, q, dim=1)   # [2, B]
    p_low  = pcts[0].view(B, 1, 1, 1)
    p_high = pcts[1].view(B, 1, 1, 1)

    denom = (p_high - p_low).clamp(min=1e-6)
    vols = (vols - p_low).clamp(min=0.0) / denom   # [0, 1]

    # ImageNet single-channel: mean 0.449, std 0.226
    vols = (vols - 0.449) / 0.226

    return vols   # [B, 24, H, W] float32


# ─────────────────────────────────────────────────────────────────────────────
# 2.5D Tri-slice Construction
# ─────────────────────────────────────────────────────────────────────────────
def build_25d_slices(vols: torch.Tensor) -> torch.Tensor:
    """
    Build 2.5D 3-channel input: for slice d use [d-1, d, d+1] with edge
    replication.

    Parameters
    ----------
    vols : [B, 24, H, W]  float32

    Returns
    -------
    [B*24, 3, H, W]  float32
    """
    B, D, H, W = vols.shape
    # Replicate-pad along depth: [B, 1, 24, H, W] -> [B, 1, 26, H, W]
    v = vols.unsqueeze(1)
    vp = F.pad(v, (0, 0, 0, 0, 1, 1), mode="replicate").squeeze(1)  # [B, 26, H, W]

    prev = vp[:, 0:D]    # [B, D, H, W]
    curr = vp[:, 1:D+1]
    nxt  = vp[:, 2:D+2]

    # Stack channels: [B, D, 3, H, W]
    tri = torch.stack([prev, curr, nxt], dim=2)  # [B, D, 3, H, W]
    return tri.view(B * D, 3, H, W)              # [B*24, 3, H, W]


# ─────────────────────────────────────────────────────────────────────────────
# GPU Augmentation (per-series, shared across all 24 slices)
# ─────────────────────────────────────────────────────────────────────────────
class GPUSeriesAugmentor(nn.Module):
    """
    GPU augmentation for a batch of series volumes [B, 24, H, W].

    One random parameter set is drawn PER SERIES and applied consistently
    to all 24 slices, treating depth as the batch dimension for affine_grid.

    Augmentations:
      - Affine: rotation ±8°, scale 0.95–1.05, translation ±4%
      - Brightness/contrast ±12%
      - Gamma 0.85–1.20
      - Light Gaussian noise
      - Horizontal flip p=0.5 for Coronal/Axial ONLY
    """

    def __init__(self) -> None:
        super().__init__()

    @torch.no_grad()
    def forward(
        self,
        vols: torch.Tensor,        # [B, 24, H, W] float32 on GPU
        planes: torch.Tensor,      # [B] long on GPU
    ) -> torch.Tensor:
        B, D, H, W = vols.shape
        device = vols.device

        # ── Affine parameters (one per series) ───────────────────────────────
        angles = (torch.rand(B, device=device) * 16.0 - 8.0) * (np.pi / 180.0)
        scales = torch.rand(B, device=device) * 0.10 + 0.95     # [0.95, 1.05]
        tx = torch.rand(B, device=device) * 0.08 - 0.04         # ±4%
        ty = torch.rand(B, device=device) * 0.08 - 0.04

        cos_a = torch.cos(angles) * scales
        sin_a = torch.sin(angles) * scales

        # ── Horizontal flip (Coronal/Axial only) ─────────────────────────────
        can_flip = (planes == 1) | (planes == 2)
        do_flip  = (torch.rand(B, device=device) < 0.5) & can_flip
        ff = torch.where(do_flip, -1.0, 1.0)  # flip factor

        # ── Build theta [B, 2, 3] ────────────────────────────────────────────
        theta = torch.zeros(B, 2, 3, device=device, dtype=torch.float32)
        theta[:, 0, 0] = cos_a * ff
        theta[:, 0, 1] = -sin_a
        theta[:, 0, 2] = tx
        theta[:, 1, 0] = sin_a * ff
        theta[:, 1, 1] = cos_a
        theta[:, 1, 2] = ty

        # Repeat theta for each slice: [B*D, 2, 3]
        theta_exp = theta.unsqueeze(1).expand(B, D, 2, 3).reshape(B * D, 2, 3)

        # Flatten to [B*D, 1, H, W] for affine_grid
        x_flat = vols.view(B * D, 1, H, W)
        grid = F.affine_grid(theta_exp, x_flat.size(), align_corners=False)
        x_aug = F.grid_sample(x_flat, grid, mode="bilinear",
                               padding_mode="border", align_corners=False)
        x_aug = x_aug.view(B, D, H, W)

        # ── Photometric jitter (per-series scalars broadcast over D,H,W) ──────
        contrast   = torch.rand(B, 1, 1, 1, device=device) * 0.24 + 0.88   # [0.88,1.12]
        brightness = torch.rand(B, 1, 1, 1, device=device) * 0.24 - 0.12   # [-0.12,0.12]
        x_aug = x_aug * contrast + brightness

        # Gamma (soft sign-preserving to handle negative values post-ImageNet norm)
        gamma = torch.rand(B, 1, 1, 1, device=device) * 0.35 + 0.85        # [0.85,1.20]
        x_aug = torch.sign(x_aug) * (torch.abs(x_aug).clamp(min=1e-8) ** gamma)

        # Gaussian noise
        x_aug = x_aug + torch.randn_like(x_aug) * 0.02

        return x_aug   # [B, 24, H, W] float32


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation Preview PNG (Phase 6.1)
# ─────────────────────────────────────────────────────────────────────────────
def save_augmentation_check(
    dataset: SeriesDataset,
    output_path: str = "outputs_v3/aug_check.png",
    device: Optional[torch.device] = None,
    num_per_plane: int = 2,
) -> None:
    """
    Save a grid of original vs augmented mid-slices for Sag, Cor, Axial.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    augmentor = GPUSeriesAugmentor().to(device)
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    # Collect one item per plane
    plane_items: Dict[int, List] = {0: [], 1: [], 2: []}
    for i in range(min(300, len(dataset))):
        row = dataset.rows.iloc[i]
        p = int(row["plane"])
        if len(plane_items[p]) < num_per_plane:
            plane_items[p].append(i)
        if all(len(v) >= num_per_plane for v in plane_items.values()):
            break

    rows_total = num_per_plane * 3
    fig, axes = plt.subplots(rows_total, 2, figsize=(8, 2.8 * rows_total))
    plt.suptitle("V3 Augmentation Check (Original | Augmented)", fontsize=13, fontweight="bold")

    row_idx = 0
    for p_id in (0, 1, 2):
        plane_name = ID_TO_PLANE[p_id]
        for item_idx in plane_items[p_id]:
            item = dataset[item_idx]
            vol_u8 = item["volume"].unsqueeze(0).to(device)  # [1,24,224,224] uint8
            plane_t = torch.tensor([item["plane"]], dtype=torch.long, device=device)

            vol_f = gpu_normalize_volume(vol_u8)              # [1,24,224,224] float32
            vol_aug = augmentor(vol_f, plane_t)               # augmented

            mid = 12
            orig_slice = vol_f[0, mid].cpu().numpy()
            aug_slice  = vol_aug[0, mid].cpu().numpy()

            ax0, ax1 = axes[row_idx, 0], axes[row_idx, 1]
            ax0.imshow(orig_slice, cmap="gray"); ax0.set_title(f"Orig {plane_name}", fontsize=8); ax0.axis("off")
            ax1.imshow(aug_slice,  cmap="gray"); ax1.set_title(f"Aug {plane_name}",  fontsize=8); ax1.axis("off")
            row_idx += 1

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[Aug Check] Saved to {output_path}")


# Add numpy import (used in augmentor)
import numpy as np
