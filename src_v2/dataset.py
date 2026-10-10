"""
src_v2/dataset.py
=================
Study-level PyTorch Dataset and GPU augmentation pipeline for RSNA Knee MRI classification.

Key Features:
  - One item = one STUDY (contains 3 to 14 series across Sagittal, Coronal, Axial planes).
  - Training: at most 2 series per plane per study (sampled randomly on-the-fly, <= 6 series).
  - Evaluation: ALL series of the study (deterministic).
  - Workers return uint8 tensors ONLY — no float ops in workers (avoids Windows shared-memory exhaustion).
  - GPU normalization: 0.5th-99.5th percentile clipping + [0, 1] scaling + ImageNet mean/std on GPU.
  - Batch collation with padding to max series count S and series_mask generation.
  - 2.5D tri-slice channel construction on GPU.
  - GPU-accelerated spatial affine augmentation (slices treated as channels for consistent 3D transforms).
  - Anatomically compliant horizontal flipping (Coronal & Axial only; NEVER Sagittal).
  - Token dropout (10% slice tokens randomly masked in Transformer encoder).
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# Plane mapping
PLANE_TO_ID = {"Sagittal": 0, "Coronal": 1, "Axial": 2}
ID_TO_PLANE = {0: "Sagittal", 1: "Coronal", 2: "Axial"}

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture"
]


class StudyKneeDataset(Dataset):
    """
    Dataset where each item represents an entire patient Study.

    Parameters
    ----------
    studies_df : pd.DataFrame
        DataFrame with StudyInstanceUID and the 12 soft target columns.
    series_df : pd.DataFrame
        DataFrame with StudyInstanceUID, SeriesInstanceUID, Anatomical_Plane, Fluid_Sensitive, Fat_Suppression.
    base_dir : str
        Directory containing <SeriesInstanceUID>.npz files.
    labels : list of str
        List of target column names (12 classes).
    is_train : bool
        If True, samples at most 2 series per plane randomly each epoch.
        If False, uses all series in deterministic order.
    max_series_per_plane : int
        Max series to keep per anatomical plane in training (default: 2).
    """

    def __init__(
        self,
        studies_df: pd.DataFrame,
        series_df: pd.DataFrame,
        base_dir: str = "D:/rsna_knee_dataset",
        labels: Optional[List[str]] = None,
        is_train: bool = False,
        max_series_per_plane: int = 2,
    ) -> None:
        self.studies_df = studies_df.reset_index(drop=True)
        self.base_dir = Path(base_dir)
        self.labels = labels or LABELS
        self.is_train = is_train
        self.max_series_per_plane = max_series_per_plane

        # Pre-group series by StudyInstanceUID for fast O(1) lookup
        # Store series metadata as lightweight tuples: (series_uid, plane_id, fluid_sens, fat_supp)
        grouped = {}
        for row in series_df.itertuples(index=False):
            study_uid = str(row.StudyInstanceUID)
            series_uid = str(row.SeriesInstanceUID)
            plane_str = getattr(row, "Anatomical_Plane", "Sagittal")
            plane_id = PLANE_TO_ID.get(plane_str, 0)
            fluid_sens = int(getattr(row, "Fluid_Sensitive", 0))
            fat_supp = int(getattr(row, "Fat_Suppression", fluid_sens))

            if study_uid not in grouped:
                grouped[study_uid] = {0: [], 1: [], 2: []}
            grouped[study_uid][plane_id].append((series_uid, plane_id, fluid_sens, fat_supp))

        self.grouped_series = grouped

    def __len__(self) -> int:
        return len(self.studies_df)

    def _load_volume_uint8(self, series_uid: str) -> np.ndarray:
        """
        Load .npz volume and return raw uint8 array [24, 224, 224].

        All float ops (percentile clipping, scaling, ImageNet normalization) are
        deliberately deferred to gpu_normalize_volumes() on the GPU so that
        DataLoader workers never create large float32 shared-memory buffers.
        This prevents the Windows ERROR_COMMITMENT_LIMIT (error 1455) crash.
        """
        path = self.base_dir / f"{series_uid}.npz"
        with np.load(path) as f:
            vol = f[f.files[0]]  # already uint8 from preprocessing
        # Ensure uint8 (guard against datasets stored as uint16 / float)
        if vol.dtype != np.uint8:
            vol = np.clip(vol, 0, 255).astype(np.uint8)
        return vol  # [24, 224, 224]  uint8

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, int]]:
        row = self.studies_df.iloc[idx]
        study_uid = str(row["StudyInstanceUID"])
        target_vec = row[self.labels].values.astype(np.float32) if set(self.labels).issubset(row.index) else np.zeros(len(self.labels), dtype=np.float32)

        plane_dict = self.grouped_series.get(study_uid, {0: [], 1: [], 2: []})
        selected_series = []

        if self.is_train:
            # Randomly sample at most max_series_per_plane per plane
            for p_id in (0, 1, 2):
                series_list = plane_dict[p_id]
                if len(series_list) > self.max_series_per_plane:
                    selected_series.extend(random.sample(series_list, self.max_series_per_plane))
                else:
                    selected_series.extend(series_list)
        else:
            # Deterministic: include ALL series ordered by plane
            for p_id in (0, 1, 2):
                selected_series.extend(plane_dict[p_id])

        # If a study has no series (edge case), fallback to empty placeholder
        if not selected_series:
            # Fallback 1 dummy series (zeros as uint8)
            selected_series = [("dummy", 0, 0, 0)]
            vol_list = [np.zeros((24, 224, 224), dtype=np.uint8)]
        else:
            vol_list = [self._load_volume_uint8(s[0]) for s in selected_series]

        # Stack into uint8 tensor [S, 24, 224, 224]  — 4x smaller than float32 in shared memory
        vols_tensor = torch.from_numpy(np.stack(vol_list, axis=0))  # [S, 24, 224, 224]  uint8
        planes_tensor = torch.tensor([s[1] for s in selected_series], dtype=torch.long)  # [S]
        fluid_tensor = torch.tensor([s[2] for s in selected_series], dtype=torch.long)   # [S]
        fat_tensor = torch.tensor([s[3] for s in selected_series], dtype=torch.long)     # [S]
        targets_tensor = torch.from_numpy(target_vec)                                    # [12]

        return {
            "study_uid": study_uid,
            "volumes": vols_tensor,    # uint8 [S, 24, 224, 224]
            "planes": planes_tensor,
            "fluid": fluid_tensor,
            "fat": fat_tensor,
            "targets": targets_tensor,
            "num_series": len(selected_series),
        }


def collate_study_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate variable series count per study by padding to the batch's max series count S_max.

    Volumes are kept as uint8 here and cast to float32 on the GPU inside
    gpu_normalize_volumes(). This keeps the collated tensor ~4x smaller,
    avoiding Windows shared-memory pressure that caused the 1455 crash.

    Returns
    -------
    dict with:
        volumes: [B, S_max, 24, 224, 224] (uint8)  <- normalized later on GPU
        planes: [B, S_max] (long, padded with 0)
        fluid: [B, S_max] (long, padded with 0)
        fat: [B, S_max] (long, padded with 0)
        series_mask: [B, S_max] (bool, True for valid series, False for padded)
        targets: [B, 12] (float32)
        study_uids: List[str]
    """
    batch_size = len(batch)
    max_s = max(item["num_series"] for item in batch)
    d, h, w = 24, 224, 224

    # uint8 zeros — 4x less shared memory than float32
    padded_vols = torch.zeros((batch_size, max_s, d, h, w), dtype=torch.uint8)
    padded_planes = torch.zeros((batch_size, max_s), dtype=torch.long)
    padded_fluid = torch.zeros((batch_size, max_s), dtype=torch.long)
    padded_fat = torch.zeros((batch_size, max_s), dtype=torch.long)
    series_mask = torch.zeros((batch_size, max_s), dtype=torch.bool)
    targets = torch.stack([item["targets"] for item in batch], dim=0)  # [B, 12]
    study_uids = [item["study_uid"] for item in batch]

    for b, item in enumerate(batch):
        s_count = item["num_series"]
        padded_vols[b, :s_count] = item["volumes"]
        padded_planes[b, :s_count] = item["planes"]
        padded_fluid[b, :s_count] = item["fluid"]
        padded_fat[b, :s_count] = item["fat"]
        series_mask[b, :s_count] = True

    return {
        "volumes": padded_vols,      # uint8 [B, S_max, 24, 224, 224]
        "planes": padded_planes,
        "fluid": padded_fluid,
        "fat": padded_fat,
        "series_mask": series_mask,
        "targets": targets,
        "study_uids": study_uids,
    }


# ─────────────────────────────────────────────────────────────────────────────
# GPU Normalization (replaces per-worker float ops)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def gpu_normalize_volumes(vols_uint8: torch.Tensor, series_mask: torch.Tensor) -> torch.Tensor:
    """
    Convert uint8 volume batch to float32 with per-volume percentile normalization
    and ImageNet standardization, executed entirely on the GPU.

    Parameters
    ----------
    vols_uint8 : torch.Tensor  shape [B, S, 24, H, W]  dtype=uint8  (on GPU)
    series_mask : torch.Tensor shape [B, S]             dtype=bool   (on GPU)
        True for valid (real) series, False for zero-padded slots.

    Returns
    -------
    torch.Tensor  shape [B, S, 24, H, W]  dtype=float32
        Padded slots remain zero after normalization.
    """
    B, S, D, H, W = vols_uint8.shape
    # Cast to float32 on the GPU (this is the ONLY float conversion)
    vols = vols_uint8.float()  # [B, S, 24, H, W]

    # Per-volume percentile clipping + [0, 1] scaling
    # Reshape to [B*S, -1] so we can quantile over the spatial dims
    flat = vols.view(B * S, -1)                # [B*S, 24*H*W]

    # torch.quantile is available from PyTorch 1.7+
    q = torch.tensor([0.005, 0.995], device=vols.device, dtype=torch.float32)
    percentiles = torch.quantile(flat, q, dim=1)  # [2, B*S]
    p_low  = percentiles[0].view(B * S, 1, 1, 1, 1)  # [B*S, 1, 1, 1, 1]
    p_high = percentiles[1].view(B * S, 1, 1, 1, 1)

    vols_r = vols.view(B * S, D, H, W).unsqueeze(1)   # [B*S, 1, D, H, W] for broadcasting

    denom = (p_high - p_low).clamp(min=1e-6)
    vols_r = (vols_r - p_low).clamp(min=0.0) / denom  # [0, 1]

    # ImageNet single-channel normalization (mean 0.449, std 0.226)
    vols_r = (vols_r - 0.449) / 0.226

    vols_out = vols_r.squeeze(1).view(B, S, D, H, W)  # [B, S, D, H, W]

    # Zero out padded series so they don't contribute to attention
    pad_mask = ~series_mask                                          # [B, S]  True = padded
    vols_out = vols_out * (~pad_mask).float().view(B, S, 1, 1, 1)   # broadcast over D, H, W

    return vols_out  # float32 [B, S, 24, H, W]


# ─────────────────────────────────────────────────────────────────────────────
# 2.5D GPU Tri-Slice Construction
# ─────────────────────────────────────────────────────────────────────────────
def build_25d_slices(vols: torch.Tensor) -> torch.Tensor:
    """
    Construct 2.5D 3-channel input [d-1, d, d+1] with edge replication for every slice d.

    Parameters
    ----------
    vols : torch.Tensor  shape [N, 24, H, W]

    Returns
    -------
    torch.Tensor  shape [N, 24, 3, H, W]
    """
    N, D, H, W = vols.shape
    assert D == 24, f"Expected D=24 slices, got {D}"

    # Pad depth by 1 on both ends via replicate mode: [N, 1, 24, H, W] -> [N, 1, 26, H, W]
    vols_p = vols.unsqueeze(1)
    vols_padded = F.pad(vols_p, (0, 0, 0, 0, 1, 1), mode="replicate").squeeze(1)  # [N, 26, H, W]

    # Tri-slices for d in 0..23 -> indices [d, d+1, d+2] in vols_padded
    tri_slices = []
    for d in range(D):
        ch_prev = vols_padded[:, d]      # d-1 in original
        ch_curr = vols_padded[:, d + 1]  # d in original
        ch_next = vols_padded[:, d + 2]  # d+1 in original
        tri = torch.stack([ch_prev, ch_curr, ch_next], dim=1)  # [N, 3, H, W]
        tri_slices.append(tri)

    return torch.stack(tri_slices, dim=1)  # [N, 24, 3, H, W]


# ─────────────────────────────────────────────────────────────────────────────
# GPU Augmentation Pipeline (Affine, Photometric, Selective Flip)
# ─────────────────────────────────────────────────────────────────────────────
class GPUAugmentor(nn.Module):
    """
    GPU spatial and photometric augmentation applied per series.
    Treats the 24 depth slices as 24 channels to ensure 100% consistent 3D spatial transformation.
    """

    def __init__(self) -> None:
        super().__init__()

    @torch.no_grad()
    def forward(
        self,
        vols: torch.Tensor,
        planes: torch.Tensor,
        series_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        vols : torch.Tensor  shape [B, S, 24, H, W]
        planes : torch.Tensor shape [B, S]
        series_mask : torch.Tensor shape [B, S]

        Returns
        -------
        torch.Tensor shape [B, S, 24, H, W] (augmented on GPU)
        """
        B, S, D, H, W = vols.shape
        device = vols.device

        # Reshape to [B*S, 24, H, W] to transform all series in parallel
        x = vols.view(B * S, D, H, W)
        flat_mask = series_mask.view(B * S)
        flat_planes = planes.view(B * S)

        # ── 1. Random Affine (Rotation +-10 deg, Scale 0.9-1.1, Translation +-6%) ──
        # Generate random parameters per series
        angles = (torch.rand(B * S, device=device) * 20.0 - 10.0) * (np.pi / 180.0)  # [-10, +10] deg in rad
        scales = torch.rand(B * S, device=device) * 0.2 + 0.9                         # [0.9, 1.1]
        tx = torch.rand(B * S, device=device) * 0.12 - 0.06                           # [-0.06, +0.06]
        ty = torch.rand(B * S, device=device) * 0.12 - 0.06                           # [-0.06, +0.06]

        cos_a = torch.cos(angles) * scales
        sin_a = torch.sin(angles) * scales

        # ── 2. Anatomically Compliant Horizontal Flip ────────────────────────────
        # Flip with p=0.5 ONLY for Coronal (plane=1) and Axial (plane=2), NEVER Sagittal (plane=0)
        flip_prob = torch.rand(B * S, device=device)
        can_flip = (flat_planes == 1) | (flat_planes == 2)
        do_flip = (flip_prob > 0.5) & can_flip

        # If do_flip, negate the horizontal scaling in affine matrix
        flip_factor = torch.where(do_flip, -1.0, 1.0)
        r00 = cos_a * flip_factor
        r01 = -sin_a
        r10 = sin_a * flip_factor
        r11 = cos_a

        # Construct 2x3 affine theta matrices [B*S, 2, 3]
        theta = torch.zeros(B * S, 2, 3, device=device, dtype=torch.float32)
        theta[:, 0, 0] = r00
        theta[:, 0, 1] = r01
        theta[:, 0, 2] = tx
        theta[:, 1, 0] = r10
        theta[:, 1, 1] = r11
        theta[:, 1, 2] = ty

        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x_aug = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)

        # ── 3. Photometric Jitter (Brightness/Contrast +-15%, Gamma 0.8-1.25, Gaussian Noise) ──
        # Brightness & contrast
        contrast = torch.rand(B * S, 1, 1, 1, device=device) * 0.30 + 0.85   # [0.85, 1.15]
        brightness = torch.rand(B * S, 1, 1, 1, device=device) * 0.30 - 0.15 # [-0.15, +0.15]
        x_aug = x_aug * contrast + brightness

        # Gamma power transform (approximate: exponent in [0.8, 1.25])
        # Note: since data is ImageNet standardized, apply non-linear gamma gently
        gamma = torch.rand(B * S, 1, 1, 1, device=device) * 0.45 + 0.80      # [0.80, 1.25]
        # Soft sign-preserving gamma
        x_aug = torch.sign(x_aug) * (torch.abs(x_aug) ** gamma)

        # Light Gaussian noise (std ~ 0.03)
        noise = torch.randn_like(x_aug) * 0.03
        x_aug = x_aug + noise

        # Preserve un-augmented zeros for padded series
        x_final = torch.where(flat_mask.view(-1, 1, 1, 1), x_aug, x)
        return x_final.view(B, S, D, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# Visual Inspection Utility (outputs_v2/aug_check.png)
# ─────────────────────────────────────────────────────────────────────────────
def save_augmentation_check(
    dataset: StudyKneeDataset,
    output_path: str = "outputs_v2/aug_check.png",
    num_samples: int = 4,
) -> None:
    """
    Generate and save a visual comparison PNG showing original vs augmented slices
    across multiple anatomical planes.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    augmentor = GPUAugmentor().to(device)

    fig, axes = plt.subplots(num_samples, 6, figsize=(18, 3.2 * num_samples))
    plt.suptitle("RSNA Knee MRI V2 — Data Augmentation Verification Grid", fontsize=14, fontweight="bold", y=0.99)

    for i in range(num_samples):
        item = dataset[i]
        vols = item["volumes"].unsqueeze(0).to(device)   # [1, S, 24, 224, 224] uint8
        planes = item["planes"].unsqueeze(0).to(device)  # [1, S]
        mask = torch.ones((1, item["num_series"]), dtype=torch.bool, device=device)

        vols_norm = gpu_normalize_volumes(vols, mask)    # float32 [1, S, 24, 224, 224]
        vols_aug = augmentor(vols_norm, planes, mask)

        # Pick up to 3 series from the study (Sagittal, Coronal, Axial if available)
        s_count = min(3, item["num_series"])
        for s_idx in range(s_count):
            p_id = int(item["planes"][s_idx])
            p_name = ID_TO_PLANE.get(p_id, f"Plane {p_id}")
            mid_slice_orig = vols[0, s_idx, 12].cpu().numpy()
            mid_slice_aug = vols_aug[0, s_idx, 12].cpu().numpy()

            ax_orig = axes[i, s_idx * 2]
            ax_aug = axes[i, s_idx * 2 + 1]

            ax_orig.imshow(mid_slice_orig, cmap="gray")
            ax_orig.set_title(f"Orig: {p_name} (S{s_idx})", fontsize=9)
            ax_orig.axis("off")

            ax_aug.imshow(mid_slice_aug, cmap="gray")
            ax_aug.set_title(f"Aug: {p_name} (S{s_idx})", fontsize=9, color="#00ffaa" if p_id != 0 else "#ffaa00")
            ax_aug.axis("off")

        # Clear remaining subplots if fewer than 3 series
        for extra in range(s_count * 2, 6):
            axes[i, extra].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Aug Check] Visual verification grid saved to: {output_path}")
