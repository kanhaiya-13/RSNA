"""
src/dataset.py
==============
PyTorch Dataset for multi-label knee MRI classification.

Handles:
  - Loading 3D .npz volumes from disk (dynamic key detection)
  - Min-max normalization
  - Uniform depth resampling to FIXED_DEPTH slices
  - Optional spatial resize to (IMG_HEIGHT, IMG_WIDTH)
  - Optional horizontal-flip augmentation
  - Soft probability targets (float32)
  - Optional per-sample weighting via a 'sample_weight' column
"""

import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class KneeMRIDataset(Dataset):
    """
    Dataset for multi-label knee MRI classification with soft probability targets.

    Each item is one (SeriesInstanceUID, StudyInstanceUID) pair.
    The volume is loaded from <base_dir>/<SeriesInstanceUID>.npz,
    uniformly resampled to ``fixed_depth`` slices, resized spatially, and
    returned as a float32 tensor of shape [1, D, H, W].

    If a ``sample_weight`` column exists in ``df``, ``__getitem__`` returns a
    3-tuple ``(volume_tensor, label_tensor, weight_tensor)``; otherwise a
    2-tuple ``(volume_tensor, label_tensor)``.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: ``SeriesInstanceUID``, label columns, and
        optionally ``sample_weight``.
    base_dir : str
        Root directory containing ``<SeriesInstanceUID>.npz`` files.
    labels : list of str
        Ordered list of label column names (12 for this task).
    fixed_depth : int
        Target number of depth slices. Volumes are up-/down-sampled to this.
    img_h, img_w : int
        Target spatial resolution. Fast-path skips resize if already matching.
    augment : bool
        If True, applies random horizontal flip during ``__getitem__``.
    """

    def __init__(
        self,
        df,
        base_dir:    str       = "D:/rsna_knee_dataset",
        labels:      List[str] = None,
        fixed_depth: int       = 24,
        img_h:       int       = 224,
        img_w:       int       = 224,
        augment:     bool      = False,
    ) -> None:
        self.df          = df.reset_index(drop=True)
        self.base_dir    = Path(base_dir)
        self.labels      = labels or []
        self.fixed_depth = fixed_depth
        self.img_h       = img_h
        self.img_w       = img_w
        self.augment     = augment
        self.has_weight  = "sample_weight" in self.df.columns

    # ─────────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.df)

    # ─────────────────────────────────────────────────────────────
    def _load_volume(self, series_uid: str) -> np.ndarray:
        """
        Load .npz and return a float32 array of shape [D, H, W].

        Uses ``f.files[0]`` to support any key name stored in the archive
        (e.g. ``'data'``, ``'arr_0'``, or any custom key).
        """
        path = self.base_dir / f"{series_uid}.npz"
        with np.load(path) as f:
            vol = f[f.files[0]].astype(np.float32)   # [D, H, W]
        return vol

    # ─────────────────────────────────────────────────────────────
    def _normalize(self, vol: np.ndarray) -> np.ndarray:
        """Min-max scale to [0, 1]. Returns zeros for flat (empty) volumes."""
        v_min, v_max = vol.min(), vol.max()
        if v_max - v_min < 1e-6:
            return np.zeros_like(vol)
        return (vol - v_min) / (v_max - v_min)

    # ─────────────────────────────────────────────────────────────
    def _sample_depth(self, vol: np.ndarray) -> np.ndarray:
        """
        Uniformly resample the depth axis to ``self.fixed_depth`` slices via
        linear interpolation.  Fast-path: returns ``vol`` unchanged if the
        depth already matches (the RSNA processed dataset ships at D=24).

        Parameters
        ----------
        vol : np.ndarray  shape [D, H, W]

        Returns
        -------
        np.ndarray  shape [fixed_depth, H, W]
        """
        D = vol.shape[0]
        if D == self.fixed_depth:
            return vol

        indices = np.linspace(0, D - 1, self.fixed_depth)
        lo = np.floor(indices).astype(int)
        hi = np.minimum(lo + 1, D - 1)
        w  = (indices - lo).astype(np.float32)   # blend weight ∈ [0, 1)
        return (
            vol[lo] * (1.0 - w[:, None, None]) +
            vol[hi] *         w[:, None, None]
        ).astype(np.float32)

    # ─────────────────────────────────────────────────────────────
    def _resize_slices(self, vol: np.ndarray) -> torch.Tensor:
        """
        Resize H × W of every slice to (``img_h``, ``img_w``) via trilinear
        interpolation.  Fast-path skips resize if the spatial dims already
        match (the RSNA processed dataset ships at 224 × 224).

        Parameters
        ----------
        vol : np.ndarray  shape [D, H, W]

        Returns
        -------
        torch.Tensor  shape [D, img_h, img_w]
        """
        D, H, W = vol.shape
        t = torch.from_numpy(vol)   # [D, H, W]
        if H == self.img_h and W == self.img_w:
            return t
        t = t.unsqueeze(0).unsqueeze(0)   # [1, 1, D, H, W]
        t = F.interpolate(
            t,
            size=(D, self.img_h, self.img_w),
            mode="trilinear",
            align_corners=False,
        )
        return t.squeeze(0).squeeze(0)    # [D, img_h, img_w]

    # ─────────────────────────────────────────────────────────────
    def __getitem__(self, idx: int):
        """
        Returns
        -------
        Without sample_weight : (volume_tensor [1,D,H,W], label_tensor [C])
        With    sample_weight : (volume_tensor [1,D,H,W], label_tensor [C],
                                 weight_tensor scalar)
        """
        row        = self.df.iloc[idx]
        series_uid = str(row["SeriesInstanceUID"])
        label_vec  = row[self.labels].values.astype(np.float32)   # soft targets

        # ── Volume pipeline ──────────────────────────────────────
        vol = self._load_volume(series_uid)   # [D, H, W]
        vol = self._normalize(vol)            # [D, H, W]  in [0, 1]
        vol = self._sample_depth(vol)         # [fixed_depth, H, W]
        vol = self._resize_slices(vol)        # [fixed_depth, img_h, img_w]

        # ── Augmentation ─────────────────────────────────────────
        if self.augment and random.random() > 0.5:
            vol = vol.flip(dims=[-1])         # horizontal flip

        # ── Tensors ──────────────────────────────────────────────
        volume_tensor = vol.unsqueeze(0).float()                        # [1, D, H, W]
        label_tensor  = torch.tensor(label_vec, dtype=torch.float32)   # [C]

        if self.has_weight:
            weight_tensor = torch.tensor(float(row["sample_weight"]), dtype=torch.float32)
            return volume_tensor, label_tensor, weight_tensor

        return volume_tensor, label_tensor
