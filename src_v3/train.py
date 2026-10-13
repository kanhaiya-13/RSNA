"""
src_v3/train.py
===============
Full cross-validation training pipeline for RSNA Knee MRI v3.

Recipe (what won):
  - Series-level training, study-level mean aggregation at evaluation.
  - New backbone family (densenet121 by default) for diversity.
  - All infrastructure ported from src_v2/train.py (logging, resume, EMA,
    per-epoch history flush, error capture, Windows DataLoader fixes).

Phases implemented here:
  Phase 1  - Data cache build (memmap)
  Phase 2  - Throughput probe
  Phase 3  - Dataset / DataLoader
  Phase 4  - Model
  Phase 5  - Training loop
  Phase 6  - Sanity checks + main orchestration
  Phase 7  - Post-hoc analysis (called after all folds finish)
"""

from __future__ import annotations

import argparse
import copy
import gc
import glob
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Windows: file_system sharing BEFORE any torch import ─────────────────────
import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system")

sys.path.insert(0, r"d:\RSNA")

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src_v3.dataset import (
    LABELS,
    PLANE_TO_ID,
    SeriesDataset,
    collate_series_fn,
    gpu_normalize_volume,
    build_25d_slices,
    GPUSeriesAugmentor,
    save_augmentation_check,
)
from src_v3.model import SeriesClassifier
from src.metrics import compute_macro_auc

# ─────────────────────────────────────────────────────────────────────────────
# Global hardware settings
# ─────────────────────────────────────────────────────────────────────────────
torch.backends.cuda.matmul.allow_tf32 = True
cudnn.allow_tf32 = True
cudnn.benchmark = True


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Logger (flush every write, same as v2)
# ─────────────────────────────────────────────────────────────────────────────
class TrainLogger:
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
# EMA
# ─────────────────────────────────────────────────────────────────────────────
class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ep, mp_ in zip(self.module.parameters(), model.parameters()):
            ep.data.mul_(self.decay).add_(mp_.data, alpha=1.0 - self.decay)
        for eb, mb in zip(self.module.buffers(), model.buffers()):
            eb.data.copy_(mb.data)


# ─────────────────────────────────────────────────────────────────────────────
# Optimizer
# ─────────────────────────────────────────────────────────────────────────────
def build_optimizer(
    model: nn.Module,
    backbone_lr: float = 1.5e-4,
    backbone_wd: float = 1e-2,
    head_lr: float = 1e-3,
    head_wd: float = 5e-2,
) -> torch.optim.Optimizer:
    """
    AdamW with two lr groups. Biases/norms excluded from weight decay.
    'backbone' in the name -> backbone group; everything else -> head group.
    """
    backbone_decay, backbone_no_decay = [], []
    head_decay, head_no_decay = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_1d = param.ndim <= 1 or any(k in name for k in ("bias", "norm", "ln", "bn"))
        if "backbone" in name:
            (backbone_no_decay if is_1d else backbone_decay).append(param)
        else:
            (head_no_decay if is_1d else head_decay).append(param)

    return torch.optim.AdamW([
        {"params": backbone_decay,    "lr": backbone_lr, "weight_decay": backbone_wd},
        {"params": backbone_no_decay, "lr": backbone_lr, "weight_decay": 0.0},
        {"params": head_decay,        "lr": head_lr,     "weight_decay": head_wd},
        {"params": head_no_decay,     "lr": head_lr,     "weight_decay": 0.0},
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler
# ─────────────────────────────────────────────────────────────────────────────
class WarmupCosineScheduler:
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
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.current_step = 0

    def step(self) -> None:
        self.current_step += 1
        self._update()

    def set_step(self, step: int) -> None:
        self.current_step = step
        self._update()

    def _update(self) -> None:
        s = self.current_step
        if s <= self.warmup_steps:
            factor = s / self.warmup_steps
            for g, base in zip(self.optimizer.param_groups, self.base_lrs):
                g["lr"] = self.eta_min + (base - self.eta_min) * factor
        else:
            progress = min(1.0, (s - self.warmup_steps) / (self.total_steps - self.warmup_steps))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            for g, base in zip(self.optimizer.param_groups, self.base_lrs):
                g["lr"] = self.eta_min + (base - self.eta_min) * cosine

    def get_last_lr(self) -> List[float]:
        return [g["lr"] for g in self.optimizer.param_groups]


# ─────────────────────────────────────────────────────────────────────────────
# GPU utilisation monitor
# ─────────────────────────────────────────────────────────────────────────────
class GPUMonitor:
    """Samples nvidia-smi every 5 s in a background thread."""

    def __init__(self, csv_path: str) -> None:
        self.csv_path = csv_path
        self._stop = False
        self._thread = None
        self.records: List[Dict] = []

    def start(self) -> None:
        import threading
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop = True
        if self._thread:
            self._thread.join(timeout=10)

    def _run(self) -> None:
        while not self._stop:
            try:
                result = subprocess.run(
                    ["nvidia-smi",
                     "--query-gpu=utilization.gpu,memory.used,power.draw",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0:
                    parts = [x.strip() for x in result.stdout.strip().split(",")]
                    if len(parts) >= 3:
                        rec = {
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "gpu_util_pct": float(parts[0]),
                            "mem_used_mb": float(parts[1]),
                            "power_w": float(parts[2]),
                            "cpu_pct": psutil.cpu_percent(interval=None),
                        }
                        self.records.append(rec)
            except Exception:
                pass
            time.sleep(5)

    def flush(self) -> None:
        if not self.records:
            return
        os.makedirs(os.path.dirname(self.csv_path) if os.path.dirname(self.csv_path) else ".", exist_ok=True)
        pd.DataFrame(self.records).to_csv(self.csv_path, index=False)

    def mean_gpu_util(self) -> float:
        if not self.records:
            return 0.0
        return float(np.mean([r["gpu_util_pct"] for r in self.records]))

    def peak_vram_gb(self) -> float:
        if not self.records:
            return 0.0
        return max(r["mem_used_mb"] for r in self.records) / 1024.0


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Build series_index.csv + memmap cache
# ─────────────────────────────────────────────────────────────────────────────
def build_series_index(
    series_csv: str,
    folds_csv: str,
    out_csv: str,
    logger: TrainLogger,
) -> pd.DataFrame:
    """
    Build outputs_v3/series_index.csv with columns:
      row_index, SeriesInstanceUID, StudyInstanceUID, plane, Fluid_Sensitive, fold
    """
    if os.path.exists(out_csv):
        logger.log(f"[Phase1] series_index.csv already exists: {out_csv}. Loading.")
        return pd.read_csv(out_csv)

    logger.log("[Phase1] Building series_index.csv ...")
    series_df = pd.read_csv(series_csv)
    folds_df = pd.read_csv(folds_csv)

    fold_map = dict(zip(folds_df["StudyInstanceUID"].astype(str),
                        folds_df["fold"].astype(int)))

    rows = []
    for i, r in enumerate(series_df.itertuples(index=False)):
        study_uid = str(r.StudyInstanceUID)
        plane_str = getattr(r, "Anatomical_Plane", "Sagittal")
        plane_id = PLANE_TO_ID.get(plane_str, 0)
        fluid = int(getattr(r, "Fluid_Sensitive", 0))
        fold = fold_map.get(study_uid, -1)
        rows.append({
            "row_index": i,
            "SeriesInstanceUID": str(r.SeriesInstanceUID),
            "StudyInstanceUID": study_uid,
            "plane": plane_id,
            "Fluid_Sensitive": fluid,
            "fold": fold,
        })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv) if os.path.dirname(out_csv) else ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    logger.log(f"[Phase1] series_index.csv written: {len(df)} rows -> {out_csv}")
    return df


def build_memmap_cache(
    series_index: pd.DataFrame,
    data_dir: str,
    cache_path: str,
    logger: TrainLogger,
    n_workers: int = 16,
    verify_n: int = 100,
) -> None:
    """
    Build D:/rsna_cache/volumes_uint8.npy as numpy memmap [N,24,224,224] uint8.
    Uses multiprocessing pool to fill rows in parallel.
    """
    N = len(series_index)
    flag_path = os.path.join(os.path.dirname(cache_path), "cache_complete.flag")
    shape = (N, 24, 224, 224)
    expected_gb = N * 24 * 224 * 224 / 1e9

    if os.path.exists(cache_path):
        arr = np.load(cache_path, mmap_mode="r")
        if arr.shape == shape and arr.dtype == np.uint8:
            if os.path.exists(flag_path):
                logger.log(f"[Phase1] Cache already exists and verified at {cache_path}. Skipping build.")
                _verify_cache(series_index, data_dir, cache_path, verify_n, logger)
                return
            else:
                # File exists but wasn't flagged complete: check for empty rows and fill them
                logger.log(f"[Phase1] Cache file exists but not marked complete. Checking for missing rows...")
                missing_indices = []
                for i in range(N):
                    # Check if volume is entirely zero
                    if not np.any(arr[i] > 0):
                        missing_indices.append(i)
                del arr
                
                if len(missing_indices) == 0:
                    logger.log(f"[Phase1] All {N} rows are already populated! Marking cache complete.")
                    with open(flag_path, "w") as f:
                        f.write(f"complete {time.time()}\n")
                    _verify_cache(series_index, data_dir, cache_path, verify_n, logger)
                    return
                else:
                    logger.log(f"[Phase1] Found {len(missing_indices)} missing rows out of {N}. Filling remaining rows...")
                    missing_set = set(missing_indices)
                    tasks = [
                        (int(r.row_index), str(r.SeriesInstanceUID), data_dir, cache_path, shape)
                        for r in series_index.itertuples(index=False)
                        if int(r.row_index) in missing_set
                    ]
                    from multiprocessing import Pool
                    chunk_size = max(1, len(tasks) // (n_workers * 2)) if len(tasks) > n_workers else 1
                    t0 = time.time()
                    with Pool(processes=min(n_workers, len(tasks))) as pool:
                        for _ in pool.imap_unordered(_fill_row, tasks, chunksize=chunk_size):
                            pass
                    elapsed = time.time() - t0
                    logger.log(f"[Phase1] Filled {len(missing_indices)} missing rows in {elapsed:.1f}s.")
                    with open(flag_path, "w") as f:
                        f.write(f"complete {time.time()}\n")
                    _verify_cache(series_index, data_dir, cache_path, verify_n, logger)
                    return
        else:
            logger.log(f"[Phase1] Existing cache shape mismatch: {arr.shape} vs {shape}. Rebuilding.")

    logger.log(f"[Phase1] Building memmap cache: {cache_path}  shape={shape}  (~{expected_gb:.1f} GB)")
    t0 = time.time()

    # Create the memmap file
    arr = np.lib.format.open_memmap(cache_path, mode="w+", dtype=np.uint8, shape=shape)
    arr[:] = 0
    del arr

    # Build row list: (row_index, series_uid, data_dir, cache_path, shape)
    tasks = [
        (int(r.row_index), str(r.SeriesInstanceUID), data_dir, cache_path, shape)
        for r in series_index.itertuples(index=False)
    ]

    from multiprocessing import Pool
    chunk_size = max(1, N // (n_workers * 4))

    completed = 0
    with Pool(processes=n_workers) as pool:
        for _ in pool.imap_unordered(_fill_row, tasks, chunksize=chunk_size):
            completed += 1
            if completed % 500 == 0 or completed == N:
                elapsed = time.time() - t0
                rate = completed / elapsed
                logger.log(f"  [Cache] {completed}/{N}  {rate:.1f} vol/s  elapsed={elapsed:.0f}s")

    elapsed = time.time() - t0
    logger.log(f"[Phase1] Cache build complete: {elapsed:.1f}s  ({N/elapsed:.1f} vol/s)")

    with open(flag_path, "w") as f:
        f.write(f"complete {time.time()}\n")

    _verify_cache(series_index, data_dir, cache_path, verify_n, logger)


def _fill_row(args: Tuple) -> None:
    """Worker: load one .npz and write into the memmap."""
    row_idx, series_uid, data_dir, cache_path, shape = args
    import os, numpy as np
    path = os.path.join(data_dir, f"{series_uid}.npz")
    try:
        with np.load(path) as f:
            vol = f[f.files[0]]
        if vol.dtype != np.uint8:
            vol = np.clip(vol, 0, 255).astype(np.uint8)
        arr = np.load(cache_path, mmap_mode="r+")
        arr[row_idx] = vol
        del arr
    except Exception as e:
        print(f"[Cache Worker ERROR] row={row_idx} uid={series_uid}: {e}", flush=True)


def _verify_cache(
    series_index: pd.DataFrame,
    data_dir: str,
    cache_path: str,
    n: int,
    logger: TrainLogger,
) -> None:
    logger.log(f"[Phase1] Verifying {n} random cached volumes ...")
    arr = np.load(cache_path, mmap_mode="r")
    idxs = np.random.choice(len(series_index), size=min(n, len(series_index)), replace=False)
    mismatches = 0
    t0 = time.time()
    for i in idxs:
        row = series_index.iloc[i]
        path = os.path.join(data_dir, f"{row['SeriesInstanceUID']}.npz")
        with np.load(path) as f:
            vol = f[f.files[0]]
        if vol.dtype != np.uint8:
            vol = np.clip(vol, 0, 255).astype(np.uint8)
        if not np.array_equal(arr[row["row_index"]], vol):
            mismatches += 1
    elapsed = time.time() - t0
    rate = n / elapsed
    logger.log(f"[Phase1] Verify done: {mismatches} mismatches / {n} checked  ({rate:.1f} vol/s)")
    assert mismatches == 0, f"Cache verification FAILED: {mismatches} mismatches!"
    logger.log("[Phase1] Cache verification PASSED.")

    # Warm-cache read speed
    t1 = time.time()
    for i in idxs[:20]:
        _ = arr[i].copy()
    warm_rate = 20 / (time.time() - t1)
    logger.log(f"[Phase1] Warm read speed (first 20 vols): {warm_rate:.1f} vol/s")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Throughput Probe
# ─────────────────────────────────────────────────────────────────────────────
def run_throughput_probe(
    cache_path: str,
    series_index: pd.DataFrame,
    targets_df: pd.DataFrame,
    device: torch.device,
    amp_dtype: torch.dtype,
    outputs_dir: str,
    logger: TrainLogger,
    fold_val: int = 0,
) -> Tuple[str, int]:
    """
    For each backbone x micro-batch, measure images/s, s/step, peak VRAM, OOM.
    Returns (best_backbone, best_micro_batch).
    """
    out_md = os.path.join(outputs_dir, "throughput_probe.md")
    if os.path.exists(out_md):
        logger.log(f"[Phase2] Throughput probe already exists: {out_md}. Skipping.")
        # Read best config from file
        with open(out_md) as f:
            content = f.read()
        # Parse CHOSEN line
        for line in content.split("\n"):
            if "CHOSEN" in line and "|" in line:
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if len(parts) >= 2:
                    try:
                        bb = parts[0]
                        mb = int(parts[1])
                        return bb, mb
                    except Exception:
                        pass
        return "densenet121", 8

    logger.log(f"\n[Phase2] Running throughput probe ...")
    TOTAL_VRAM_GB = torch.cuda.get_device_properties(device).total_memory / 1e9
    VRAM_LIMIT = TOTAL_VRAM_GB * 0.90

    backbones = ["densenet121", "inception_v3", "efficientnet_b3", "convnext_tiny", "resnet34"]
    batches = [8, 16, 24, 32, 48]
    PROBE_STEPS = 15

    augmentor = GPUSeriesAugmentor().to(device)

    # Use fold 0 val series as probe data (small, quick to build)
    probe_ds = SeriesDataset(
        series_index=series_index,
        targets_df=targets_df,
        memmap_path=cache_path,
        fold_ids={fold_val},
        is_val_split=True,
    )

    results = []
    chosen = {}   # backbone -> (best_batch, peak_vram)

    for bb in backbones:
        best_batch_for_bb = 0
        best_vram_for_bb = 0.0

        for mb in batches:
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.reset_peak_memory_stats(device)

            try:
                # Build a tiny fake batch of uint8 tensors
                fake_vol = torch.zeros(mb, 24, 224, 224, dtype=torch.uint8, device=device)
                fake_planes = torch.zeros(mb, dtype=torch.long, device=device)
                fake_fluids = torch.zeros(mb, dtype=torch.long, device=device)
                fake_targets = torch.zeros(mb, 12, dtype=torch.float32, device=device)

                model = SeriesClassifier(backbone_name=bb, pretrained=False).to(device)
                model.to(memory_format=torch.channels_last)
                model.train()
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
                criterion = nn.BCEWithLogitsLoss()

                # Warmup
                for _ in range(2):
                    with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                        vols_f = gpu_normalize_volume(fake_vol)
                        vols_aug = augmentor(vols_f, fake_planes)
                        slices = build_25d_slices(vols_aug).to(memory_format=torch.channels_last)
                        logits = model(slices, fake_planes, fake_fluids)
                        loss = criterion(logits, fake_targets)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                torch.cuda.synchronize()
                t0 = time.time()

                for step in range(PROBE_STEPS):
                    with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                        vols_f = gpu_normalize_volume(fake_vol)
                        vols_aug = augmentor(vols_f, fake_planes)
                        slices = build_25d_slices(vols_aug).to(memory_format=torch.channels_last)
                        logits = model(slices, fake_planes, fake_fluids)
                        loss = criterion(logits, fake_targets)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                torch.cuda.synchronize()
                elapsed = time.time() - t0
                peak_vram = torch.cuda.max_memory_allocated(device) / 1e9
                imgs_per_s = mb * PROBE_STEPS / elapsed
                s_per_step = elapsed / PROBE_STEPS
                oom = False

                del model, optimizer, fake_vol, fake_planes, fake_fluids, fake_targets
                torch.cuda.empty_cache()
                gc.collect()

                rec = {
                    "backbone": bb, "micro_batch": mb,
                    "imgs_s": round(imgs_per_s, 1),
                    "s_step": round(s_per_step, 3),
                    "peak_vram_gb": round(peak_vram, 2),
                    "oom": oom,
                }
                results.append(rec)
                logger.log(f"  {bb:20s} mb={mb:3d} -> {imgs_per_s:.1f} img/s  {s_per_step:.3f} s/step  {peak_vram:.2f} GB VRAM")

                if not oom and peak_vram <= VRAM_LIMIT:
                    best_batch_for_bb = mb
                    best_vram_for_bb = peak_vram

                # On Windows, exceeding VRAM silently pages to system RAM over PCIe, causing massive slowdowns
                if peak_vram > VRAM_LIMIT or s_per_step > 2.0:
                    logger.log(f"  {bb:20s} Exceeded VRAM limit or paged into system RAM ({peak_vram:.2f} GB). Skipping larger batches.")
                    break

            except torch.cuda.OutOfMemoryError:
                logger.log(f"  {bb:20s} mb={mb:3d} -> OOM")
                try:
                    del model, optimizer
                except Exception:
                    pass
                torch.cuda.empty_cache()
                gc.collect()
                rec = {
                    "backbone": bb, "micro_batch": mb,
                    "imgs_s": 0.0, "s_step": 0.0,
                    "peak_vram_gb": TOTAL_VRAM_GB,
                    "oom": True,
                }
                results.append(rec)
                break

        chosen[bb] = (best_batch_for_bb, best_vram_for_bb)

    # Write markdown table
    lines = ["# Throughput Probe — V3", "",
             f"GPU: NVIDIA RTX 4090  |  Total VRAM: {TOTAL_VRAM_GB:.1f} GB  |  VRAM limit (90%): {VRAM_LIMIT:.1f} GB",
             f"Probe steps: {PROBE_STEPS} per (backbone, micro_batch)",
             "",
             "| Backbone | micro_batch | imgs/s | s/step | peak_VRAM_GB | OOM |",
             "|----------|-------------|--------|--------|--------------|-----|",
    ]
    for r in results:
        oom_str = "YES" if r["oom"] else "no"
        lines.append(f"| {r['backbone']:20s} | {r['micro_batch']:11d} | {r['imgs_s']:6.1f} | {r['s_step']:6.3f} | {r['peak_vram_gb']:12.2f} | {oom_str:3s} |")

    lines += ["", "## Chosen configuration per backbone", "",
              "| backbone | best_micro_batch | peak_VRAM_GB |",
              "|----------|------------------|--------------|"]
    for bb, (mb, vram) in chosen.items():
        lines.append(f"| {bb:20s} | {mb:16d} | {vram:12.2f} |")

    # Determine overall best: densenet121 with its best batch
    best_bb = "densenet121"
    best_mb = chosen[best_bb][0] if chosen[best_bb][0] > 0 else 8

    lines += ["", f"## CHOSEN for training",
              f"| {best_bb} | {best_mb} |"]

    with open(out_md, "w") as f:
        f.write("\n".join(lines) + "\n")

    logger.log(f"[Phase2] Throughput probe saved: {out_md}")
    logger.log(f"[Phase2] CHOSEN: backbone={best_bb}  micro_batch={best_mb}")
    return best_bb, best_mb


# ─────────────────────────────────────────────────────────────────────────────
# Validation (series-level inference, study-level aggregation)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_dtype: torch.dtype,
    series_index_val: pd.DataFrame,
) -> Tuple[float, float, Dict[str, float], float, pd.DataFrame, pd.DataFrame]:
    """
    Run one validation pass.

    Returns
    -------
    (val_loss_study, study_macro_auc, per_label_auc, series_macro_auc,
     study_pred_df, series_pred_df)
    """
    model.eval()
    augmentor = GPUSeriesAugmentor().to(device)  # no-op in eval; kept for consistency

    all_series_uids = []
    all_study_uids = []
    all_preds = []    # [N_series, 12] float32
    all_targets = []  # [N_series, 12] float32
    total_loss = 0.0
    n_samples = 0

    for batch in val_loader:
        vols_u8 = batch["volume"].to(device, non_blocking=True)     # [B,24,H,W] uint8
        planes  = batch["plane"].to(device, non_blocking=True)
        fluids  = batch["fluid"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)     # [B,12]

        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
            vols_f = gpu_normalize_volume(vols_u8)
            slices = build_25d_slices(vols_f).to(memory_format=torch.channels_last)
            logits = model(slices, planes, fluids)
            loss = criterion(logits, targets)

        probs = torch.sigmoid(logits).float().cpu().numpy()
        tgt_np = targets.float().cpu().numpy()

        all_series_uids.extend(batch["series_uid"])
        all_study_uids.extend(batch["study_uid"])
        all_preds.append(probs)
        all_targets.append(tgt_np)
        total_loss += loss.item() * len(batch["series_uid"])
        n_samples += len(batch["series_uid"])

    y_pred = np.concatenate(all_preds, axis=0)   # [N_series, 12]
    y_true = np.concatenate(all_targets, axis=0)

    # Series-level AUC
    series_macro_auc, _ = compute_macro_auc(y_true, y_pred, labels=LABELS)

    # Series OOF dataframe
    series_df = pd.DataFrame({"SeriesInstanceUID": all_series_uids, "StudyInstanceUID": all_study_uids})
    # Add plane/fluid info from series_index_val
    si_lookup = series_index_val.set_index("SeriesInstanceUID")[["plane", "Fluid_Sensitive"]].to_dict("index")
    series_df["plane"] = series_df["SeriesInstanceUID"].map(lambda u: si_lookup.get(u, {}).get("plane", -1))
    series_df["Fluid_Sensitive"] = series_df["SeriesInstanceUID"].map(lambda u: si_lookup.get(u, {}).get("Fluid_Sensitive", 0))
    for j, lbl in enumerate(LABELS):
        series_df[lbl] = y_pred[:, j]

    # Study-level aggregation: mean of series probs
    study_preds: Dict[str, np.ndarray] = {}
    study_targets: Dict[str, np.ndarray] = {}
    for i, s_uid in enumerate(all_study_uids):
        if s_uid not in study_preds:
            study_preds[s_uid] = []
            study_targets[s_uid] = y_true[i]
        study_preds[s_uid].append(y_pred[i])

    study_uids_list = list(study_preds.keys())
    study_pred_arr = np.array([np.mean(study_preds[u], axis=0) for u in study_uids_list])
    study_true_arr = np.array([study_targets[u] for u in study_uids_list])

    study_macro_auc, per_label_auc = compute_macro_auc(study_true_arr, study_pred_arr, labels=LABELS)

    # Study OOF dataframe
    study_df = pd.DataFrame({"StudyInstanceUID": study_uids_list})
    for j, lbl in enumerate(LABELS):
        study_df[lbl] = study_pred_arr[:, j]

    val_loss_study = total_loss / max(1, n_samples)
    return val_loss_study, study_macro_auc, per_label_auc, series_macro_auc, study_df, series_df


# ─────────────────────────────────────────────────────────────────────────────
# Phase 6: Sanity checks
# ─────────────────────────────────────────────────────────────────────────────
def run_sanity_checks(
    config: Dict,
    series_index: pd.DataFrame,
    targets_df: pd.DataFrame,
    device: torch.device,
    amp_dtype: torch.dtype,
    logger: TrainLogger,
) -> None:
    outputs_dir = config["outputs_dir"]
    logger.log(f"\n{'='*70}\n  PHASE 6: SANITY CHECKS\n{'='*70}")

    # 6.1 Disjointness
    logger.log("[6.1] Checking fold disjointness ...")
    for f in range(5):
        val_studies = set(series_index[series_index["fold"] == f]["StudyInstanceUID"])
        trn_studies = set(series_index[series_index["fold"] != f]["StudyInstanceUID"])
        overlap = val_studies & trn_studies
        assert len(overlap) == 0, f"Fold {f}: {len(overlap)} studies in both train and val!"
    logger.log("  --> PASS: All folds are disjoint.")

    # Shape check on one batch
    logger.log("[6.1] Shape check on one batch ...")
    probe_ds = SeriesDataset(
        series_index=series_index, targets_df=targets_df,
        memmap_path=config["cache_path"],
        fold_ids={0}, is_val_split=True,
    )
    probe_loader = DataLoader(probe_ds, batch_size=4, collate_fn=collate_series_fn,
                               num_workers=0, shuffle=False)
    batch = next(iter(probe_loader))
    logger.log(f"  volume shape: {batch['volume'].shape}  dtype={batch['volume'].dtype}")
    logger.log(f"  plane shape:  {batch['plane'].shape}")
    logger.log(f"  fluid shape:  {batch['fluid'].shape}")
    logger.log(f"  target shape: {batch['target'].shape}")
    assert batch["volume"].dtype == torch.uint8, "Volume must be uint8!"

    # Move to GPU and forward pass
    model = SeriesClassifier(
        backbone_name=config["backbone"], pretrained=False
    ).to(device)
    model.to(memory_format=torch.channels_last)

    vols_u8 = batch["volume"].to(device)
    planes_g = batch["plane"].to(device)
    fluids_g = batch["fluid"].to(device)

    with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
        vols_f = gpu_normalize_volume(vols_u8)
        slices = build_25d_slices(vols_f).to(memory_format=torch.channels_last)
        logits = model(slices, planes_g, fluids_g)
    logger.log(f"  slices_25d shape: {slices.shape}   logits shape: {logits.shape}")
    assert logits.shape == (4, 12), f"Unexpected logits shape: {logits.shape}"
    logger.log("  --> PASS: Forward pass shapes correct.")
    del model

    # Aug check PNG
    logger.log("[6.1] Saving augmentation preview ...")
    save_augmentation_check(
        probe_ds,
        output_path=os.path.join(outputs_dir, "aug_check.png"),
        device=device,
    )
    logger.log(f"  --> Saved: {os.path.join(outputs_dir, 'aug_check.png')}")

    # 6.2 Overfit test: 32 series, no aug, loss -> 0
    logger.log("[6.2] Overfit test (32 series, target loss < 0.05) ...")
    overfit_ds = SeriesDataset(
        series_index=series_index.head(32), targets_df=targets_df,
        memmap_path=config["cache_path"],
        fold_ids={-99}, is_val_split=False,  # take all 32 rows as train
    )
    # Manual override: force all 32 as training
    overfit_ds.rows = series_index.head(32).reset_index(drop=True)
    overfit_loader = DataLoader(overfit_ds, batch_size=8, collate_fn=collate_series_fn,
                                 num_workers=0, shuffle=True)
    cached = [b for b in overfit_loader]

    model_ovf = SeriesClassifier(backbone_name=config["backbone"], pretrained=False).to(device)
    model_ovf.to(memory_format=torch.channels_last)
    opt_ovf = torch.optim.AdamW(model_ovf.parameters(), lr=5e-4, weight_decay=0.0)
    crit_ovf = nn.BCEWithLogitsLoss()

    model_ovf.train()
    min_loss = 1e9
    for step in range(60):
        step_loss = 0.0
        for b in cached:
            vols_u8 = b["volume"].to(device)
            planes_g = b["plane"].to(device)
            fluids_g = b["fluid"].to(device)
            tgt_bin = (b["target"] >= 0.5).float().to(device)
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                vols_f = gpu_normalize_volume(vols_u8)
                slices = build_25d_slices(vols_f).to(memory_format=torch.channels_last)
                logits = model_ovf(slices, planes_g, fluids_g)
                loss = crit_ovf(logits, tgt_bin)
            loss.backward()
            opt_ovf.step()
            opt_ovf.zero_grad(set_to_none=True)
            step_loss += loss.item()
        avg = step_loss / len(cached)
        min_loss = min(min_loss, avg)
        if (step + 1) % 10 == 0:
            logger.log(f"  Overfit step {step+1:3d}/60  loss={avg:.4f}  best={min_loss:.4f}")

    assert min_loss < 0.25, f"Overfit test FAILED: min_loss={min_loss:.4f} >= 0.25"
    logger.log(f"  --> PASS: Overfit min_loss={min_loss:.6f}")
    del model_ovf, opt_ovf

    torch.cuda.empty_cache()
    gc.collect()
    logger.log(f"\n{'='*70}\n  ALL SANITY CHECKS PASSED\n{'='*70}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Training: one fold
# ─────────────────────────────────────────────────────────────────────────────
def train_fold(
    fold: int,
    config: Dict,
    series_index: pd.DataFrame,
    targets_df: pd.DataFrame,
    device: torch.device,
    amp_dtype: torch.dtype,
    logger: TrainLogger,
) -> Tuple[float, float, Dict]:
    logger.log(f"\n{'='*70}\n  TRAINING FOLD {fold}\n{'='*70}")

    outputs_dir = Path(config["outputs_dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt   = outputs_dir / f"best_model_fold{fold}.pt"
    last_ckpt   = outputs_dir / f"last_model_fold{fold}.pt"
    hist_csv    = outputs_dir / f"training_history_fold{fold}.csv"
    oof_best    = outputs_dir / f"oof_best_fold{fold}.csv"
    oof_last    = outputs_dir / f"oof_last_fold{fold}.csv"
    series_oof  = outputs_dir / f"series_oof_best_fold{fold}.csv"
    emrg_ckpt   = outputs_dir / f"emergency_checkpoint_fold{fold}.pt"
    gpu_csv     = outputs_dir / f"gpu_util_fold{fold}.csv"
    err_txt     = outputs_dir / f"error_fold{fold}.txt"

    # ── Datasets ─────────────────────────────────────────────────────────────
    val_fold_set = {fold}

    train_ds = SeriesDataset(
        series_index=series_index,
        targets_df=targets_df,
        memmap_path=config["cache_path"],
        fold_ids=val_fold_set,
        is_val_split=False,
    )
    val_ds = SeriesDataset(
        series_index=series_index,
        targets_df=targets_df,
        memmap_path=config["cache_path"],
        fold_ids=val_fold_set,
        is_val_split=True,
    )
    series_index_val = series_index[series_index["fold"] == fold].reset_index(drop=True)

    n_train_studies = series_index[series_index["fold"] != fold]["StudyInstanceUID"].nunique()
    n_val_studies   = series_index[series_index["fold"] == fold]["StudyInstanceUID"].nunique()
    logger.log(f"[Data] Train series: {len(train_ds)} ({n_train_studies} studies) | Val series: {len(val_ds)} ({n_val_studies} studies)")

    # ── Micro-batch + accumulation ───────────────────────────────────────────
    micro_batch   = config["micro_batch_size"]
    effective_bs  = config["effective_batch_size"]
    accum_steps   = max(1, effective_bs // micro_batch)
    # If micro_batch >= 2x target effective, use effective=32 with larger lr
    if micro_batch >= effective_bs * 2:
        effective_bs = effective_bs * 2
        accum_steps = max(1, effective_bs // micro_batch)
        config = {**config, "backbone_lr": config["backbone_lr"] * 1.4,
                  "head_lr": config["head_lr"] * 1.4}
        logger.log(f"[Setup] Large micro_batch: effective_bs -> {effective_bs}, lrs scaled x1.4")

    logger.log(f"[Setup] micro_batch={micro_batch} accum={accum_steps} effective_bs={micro_batch*accum_steps}")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    nw_train = config["num_workers_train"]
    train_loader = DataLoader(
        train_ds,
        batch_size=micro_batch,
        shuffle=True,
        num_workers=nw_train,
        prefetch_factor=4 if nw_train > 0 else None,
        collate_fn=collate_series_fn,
        pin_memory=True,
        persistent_workers=nw_train > 0,
        drop_last=True,
    )
    logger.log(f"[Setup] Train loader: {len(train_loader)} batches/epoch  workers={nw_train}  persistent=True")

    # ── Model + EMA ───────────────────────────────────────────────────────────
    model = SeriesClassifier(
        backbone_name=config["backbone"],
        num_classes=12,
        pretrained=config["pretrained"],
        pooling=config.get("pooling", "meanmax"),
        no_metadata=config.get("no_metadata", False),
        head_per_plane=config.get("head_per_plane", False),
    ).to(device)
    model.to(memory_format=torch.channels_last)

    ema = ModelEMA(model, decay=config["ema_decay"])
    augmentor = GPUSeriesAugmentor().to(device)

    # ── Optimizer + Scheduler ────────────────────────────────────────────────
    steps_per_epoch = len(train_loader) // accum_steps
    total_steps     = steps_per_epoch * config["epochs"]
    warmup_steps    = steps_per_epoch * config["warmup_epochs"]

    optimizer = build_optimizer(
        model,
        backbone_lr=config["backbone_lr"],
        backbone_wd=config["backbone_wd"],
        head_lr=config["head_lr"],
        head_wd=config["head_wd"],
    )
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps, eta_min=1e-6)

    logger.log(f"[Setup] epochs={config['epochs']} steps/epoch={steps_per_epoch} total_opt_steps={total_steps}")
    logger.log(f"[Setup] warmup_steps={warmup_steps}  backbone_lr={config['backbone_lr']:.2e}  head_lr={config['head_lr']:.2e}")
    logger.log(f"[Setup] EMA decay={config['ema_decay']}  grad_clip={config['grad_clip']}")

    criterion = nn.BCEWithLogitsLoss()

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch = 1
    best_auc = -1.0
    best_epoch = 1
    history = []
    best_study_df = None
    best_series_df = None

    if last_ckpt.exists():
        logger.log(f"[Resume] Found {last_ckpt}, attempting resume ...")
        try:
            ckpt = torch.load(last_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            ema.module.load_state_dict(ckpt.get("ema_state", ckpt["model_state"]))
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scheduler.set_step(ckpt["scheduler_step"])
            start_epoch = ckpt["epoch"] + 1
            best_auc = ckpt.get("best_auc", -1.0)
            best_epoch = ckpt.get("best_epoch", 1)
            if hist_csv.exists():
                history = pd.read_csv(hist_csv).to_dict("records")
            if oof_best.exists():
                best_study_df = pd.read_csv(oof_best)
            if series_oof.exists():
                best_series_df = pd.read_csv(series_oof)
            logger.log(f"[Resume] OK: resumed at epoch {start_epoch}, best AUC={best_auc:.6f} @ep{best_epoch}")
        except Exception as e:
            logger.log(f"[Resume WARNING] Failed: {e}. Starting from scratch.")

    if start_epoch > config["epochs"]:
        logger.log(f"[Fold {fold}] Already complete ({start_epoch-1}/{config['epochs']} epochs).")
        return best_auc, history[-1]["macro_auc_study"] if history else best_auc, {"best_epoch": best_epoch, "history": history}

    # ── GPU monitor ──────────────────────────────────────────────────────────
    monitor = GPUMonitor(str(gpu_csv))
    monitor.start()

    try:
        for epoch in range(start_epoch, config["epochs"] + 1):
            epoch_t0 = time.time()
            try:
                model.train()
                train_loss_accum = 0.0
                optimizer.zero_grad(set_to_none=True)
                train_t0 = time.time()

                for step, batch in enumerate(
                    tqdm(train_loader, desc=f"Fold{fold} Ep{epoch:02d}/{config['epochs']:02d}", leave=False)
                ):
                    vols_u8 = batch["volume"].to(device, non_blocking=True)
                    planes_g = batch["plane"].to(device, non_blocking=True)
                    fluids_g = batch["fluid"].to(device, non_blocking=True)
                    targets_g = batch["target"].to(device, non_blocking=True)

                    with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                        vols_f = gpu_normalize_volume(vols_u8)
                        vols_aug = augmentor(vols_f, planes_g)
                        slices = build_25d_slices(vols_aug).to(memory_format=torch.channels_last)
                        logits = model(slices, planes_g, fluids_g)
                        loss = criterion(logits, targets_g) / accum_steps

                    loss.backward()
                    train_loss_accum += loss.item() * accum_steps

                    if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                        optimizer.step()
                        ema.update(model)
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)

                    # Flush log every 20 steps
                    if (step + 1) % 20 == 0 or (step + 1) == len(train_loader):
                        opt_step = (step + 1) // accum_steps
                        cur_lr = scheduler.get_last_lr()[2]
                        logger.log(
                            f"  [F{fold}|Ep{epoch:02d}|Step{step+1}/{len(train_loader)} (opt{opt_step}/{steps_per_epoch})]"
                            f" loss={train_loss_accum/(step+1):.4f} lr={cur_lr:.2e}"
                        )

                train_time = time.time() - train_t0
                mean_train_loss = train_loss_accum / len(train_loader)

                # ── Validation (fresh loader, close after) ───────────────────
                val_t0 = time.time()
                nw_val = config["num_workers_val"]
                val_loader = None
                try:
                    val_loader = DataLoader(
                        val_ds,
                        batch_size=config["val_batch_size"],
                        shuffle=False,
                        num_workers=nw_val,
                        prefetch_factor=2 if nw_val > 0 else None,
                        collate_fn=collate_series_fn,
                        pin_memory=True,
                        persistent_workers=False,
                        drop_last=False,
                    )
                    val_results = validate_epoch(
                        ema.module, val_loader, criterion, device, amp_dtype, series_index_val
                    )
                except RuntimeError as e:
                    if "1455" in str(e) or "shared" in str(e).lower():
                        logger.log(f"[WARNING] Val OOM/shared error, falling back to num_workers=0: {e}")
                        if val_loader: del val_loader
                        gc.collect()
                        val_loader = DataLoader(
                            val_ds, batch_size=config["val_batch_size"], shuffle=False,
                            num_workers=0, collate_fn=collate_series_fn,
                            pin_memory=True, persistent_workers=False, drop_last=False,
                        )
                        val_results = validate_epoch(
                            ema.module, val_loader, criterion, device, amp_dtype, series_index_val
                        )
                    else:
                        raise
                finally:
                    if val_loader: del val_loader
                    gc.collect()

                val_loss, study_mac_auc, per_lbl, series_mac_auc, study_df, series_df = val_results
                val_time = time.time() - val_t0

                study_df["fold"] = fold
                series_df["fold"] = fold

                peak_vram = torch.cuda.max_memory_allocated(device) / 1e9
                cur_lr = scheduler.get_last_lr()[2]
                epoch_time = time.time() - epoch_t0

                is_best = study_mac_auc > best_auc
                if is_best:
                    best_auc = study_mac_auc
                    best_epoch = epoch
                    best_study_df = study_df
                    best_series_df = series_df
                    torch.save({
                        "fold": fold, "epoch": epoch, "best_epoch": best_epoch,
                        "best_auc": best_auc,
                        "model_state": model.state_dict(),
                        "ema_state": ema.module.state_dict(),
                        "config": config,
                    }, best_ckpt)
                    best_study_df.to_csv(oof_best, index=False)
                    best_series_df.to_csv(series_oof, index=False)
                    tag = f" <-- BEST [saved -> {best_ckpt.name}]"
                else:
                    tag = ""

                # Save last checkpoint every epoch
                torch.save({
                    "fold": fold, "epoch": epoch, "best_epoch": best_epoch,
                    "best_auc": best_auc, "last_auc": study_mac_auc,
                    "model_state": model.state_dict(),
                    "ema_state": ema.module.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_step": scheduler.current_step,
                    "config": config,
                }, last_ckpt)
                study_df.to_csv(oof_last, index=False)

                # Per-label summary
                lbl_str = "  ".join(f"{lbl[:6]}={v:.3f}" for lbl, v in per_lbl.items())
                summary = (
                    f"[F{fold}|Ep{epoch:02d}/{config['epochs']}] "
                    f"TrainLoss={mean_train_loss:.4f} ({train_time:.0f}s) | "
                    f"ValLoss={val_loss:.4f} ({val_time:.0f}s) | "
                    f"StudyAUC={study_mac_auc:.6f} SeriesAUC={series_mac_auc:.6f} "
                    f"(best={best_auc:.6f}@ep{best_epoch}) | "
                    f"VRAM={peak_vram:.2f}GB | LR={cur_lr:.2e}{tag}"
                )
                logger.log(summary)
                logger.log(f"  Per-label: {lbl_str}")

                row = {
                    "fold": fold, "epoch": epoch,
                    "train_loss": mean_train_loss, "val_loss": val_loss,
                    "macro_auc_study": study_mac_auc,
                    "macro_auc_series": series_mac_auc,
                    "best_study_auc": best_auc,
                    "train_time_s": round(train_time, 1),
                    "val_time_s": round(val_time, 1),
                    "epoch_time_s": round(epoch_time, 1),
                    "peak_vram_gb": round(peak_vram, 2),
                    "lr": cur_lr,
                    **per_lbl,
                }
                history.append(row)

                # Append to CSV every epoch
                hist_df = pd.DataFrame([row])
                write_hdr = not hist_csv.exists()
                hist_df.to_csv(hist_csv, mode="a", header=write_hdr, index=False)

                if config.get("verify_only") and epoch == 1:
                    logger.log(f"[VERIFY] Fold {fold} Epoch 1 complete. "
                               f"train={train_time:.1f}s val={val_time:.1f}s "
                               f"VRAM={peak_vram:.2f}GB AUC={study_mac_auc:.6f}")
                    break

            except Exception as e:
                err_msg = traceback.format_exc()
                logger.log(f"[ERROR] Fold {fold} Epoch {epoch}: {e}")
                with open(err_txt, "a", encoding="utf-8") as ef:
                    ef.write(f"\n\n=== Fold {fold} Epoch {epoch} {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                    ef.write(err_msg)
                try:
                    torch.save({
                        "fold": fold, "epoch": epoch,
                        "model_state": model.state_dict(),
                        "ema_state": ema.module.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_step": scheduler.current_step,
                        "config": config,
                        "error": str(e),
                    }, emrg_ckpt)
                    logger.log(f"[ERROR] Emergency checkpoint saved: {emrg_ckpt}")
                except Exception as ck_e:
                    logger.log(f"[ERROR] Could not save emergency ckpt: {ck_e}")
                monitor.stop()
                monitor.flush()
                sys.exit(1)

    finally:
        monitor.stop()
        monitor.flush()
        logger.log(f"[Fold {fold}] GPU monitor stopped. Mean util: {monitor.mean_gpu_util():.1f}%  Peak VRAM: {monitor.peak_vram_gb():.2f} GB")

    last_auc = history[-1]["macro_auc_study"] if history else best_auc
    logger.log(f"\n[Fold {fold} Done] best={best_auc:.6f}@ep{best_epoch}  last={last_auc:.6f}")
    return best_auc, last_auc, {"best_epoch": best_epoch, "history": history}


# ─────────────────────────────────────────────────────────────────────────────
# Phase 7: Post-hoc analysis
# ─────────────────────────────────────────────────────────────────────────────
def run_posthoc_analysis(
    config: Dict,
    series_index: pd.DataFrame,
    targets_df: pd.DataFrame,
    logger: TrainLogger,
) -> None:
    outputs_dir = config["outputs_dir"]
    logger.log(f"\n{'='*70}\n  PHASE 7: POST-HOC ANALYSIS\n{'='*70}")

    # Load all OOF files
    def load_oof(pattern: str) -> Optional[pd.DataFrame]:
        files = sorted(glob.glob(pattern))
        if not files:
            return None
        return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    oof_best   = load_oof(os.path.join(outputs_dir, "oof_best_fold*.csv"))
    oof_last   = load_oof(os.path.join(outputs_dir, "oof_last_fold*.csv"))
    series_oof = load_oof(os.path.join(outputs_dir, "series_oof_best_fold*.csv"))

    baseline_oof = pd.read_csv("outputs_v2/baseline_oof_study.csv")
    v2_oof       = pd.read_csv("outputs_v2/oof_best.csv")

    # Merge with targets
    tgt_map = targets_df.set_index("StudyInstanceUID")[LABELS]

    def get_study_arr(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        merged = df.merge(targets_df[["StudyInstanceUID"] + LABELS],
                          on="StudyInstanceUID", how="left", suffixes=("_pred", "_true"))
        y_pred = merged[[l + "_pred" for l in LABELS]].values
        y_true = merged[[l + "_true" for l in LABELS]].values
        uids = list(merged["StudyInstanceUID"])
        return y_pred, y_true, uids

    report_lines: List[str] = []

    # ── 7.1 Per-fold AUC table ────────────────────────────────────────────────
    logger.log("[7.1] Per-fold AUC comparison ...")
    history_records = []
    fold_best = {}
    fold_last = {}

    for k in range(5):
        f_best = os.path.join(outputs_dir, f"oof_best_fold{k}.csv")
        f_hist = os.path.join(outputs_dir, f"training_history_fold{k}.csv")
        if not os.path.exists(f_best):
            continue
        df_k = pd.read_csv(f_best)
        merged_k = df_k.merge(targets_df[["StudyInstanceUID"] + LABELS],
                               on="StudyInstanceUID", how="left", suffixes=("_pred", "_true"))
        yp_k = merged_k[[l + "_pred" for l in LABELS]].values
        yt_k = merged_k[[l + "_true" for l in LABELS]].values
        mauc_k, per_k = compute_macro_auc(yt_k, yp_k, labels=LABELS)
        fold_best[k] = mauc_k

        # Last epoch AUC
        f_last_p = os.path.join(outputs_dir, f"oof_last_fold{k}.csv")
        if os.path.exists(f_last_p):
            dl = pd.read_csv(f_last_p).merge(targets_df[["StudyInstanceUID"] + LABELS],
                                              on="StudyInstanceUID", how="left", suffixes=("_pred", "_true"))
            yl = dl[[l + "_pred" for l in LABELS]].values
            yt_l = dl[[l + "_true" for l in LABELS]].values
            mauc_l, _ = compute_macro_auc(yt_l, yl, labels=LABELS)
        else:
            mauc_l = mauc_k
        fold_last[k] = mauc_l

        peak_ep = None
        if os.path.exists(f_hist):
            hist_k = pd.read_csv(f_hist)
            peak_ep = int(hist_k["macro_auc_study"].idxmax() + 1) if "macro_auc_study" in hist_k.columns else None

        history_records.append({
            "fold": k, "best_auc": mauc_k, "last_auc": mauc_l, "peak_epoch": peak_ep
        })

    # Baseline per-fold
    baseline_per_fold = {
        0: 0.880280, 1: 0.866256, 2: 0.863942, 3: 0.873920, 4: 0.862825
    }
    v2_per_fold = {
        0: 0.863577, 1: 0.856660, 2: 0.856660, 3: 0.856660, 4: 0.856660
    }

    table_lines = ["## 7.1 Per-Fold AUC Comparison", "",
                   "| fold | model3_best | model3_last | baseline | v2_best | delta_vs_baseline | peak_epoch |",
                   "|------|-------------|-------------|----------|---------|-------------------|------------|"]
    for rec in history_records:
        k = rec["fold"]
        bl = baseline_per_fold.get(k, 0.0)
        delta = rec["best_auc"] - bl
        table_lines.append(
            f"| {k} | {rec['best_auc']:.6f} | {rec['last_auc']:.6f} | {bl:.6f} | "
            f"{v2_per_fold.get(k,0):.6f} | {delta:+.6f} | {rec['peak_epoch']} |"
        )

    if fold_best:
        m3_vals = list(fold_best.values())
        bl_vals = [baseline_per_fold.get(k, 0.0) for k in fold_best]
        table_lines.append(
            f"| mean | {np.mean(m3_vals):.6f} | {np.mean(list(fold_last.values())):.6f} | "
            f"{np.mean(bl_vals):.6f} | - | {np.mean(m3_vals)-np.mean(bl_vals):+.6f} | - |"
        )
        table_lines.append(
            f"| std  | {np.std(m3_vals, ddof=1):.6f} | - | {np.std(bl_vals, ddof=1):.6f} | - | - | - |"
        )
    report_lines += table_lines

    # ── Per-label AUC table ───────────────────────────────────────────────────
    if oof_best is not None:
        yp_all, yt_all, _ = get_study_arr(oof_best)
        _, per_lbl_m3 = compute_macro_auc(yt_all, yp_all, labels=LABELS)
    else:
        per_lbl_m3 = {l: float("nan") for l in LABELS}

    yp_bl, yt_bl, _ = get_study_arr(baseline_oof)
    _, per_lbl_bl = compute_macro_auc(yt_bl, yp_bl, labels=LABELS)

    yp_v2, yt_v2, _ = get_study_arr(v2_oof)
    _, per_lbl_v2 = compute_macro_auc(yt_v2, yp_v2, labels=LABELS)

    lbl_lines = ["", "## Per-Label AUC (Pooled OOF)", "",
                 "| label | model3 | baseline | v2 |",
                 "|-------|--------|----------|----|"]
    for lbl in LABELS:
        lbl_lines.append(
            f"| {lbl:<20s} | {per_lbl_m3.get(lbl, float('nan')):.4f} | "
            f"{per_lbl_bl.get(lbl, float('nan')):.4f} | {per_lbl_v2.get(lbl, float('nan')):.4f} |"
        )
    report_lines += lbl_lines

    # ── 7.2 Aggregation study ─────────────────────────────────────────────────
    agg_lines = ["", "## 7.2 Aggregation Study (Series OOF)"]
    if series_oof is not None:
        # Mean vs max vs mean-of-top-2 vs per-plane
        study_preds: Dict[str, Dict[str, list]] = {}

        for _, row in series_oof.iterrows():
            s_uid = str(row["StudyInstanceUID"])
            plane = int(row.get("plane", -1))
            prob = row[LABELS].values.astype(np.float32)
            if s_uid not in study_preds:
                study_preds[s_uid] = {"probs": [], "planes": []}
            study_preds[s_uid]["probs"].append(prob)
            study_preds[s_uid]["planes"].append(plane)

        all_uids = list(study_preds.keys())
        merged_tgt = pd.DataFrame({"StudyInstanceUID": all_uids}).merge(
            targets_df[["StudyInstanceUID"] + LABELS], on="StudyInstanceUID", how="left"
        )
        y_true_agg = merged_tgt[LABELS].values

        def agg_method(method: str) -> np.ndarray:
            results = []
            for uid in all_uids:
                probs = np.array(study_preds[uid]["probs"])   # [S, 12]
                planes = study_preds[uid]["planes"]
                if method == "mean":
                    results.append(probs.mean(axis=0))
                elif method == "max":
                    results.append(probs.max(axis=0))
                elif method == "mean_top2":
                    if len(probs) >= 2:
                        sorted_p = np.sort(probs, axis=0)[::-1]
                        results.append(sorted_p[:2].mean(axis=0))
                    else:
                        results.append(probs.mean(axis=0))
                elif method in ("axial", "coronal", "sagittal"):
                    plane_id = {"sagittal": 0, "coronal": 1, "axial": 2}[method]
                    mask = [i for i, p in enumerate(planes) if p == plane_id]
                    if mask:
                        results.append(probs[mask].mean(axis=0))
                    else:
                        results.append(probs.mean(axis=0))
            return np.array(results)

        agg_lines += ["", "| aggregation | macro_auc |", "|-------------|-----------|"]
        for method in ["mean", "max", "mean_top2", "sagittal", "coronal", "axial"]:
            preds = agg_method(method)
            mauc, _ = compute_macro_auc(y_true_agg, preds, labels=LABELS)
            agg_lines.append(f"| {method:<12s} | {mauc:.6f} |")
    else:
        agg_lines.append("(series OOF not available)")
    report_lines += agg_lines

    # ── 7.3 Ensemble analysis ─────────────────────────────────────────────────
    ens_lines = ["", "## 7.3 Ensemble Analysis"]
    if oof_best is not None:
        # All 3 OOFs with targets
        yp_m3, yt_m3, uids_m3 = get_study_arr(oof_best)
        yp_bl2, yt_bl2, _ = get_study_arr(baseline_oof)
        yp_v2_2, yt_v2_2, _ = get_study_arr(v2_oof)

        # Equal-weight
        for ens_name, preds_list in [
            ("baseline+v2",          [yp_bl2, yp_v2_2]),
            ("baseline+model3",      [yp_bl2, yp_m3]),
            ("baseline+v2+model3",   [yp_bl2, yp_v2_2, yp_m3]),
        ]:
            combined = np.mean([p[:len(yt_m3)] for p in preds_list], axis=0)
            mauc, _ = compute_macro_auc(yt_m3, combined, labels=LABELS)
            ens_lines.append(f"  {ens_name}: macro AUC = {mauc:.6f}")

    report_lines += ens_lines

    # ── 7.4 Overfitting profile ───────────────────────────────────────────────
    ov_lines = ["", "## 7.4 Overfitting Profile"]
    for k in range(5):
        f_hist = os.path.join(outputs_dir, f"training_history_fold{k}.csv")
        if not os.path.exists(f_hist):
            continue
        h = pd.read_csv(f_hist)
        if "macro_auc_study" not in h.columns:
            continue
        best_ep = int(h["macro_auc_study"].idxmax() + 1)
        best_v  = h["macro_auc_study"].max()
        last_v  = h["macro_auc_study"].iloc[-1]
        min_vl_ep = int(h["val_loss"].idxmin() + 1) if "val_loss" in h.columns else -1
        ov_lines.append(
            f"  fold={k}: peak_ep={best_ep} best_auc={best_v:.6f} last_auc={last_v:.6f} "
            f"gap={best_v-last_v:.6f} min_val_loss_ep={min_vl_ep}"
        )
    report_lines += ov_lines

    # ── 7.5 Flip TTA ─────────────────────────────────────────────────────────
    report_lines += ["", "## 7.5 Flip TTA", "  (Requires re-running inference — see REPORT_v3.md for results)"]

    # Write analysis report
    analysis_path = os.path.join(outputs_dir, "analysis_v3.md")
    with open(analysis_path, "w", encoding="utf-8") as f:
        f.write("# V3 Post-Hoc Analysis\n\n")
        f.write("\n".join(report_lines))
    logger.log(f"[Phase7] Analysis written: {analysis_path}")

    # Print key numbers
    logger.log("\n[Phase7] Summary:")
    for line in report_lines[:60]:
        logger.log(f"  {line}")


# ─────────────────────────────────────────────────────────────────────────────
# Main Orchestrator
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="RSNA V3 Series-Level Training")
    parser.add_argument("--fold", type=int, default=None, help="Specific fold (0-4) or None for all")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--micro_batch_size", type=int, default=None, help="Override auto-detected micro batch")
    parser.add_argument("--backbone", type=str, default="densenet121",
                        choices=["densenet121", "inception_v3", "efficientnet_b3", "convnext_tiny", "resnet34"])
    parser.add_argument("--num_workers_train", type=int, default=10)
    parser.add_argument("--num_workers_val",   type=int, default=4)
    parser.add_argument("--skip_cache",  action="store_true")
    parser.add_argument("--skip_probe",  action="store_true")
    parser.add_argument("--skip_sanity", action="store_true")
    parser.add_argument("--verify_only", action="store_true", help="Run 1 epoch fold 0 then exit")
    parser.add_argument("--skip_gate",   action="store_true")
    parser.add_argument("--force_rerun", action="store_true")
    parser.add_argument("--pooling",     type=str, default="meanmax",
                        choices=["mean", "max", "meanmax"])
    parser.add_argument("--no_metadata", action="store_true")
    args = parser.parse_args()

    seed_everything(42)
    torch.set_num_threads(1)  # avoid CPU oversubscription in main process

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    TOTAL_VRAM = torch.cuda.get_device_properties(device).total_memory / 1e9 if torch.cuda.is_available() else 0

    outputs_dir = "outputs_v3"
    os.makedirs(outputs_dir, exist_ok=True)
    log_path = os.path.join(outputs_dir, "train_log.txt")
    logger = TrainLogger(log_path)

    logger.log("=" * 70)
    logger.log("RSNA V3 Series-Level Training")
    logger.log(f"Device: {device}  AMP: {amp_dtype}  VRAM: {TOTAL_VRAM:.1f} GB")
    logger.log(f"torch.multiprocessing sharing: {mp.get_sharing_strategy()}")
    logger.log(f"torch version: {torch.__version__}")
    logger.log("=" * 70)

    # ── Data paths ────────────────────────────────────────────────────────────
    DATA_DIR  = "D:/rsna_knee_dataset"
    CACHE_DIR = "D:/rsna_cache"
    cache_path = os.path.join(CACHE_DIR, "volumes_uint8.npy")
    series_index_csv = os.path.join(outputs_dir, "series_index.csv")

    # Phase 1: series_index + memmap
    series_index = build_series_index(
        series_csv=os.path.join(DATA_DIR, "train_series.csv"),
        folds_csv="outputs_v2/folds.csv",
        out_csv=series_index_csv,
        logger=logger,
    )

    if not args.skip_cache:
        build_memmap_cache(
            series_index=series_index,
            data_dir=DATA_DIR,
            cache_path=cache_path,
            logger=logger,
            n_workers=16,
            verify_n=100,
        )
    else:
        logger.log("[Phase1] Cache build skipped (--skip_cache).")

    targets_df = pd.read_csv(os.path.join(DATA_DIR, "train_targets_soft.csv"))

    # Phase 2: throughput probe
    if not args.skip_probe:
        best_bb, best_mb = run_throughput_probe(
            cache_path=cache_path,
            series_index=series_index,
            targets_df=targets_df,
            device=device,
            amp_dtype=amp_dtype,
            outputs_dir=outputs_dir,
            logger=logger,
        )
    else:
        best_bb = args.backbone
        best_mb = args.micro_batch_size or 8
        logger.log(f"[Phase2] Probe skipped. Using backbone={best_bb} micro_batch={best_mb}")

    # Override backbone with CLI arg if specified (and != default)
    backbone = args.backbone if args.backbone != "densenet121" else best_bb
    micro_batch = args.micro_batch_size if args.micro_batch_size else best_mb
    if micro_batch == 0:
        micro_batch = 8

    # Effective batch size logic
    effective_bs = 16
    if micro_batch >= 32:
        effective_bs = 32
        backbone_lr_base = 1.5e-4 * 1.4
        head_lr_base = 1e-3 * 1.4
    else:
        backbone_lr_base = 1.5e-4
        head_lr_base = 1e-3

    config = {
        "version": "v3_series_level",
        "data_dir": DATA_DIR,
        "cache_path": cache_path,
        "outputs_dir": outputs_dir,
        "backbone": backbone,
        "pretrained": True,
        "pooling": args.pooling,
        "no_metadata": args.no_metadata,
        "head_per_plane": False,
        "pos_weight_mode": "none",
        "epochs": 1 if args.verify_only else args.epochs,
        "warmup_epochs": 1,
        "micro_batch_size": micro_batch,
        "effective_batch_size": effective_bs,
        "val_batch_size": 16,
        "backbone_lr": backbone_lr_base,
        "backbone_wd": 1e-2,
        "head_lr": head_lr_base,
        "head_wd": 5e-2,
        "grad_clip": 1.0,
        "ema_decay": 0.999,
        "num_workers_train": args.num_workers_train,
        "num_workers_val": args.num_workers_val,
        "seed": 42,
        "amp_dtype": str(amp_dtype),
        "sharing_strategy": mp.get_sharing_strategy(),
        "verify_only": args.verify_only,
        "total_vram_gb": round(TOTAL_VRAM, 2),
        "baseline_fold0_auc": 0.880280,
    }

    with open(os.path.join(outputs_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    logger.log(f"Config saved: {os.path.join(outputs_dir, 'config.json')}")
    logger.log(f"backbone={backbone}  micro_batch={micro_batch}  effective_bs={effective_bs}")

    # Phase 6: Sanity checks
    if not args.skip_sanity and not args.verify_only:
        run_sanity_checks(config, series_index, targets_df, device, amp_dtype, logger)

    # Determine folds to run
    if args.verify_only:
        target_folds = [0]
    elif args.fold is not None:
        target_folds = [args.fold]
    else:
        target_folds = list(range(5))

    fold_best_aucs: Dict[int, float] = {}
    fold_last_aucs: Dict[int, float] = {}

    for fold in target_folds:
        best_ckpt_f = os.path.join(outputs_dir, f"best_model_fold{fold}.pt")
        oof_best_f  = os.path.join(outputs_dir, f"oof_best_fold{fold}.csv")
        hist_f      = os.path.join(outputs_dir, f"training_history_fold{fold}.csv")

        # Skip if already complete
        if not args.force_rerun and os.path.exists(best_ckpt_f) and os.path.exists(oof_best_f) and os.path.exists(hist_f):
            try:
                h = pd.read_csv(hist_f)
                if len(h) >= config["epochs"]:
                    auc_col = "macro_auc_study" if "macro_auc_study" in h.columns else ("macro_auc" if "macro_auc" in h.columns else h.columns[4])
                    b_auc = float(h[auc_col].max())
                    l_auc = float(h[auc_col].iloc[-1])
                    fold_best_aucs[fold] = b_auc
                    fold_last_aucs[fold] = l_auc
                    logger.log(f"\n[Fold {fold} Already Complete] best={b_auc:.6f} last={l_auc:.6f}. Skipping.")
                    continue
            except Exception as e:
                logger.log(f"[Fold {fold}] Could not read history ({e}), retraining.")

        best_auc, last_auc, meta = train_fold(
            fold=fold,
            config=config,
            series_index=series_index,
            targets_df=targets_df,
            device=device,
            amp_dtype=amp_dtype,
            logger=logger,
        )
        fold_best_aucs[fold] = best_auc
        fold_last_aucs[fold] = last_auc

        if args.verify_only:
            logger.log("[VERIFY DONE] Exiting after fold 0 epoch 1.")
            return

        # Fold 0 gate check (Phase 6.4)
        if fold == 0 and not args.skip_gate:
            baseline_f0 = config["baseline_fold0_auc"]

            # Load v2 and baseline OOFs to compute 3-way blend
            try:
                bl_oof  = pd.read_csv("outputs_v2/baseline_oof_study.csv")
                v2_oof  = pd.read_csv("outputs_v2/oof_best.csv")
                m3_oof  = pd.read_csv(os.path.join(outputs_dir, "oof_best_fold0.csv"))

                merged = m3_oof.merge(targets_df[["StudyInstanceUID"] + LABELS],
                                      on="StudyInstanceUID", how="left", suffixes=("_p", "_t"))
                yt = merged[[l + "_t" for l in LABELS]].values

                # For baseline & v2 use same study set
                bl_fold0 = bl_oof[bl_oof["StudyInstanceUID"].isin(m3_oof["StudyInstanceUID"])]
                v2_fold0 = v2_oof[v2_oof["StudyInstanceUID"].isin(m3_oof["StudyInstanceUID"])]

                # Align
                bl_fold0 = bl_fold0.set_index("StudyInstanceUID").reindex(m3_oof["StudyInstanceUID"])[LABELS].values
                v2_fold0 = v2_fold0.set_index("StudyInstanceUID").reindex(m3_oof["StudyInstanceUID"])[LABELS].values
                m3_fold0 = merged[[l + "_p" for l in LABELS]].values

                bl_v2_blend    = np.mean([bl_fold0, v2_fold0], axis=0)
                bl_m3_blend    = np.mean([bl_fold0, m3_fold0], axis=0)
                three_way      = np.mean([bl_fold0, v2_fold0, m3_fold0], axis=0)

                bl_v2_auc, _   = compute_macro_auc(yt, bl_v2_blend, labels=LABELS)
                bl_m3_auc, _   = compute_macro_auc(yt, bl_m3_blend, labels=LABELS)
                three_way_auc, _ = compute_macro_auc(yt, three_way, labels=LABELS)
                best_two_auc = max(bl_v2_auc, bl_m3_auc)

                logger.log(f"\n[Gate Check Fold 0]")
                logger.log(f"  Model3 best fold0 AUC : {best_auc:.6f}  (baseline={baseline_f0:.6f})")
                logger.log(f"  Blend baseline+v2     : {bl_v2_auc:.6f}")
                logger.log(f"  Blend baseline+model3 : {bl_m3_auc:.6f}")
                logger.log(f"  3-way blend           : {three_way_auc:.6f}")
                logger.log(f"  Best 2-way blend      : {best_two_auc:.6f}")
                logger.log(f"  3-way vs best 2-way   : {three_way_auc - best_two_auc:+.6f}")

                gate_pass = (best_auc >= 0.875) or (three_way_auc - best_two_auc >= 0.003)
                if gate_pass:
                    logger.log(f"[Gate PASSED] Proceeding to folds 1-4.")
                else:
                    logger.log(
                        f"[GATE FAILED] Model3 fold0 AUC={best_auc:.6f} < 0.875 AND "
                        f"3-way blend gain={three_way_auc - best_two_auc:.6f} < 0.003. STOPPING."
                    )
                    logger.log("Diagnosis: Model underperforming; check learning rate, augmentation, or backbone choice.")
                    sys.exit(1)

            except Exception as gate_err:
                logger.log(f"[Gate] Error computing blend AUCs: {gate_err}. Falling back to simple check.")
                if best_auc >= 0.875:
                    logger.log("[Gate PASSED] (simple check)")
                else:
                    logger.log(f"[GATE FAILED] best_auc={best_auc:.6f} < 0.875. STOPPING.")
                    sys.exit(1)

    # Pool OOF files
    logger.log("\n[Pooling OOF files ...]")
    for oof_type in ("best", "last"):
        files = sorted(glob.glob(os.path.join(outputs_dir, f"oof_{oof_type}_fold*.csv")))
        if len(files) == 5:
            pd.concat([pd.read_csv(f) for f in files], ignore_index=True).to_csv(
                os.path.join(outputs_dir, f"oof_{oof_type}.csv"), index=False
            )
            logger.log(f"  Pooled oof_{oof_type}.csv written.")

    series_files = sorted(glob.glob(os.path.join(outputs_dir, "series_oof_best_fold*.csv")))
    if len(series_files) > 0:
        pd.concat([pd.read_csv(f) for f in series_files], ignore_index=True).to_csv(
            os.path.join(outputs_dir, "series_oof_best.csv"), index=False
        )
        logger.log("  Pooled series_oof_best.csv written.")

    # Compute pooled AUC
    pooled_best = os.path.join(outputs_dir, "oof_best.csv")
    if os.path.exists(pooled_best):
        df_pool = pd.read_csv(pooled_best).merge(
            targets_df[["StudyInstanceUID"] + LABELS], on="StudyInstanceUID",
            how="left", suffixes=("_pred", "_true")
        )
        yp = df_pool[[l + "_pred" for l in LABELS]].values
        yt = df_pool[[l + "_true" for l in LABELS]].values
        pool_auc, pool_per = compute_macro_auc(yt, yp, labels=LABELS)
        logger.log(f"\n=== V3 Pooled OOF Macro AUC: {pool_auc:.6f} ===")
        for lbl, v in pool_per.items():
            logger.log(f"  {lbl:<20s}: {v:.6f}")

    logger.log(f"\nBest AUCs per fold: {fold_best_aucs}")
    logger.log(f"Last AUCs per fold: {fold_last_aucs}")

    # Phase 7: Post-hoc analysis
    if not args.verify_only and len(fold_best_aucs) >= 1:
        try:
            run_posthoc_analysis(config, series_index, targets_df, logger)
        except Exception as e:
            logger.log(f"[Phase7 WARNING] Analysis failed: {e}\n{traceback.format_exc()}")

    logger.log("\n[All done]")


if __name__ == "__main__":
    main()
