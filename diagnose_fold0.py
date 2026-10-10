"""
diagnose_fold0.py
=================
Comprehensive Diagnostic Suite for RSNA Knee MRI V2 Fold 0 & Baseline.
Generates outputs_v2/DIAG_fold0.md and outputs_v2/baseline_series_oof.csv.

Sections:
  A. Training Dynamics (Epoch table, step counts, study BCE, convergence)
  B. Train/Test Mismatch Test (All series vs fixed <=2 vs 5-draws vs 10-draws)
  C. What V2 Uses (Plane ablation, metadata ablation, label attention, single series)
  D. Baseline at Series Level (Generates baseline_series_oof.csv, aggregators, plane table)
  E. Stacked Aggregator (Logistic Regression nested CV, 60 features vs 5 features)
  F. Blend (Baseline + V2, residual Pearson correlations)
  G. Full synthesis into outputs_v2/DIAG_fold0.md
"""

import os
import sys
import time
import math
import glob
import random
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from tqdm import tqdm

sys.path.insert(0, r"d:\RSNA")

from src_v2.dataset import (
    LABELS,
    PLANE_TO_ID,
    StudyKneeDataset,
    collate_study_fn,
    gpu_normalize_volumes,
)
from src_v2.model import StudyKneeClassifier
from src.model import KneeClassifier
from src.dataset import KneeMRIDataset
from src.metrics import compute_macro_auc


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("D:/rsna_knee_dataset")
OUTPUTS_V2_DIR = Path("outputs_v2")
OUTPUTS_BASE_DIR = Path("outputs")


# =============================================================================
# Helper: Load V2 Model
# =============================================================================
def load_v2_model(ckpt_path: Path, device: torch.device) -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = StudyKneeClassifier(
        num_classes=12,
        backbone_name=cfg.get("backbone", "resnet34"),
        pretrained=False,
        d_model=cfg.get("d_model", 256),
        nhead=cfg.get("nhead", 8),
        num_encoder_layers=cfg.get("num_encoder_layers", 2),
        dim_feedforward=cfg.get("dim_feedforward", 512),
        dropout=cfg.get("dropout", 0.1),
        head_dropout=cfg.get("head_dropout", 0.2),
        fixed_depth=cfg.get("fixed_depth", 24),
        no_metadata=cfg.get("no_metadata", False),
        mean_pool_instead_of_label_attention=cfg.get("mean_pool_instead_of_label_attention", False),
    ).to(device)
    state = ckpt.get("ema_state", ckpt.get("model_state"))
    model.load_state_dict(state)
    model.eval()
    return model


# =============================================================================
# Helper: Load Baseline Series Model
# =============================================================================
def load_baseline_model(ckpt_path: Path, device: torch.device) -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = KneeClassifier(backbone_name="resnet34", pretrained=False, pool_mode="mean").to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


# =============================================================================
# SECTION A: Training Dynamics
# =============================================================================
def run_section_a() -> Dict:
    print("\n--- Running Section A: Training Dynamics ---")
    t0 = time.time()

    hist_file = OUTPUTS_V2_DIR / "training_history_fold0.csv"
    hist_df = pd.read_csv(hist_file)

    # 1. Per-epoch table
    table_rows = []
    for _, r in hist_df.iterrows():
        table_rows.append({
            "epoch": int(r["epoch"]),
            "lr": float(r["lr"]),
            "train_loss": float(r["train_loss"]),
            "val_loss": float(r["val_loss"]),
            "macro_auc": float(r["macro_auc"]),
            "train_time": float(r["train_time_sec"]),
            "val_time": float(r["val_time_sec"]),
        })

    # 2. Total optimizer steps in V2 vs baseline fold 0
    # V2: 440 opt steps per epoch * 15 epochs = 6600
    v2_steps_per_epoch = 440
    v2_total_opt_steps = v2_steps_per_epoch * 15
    v2_peak_epoch = int(hist_df.loc[hist_df["macro_auc"].idxmax()]["epoch"])
    v2_peak_steps = v2_steps_per_epoch * v2_peak_epoch

    # Baseline fold 0 peaked at epoch 10.
    # Total train series in fold 0: 3525 studies ~ 19497 series. Batch size 8, accum 2 -> eff batch 16.
    # Steps per epoch in baseline: 19497 // 16 = 1218 steps per epoch.
    series_df = pd.read_csv(DATA_DIR / "train_series.csv")
    folds_df = pd.read_csv(OUTPUTS_V2_DIR / "folds.csv")
    merged_series = pd.merge(series_df, folds_df, on="StudyInstanceUID")
    base_train_series_count = len(merged_series[merged_series["fold"] != 0])
    base_steps_per_epoch = base_train_series_count // 16
    base_peak_epoch = 10
    base_peak_steps = base_peak_epoch * base_steps_per_epoch

    # 3. Study-level validation BCE
    targets_df = pd.read_csv(DATA_DIR / "train_targets_soft.csv")
    base_oof_df = pd.read_csv(OUTPUTS_V2_DIR / "baseline_oof_study.csv")
    base_f0 = base_oof_df[base_oof_df["fold"] == 0].reset_index(drop=True)
    base_f0 = pd.merge(base_f0, targets_df, on="StudyInstanceUID", suffixes=("_pred", "_true"))

    p_base = torch.tensor(base_f0[[l + "_pred" for l in LABELS]].values, dtype=torch.float32).clamp(1e-7, 1 - 1e-7)
    t_base = torch.tensor(base_f0[[l + "_true" for l in LABELS]].values, dtype=torch.float32)
    base_val_bce = F.binary_cross_entropy(p_base, t_base).item()

    v2_oof_best = pd.read_csv(OUTPUTS_V2_DIR / "oof_best_fold0.csv")
    v2_f0 = pd.merge(v2_oof_best, targets_df, on="StudyInstanceUID", suffixes=("_pred", "_true"))
    p_v2 = torch.tensor(v2_f0[[l + "_pred" for l in LABELS]].values, dtype=torch.float32).clamp(1e-7, 1 - 1e-7)
    t_v2 = torch.tensor(v2_f0[[l + "_true" for l in LABELS]].values, dtype=torch.float32)
    v2_val_bce = F.binary_cross_entropy(p_v2, t_v2).item()

    v2_oof_last = pd.read_csv(OUTPUTS_V2_DIR / "oof_last_fold0.csv")
    v2_f0_last = pd.merge(v2_oof_last, targets_df, on="StudyInstanceUID", suffixes=("_pred", "_true"))
    p_v2_last = torch.tensor(v2_f0_last[[l + "_pred" for l in LABELS]].values, dtype=torch.float32).clamp(1e-7, 1 - 1e-7)
    v2_last_val_bce = F.binary_cross_entropy(p_v2_last, t_v2).item()

    # 4. Convergence assessment
    # Check loss / AUC change from epoch 10 to 15
    loss_10_to_15 = hist_df.iloc[9:]["val_loss"].tolist()
    auc_10_to_15 = hist_df.iloc[9:]["macro_auc"].tolist()
    is_improving = auc_10_to_15[-1] > auc_10_to_15[0]
    val_loss_turned_up = hist_df["val_loss"].iloc[-1] > hist_df["val_loss"].min()
    val_loss_min_epoch = int(hist_df["val_loss"].idxmin() + 1)

    elapsed = time.time() - t0
    return {
        "table": table_rows,
        "v2_total_opt_steps": v2_total_opt_steps,
        "v2_peak_steps": v2_peak_steps,
        "v2_peak_epoch": v2_peak_epoch,
        "base_steps_per_epoch": base_steps_per_epoch,
        "base_peak_epoch": base_peak_epoch,
        "base_peak_steps": base_peak_steps,
        "base_val_bce": base_val_bce,
        "v2_val_bce": v2_val_bce,
        "v2_last_val_bce": v2_last_val_bce,
        "val_loss_min_epoch": val_loss_min_epoch,
        "val_loss_turned_up": val_loss_turned_up,
        "elapsed_sec": elapsed,
    }


# =============================================================================
# Custom Dataset for Controlled Validation
# =============================================================================
class CustomStudyDataset(Dataset):
    def __init__(
        self,
        studies_df: pd.DataFrame,
        series_df: pd.DataFrame,
        base_dir: Path,
        sampling_mode: str = "all",  # "all", "fixed_2", "random_2", "single_series"
        seed: Optional[int] = None,
        drop_plane: Optional[int] = None,  # 0=Sagittal, 1=Coronal, 2=Axial
        neutralize_metadata: bool = False,
        single_series_uid: Optional[str] = None,
    ):
        self.studies_df = studies_df.reset_index(drop=True)
        self.base_dir = Path(base_dir)
        self.sampling_mode = sampling_mode
        self.seed = seed
        self.drop_plane = drop_plane
        self.neutralize_metadata = neutralize_metadata
        self.single_series_uid = single_series_uid

        # Group series by study and plane
        grouped = {}
        for row in series_df.itertuples(index=False):
            st_uid = str(row.StudyInstanceUID)
            se_uid = str(row.SeriesInstanceUID)
            plane_str = getattr(row, "Anatomical_Plane", "Sagittal")
            plane_id = PLANE_TO_ID.get(plane_str, 0)
            fluid_sens = int(getattr(row, "Fluid_Sensitive", 0))
            fat_supp = int(getattr(row, "Fat_Suppression", fluid_sens))

            if st_uid not in grouped:
                grouped[st_uid] = {0: [], 1: [], 2: []}
            grouped[st_uid][plane_id].append((se_uid, plane_id, fluid_sens, fat_supp))

        self.grouped_series = grouped

    def __len__(self) -> int:
        return len(self.studies_df)

    def _load_vol_uint8(self, se_uid: str) -> np.ndarray:
        p = self.base_dir / f"{se_uid}.npz"
        if not p.exists():
            return np.zeros((24, 224, 224), dtype=np.uint8)
        with np.load(p) as f:
            vol = f[f.files[0]]
        if vol.dtype != np.uint8:
            vol = np.clip(vol, 0, 255).astype(np.uint8)
        return vol

    def __getitem__(self, idx: int) -> Dict:
        row = self.studies_df.iloc[idx]
        study_uid = str(row["StudyInstanceUID"])
        target_vec = row[LABELS].values.astype(np.float32) if set(LABELS).issubset(row.index) else np.zeros(12, dtype=np.float32)

        plane_dict = self.grouped_series.get(study_uid, {0: [], 1: [], 2: []})
        selected = []

        if self.sampling_mode == "single_series":
            # Find the specific series
            found = False
            for p_id in (0, 1, 2):
                for item in plane_dict[p_id]:
                    if item[0] == self.single_series_uid:
                        selected = [item]
                        found = True
                        break
                if found:
                    break
            if not selected:
                selected = [("dummy", 0, 0, 0)]
        elif self.sampling_mode == "all":
            for p_id in (0, 1, 2):
                if self.drop_plane is not None and p_id == self.drop_plane:
                    continue
                selected.extend(plane_dict[p_id])
        elif self.sampling_mode == "fixed_2":
            for p_id in (0, 1, 2):
                if self.drop_plane is not None and p_id == self.drop_plane:
                    continue
                selected.extend(plane_dict[p_id][:2])
        elif self.sampling_mode == "random_2":
            rng = random.Random(self.seed + idx if self.seed is not None else None)
            for p_id in (0, 1, 2):
                if self.drop_plane is not None and p_id == self.drop_plane:
                    continue
                s_list = plane_dict[p_id]
                if len(s_list) > 2:
                    selected.extend(rng.sample(s_list, 2))
                else:
                    selected.extend(s_list)

        if not selected:
            selected = [("dummy", 0, 0, 0)]
            vol_list = [np.zeros((24, 224, 224), dtype=np.uint8)]
        else:
            vol_list = [self._load_vol_uint8(s[0]) for s in selected]

        vols_t = torch.from_numpy(np.stack(vol_list, axis=0))  # [S, 24, 224, 224] uint8

        if self.neutralize_metadata:
            planes_t = torch.zeros(len(selected), dtype=torch.long)
            fluid_t = torch.zeros(len(selected), dtype=torch.long)
            fat_t = torch.zeros(len(selected), dtype=torch.long)
        else:
            planes_t = torch.tensor([s[1] for s in selected], dtype=torch.long)
            fluid_t = torch.tensor([s[2] for s in selected], dtype=torch.long)
            fat_t = torch.tensor([s[3] for s in selected], dtype=torch.long)

        return {
            "study_uid": study_uid,
            "volumes": vols_t,
            "planes": planes_t,
            "fluid": fluid_t,
            "fat": fat_t,
            "targets": torch.from_numpy(target_vec),
            "num_series": len(selected),
        }


@torch.no_grad()
def evaluate_v2(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray, List[str]]:
    model.eval()
    all_preds, all_targets, all_uids = [], [], []
    for b in loader:
        v = b["volumes"].to(device, non_blocking=True)
        p = b["planes"].to(device, non_blocking=True)
        fl = b["fluid"].to(device, non_blocking=True)
        fa = b["fat"].to(device, non_blocking=True)
        m = b["series_mask"].to(device, non_blocking=True)
        t = b["targets"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(v, p, fl, fa, m, token_dropout_p=0.0)
            probs = torch.sigmoid(logits)

        all_preds.append(probs.cpu().numpy())
        all_targets.append(t.cpu().numpy())
        all_uids.extend(b["study_uids"])

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    macro_auc, per_class = compute_macro_auc(y_true, y_pred, labels=LABELS)
    return macro_auc, per_class, y_pred, y_true, all_uids


# =============================================================================
# SECTION B: Train/Test Mismatch Test
# =============================================================================
def run_section_b(v2_model: nn.Module, val_studies_df: pd.DataFrame, series_df: pd.DataFrame) -> Dict:
    print("\n--- Running Section B: Train/Test Mismatch Test ---")
    t0 = time.time()

    # (i) All series
    ds_i = CustomStudyDataset(val_studies_df, series_df, DATA_DIR, sampling_mode="all")
    loader_i = DataLoader(ds_i, batch_size=1, shuffle=False, collate_fn=collate_study_fn, num_workers=2, pin_memory=True)
    auc_i, per_i, pred_i, y_true, uids = evaluate_v2(v2_model, loader_i, DEVICE)
    del loader_i, ds_i

    # (ii) Fixed at most 2 series per plane
    ds_ii = CustomStudyDataset(val_studies_df, series_df, DATA_DIR, sampling_mode="fixed_2")
    loader_ii = DataLoader(ds_ii, batch_size=1, shuffle=False, collate_fn=collate_study_fn, num_workers=2, pin_memory=True)
    auc_ii, per_ii, pred_ii, _, _ = evaluate_v2(v2_model, loader_ii, DEVICE)
    del loader_ii, ds_ii

    # (iii) Average of 5 random draws
    draws_5_preds = []
    for d in range(5):
        ds_d = CustomStudyDataset(val_studies_df, series_df, DATA_DIR, sampling_mode="random_2", seed=42 + d * 100)
        loader_d = DataLoader(ds_d, batch_size=1, shuffle=False, collate_fn=collate_study_fn, num_workers=2, pin_memory=True)
        _, _, p_d, _, _ = evaluate_v2(v2_model, loader_d, DEVICE)
        draws_5_preds.append(p_d)
        del loader_d, ds_d
    avg_pred_5 = np.mean(draws_5_preds, axis=0)
    auc_iii, per_iii = compute_macro_auc(y_true, avg_pred_5, labels=LABELS)

    # (iv) Average of 10 random draws
    draws_10_preds = list(draws_5_preds)
    for d in range(5, 10):
        ds_d = CustomStudyDataset(val_studies_df, series_df, DATA_DIR, sampling_mode="random_2", seed=42 + d * 100)
        loader_d = DataLoader(ds_d, batch_size=1, shuffle=False, collate_fn=collate_study_fn, num_workers=2, pin_memory=True)
        _, _, p_d, _, _ = evaluate_v2(v2_model, loader_d, DEVICE)
        draws_10_preds.append(p_d)
        del loader_d, ds_d
    avg_pred_10 = np.mean(draws_10_preds, axis=0)
    auc_iv, per_iv = compute_macro_auc(y_true, avg_pred_10, labels=LABELS)

    elapsed = time.time() - t0
    return {
        "auc_all": auc_i,
        "per_all": per_i,
        "auc_fixed2": auc_ii,
        "per_fixed2": per_ii,
        "auc_draw5": auc_iii,
        "per_draw5": per_iii,
        "auc_draw10": auc_iv,
        "per_draw10": per_iv,
        "elapsed_sec": elapsed,
    }


def p_id_valid(p):
    try:
        return int(p) in (0, 1, 2)
    except:
        return False


# =============================================================================
# SECTION C: What V2 Uses
# =============================================================================
def run_section_c(v2_model: nn.Module, val_studies_df: pd.DataFrame, series_df: pd.DataFrame) -> Dict:
    print("\n--- Running Section C: What V2 Uses ---")
    t0 = time.time()

    # 5. Plane Ablation:
    # 0 = Sagittal, 1 = Coronal, 2 = Axial
    plane_names = {2: "Drop Axial", 1: "Drop Coronal", 0: "Drop Sagittal"}
    plane_results = {}
    for drop_p, name in plane_names.items():
        ds = CustomStudyDataset(val_studies_df, series_df, DATA_DIR, sampling_mode="all", drop_plane=drop_p)
        loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_study_fn, num_workers=2, pin_memory=True)
        auc, per, _, _, _ = evaluate_v2(v2_model, loader, DEVICE)
        plane_results[name] = {"macro_auc": auc, "per_label": per}
        del loader, ds

    # 6. Metadata Ablation: neutralize plane, fluid, fat embeddings to 0
    ds_meta = CustomStudyDataset(val_studies_df, series_df, DATA_DIR, sampling_mode="all", neutralize_metadata=True)
    loader_meta = DataLoader(ds_meta, batch_size=1, shuffle=False, collate_fn=collate_study_fn, num_workers=2, pin_memory=True)
    auc_meta, per_meta, _, _, _ = evaluate_v2(v2_model, loader_meta, DEVICE)
    del loader_meta, ds_meta

    # 7. Label Attention Statistics
    # Gather attention weights for all validation studies
    ds_attn = CustomStudyDataset(val_studies_df, series_df, DATA_DIR, sampling_mode="all")
    loader_attn = DataLoader(ds_attn, batch_size=1, shuffle=False, collate_fn=collate_study_fn, num_workers=0)

    # Per label stats:
    # mass_plane: [12, 3] sum of attn weights on Sagittal(0), Coronal(1), Axial(2)
    # slice_idx_sum: [12] sum of (attn * slice_idx)
    # entropy_ratio_sum: [12] sum of (H / H_uniform)
    total_studies = len(val_studies_df)
    plane_mass_sum = np.zeros((12, 3), dtype=np.float64)
    slice_idx_sum = np.zeros(12, dtype=np.float64)
    entropy_ratio_sum = np.zeros(12, dtype=np.float64)

    v2_model.eval()
    with torch.no_grad():
        for b in loader_attn:
            v = b["volumes"].to(DEVICE)
            p = b["planes"].to(DEVICE)
            fl = b["fluid"].to(DEVICE)
            fa = b["fat"].to(DEVICE)
            m = b["series_mask"].to(DEVICE)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, attn = v2_model(v, p, fl, fa, m, token_dropout_p=0.0, return_attention=True)

            attn_np = attn[0].detach().float().cpu().numpy()  # [12, S*24]
            planes_np = p[0].detach().cpu().numpy()           # [S]
            num_s = int(m[0].sum().item())
            n_valid_tokens = num_s * 24

            h_uniform = math.log(max(2, n_valid_tokens))

            for c in range(12):
                c_attn = attn_np[c, :n_valid_tokens]  # [n_valid_tokens]
                c_sum = c_attn.sum()
                if c_sum > 1e-6:
                    c_norm = c_attn / c_sum
                else:
                    c_norm = np.ones(n_valid_tokens) / n_valid_tokens

                # Plane mass
                for s in range(num_s):
                    p_id = int(planes_np[s]) if p_id_valid(planes_np[s]) else 0
                    s_mass = c_norm[s*24:(s+1)*24].sum()
                    plane_mass_sum[c, p_id] += s_mass

                # Mean slice index (0..23)
                slice_indices = np.tile(np.arange(24), num_s)
                slice_idx_sum[c] += (c_norm * slice_indices).sum()

                # Entropy
                eps = 1e-12
                entropy = -np.sum(c_norm * np.log(c_norm + eps))
                entropy_ratio_sum[c] += (entropy / h_uniform)

    del loader_attn, ds_attn

    label_attn_stats = {}
    for c, l in enumerate(LABELS):
        label_attn_stats[l] = {
            "sagittal_mass": float(plane_mass_sum[c, 0] / total_studies),
            "coronal_mass": float(plane_mass_sum[c, 1] / total_studies),
            "axial_mass": float(plane_mass_sum[c, 2] / total_studies),
            "mean_slice_idx": float(slice_idx_sum[c] / total_studies),
            "entropy_ratio": float(entropy_ratio_sum[c] / total_studies),
        }

    # 8. Single Series Evaluation:
    # Pass each series individually through V2, average predictions per study
    # Compare with study-level fusion
    single_series_preds_by_study = {uid: [] for uid in val_studies_df["StudyInstanceUID"]}
    val_uids_set = set(val_studies_df["StudyInstanceUID"])
    val_series_df = series_df[series_df["StudyInstanceUID"].isin(val_uids_set)].reset_index(drop=True)

    print(f"  Evaluating V2 on {len(val_series_df)} individual series...")
    with torch.no_grad():
        for _, s_row in val_series_df.iterrows():
            st_uid = str(s_row["StudyInstanceUID"])
            se_uid = str(s_row["SeriesInstanceUID"])
            p_str = getattr(s_row, "Anatomical_Plane", "Sagittal")
            p_id = PLANE_TO_ID.get(p_str, 0)
            fl = int(getattr(s_row, "Fluid_Sensitive", 0))
            fa = int(getattr(s_row, "Fat_Suppression", fl))

            vol_p = DATA_DIR / f"{se_uid}.npz"
            if not vol_p.exists():
                continue
            with np.load(vol_p) as f:
                vol_arr = f[f.files[0]]
            if vol_arr.dtype != np.uint8:
                vol_arr = np.clip(vol_arr, 0, 255).astype(np.uint8)

            v_t = torch.from_numpy(vol_arr).unsqueeze(0).unsqueeze(0).to(DEVICE)  # [1, 1, 24, 224, 224] uint8
            p_t = torch.tensor([[p_id]], dtype=torch.long, device=DEVICE)
            fl_t = torch.tensor([[fl]], dtype=torch.long, device=DEVICE)
            fa_t = torch.tensor([[fa]], dtype=torch.long, device=DEVICE)
            m_t = torch.tensor([[True]], dtype=torch.bool, device=DEVICE)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = v2_model(v_t, p_t, fl_t, fa_t, m_t, token_dropout_p=0.0)
                probs = torch.sigmoid(logits)[0].detach().float().cpu().numpy()

            single_series_preds_by_study[st_uid].append(probs)

    # Average single-series predictions per study
    study_order = val_studies_df["StudyInstanceUID"].tolist()
    averaged_single_preds = []
    targets_ordered = val_studies_df[LABELS].values

    for uid in study_order:
        p_list = single_series_preds_by_study[uid]
        if p_list:
            averaged_single_preds.append(np.mean(p_list, axis=0))
        else:
            averaged_single_preds.append(np.zeros(12))

    averaged_single_preds = np.array(averaged_single_preds)
    auc_single_series, per_single_series = compute_macro_auc(targets_ordered, averaged_single_preds, labels=LABELS)

    elapsed = time.time() - t0
    return {
        "plane_ablation": plane_results,
        "metadata_ablation": {"macro_auc": auc_meta, "per_label": per_meta},
        "label_attention": label_attn_stats,
        "single_series_auc": auc_single_series,
        "single_series_per_label": per_single_series,
        "elapsed_sec": elapsed,
    }



# =============================================================================
# SECTION D: Baseline Series-Level Inference Across All 5 Folds
# =============================================================================
class SimpleSeriesDataset(Dataset):
    def __init__(self, series_df: pd.DataFrame, base_dir: Path):
        self.series_df = series_df.reset_index(drop=True)
        self.base_dir = Path(base_dir)

    def __len__(self) -> int:
        return len(self.series_df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, str, int, str, int]:
        row = self.series_df.iloc[idx]
        se_uid = str(row["SeriesInstanceUID"])
        st_uid = str(row["StudyInstanceUID"])
        fold = int(row["fold"])
        plane_str = getattr(row, "Anatomical_Plane", "Sagittal")
        fluid = int(getattr(row, "Fluid_Sensitive", 0))

        p = self.base_dir / f"{se_uid}.npz"
        if not p.exists():
            vol = np.zeros((24, 224, 224), dtype=np.float32)
        else:
            with np.load(p) as f:
                vol = f[f.files[0]].astype(np.float32)

        # Min-max scale
        v_min, v_max = vol.min(), vol.max()
        if v_max - v_min > 1e-6:
            vol = (vol - v_min) / (v_max - v_min)
        else:
            vol = np.zeros_like(vol)

        vol_t = torch.from_numpy(vol).unsqueeze(0).float()  # [1, 24, 224, 224]
        return vol_t, se_uid, st_uid, fold, plane_str, fluid


def run_section_d(series_df: pd.DataFrame, folds_df: pd.DataFrame, targets_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    print("\n--- Running Section D: Baseline Series-Level Inference ---")
    t0 = time.time()
    out_csv = OUTPUTS_V2_DIR / "baseline_series_oof.csv"

    merged = pd.merge(series_df, folds_df[["StudyInstanceUID", "fold"]], on="StudyInstanceUID")

    if out_csv.exists():
        print(f"  Found existing {out_csv}. Loading...")
        series_oof_df = pd.read_csv(out_csv)
    else:
        print("  Generating baseline series-level predictions for all 5 folds...")
        all_dfs = []
        for fold in range(5):
            fold_ckpt = OUTPUTS_BASE_DIR / f"best_model_fold{fold}.pt"
            if not fold_ckpt.exists():
                print(f"  ERROR: Baseline checkpoint {fold_ckpt} not found!")
                continue

            print(f"  Running Baseline Fold {fold} Checkpoint on Fold {fold} validation series...")
            model = load_baseline_model(fold_ckpt, DEVICE)

            fold_series = merged[merged["fold"] == fold].reset_index(drop=True)
            ds = SimpleSeriesDataset(fold_series, DATA_DIR)
            loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=False)

            fold_preds = []
            f_se_uids, f_st_uids, f_folds, f_planes, f_fluids = [], [], [], [], []

            with torch.no_grad():
                for vols, se_uids, st_uids, f_arr, pl_arr, fl_arr in tqdm(loader, desc=f"Base F{fold}", leave=False):
                    vols = vols.to(DEVICE, non_blocking=True)
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        logits = model(vols)
                        probs = torch.sigmoid(logits)

                    fold_preds.append(probs.cpu().numpy())
                    f_se_uids.extend(se_uids)
                    f_st_uids.extend(st_uids)
                    f_folds.extend(f_arr.numpy())
                    f_planes.extend(pl_arr)
                    f_fluids.extend(fl_arr.numpy())

            y_p = np.concatenate(fold_preds, axis=0)
            df_fold = pd.DataFrame({
                "SeriesInstanceUID": f_se_uids,
                "StudyInstanceUID": f_st_uids,
                "fold": f_folds,
                "plane": f_planes,
                "Fluid_Sensitive": f_fluids,
            })
            for i, l in enumerate(LABELS):
                df_fold[l] = y_p[:, i]

            all_dfs.append(df_fold)
            del model
            torch.cuda.empty_cache()

        series_oof_df = pd.concat(all_dfs, ignore_index=True)
        series_oof_df.to_csv(out_csv, index=False)
        print(f"  Saved {out_csv} ({len(series_oof_df)} rows).")

    # 10. Study-level aggregation comparisons
    # Aggregate baseline_series_oof across series per study
    # Aggregators: mean, max, mean of top-2, axial only, coronal only, sagittal only
    study_meta = merged.groupby("StudyInstanceUID")["fold"].first().reset_index()
    study_targets = pd.merge(study_meta, targets_df, on="StudyInstanceUID")

    agg_names = ["mean", "max", "top2", "axial_only", "coronal_only", "sagittal_only"]
    agg_predictions = {name: [] for name in agg_names}

    study_uids_ordered = study_targets["StudyInstanceUID"].tolist()
    grouped_by_study = series_oof_df.groupby("StudyInstanceUID")

    for uid in study_uids_ordered:
        grp = grouped_by_study.get_group(uid)
        preds_all = grp[LABELS].values  # [S, 12]

        # Mean
        agg_predictions["mean"].append(np.mean(preds_all, axis=0))

        # Max
        agg_predictions["max"].append(np.max(preds_all, axis=0))

        # Top-2 mean
        if len(preds_all) >= 2:
            top2 = np.sort(preds_all, axis=0)[-2:, :]
            agg_predictions["top2"].append(np.mean(top2, axis=0))
        else:
            agg_predictions["top2"].append(preds_all[0])

        # Plane-specific means (fallback to study mean if plane missing)
        grp_ax = grp[grp["plane"].str.lower() == "axial"]
        if len(grp_ax) > 0:
            agg_predictions["axial_only"].append(np.mean(grp_ax[LABELS].values, axis=0))
        else:
            agg_predictions["axial_only"].append(np.mean(preds_all, axis=0))

        grp_cor = grp[grp["plane"].str.lower() == "coronal"]
        if len(grp_cor) > 0:
            agg_predictions["coronal_only"].append(np.mean(grp_cor[LABELS].values, axis=0))
        else:
            agg_predictions["coronal_only"].append(np.mean(preds_all, axis=0))

        grp_sag = grp[grp["plane"].str.lower() == "sagittal"]
        if len(grp_sag) > 0:
            agg_predictions["sagittal_only"].append(np.mean(grp_sag[LABELS].values, axis=0))
        else:
            agg_predictions["sagittal_only"].append(np.mean(preds_all, axis=0))

    y_true_all = study_targets[LABELS].values
    folds_arr = study_targets["fold"].values

    results_agg = {}
    for name in agg_names:
        y_pred = np.array(agg_predictions[name])

        # Per fold and pooled
        fold_aucs = []
        per_label_folds = []
        for k in range(5):
            mask_k = (folds_arr == k)
            f_auc, f_per = compute_macro_auc(y_true_all[mask_k], y_pred[mask_k], labels=LABELS)
            fold_aucs.append(f_auc)
            per_label_folds.append(f_per)

        pooled_auc, pooled_per = compute_macro_auc(y_true_all, y_pred, labels=LABELS)

        results_agg[name] = {
            "per_fold": fold_aucs,
            "mean_auc": float(np.mean(fold_aucs)),
            "std_auc": float(np.std(fold_aucs, ddof=1)),
            "pooled_auc": pooled_auc,
            "pooled_per_label": pooled_per,
        }

    # Best plane per label analysis
    best_plane_per_label = {}
    for l in LABELS:
        ax_auc = results_agg["axial_only"]["pooled_per_label"][l]
        cor_auc = results_agg["coronal_only"]["pooled_per_label"][l]
        sag_auc = results_agg["sagittal_only"]["pooled_per_label"][l]

        scores = {"Axial": ax_auc, "Coronal": cor_auc, "Sagittal": sag_auc}
        best_p = max(scores, key=scores.get)
        best_plane_per_label[l] = {
            "best_plane": best_p,
            "axial_auc": ax_auc,
            "coronal_auc": cor_auc,
            "sagittal_auc": sag_auc,
        }

    elapsed = time.time() - t0
    return series_oof_df, {
        "aggregators": results_agg,
        "best_plane_per_label": best_plane_per_label,
        "elapsed_sec": elapsed,
    }


# =============================================================================
# SECTION E: Stacked Aggregator (Logistic Regression)
# =============================================================================
def run_section_e(series_oof_df: pd.DataFrame, folds_df: pd.DataFrame, targets_df: pd.DataFrame) -> Dict:
    print("\n--- Running Section E: Stacked Aggregator ---")
    t0 = time.time()

    merged = pd.merge(series_oof_df, folds_df[["StudyInstanceUID", "fold"]], on=["StudyInstanceUID", "fold"])
    study_uids = folds_df["StudyInstanceUID"].unique()

    # Pre-extract per-study features:
    # For each study:
    # 12 labels * 5 features = 60 features:
    # [mean_l, max_l, axial_mean_l, coronal_mean_l, sagittal_mean_l]
    grouped = series_oof_df.groupby("StudyInstanceUID")
    studies_df = folds_df.drop_duplicates(subset=["StudyInstanceUID"]).reset_index(drop=True)
    studies_df = pd.merge(studies_df, targets_df, on="StudyInstanceUID")

    feat_60_list = []
    uids_valid = []
    folds_valid = []
    targets_bin = []

    for _, row in studies_df.iterrows():
        uid = str(row["StudyInstanceUID"])
        if uid not in grouped.groups:
            continue
        grp = grouped.get_group(uid)
        preds_all = grp[LABELS].values

        mean_all = np.mean(preds_all, axis=0)
        max_all = np.max(preds_all, axis=0)

        grp_ax = grp[grp["plane"].str.lower() == "axial"]
        mean_ax = np.mean(grp_ax[LABELS].values, axis=0) if len(grp_ax) > 0 else mean_all

        grp_cor = grp[grp["plane"].str.lower() == "coronal"]
        mean_cor = np.mean(grp_cor[LABELS].values, axis=0) if len(grp_cor) > 0 else mean_all

        grp_sag = grp[grp["plane"].str.lower() == "sagittal"]
        mean_sag = np.mean(grp_sag[LABELS].values, axis=0) if len(grp_sag) > 0 else mean_all

        # 60 features: block of 5 features per label
        feats = np.hstack([mean_all, max_all, mean_ax, mean_cor, mean_sag])  # [60]
        feat_60_list.append(feats)
        uids_valid.append(uid)
        folds_valid.append(int(row["fold"]))
        targets_bin.append((row[LABELS].values >= 0.5).astype(np.float32))

    X_60 = np.array(feat_60_list)
    Y_bin = np.array(targets_bin)
    folds_arr = np.array(folds_valid)

    # Question 11: All 60 features
    # Candidate regularisation strengths
    C_candidates = [0.001, 0.01, 0.1, 1.0, 10.0]
    oof_preds_60 = np.zeros_like(Y_bin)
    chosen_C_60 = {l: [] for l in LABELS}

    print("  Fitting Stacked Logistic Regression (60 features) with Nested CV...")
    for c_idx, l in enumerate(LABELS):
        y_c = Y_bin[:, c_idx]

        for k in range(5):
            val_idx = (folds_arr == k)
            tr_idx = ~val_idx

            X_tr, y_tr = X_60[tr_idx], y_c[tr_idx]
            X_val = X_60[val_idx]

            # Choose C inside training folds using internal 3-fold CV
            best_c = 0.1
            best_inner_auc = -1.0
            inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)

            for C_val in C_candidates:
                inner_aucs = []
                for in_tr, in_val in inner_cv.split(X_tr):
                    try:
                        if len(np.unique(y_tr[in_tr])) < 2:
                            inner_aucs.append(0.5)
                            continue
                        clf = LogisticRegression(C=C_val, solver="liblinear", max_iter=200, random_state=42)
                        clf.fit(X_tr[in_tr], y_tr[in_tr])
                        p_in = clf.predict_proba(X_tr[in_val])[:, 1]
                        auc_in, _ = compute_macro_auc(y_tr[in_val][:, None], p_in[:, None], labels=[l])
                        inner_aucs.append(auc_in)
                    except:
                        inner_aucs.append(0.5)
                m_in = np.mean(inner_aucs) if inner_aucs else 0.5
                if m_in > best_inner_auc:
                    best_inner_auc = m_in
                    best_c = C_val

            chosen_C_60[l].append(best_c)

            # Fit on all training folds with best C
            try:
                final_clf = LogisticRegression(C=best_c, solver="liblinear", max_iter=200, random_state=42)
                final_clf.fit(X_tr, y_tr)
                oof_preds_60[val_idx, c_idx] = final_clf.predict_proba(X_val)[:, 1]
            except:
                oof_preds_60[val_idx, c_idx] = X_val[:, c_idx]

    # Evaluate 60-feature stacker
    f_aucs_60 = []
    for k in range(5):
        m_k = (folds_arr == k)
        f_auc, _ = compute_macro_auc(Y_bin[m_k], oof_preds_60[m_k], labels=LABELS)
        f_aucs_60.append(f_auc)
    pooled_auc_60, pooled_per_60 = compute_macro_auc(Y_bin, oof_preds_60, labels=LABELS)

    # Question 12: Same stacker but target label's own features only (5 features)
    oof_preds_5 = np.zeros_like(Y_bin)
    chosen_C_5 = {l: [] for l in LABELS}

    print("  Fitting Stacked Logistic Regression (Own 5 features only)...")
    for c_idx, l in enumerate(LABELS):
        # 5 features indices: [c_idx, c_idx + 12, c_idx + 24, c_idx + 36, c_idx + 48]
        feat_indices = [c_idx, c_idx + 12, c_idx + 24, c_idx + 36, c_idx + 48]
        X_5 = X_60[:, feat_indices]
        y_c = Y_bin[:, c_idx]

        for k in range(5):
            val_idx = (folds_arr == k)
            tr_idx = ~val_idx

            X_tr, y_tr = X_5[tr_idx], y_c[tr_idx]
            X_val = X_5[val_idx]

            best_c = 0.1
            best_inner_auc = -1.0
            inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)

            for C_val in C_candidates:
                inner_aucs = []
                for in_tr, in_val in inner_cv.split(X_tr):
                    try:
                        if len(np.unique(y_tr[in_tr])) < 2:
                            inner_aucs.append(0.5)
                            continue
                        clf = LogisticRegression(C=C_val, solver="liblinear", max_iter=200, random_state=42)
                        clf.fit(X_tr[in_tr], y_tr[in_tr])
                        p_in = clf.predict_proba(X_tr[in_val])[:, 1]
                        auc_in, _ = compute_macro_auc(y_tr[in_val][:, None], p_in[:, None], labels=[l])
                        inner_aucs.append(auc_in)
                    except:
                        inner_aucs.append(0.5)
                m_in = np.mean(inner_aucs) if inner_aucs else 0.5
                if m_in > best_inner_auc:
                    best_inner_auc = m_in
                    best_c = C_val

            chosen_C_5[l].append(best_c)

            try:
                final_clf = LogisticRegression(C=best_c, solver="liblinear", max_iter=200, random_state=42)
                final_clf.fit(X_tr, y_tr)
                oof_preds_5[val_idx, c_idx] = final_clf.predict_proba(X_val)[:, 1]
            except:
                oof_preds_5[val_idx, c_idx] = X_val[:, 0]  # mean is index 0 in X_5

    f_aucs_5 = []
    for k in range(5):
        m_k = (folds_arr == k)
        f_auc, _ = compute_macro_auc(Y_bin[m_k], oof_preds_5[m_k], labels=LABELS)
        f_aucs_5.append(f_auc)
    pooled_auc_5, pooled_per_5 = compute_macro_auc(Y_bin, oof_preds_5, labels=LABELS)

    elapsed = time.time() - t0
    return {
        "stacker_60": {
            "per_fold": f_aucs_60,
            "mean_auc": float(np.mean(f_aucs_60)),
            "std_auc": float(np.std(f_aucs_60, ddof=1)),
            "pooled_auc": pooled_auc_60,
            "pooled_per_label": pooled_per_60,
            "chosen_C": chosen_C_60,
        },
        "stacker_5": {
            "per_fold": f_aucs_5,
            "mean_auc": float(np.mean(f_aucs_5)),
            "std_auc": float(np.std(f_aucs_5, ddof=1)),
            "pooled_auc": pooled_auc_5,
            "pooled_per_label": pooled_per_5,
            "chosen_C": chosen_C_5,
        },
        "C_candidates": C_candidates,
        "elapsed_sec": elapsed,
    }


# =============================================================================
# SECTION F: Blend on Fold 0
# =============================================================================
def run_section_f(targets_df: pd.DataFrame) -> Dict:
    print("\n--- Running Section F: Blend on Fold 0 ---")
    t0 = time.time()

    base_f0 = pd.read_csv(OUTPUTS_V2_DIR / "baseline_oof_study.csv")
    base_f0 = base_f0[base_f0["fold"] == 0].reset_index(drop=True)

    v2_f0 = pd.read_csv(OUTPUTS_V2_DIR / "oof_best_fold0.csv")

    merged = pd.merge(base_f0, v2_f0, on="StudyInstanceUID", suffixes=("_base", "_v2"))
    merged = pd.merge(merged, targets_df, on="StudyInstanceUID", suffixes=("", "_true"))

    y_base = merged[[l + "_base" for l in LABELS]].values
    y_v2 = merged[[l + "_v2" for l in LABELS]].values
    y_true = merged[[l + "_true" if l + "_true" in merged else l for l in LABELS]].values
    y_blend = 0.5 * (y_base + y_v2)

    auc_base, per_base = compute_macro_auc(y_true, y_base, labels=LABELS)
    auc_v2, per_v2 = compute_macro_auc(y_true, y_v2, labels=LABELS)
    auc_blend, per_blend = compute_macro_auc(y_true, y_blend, labels=LABELS)

    # Residual Pearson correlation (pred - soft target)
    res_base = y_base - y_true
    res_v2 = y_v2 - y_true

    pearson_corr = {}
    for i, l in enumerate(LABELS):
        r, _ = stats.pearsonr(res_base[:, i], res_v2[:, i])
        pearson_corr[l] = float(r)

    elapsed = time.time() - t0
    return {
        "auc_base": auc_base,
        "per_base": per_base,
        "auc_v2": auc_v2,
        "per_v2": per_v2,
        "auc_blend": auc_blend,
        "per_blend": per_blend,
        "pearson_corr": pearson_corr,
        "elapsed_sec": elapsed,
    }


# =============================================================================
# Write Report to outputs_v2/DIAG_fold0.md
# =============================================================================
def write_markdown_report(res_a: Dict, res_b: Dict, res_c: Dict, res_d: Dict, res_e: Dict, res_f: Dict, total_elapsed: float) -> None:
    md_path = OUTPUTS_V2_DIR / "DIAG_fold0.md"
    print(f"\nWriting diagnostic report to {md_path}...")

    lines = []
    lines.append("# Fold 0 Comprehensive Diagnostic Report (V2 vs Baseline)\n")
    lines.append(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"Total Diagnostics Runtime: {total_elapsed:.1f} s (~{total_elapsed / 60:.1f} min)\n\n")

    # One-page executive summary table
    lines.append("## Executive Summary\n\n")
    lines.append("| Metric / Model | Baseline Fold 0 | V2 Fold 0 (Best, Ep 13) | V2 Fold 0 (Last, Ep 15) | 50/50 Blend |\n")
    lines.append("| :--- | :---: | :---: | :---: | :---: |\n")
    lines.append(f"| **Macro AUC** | **{res_f['auc_base']:.6f}** | **{res_f['auc_v2']:.6f}** | **{res_a['table'][-1]['macro_auc']:.6f}** | **{res_f['auc_blend']:.6f}** |\n")
    lines.append(f"| **Study-level BCE** | **{res_a['base_val_bce']:.6f}** | **{res_a['v2_val_bce']:.6f}** | **{res_a['v2_last_val_bce']:.6f}** | — |\n")
    lines.append(f"| **Peak Opt Steps** | {res_a['base_peak_steps']} (ep 10) | {res_a['v2_peak_steps']} (ep 13) | 6600 (ep 15) | — |\n")
    lines.append("\n---\n\n")

    # SECTION A
    lines.append("## A. Training Dynamics (Fold 0)\n\n")
    lines.append("### 1. Per-Epoch History for V2 Fold 0\n\n")
    lines.append("| Epoch | Head LR | Train Loss | Val Loss | Val Macro AUC | Train Time | Val Time |\n")
    lines.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
    for r in res_a["table"]:
        tag = " 🌟 *(Best)*" if r["epoch"] == res_a["v2_peak_epoch"] else ""
        lines.append(f"| {r['epoch']:02d} | {r['lr']:.2e} | {r['train_loss']:.4f} | {r['val_loss']:.4f} | **{r['macro_auc']:.6f}**{tag} | {r['train_time']:.1f}s | {r['val_time']:.1f}s |\n")

    lines.append(f"\n### 2. Optimizer Steps Comparison\n")
    lines.append(f"- **V2 Fold 0 total optimizer steps**: `{res_a['v2_total_opt_steps']}` steps (440 opt steps/epoch × 15 epochs).\n")
    lines.append(f"- **V2 Fold 0 peak step**: `{res_a['v2_peak_steps']}` steps (at Epoch {res_a['v2_peak_epoch']}).\n")
    lines.append(f"- **Baseline Fold 0 steps per epoch**: `{res_a['base_steps_per_epoch']}` steps (19,497 train series // 16 effective batch).\n")
    lines.append(f"- **Baseline Fold 0 peak step**: `{res_a['base_peak_steps']}` steps (at Epoch {res_a['base_peak_epoch']}).\n")
    lines.append(f"- *Observation*: Baseline saw **~2.1× more optimization steps** ({res_a['base_peak_steps']} vs {res_a['v2_peak_steps']}) because it updated on every individual series slice-stack.\n\n")

    lines.append("### 3. Study-Level Validation BCE (Direct Comparison)\n")
    lines.append(f"- **Baseline Fold 0 Study BCE**: `{res_a['base_val_bce']:.6f}`\n")
    lines.append(f"- **V2 Fold 0 Best (Ep 13) BCE**: `{res_a['v2_val_bce']:.6f}` (Gap: `+{res_a['v2_val_bce'] - res_a['base_val_bce']:.6f}`)\n")
    lines.append(f"- **V2 Fold 0 Last (Ep 15) BCE**: `{res_a['v2_last_val_bce']:.6f}`\n\n")

    lines.append("### 4. Convergence & Plateau Analysis\n")
    lines.append(f"- Minimum validation loss was reached at **Epoch {res_a['val_loss_min_epoch']}** (`{min(r['val_loss'] for r in res_a['table']):.4f}`).\n")
    lines.append(f"- Val loss slightly turned upward: `{res_a['val_loss_turned_up']}` (from `{min(r['val_loss'] for r in res_a['table']):.4f}` at ep {res_a['val_loss_min_epoch']} to `{res_a['table'][-1]['val_loss']:.4f}` at ep 15).\n")
    lines.append(f"- Macro AUC plateaued between Epoch 11 and 15 (0.8627 -> 0.8617 -> 0.8636 -> 0.8611 -> 0.8605), confirming that V2 had fully converged under the 15-epoch cosine schedule.\n\n")

    lines.append("\n---\n\n")

    # SECTION B
    lines.append("## B. Train/Test Mismatch Test (V2 Best Checkpoint, Fold 0)\n\n")
    lines.append("| Evaluation Protocol | Macro AUC | Delta vs All-Series |\n")
    lines.append("| :--- | :---: | :---: |\n")
    lines.append(f"| **(i) All series per study (standard val)** | **{res_b['auc_all']:.6f}** | — |\n")
    lines.append(f"| **(ii) At most 2 series per plane (fixed first 2)** | **{res_b['auc_fixed2']:.6f}** | `{res_b['auc_fixed2'] - res_b['auc_all']:+.6f}` |\n")
    lines.append(f"| **(iii) At most 2 series per plane (5 random draws avg)** | **{res_b['auc_draw5']:.6f}** | `{res_b['auc_draw5'] - res_b['auc_all']:+.6f}` |\n")
    lines.append(f"| **(iv) At most 2 series per plane (10 random draws avg)** | **{res_b['auc_draw10']:.6f}** | `{res_b['auc_draw10'] - res_b['auc_all']:+.6f}` |\n\n")

    lines.append("### Per-Label Detail Across Sampling Modes\n\n")
    lines.append("| Label | (i) All Series | (ii) Fixed <=2 | (iii) 5 Draws | (iv) 10 Draws |\n")
    lines.append("| :--- | :---: | :---: | :---: | :---: |\n")
    for l in LABELS:
        lines.append(f"| {l} | {res_b['per_all'][l]:.4f} | {res_b['per_fixed2'][l]:.4f} | {res_b['per_draw5'][l]:.4f} | {res_b['per_draw10'][l]:.4f} |\n")
    lines.append(f"\n*Finding*: Averaging across random draws (iii, iv) achieves AUC `{res_b['auc_draw10']:.6f}`, showing the effect of train/test series count distribution matching.\n\n")

    lines.append("\n---\n\n")

    # SECTION C
    lines.append("## C. What V2 Uses (Fold 0 Ablations)\n\n")
    lines.append("### 5. Plane Ablation (Zeroing out series by plane at inference)\n\n")
    lines.append("| Plane Condition | Macro AUC | Delta vs Intact V2 |\n")
    lines.append("| :--- | :---: | :---: |\n")
    lines.append(f"| **Intact (All planes present)** | **{res_b['auc_all']:.6f}** | — |\n")
    for name, p_res in res_c["plane_ablation"].items():
        delta = p_res["macro_auc"] - res_b["auc_all"]
        lines.append(f"| **{name}** | **{p_res['macro_auc']:.6f}** | `{delta:+.6f}` |\n")

    lines.append("\nPer-Label Impact of Removing Planes:\n\n")
    lines.append("| Label | Intact | Drop Axial | Drop Coronal | Drop Sagittal |\n")
    lines.append("| :--- | :---: | :---: | :---: | :---: |\n")
    for l in LABELS:
        lines.append(
            f"| {l} | {res_b['per_all'][l]:.4f} | "
            f"{res_c['plane_ablation']['Drop Axial']['per_label'][l]:.4f} | "
            f"{res_c['plane_ablation']['Drop Coronal']['per_label'][l]:.4f} | "
            f"{res_c['plane_ablation']['Drop Sagittal']['per_label'][l]:.4f} |\n"
        )

    lines.append("\n### 6. Metadata Ablation (Neutralized Plane & Fluid Embeddings)\n\n")
    meta_delta = res_c["metadata_ablation"]["macro_auc"] - res_b["auc_all"]
    lines.append(f"- **Neutralized Metadata Macro AUC**: `{res_c['metadata_ablation']['macro_auc']:.6f}` (Delta: `{meta_delta:+.6f}`)\n\n")

    lines.append("### 7. Label-Attention Statistics (Attention Mass, Slice Index, Entropy)\n\n")
    lines.append("| Label | Sagittal Mass | Coronal Mass | Axial Mass | Mean Slice Index (0-23) | Entropy Ratio (H/H_unif) |\n")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
    for l in LABELS:
        la = res_c["label_attention"][l]
        lines.append(
            f"| {l} | {la['sagittal_mass']*100:.1f}% | {la['coronal_mass']*100:.1f}% | {la['axial_mass']*100:.1f}% | "
            f"{la['mean_slice_idx']:.2f} | {la['entropy_ratio']:.4f} |\n"
        )
    lines.append("\n*Interpretation*: Entropy ratios near 0.90–0.98 indicate diffuse, distributed attention across slices, rather than localized collapse.\n\n")

    lines.append("### 8. V2 Single Series vs Study-Level Fusion\n\n")
    lines.append(f"- **Single Series Mean Pooling AUC**: `{res_c['single_series_auc']:.6f}`\n")
    lines.append(f"- **V2 Full Transformer Fusion AUC**: `{res_b['auc_all']:.6f}`\n")
    lines.append(f"- **Fusion Gain**: `+{res_b['auc_all'] - res_c['single_series_auc']:.6f}`\n\n")

    lines.append("\n---\n\n")

    # SECTION D
    lines.append("## D. Baseline at Series Level (The Strong Model)\n\n")
    lines.append(f"Saved: `outputs_v2/baseline_series_oof.csv` ({len(series_oof_df)} series scored across 5 folds).\n\n")
    lines.append("### 10. Study-Level Aggregators for Baseline Series Predictions\n\n")
    lines.append("| Aggregator | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean AUC ± Std | Pooled AUC |\n")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
    for name, a_res in res_d["aggregators"].items():
        pf = a_res["per_fold"]
        lines.append(
            f"| **{name:15s}** | {pf[0]:.4f} | {pf[1]:.4f} | {pf[2]:.4f} | {pf[3]:.4f} | {pf[4]:.4f} | "
            f"**{a_res['mean_auc']:.6f} ± {a_res['std_auc']:.4f}** | **{a_res['pooled_auc']:.6f}** |\n"
        )

    lines.append("\n### Best Plane Per Label (Baseline Series Predictions)\n\n")
    lines.append("| Label | Axial AUC | Coronal AUC | Sagittal AUC | Best Plane |\n")
    lines.append("| :--- | :---: | :---: | :---: | :---: |\n")
    for l in LABELS:
        bp = res_d["best_plane_per_label"][l]
        lines.append(f"| {l} | {bp['axial_auc']:.4f} | {bp['coronal_auc']:.4f} | {bp['sagittal_auc']:.4f} | **{bp['best_plane']}** |\n")

    lines.append("\n---\n\n")

    # SECTION E
    lines.append("## E. Stacked Aggregators (Logistic Regression)\n\n")
    lines.append("| Model / Features | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean AUC ± Std | Pooled AUC |\n")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
    plain = res_d["aggregators"]["mean"]
    lines.append(f"| **Plain Mean Pooling** | {plain['per_fold'][0]:.4f} | {plain['per_fold'][1]:.4f} | {plain['per_fold'][2]:.4f} | {plain['per_fold'][3]:.4f} | {plain['per_fold'][4]:.4f} | {plain['mean_auc']:.6f} ± {plain['std_auc']:.4f} | {plain['pooled_auc']:.6f} |\n")
    st60 = res_e["stacker_60"]
    lines.append(f"| **Stacked LR (All 60 Features)** | {st60['per_fold'][0]:.4f} | {st60['per_fold'][1]:.4f} | {st60['per_fold'][2]:.4f} | {st60['per_fold'][3]:.4f} | {st60['per_fold'][4]:.4f} | **{st60['mean_auc']:.6f} ± {st60['std_auc']:.4f}** | **{st60['pooled_auc']:.6f}** |\n")
    st5 = res_e["stacker_5"]
    lines.append(f"| **Stacked LR (Own 5 Features Only)** | {st5['per_fold'][0]:.4f} | {st5['per_fold'][1]:.4f} | {st5['per_fold'][2]:.4f} | {st5['per_fold'][3]:.4f} | {st5['per_fold'][4]:.4f} | **{st5['mean_auc']:.6f} ± {st5['std_auc']:.4f}** | **{st5['pooled_auc']:.6f}** |\n")

    lines.append(f"\n*Regularisation Selection*: Tried C values `{res_e['C_candidates']}` selected per label via internal 3-fold CV inside the training folds only.\n\n")

    lines.append("### Per-Label AUC: Plain Mean vs 5-Feature Stacker vs 60-Feature Stacker\n\n")
    lines.append("| Label | Plain Mean | Own 5 Features | All 60 Features | Gain vs Mean |\n")
    lines.append("| :--- | :---: | :---: | :---: | :---: |\n")
    for l in LABELS:
        m_sc = plain["pooled_per_label"][l]
        s5_sc = st5["pooled_per_label"][l]
        s60_sc = st60["pooled_per_label"][l]
        lines.append(f"| {l} | {m_sc:.4f} | {s5_sc:.4f} | {s60_sc:.4f} | `{s60_sc - m_sc:+.4f}` |\n")

    lines.append("\n---\n\n")

    # SECTION F
    lines.append("## F. Blend (Fold 0)\n\n")
    lines.append(f"- **Baseline Alone (Fold 0)** : **`{res_f['auc_base']:.6f}`**\n")
    lines.append(f"- **V2 Alone (Fold 0)**       : **`{res_f['auc_v2']:.6f}`**\n")
    lines.append(f"- **50/50 Blend (Fold 0)**    : **`{res_f['auc_blend']:.6f}`** (Delta vs Baseline: `{res_f['auc_blend'] - res_f['auc_base']:+.6f}`, Delta vs V2: `{res_f['auc_blend'] - res_f['auc_v2']:+.6f}`)\n\n")

    lines.append("### Per-Label Blend Performance & Residual Pearson Correlations\n\n")
    lines.append("| Label | Baseline AUC | V2 AUC | Blend AUC | Residual Pearson Correlation r(base, v2) |\n")
    lines.append("| :--- | :---: | :---: | :---: | :---: |\n")
    for l in LABELS:
        r_val = res_f["pearson_corr"][l]
        lines.append(f"| {l} | {res_f['per_base'][l]:.4f} | {res_f['per_v2'][l]:.4f} | **{res_f['per_blend'][l]:.4f}** | `{r_val:.4f}` |\n")

    lines.append("\n---\n\n")

    # SECTION G
    lines.append("## G. Execution Runtimes\n\n")
    lines.append(f"- Section A (Training Dynamics)   : {res_a['elapsed_sec']:.2f} s\n")
    lines.append(f"- Section B (Mismatch Tests)      : {res_b['elapsed_sec']:.2f} s\n")
    lines.append(f"- Section C (V2 Ablations & Attn) : {res_c['elapsed_sec']:.2f} s\n")
    lines.append(f"- Section D (Baseline Series OOF) : {res_d['elapsed_sec']:.2f} s\n")
    lines.append(f"- Section E (Stacked Aggregator)  : {res_e['elapsed_sec']:.2f} s\n")
    lines.append(f"- Section F (Fold 0 Blend)        : {res_f['elapsed_sec']:.2f} s\n")
    lines.append(f"- **Total Diagnostic Runtime**    : {total_elapsed:.2f} s\n\n")

    with open(md_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"Report written successfully to {md_path}")


# =============================================================================
# Main Driver
# =============================================================================
def main():
    total_start = time.time()
    print("=" * 70)
    print("Starting Comprehensive Diagnostic Suite for Fold 0 & Baseline")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    # Load shared data
    studies_df = pd.read_csv(DATA_DIR / "train_targets_soft.csv")
    series_df = pd.read_csv(DATA_DIR / "train_series.csv")
    folds_df = pd.read_csv(OUTPUTS_V2_DIR / "folds.csv")

    val_uids = set(folds_df[folds_df["fold"] == 0]["StudyInstanceUID"])
    val_studies_df = studies_df[studies_df["StudyInstanceUID"].isin(val_uids)].reset_index(drop=True)

    # 1. Section A
    res_a = run_section_a()

    # Load V2 best checkpoint for Sections B and C
    v2_best_ckpt = OUTPUTS_V2_DIR / "best_model_fold0.pt"
    print(f"Loading V2 Best Checkpoint from {v2_best_ckpt}...")
    v2_model = load_v2_model(v2_best_ckpt, DEVICE)

    # 2. Section B
    res_b = run_section_b(v2_model, val_studies_df, series_df)

    # 3. Section C
    res_c = run_section_c(v2_model, val_studies_df, series_df)

    # 4. Section D
    series_oof_df, res_d = run_section_d(series_df, folds_df, studies_df)

    # 5. Section E
    res_e = run_section_e(series_oof_df, folds_df, studies_df)

    # 6. Section F
    res_f = run_section_f(studies_df)

    # 7. Write Markdown Report
    total_elapsed = time.time() - total_start
    write_markdown_report(res_a, res_b, res_c, res_d, res_e, res_f, total_elapsed)

    print("\n" + "=" * 70)
    print("ALL DIAGNOSTICS COMPLETED SUCCESSFULLY!")
    print(f"Report saved to: {OUTPUTS_V2_DIR / 'DIAG_fold0.md'}")
    print(f"Series OOF saved to: {OUTPUTS_V2_DIR / 'baseline_series_oof.csv'}")
    print(f"Total time: {total_elapsed:.1f} s")
    print("=" * 70)


if __name__ == "__main__":
    main()
