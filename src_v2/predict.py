"""
src_v2/predict.py
=================
Inference and Ensembling Script for RSNA Knee MRI V2.

Features:
  - Loads 5-fold best EMA checkpoints from outputs_v2/best_model_fold*.pt.
  - Scores studies from a series CSV and volume directory.
  - Averages sigmoid probabilities across 5 folds.
  - Supports Test-Time Augmentation (--tta): horizontal flip on Coronal & Axial series only (never Sagittal).
  - Evaluates TTA impact on OOF validation folds.
"""

import argparse
import glob
import os
import sys
sys.path.insert(0, r"d:\RSNA")
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src_v2.dataset import (
    LABELS,
    StudyKneeDataset,
    collate_study_fn,
)
from src_v2.model import StudyKneeClassifier
from src.metrics import compute_macro_auc


def load_model_from_checkpoint(ckpt_path: str, device: torch.device) -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = StudyKneeClassifier(
        num_classes=12,
        backbone_name=cfg["backbone"],
        pretrained=False,
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_encoder_layers=cfg["num_encoder_layers"],
        dim_feedforward=cfg["dim_feedforward"],
        dropout=cfg["dropout"],
        head_dropout=cfg["head_dropout"],
        fixed_depth=cfg["fixed_depth"],
        no_metadata=cfg["no_metadata"],
        mean_pool_instead_of_label_attention=cfg["mean_pool_instead_of_label_attention"],
    ).to(device)
    state = ckpt.get("ema_state", ckpt.get("model_state"))
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def predict_dataset(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_tta: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    """
    Run inference on a DataLoader.
    If use_tta is True, averages standard predictions with horizontally flipped Coronal/Axial series.
    """
    model.eval()
    all_preds = []
    study_uids = []

    for batch in loader:
        vols = batch["volumes"].to(device, non_blocking=True)
        planes = batch["planes"].to(device, non_blocking=True)
        fluid = batch["fluid"].to(device, non_blocking=True)
        fat = batch["fat"].to(device, non_blocking=True)
        mask = batch["series_mask"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
            logits = model(vols, planes, fluid, fat, mask, token_dropout_p=0.0)
            probs = torch.sigmoid(logits)

            if use_tta:
                # Horizontally flip coronal (plane=1) and axial (plane=2) series only
                vols_tta = vols.clone()
                B, S, D, H, W = vols.shape
                for b in range(B):
                    for s in range(S):
                        if mask[b, s] and int(planes[b, s]) in (1, 2):
                            vols_tta[b, s] = vols[b, s].flip(dims=[-1])

                logits_tta = model(vols_tta, planes, fluid, fat, mask, token_dropout_p=0.0)
                probs_tta = torch.sigmoid(logits_tta)
                probs = 0.5 * (probs + probs_tta)

        all_preds.append(probs.cpu().numpy())
        study_uids.extend(batch["study_uids"])

    return np.concatenate(all_preds, axis=0), study_uids


def evaluate_oof_tta(
    outputs_dir: str = "outputs_v2",
    data_dir: str = "D:/rsna_knee_dataset",
) -> Dict[str, float]:
    """
    Evaluate TTA on all 5 validation folds and print comparison.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    studies_df = pd.read_csv(os.path.join(data_dir, "train_targets_soft.csv"))
    series_df = pd.read_csv(os.path.join(data_dir, "train_series.csv"))
    folds_df = pd.read_csv(os.path.join(outputs_dir, "folds.csv"))

    normal_preds_list = []
    tta_preds_list = []
    targets_list = []

    print("\n" + "="*70 + "\n  EVALUATING TEST-TIME AUGMENTATION (TTA) ON OOF PREDICTIONS\n" + "="*70)

    for fold in range(5):
        ckpt_path = os.path.join(outputs_dir, f"best_model_fold{fold}.pt")
        if not os.path.exists(ckpt_path):
            print(f"Skipping fold {fold}: {ckpt_path} not found.")
            continue

        model = load_model_from_checkpoint(ckpt_path, device)
        val_uids = set(folds_df[folds_df["fold"] == fold]["StudyInstanceUID"])
        val_studies_df = studies_df[studies_df["StudyInstanceUID"].isin(val_uids)].reset_index(drop=True)

        val_ds = StudyKneeDataset(val_studies_df, series_df, base_dir=data_dir, labels=LABELS, is_train=False)
        val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate_study_fn, num_workers=2)

        preds_norm, _ = predict_dataset(model, val_loader, device, amp_dtype, use_tta=False)
        preds_tta, _ = predict_dataset(model, val_loader, device, amp_dtype, use_tta=True)
        y_true = val_studies_df[LABELS].values

        auc_norm, _ = compute_macro_auc(y_true, preds_norm, labels=LABELS)
        auc_tta, _ = compute_macro_auc(y_true, preds_tta, labels=LABELS)

        print(f"Fold {fold}: Normal Macro AUC = {auc_norm:.6f} | TTA Macro AUC = {auc_tta:.6f} | Delta = {auc_tta - auc_norm:+.6f}")

        normal_preds_list.append(preds_norm)
        tta_preds_list.append(preds_tta)
        targets_list.append(y_true)

    if normal_preds_list:
        all_norm = np.concatenate(normal_preds_list, axis=0)
        all_tta = np.concatenate(tta_preds_list, axis=0)
        all_true = np.concatenate(targets_list, axis=0)

        oof_auc_norm, _ = compute_macro_auc(all_true, all_norm, labels=LABELS)
        oof_auc_tta, _ = compute_macro_auc(all_true, all_tta, labels=LABELS)

        print("-" * 70)
        print(f"Overall OOF Macro AUC (Normal) : {oof_auc_norm:.6f}")
        print(f"Overall OOF Macro AUC (TTA)    : {oof_auc_tta:.6f}")
        print(f"Overall TTA Delta              : {oof_auc_tta - oof_auc_norm:+.6f}")
        print("=" * 70 + "\n")
        return {"oof_auc_norm": oof_auc_norm, "oof_auc_tta": oof_auc_tta, "delta": oof_auc_tta - oof_auc_norm}
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="RSNA Knee MRI V2 Inference & Submission Generator")
    parser.add_argument("--series_csv", type=str, default="D:/rsna_knee_dataset/train_series.csv", help="Series CSV to predict")
    parser.add_argument("--data_dir", type=str, default="D:/rsna_knee_dataset", help="Path to .npz volumes")
    parser.add_argument("--outputs_dir", type=str, default="outputs_v2", help="Directory with checkpoints")
    parser.add_argument("--output_csv", type=str, default="outputs_v2/submission.csv", help="Output submission CSV path")
    parser.add_argument("--tta", action="store_true", help="Enable Test-Time Augmentation")
    parser.add_argument("--eval_oof_tta", action="store_true", help="Run TTA comparison on OOF sets")
    args = parser.parse_args()

    if args.eval_oof_tta:
        evaluate_oof_tta(outputs_dir=args.outputs_dir, data_dir=args.data_dir)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    series_df = pd.read_csv(args.series_csv)
    unique_studies = series_df[["StudyInstanceUID"]].drop_duplicates().reset_index(drop=True)
    # Add dummy labels if not present
    for l in LABELS:
        if l not in unique_studies.columns:
            unique_studies[l] = 0.0

    test_ds = StudyKneeDataset(unique_studies, series_df, base_dir=args.data_dir, labels=LABELS, is_train=False)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, collate_fn=collate_study_fn, num_workers=2)

    fold_ckpts = sorted(glob.glob(os.path.join(args.outputs_dir, "best_model_fold*.pt")))
    if not fold_ckpts:
        raise FileNotFoundError(f"No checkpoints found matching {args.outputs_dir}/best_model_fold*.pt")

    print(f"[Inference] Ensembling {len(fold_ckpts)} fold models with TTA={args.tta} ...")
    ensemble_preds = []

    for ckpt_path in fold_ckpts:
        print(f"  --> Predicting with {os.path.basename(ckpt_path)} ...")
        model = load_model_from_checkpoint(ckpt_path, device)
        preds, study_uids = predict_dataset(model, test_loader, device, amp_dtype, use_tta=args.tta)
        ensemble_preds.append(preds)

    # Average probabilities across folds
    mean_preds = np.mean(ensemble_preds, axis=0)

    sub_df = pd.DataFrame({"StudyInstanceUID": study_uids})
    for i, l in enumerate(LABELS):
        sub_df[l] = mean_preds[:, i]

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    sub_df.to_csv(args.output_csv, index=False)
    print(f"\n[Inference] Saved ensemble predictions to: {args.output_csv} (Shape: {sub_df.shape})")


if __name__ == "__main__":
    main()
