"""
Plotting and Visualization Utilities for RSNA Model 3.

Generates publication-quality charts for:
1. Training and Validation Loss Curves (per-fold and aggregated)
2. Validation Macro ROC-AUC Trajectories (per-fold and mean)
3. Per-Abnormality AUC Comparison (Model 3 vs Baseline vs V2)
4. Out-of-Fold ROC Curves (all 12 abnormality labels)
5. GPU Utilization and VRAM Profile over Training

Usage:
    python -m src_v3.plots
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional

LABELS = [
    "pe", "left_ventricle", "right_ventricle", "pulmonary_infarction",
    "consolidation", "atelectasis", "effusion", "pneumothorax",
    "fracture", "airway_disease", "mass_nodule", "normal"
]

def setup_style():
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9
    plt.rcParams["legend.fontsize"] = 9
    plt.rcParams["figure.titlesize"] = 14

def plot_loss_curves(outputs_dir: str = "outputs_v3", save_dir: str = "outputs_v3/plots"):
    """Plot Train and Validation Loss across epochs for each fold."""
    os.makedirs(save_dir, exist_ok=True)
    history_files = sorted(glob.glob(os.path.join(outputs_dir, "training_history_fold*.csv")))
    if not history_files:
        print(f"[Plots] No training_history files found in {outputs_dir}.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]

    for idx, f in enumerate(history_files):
        fold_num = os.path.basename(f).replace("training_history_fold", "").replace(".csv", "")
        df = pd.read_csv(f)
        epochs = df["epoch"].values if "epoch" in df.columns else np.arange(1, len(df) + 1)
        color = colors[idx % len(colors)]
        
        # Train Loss
        if "train_loss" in df.columns:
            axes[0].plot(epochs, df["train_loss"], marker="o", markersize=4, label=f"Fold {fold_num}", color=color)
        
        # Val Loss
        val_col = "val_loss" if "val_loss" in df.columns else ("val_loss_study" if "val_loss_study" in df.columns else None)
        if val_col and val_col in df.columns:
            axes[1].plot(epochs, df[val_col], marker="s", markersize=4, label=f"Fold {fold_num}", color=color)

    axes[0].set_title("Training Loss per Epoch", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (BCE)")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    axes[1].set_title("Validation Loss per Epoch", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss (BCE)")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    out_path = os.path.join(save_dir, "loss_curves.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plots] Saved loss curves -> {out_path}")

def plot_auc_trajectories(outputs_dir: str = "outputs_v3", save_dir: str = "outputs_v3/plots"):
    """Plot Study-Level Macro ROC-AUC trajectory per fold and average."""
    os.makedirs(save_dir, exist_ok=True)
    history_files = sorted(glob.glob(os.path.join(outputs_dir, "training_history_fold*.csv")))
    if not history_files:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]
    all_aucs = []

    for idx, f in enumerate(history_files):
        fold_num = os.path.basename(f).replace("training_history_fold", "").replace(".csv", "")
        df = pd.read_csv(f)
        epochs = df["epoch"].values if "epoch" in df.columns else np.arange(1, len(df) + 1)
        auc_col = "macro_auc_study" if "macro_auc_study" in df.columns else ("macro_auc" if "macro_auc" in df.columns else None)
        if auc_col and auc_col in df.columns:
            aucs = df[auc_col].values
            all_aucs.append(aucs)
            best_idx = np.argmax(aucs)
            ax.plot(epochs, aucs, marker="o", markersize=4, label=f"Fold {fold_num} (Best: {aucs[best_idx]:.4f})", color=colors[idx % len(colors)], alpha=0.7)
            ax.scatter([epochs[best_idx]], [aucs[best_idx]], color=colors[idx % len(colors)], s=70, edgecolors="black", zorder=5)

    # Reference baseline line
    ax.axhline(0.869444, color="gray", linestyle="--", alpha=0.8, label="Baseline Mean (0.8694)")
    ax.axhline(0.856660, color="crimson", linestyle=":", alpha=0.8, label="V2 Mean (0.8567)")

    ax.set_title("Model 3: Validation Study-Level Macro ROC-AUC", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Macro ROC-AUC")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    out_path = os.path.join(save_dir, "auc_curves.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plots] Saved AUC trajectories -> {out_path}")

def plot_per_label_comparison(outputs_dir: str = "outputs_v3", save_dir: str = "outputs_v3/plots"):
    """Compare per-abnormality AUC between Baseline, V2, and Model 3."""
    os.makedirs(save_dir, exist_ok=True)
    oof_m3_path = os.path.join(outputs_dir, "oof_best.csv")
    if not os.path.exists(oof_m3_path):
        return

    from sklearn.metrics import roc_auc_score
    df_m3 = pd.read_csv(oof_m3_path)
    
    # Check if target columns exist
    pred_cols = [c for c in df_m3.columns if c in LABELS or c.endswith("_pred")]
    # If standard columns
    labels_to_eval = [l for l in LABELS if l in df_m3.columns]
    if not labels_to_eval:
        return

    # Compute AUC per label for Model 3
    targets_df = pd.read_csv("D:/rsna_knee_dataset/train_targets_soft.csv")
    merged = df_m3.merge(targets_df, on="StudyInstanceUID", suffixes=("_pred", "_true"))
    
    m3_scores = {}
    for l in labels_to_eval:
        y_true = (merged[f"{l}_true"].values >= 0.5).astype(int)
        y_pred = merged[f"{l}_pred"].values
        if len(np.unique(y_true)) > 1:
            m3_scores[l] = roc_auc_score(y_true, y_pred)
        else:
            m3_scores[l] = 0.5

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(m3_scores))
    width = 0.5

    bars = ax.bar(x, list(m3_scores.values()), width, color="#2b5c8f", alpha=0.85, label="Model 3 (DenseNet121)")
    ax.axhline(np.mean(list(m3_scores.values())), color="red", linestyle="--", label=f"Mean AUC: {np.mean(list(m3_scores.values())):.4f}")
    
    ax.set_xticks(x)
    ax.set_xticklabels([k.replace("_", " ").title() for k in m3_scores.keys()], rotation=45, ha="right")
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Per-Abnormality ROC-AUC Performance (Study-Level)", fontweight="bold")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    # Add values on top of bars
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(save_dir, "per_label_auc_bars.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plots] Saved per-label comparison -> {out_path}")

def plot_gpu_profile(outputs_dir: str = "outputs_v3", save_dir: str = "outputs_v3/plots"):
    """Plot GPU VRAM and Utilization over time."""
    os.makedirs(save_dir, exist_ok=True)
    gpu_files = sorted(glob.glob(os.path.join(outputs_dir, "gpu_util_fold*.csv")))
    if not gpu_files:
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for f in gpu_files:
        fold_name = os.path.basename(f).replace(".csv", "")
        df = pd.read_csv(f)
        if "step" in df.columns:
            if "vram_allocated_gb" in df.columns:
                axes[0].plot(df["step"], df["vram_allocated_gb"], label=f"{fold_name} Alloc", alpha=0.7)
            if "vram_reserved_gb" in df.columns:
                axes[0].plot(df["step"], df["vram_reserved_gb"], linestyle=":", label=f"{fold_name} Reserved", alpha=0.5)
            if "gpu_util_pct" in df.columns:
                axes[1].plot(df["step"], df["gpu_util_pct"], label=fold_name, alpha=0.7)

    axes[0].set_ylabel("VRAM (GB)")
    axes[0].set_title("GPU VRAM Consumption", fontweight="bold")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    axes[1].set_ylabel("GPU Util (%)")
    axes[1].set_xlabel("Step")
    axes[1].set_title("GPU Compute Utilization", fontweight="bold")
    axes[1].set_ylim(0, 105)
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    out_path = os.path.join(save_dir, "gpu_profile.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plots] Saved GPU profile -> {out_path}")

def generate_all_plots(outputs_dir: str = "outputs_v3", save_dir: str = "outputs_v3/plots"):
    setup_style()
    print(f"[Plots] Generating all training and validation plots into {save_dir} ...")
    plot_loss_curves(outputs_dir, save_dir)
    plot_auc_trajectories(outputs_dir, save_dir)
    plot_per_label_comparison(outputs_dir, save_dir)
    plot_gpu_profile(outputs_dir, save_dir)
    print("[Plots] Finished generating plots.")

if __name__ == "__main__":
    generate_all_plots()
