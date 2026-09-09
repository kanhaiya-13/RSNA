"""
analyze_training.py
===================
Post-training analysis and visualization for the RSNA Knee MRI 5-fold CV run.

Generates:
  outputs/plots/01_loss_curves.png         — train/val loss per fold
  outputs/plots/02_auc_curves.png          — macro AUC per fold
  outputs/plots/03_best_auc_per_fold.png   — bar chart of best val AUC per fold
  outputs/plots/04_per_class_auc.png       — heatmap of per-class AUC at best epoch
  outputs/plots/05_oof_summary.png         — OOF summary (mean ± std across folds)
  outputs/cv_summary.csv                  — machine-readable fold summary table

Usage
-----
    python analyze_training.py

    # Point to a different outputs directory:
    python analyze_training.py --outputs_dir outputs
"""

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless backend — works without a display
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Google Fonts fallback chain (Inter → Segoe UI → DejaVu Sans) ────────────
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Inter", "Segoe UI", "DejaVu Sans"],
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
    "savefig.dpi":       150,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "#0f1117",
    "figure.facecolor":  "#0f1117",
    "axes.facecolor":    "#1a1d27",
    "axes.edgecolor":    "#3a3d4d",
    "axes.labelcolor":   "#c8ccd8",
    "xtick.color":       "#8890a4",
    "ytick.color":       "#8890a4",
    "text.color":        "#e2e5f0",
    "grid.color":        "#2a2d3d",
    "grid.linestyle":    "--",
    "grid.linewidth":    0.6,
    "legend.framealpha": 0.15,
    "legend.edgecolor":  "#3a3d4d",
})

# ── Colour palette (one vibrant accent per fold) ─────────────────────────────
FOLD_COLORS = ["#7c6aff", "#00c2a8", "#ff6b6b", "#ffc94d", "#4dabf7"]
LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────
def load_histories(outputs_dir: Path, n_folds: int = 5) -> pd.DataFrame:
    """
    Load all fold training history CSVs into one concatenated DataFrame.
    Raises FileNotFoundError if any fold file is missing.
    """
    dfs = []
    for fold in range(n_folds):
        csv_path = outputs_dir / f"training_history_fold{fold}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"[ERROR] Missing training history for fold {fold}: {csv_path}\n"
                f"  Make sure all 5 folds have been trained before running analysis."
            )
        df = pd.read_csv(csv_path)
        # Drop any trailing blank rows
        df = df.dropna(subset=["epoch"])
        dfs.append(df)
        print(f"  [Fold {fold}] Loaded {len(df)} epochs  — max macro_auc={df['macro_auc'].max():.4f}")
    return pd.concat(dfs, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: styled axis decoration
# ─────────────────────────────────────────────────────────────────────────────
def _style_ax(ax, title: str, xlabel: str = "Epoch", ylabel: str = "") -> None:
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10, color="#e2e5f0")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(True, axis="y")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Loss Curves (train vs val per fold)
# ─────────────────────────────────────────────────────────────────────────────
def plot_loss_curves(df: pd.DataFrame, plots_dir: Path) -> None:
    n_folds = df["fold"].nunique()
    fig, axes = plt.subplots(1, n_folds, figsize=(5 * n_folds, 4.5), sharey=True)
    fig.suptitle("Training vs Validation Loss — All Folds", fontsize=15,
                 fontweight="bold", y=1.02, color="#e2e5f0")

    for fold, ax in enumerate(axes):
        fd = df[df["fold"] == fold].sort_values("epoch")
        color = FOLD_COLORS[fold]
        ax.plot(fd["epoch"], fd["train_loss"], color=color,
                lw=2, label="Train", alpha=0.9)
        ax.plot(fd["epoch"], fd["val_loss"], color=color,
                lw=2, linestyle="--", label="Val", alpha=0.9)

        # Mark best val epoch
        best_row = fd.loc[fd["macro_auc"].idxmax()]
        ax.axvline(best_row["epoch"], color=color, lw=0.8, alpha=0.4)

        _style_ax(ax, f"Fold {fold}", ylabel="BCE Loss" if fold == 0 else "")
        ax.legend(fontsize=8)

    fig.tight_layout()
    out = plots_dir / "01_loss_curves.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — Macro AUC Learning Curves
# ─────────────────────────────────────────────────────────────────────────────
def plot_auc_curves(df: pd.DataFrame, plots_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    for fold in sorted(df["fold"].unique()):
        fd = df[df["fold"] == fold].sort_values("epoch")
        color = FOLD_COLORS[fold]
        ax.plot(fd["epoch"], fd["macro_auc"], color=color,
                lw=2.2, label=f"Fold {fold}", alpha=0.9)

        # Star marker at best epoch
        best_row = fd.loc[fd["macro_auc"].idxmax()]
        ax.scatter(best_row["epoch"], best_row["macro_auc"],
                   s=80, color=color, zorder=5, marker="*")

    # Horizontal line at mean best AUC
    best_aucs = [df[df["fold"] == f]["macro_auc"].max() for f in sorted(df["fold"].unique())]
    mean_best = np.mean(best_aucs)
    ax.axhline(mean_best, color="#ffffff", lw=1, linestyle=":", alpha=0.5,
               label=f"Mean best AUC = {mean_best:.4f}")

    _style_ax(ax, "Macro ROC-AUC Over Training — All Folds",
              ylabel="Macro ROC-AUC")
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    fig.tight_layout()
    out = plots_dir / "02_auc_curves.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — Best Val AUC per Fold (bar chart + OOF line)
# ─────────────────────────────────────────────────────────────────────────────
def plot_best_auc_bars(df: pd.DataFrame, plots_dir: Path) -> dict:
    folds = sorted(df["fold"].unique())
    best_aucs = []
    best_epochs = []

    for fold in folds:
        fd = df[df["fold"] == fold]
        best_row = fd.loc[fd["macro_auc"].idxmax()]
        best_aucs.append(best_row["macro_auc"])
        best_epochs.append(int(best_row["epoch"]))

    mean_auc = np.mean(best_aucs)
    std_auc  = np.std(best_aucs)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        [f"Fold {f}" for f in folds], best_aucs,
        color=FOLD_COLORS[:len(folds)], alpha=0.85, width=0.55,
        edgecolor="#ffffff", linewidth=0.4
    )

    # Value labels on bars
    for bar, auc, ep in zip(bars, best_aucs, best_epochs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{auc:.4f}\n(ep {ep})",
                ha="center", va="bottom", fontsize=9, color="#e2e5f0")

    # OOF mean line
    ax.axhline(mean_auc, color="#ffc94d", lw=2, linestyle="--",
               label=f"OOF mean = {mean_auc:.4f}  ±  {std_auc:.4f}")

    _style_ax(ax, "Best Validation Macro AUC per Fold",
              xlabel="Fold", ylabel="Best Macro ROC-AUC")
    ax.set_ylim(min(best_aucs) * 0.995, max(best_aucs) * 1.012)
    ax.legend(fontsize=10)

    fig.tight_layout()
    out = plots_dir / "03_best_auc_per_fold.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")

    return {"folds": folds, "best_aucs": best_aucs,
            "best_epochs": best_epochs, "mean_auc": mean_auc, "std_auc": std_auc}


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4 — Per-Class AUC Heatmap (at best epoch per fold)
# ─────────────────────────────────────────────────────────────────────────────
def plot_per_class_heatmap(df: pd.DataFrame, plots_dir: Path) -> pd.DataFrame:
    folds = sorted(df["fold"].unique())
    rows = []

    for fold in folds:
        fd = df[df["fold"] == fold]
        best_row = fd.loc[fd["macro_auc"].idxmax()]
        row = {"fold": fold}
        for lbl in LABELS:
            if lbl in best_row.index:
                row[lbl] = best_row[lbl]
        rows.append(row)

    per_class_df = pd.DataFrame(rows).set_index("fold")
    # Add mean row
    per_class_df.loc["mean"] = per_class_df.mean()

    fig, ax = plt.subplots(figsize=(14, 4.5))

    data = per_class_df.values.astype(float)
    vmin, vmax = 0.65, 1.00

    im = ax.imshow(data, cmap="RdYlGn", vmin=vmin, vmax=vmax, aspect="auto")

    # Axis labels
    ax.set_xticks(range(len(per_class_df.columns)))
    ax.set_xticklabels(per_class_df.columns, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(per_class_df.index)))
    ax.set_yticklabels(
        [f"Fold {f}" if f != "mean" else "[ Mean ]" for f in per_class_df.index],
        fontsize=10
    )

    # Cell text annotations
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            txt_color = "black" if val > 0.82 else "white"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=7.5, color=txt_color, fontweight="bold")

    # Highlight mean row
    ax.axhline(len(folds) - 0.5, color="#ffc94d", lw=2)

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cbar.ax.yaxis.set_tick_params(color="#8890a4")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#8890a4")

    ax.set_title("Per-Class Validation AUC at Best Epoch  (by Fold)",
                 fontsize=13, fontweight="bold", pad=12, color="#e2e5f0")
    fig.tight_layout()

    out = plots_dir / "04_per_class_auc.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")

    return per_class_df


# ─────────────────────────────────────────────────────────────────────────────
# Plot 5 — OOF Summary: Mean ± Std per class
# ─────────────────────────────────────────────────────────────────────────────
def plot_oof_class_summary(per_class_df: pd.DataFrame, plots_dir: Path) -> None:
    # Exclude mean row for std calc
    data_df = per_class_df.drop(index="mean", errors="ignore")

    means = data_df.mean()
    stds  = data_df.std()

    # Sort by mean AUC descending
    order = means.sort_values(ascending=True).index

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.RdYlGn(
        (means[order].values - 0.65) / (1.00 - 0.65)
    )
    bars = ax.barh(range(len(order)), means[order].values,
                   xerr=stds[order].values,
                   color=colors, alpha=0.88, height=0.6,
                   error_kw=dict(ecolor="#ffffff", lw=1.2, capsize=3),
                   edgecolor="#ffffff", linewidth=0.3)

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=10)

    # Value labels
    for i, (lbl, bar) in enumerate(zip(order, bars)):
        ax.text(means[lbl] + stds[lbl] + 0.003,
                bar.get_y() + bar.get_height() / 2,
                f"{means[lbl]:.3f}", va="center", fontsize=9, color="#e2e5f0")

    # Grand mean line
    grand_mean = means.mean()
    ax.axvline(grand_mean, color="#7c6aff", lw=1.8, linestyle="--",
               label=f"Grand mean AUC = {grand_mean:.4f}")

    ax.set_xlim(0.62, 1.00)
    ax.set_xlabel("Validation ROC-AUC (mean ± std over 5 folds)", fontsize=10)
    ax.set_title("OOF Per-Class AUC Summary", fontsize=13,
                 fontweight="bold", pad=12, color="#e2e5f0")
    ax.legend(fontsize=9)
    ax.grid(True, axis="x")
    ax.spines["left"].set_visible(False)

    fig.tight_layout()
    out = plots_dir / "05_oof_summary.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# CV Summary Table
# ─────────────────────────────────────────────────────────────────────────────
def save_cv_summary(fold_stats: dict, per_class_df: pd.DataFrame,
                    outputs_dir: Path) -> None:
    folds       = fold_stats["folds"]
    best_aucs   = fold_stats["best_aucs"]
    best_epochs = fold_stats["best_epochs"]

    rows = []
    for i, fold in enumerate(folds):
        row = {
            "fold":           fold,
            "best_epoch":     best_epochs[i],
            "best_macro_auc": round(best_aucs[i], 6),
        }
        if fold in per_class_df.index:
            for col in per_class_df.columns:
                row[col] = round(per_class_df.loc[fold, col], 6)
        rows.append(row)

    # OOF summary row
    summary = {
        "fold":           "OOF",
        "best_epoch":     "—",
        "best_macro_auc": round(fold_stats["mean_auc"], 6),
    }
    for col in per_class_df.columns:
        summary[col] = round(per_class_df.loc["mean", col], 6)
    rows.append(summary)

    summary_df = pd.DataFrame(rows)
    out = outputs_dir / "cv_summary.csv"
    summary_df.to_csv(out, index=False)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Console Report
# ─────────────────────────────────────────────────────────────────────────────
def print_report(fold_stats: dict, per_class_df: pd.DataFrame) -> None:
    folds       = fold_stats["folds"]
    best_aucs   = fold_stats["best_aucs"]
    best_epochs = fold_stats["best_epochs"]
    mean_auc    = fold_stats["mean_auc"]
    std_auc     = fold_stats["std_auc"]

    print("\n" + "=" * 65)
    print("  CROSS-VALIDATION SUMMARY")
    print("=" * 65)
    print(f"  {'Fold':<8}  {'Best Epoch':<12}  {'Best macro AUC'}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*14}")
    for fold, auc, ep in zip(folds, best_aucs, best_epochs):
        print(f"  {fold:<8}  {ep:<12}  {auc:.6f}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*14}")
    print(f"  {'OOF mean':<8}  {'':12}  {mean_auc:.6f}  +/- {std_auc:.6f}")

    print("\n  Per-Class Mean AUC (across all 5 folds):")
    print(f"  {'-'*45}")
    data_df = per_class_df.drop(index="mean", errors="ignore")
    means = data_df.mean().sort_values(ascending=False)
    for lbl, auc in means.items():
        bar = "#" * int(auc * 30)
        print(f"  {lbl:<22} {auc:.4f}  {bar}")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="RSNA training analysis & visualization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--outputs_dir", type=str, default="outputs",
                        help="Directory containing training_history_fold*.csv files")
    parser.add_argument("--n_folds", type=int, default=5,
                        help="Number of folds to analyse")
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    plots_dir   = outputs_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  RSNA Knee MRI — Training Analysis")
    print(f"  Outputs dir : {outputs_dir.resolve()}")
    print(f"  Plots dir   : {plots_dir.resolve()}")
    print(f"{'='*65}\n")

    # ── Load all fold histories ─────────────────────────────────
    print("[1/6] Loading training histories ...")
    df = load_histories(outputs_dir, n_folds=args.n_folds)

    # ── Generate plots ──────────────────────────────────────────
    print("\n[2/6] Plotting loss curves ...")
    plot_loss_curves(df, plots_dir)

    print("\n[3/6] Plotting macro AUC curves ...")
    plot_auc_curves(df, plots_dir)

    print("\n[4/6] Plotting best AUC per fold ...")
    fold_stats = plot_best_auc_bars(df, plots_dir)

    print("\n[5/6] Plotting per-class AUC heatmap ...")
    per_class_df = plot_per_class_heatmap(df, plots_dir)

    print("\n[6/6] Plotting OOF class summary ...")
    plot_oof_class_summary(per_class_df, plots_dir)

    # ── Save CSV summary ────────────────────────────────────────
    print("\n[+] Saving cv_summary.csv ...")
    save_cv_summary(fold_stats, per_class_df, outputs_dir)

    # ── Console report ──────────────────────────────────────────
    print_report(fold_stats, per_class_df)

    print(f"\nAnalysis complete. All plots saved to: {plots_dir.resolve()}\n")


if __name__ == "__main__":
    main()
