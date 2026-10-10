# RSNA Knee MRI — Training Analysis Results

## Cross-Validation Summary

| Fold | Best Epoch | Best Macro AUC |
|------|-----------|----------------|
| 0    | 10        | 0.8403         |
| 1    | 8         | 0.8277         |
| 2    | 7         | 0.8296         |
| 3    | 8         | 0.8353         |
| 4    | 11        | 0.8187         |
| **OOF Mean** | — | **0.8303 ± 0.0073** |

> [!NOTE]
> All 5 folds converged quickly — best epochs range from 7–11 out of 30, meaning the model hits peak AUC early and then plateaus or slightly overfits. This is a strong signal to consider **early stopping** or **shorter training** in future runs.

---

## Plot 1 — Loss Curves (Train vs Val per Fold)

![Loss curves per fold](C:\Users\23101A0034\.gemini\antigravity-ide\brain\d06efd95-eceb-4a10-b09e-f9de99301737\01_loss_curves.png)

---

## Plot 2 — Macro AUC Learning Curves

![Macro AUC curves](C:\Users\23101A0034\.gemini\antigravity-ide\brain\d06efd95-eceb-4a10-b09e-f9de99301737\02_auc_curves.png)

---

## Plot 3 — Best Val AUC per Fold

![Best AUC per fold](C:\Users\23101A0034\.gemini\antigravity-ide\brain\d06efd95-eceb-4a10-b09e-f9de99301737\03_best_auc_per_fold.png)

---

## Plot 4 — Per-Class AUC Heatmap

![Per-class AUC heatmap](C:\Users\23101A0034\.gemini\antigravity-ide\brain\d06efd95-eceb-4a10-b09e-f9de99301737\04_per_class_auc.png)

---

## Plot 5 — OOF Per-Class AUC Summary (Mean ± Std)

![OOF class summary](C:\Users\23101A0034\.gemini\antigravity-ide\brain\d06efd95-eceb-4a10-b09e-f9de99301737\05_oof_summary.png)

---

## Key Findings

### Strengths (AUC > 0.85)
- **Medial Meniscus** — 0.866
- **Medial OA** — 0.864
- **Synovitis** — 0.862
- **Baker's** — 0.861
- **Fracture** — 0.850
- **ACL** — 0.847

### Weaker Classes (AUC < 0.80) — Need Attention
| Class | Mean AUC | Issue |
|-------|----------|-------|
| MCL | 0.762 | Lowest — likely rare/imbalanced |
| Lateral Meniscus | 0.790 | Moderate |
| Contusion | 0.798 | High variance across folds |

> [!TIP]
> **MCL** and **Lateral Meniscus** are candidates for class-weighted loss, more augmentation, or targeted oversampling in the next iteration.

### Overfitting Pattern
- Training loss continues falling to epoch 30, but val AUC peaks at epochs 7–11.
- **Gap between train_loss and val_loss widens after epoch ~10** → classic overfitting.
- Recommendation: add `EarlyStopping(patience=5)` based on macro AUC.

---

## Output Files
- Plots: [`outputs/plots/`](file:///d:/RSNA/outputs/plots/)
- Summary CSV: [`outputs/cv_summary.csv`](file:///d:/RSNA/outputs/cv_summary.csv)
- Script: [`analyze_training.py`](file:///d:/RSNA/analyze_training.py)
