"""
src/metrics.py
==============
Evaluation metrics for multi-label classification.

Provides ``compute_macro_auc`` — macro-averaged ROC-AUC that is compatible
with both hard binary labels and soft (continuous) probability targets.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score


def compute_macro_auc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute per-class and macro-averaged ROC-AUC.

    Compatible with both binary (0/1) and soft (continuous ∈ [0, 1]) targets:
    ``roc_auc_score`` treats ``y_true`` as an ordinal ranking signal, which
    works correctly for soft pseudo-labels.  Classes with only one unique value
    in ``y_true`` are skipped because AUC is undefined there.

    Parameters
    ----------
    y_true : np.ndarray  shape [N, num_classes]
        Ground-truth labels (binary or soft probabilities).
    y_pred : np.ndarray  shape [N, num_classes]
        Predicted probabilities (sigmoid-activated logits).
    labels : list of str, optional
        Human-readable class names, aligned with columns of ``y_true``/``y_pred``.
        If None, generic names ['class_0', 'class_1', ...] will be used.

    Returns
    -------
    macro_auc : float
        Mean AUC over all classes that had at least 2 unique ground-truth values.
        Returns ``float('nan')`` if no class could be evaluated.
    per_class : dict {str -> float}
        Per-class AUC values.  ``float('nan')`` for skipped classes.
    """
    if labels is None:
        labels = [f"class_{i}" for i in range(y_true.shape[1])]

    per_class:  Dict[str, float] = {}
    valid_aucs: List[float]      = []

    for i, lbl in enumerate(labels):
        col_true = y_true[:, i]
        col_pred = y_pred[:, i]

        # For continuous soft pseudo-labels (floats in [0, 1]), threshold at >= 0.5 for ROC-AUC
        if col_true.dtype.kind == 'f' and not np.all(np.isin(col_true, [0, 1])):
            col_true_eval = (col_true >= 0.5).astype(int)
        else:
            col_true_eval = col_true.astype(int)

        if len(np.unique(col_true_eval)) < 2:
            # AUC undefined when only one class is present in split
            per_class[lbl] = float("nan")
            continue

        try:
            auc = roc_auc_score(col_true_eval, col_pred)
            per_class[lbl] = float(auc)
            valid_aucs.append(float(auc))
        except Exception:
            per_class[lbl] = float("nan")

    macro_auc = float(np.mean(valid_aucs)) if valid_aucs else float("nan")
    return macro_auc, per_class
