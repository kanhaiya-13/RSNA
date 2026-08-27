"""
src/metrics.py
==============
Evaluation metrics for multi-label classification.

Provides ``compute_macro_auc`` — macro-averaged ROC-AUC that is compatible
with both hard binary labels and soft (continuous) probability targets.
"""

from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score


def compute_macro_auc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
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
    labels : list of str
        Human-readable class names, aligned with columns of ``y_true``/``y_pred``.

    Returns
    -------
    macro_auc : float
        Mean AUC over all classes that had at least 2 unique ground-truth values.
        Returns ``float('nan')`` if no class could be evaluated.
    per_class : dict {str -> float}
        Per-class AUC values.  ``float('nan')`` for skipped classes.
    """
    per_class:  Dict[str, float] = {}
    valid_aucs: List[float]      = []

    for i, lbl in enumerate(labels):
        col_true = y_true[:, i]
        col_pred = y_pred[:, i]

        if len(np.unique(col_true)) < 2:
            # AUC undefined when all ground-truth values are identical
            per_class[lbl] = float("nan")
            continue

        try:
            auc = roc_auc_score(col_true, col_pred)
            per_class[lbl] = auc
            valid_aucs.append(auc)
        except Exception:
            per_class[lbl] = float("nan")

    macro_auc = float(np.mean(valid_aucs)) if valid_aucs else float("nan")
    return macro_auc, per_class
