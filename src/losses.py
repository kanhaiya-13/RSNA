"""
src/losses.py
=============
Loss functions for multi-label classification with soft probability targets.

Provides ``compute_bce_loss`` — a BCE wrapper that optionally integrates
per-sample weights from a ``sample_weight`` column in the dataset.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_bce_loss(
    criterion: nn.BCEWithLogitsLoss,
    logits:    torch.Tensor,
    targets:   torch.Tensor,
    weights:   Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute Binary Cross-Entropy loss against soft (continuous) targets.

    ``torch.nn.BCEWithLogitsLoss`` natively supports continuous targets
    y ∈ [0, 1], so this function simply delegates to ``criterion`` when no
    per-sample weights are supplied.

    When ``weights`` is provided (shape ``[B]``), element-wise BCE is computed
    with ``reduction='none'`` (shape ``[B, C]``), each sample row is scaled by
    its corresponding weight, and the result is averaged over all elements.

    Parameters
    ----------
    criterion : nn.BCEWithLogitsLoss
        Pre-instantiated loss (``reduction='mean'`` for the un-weighted path).
    logits : torch.Tensor  shape [B, C]
        Raw (un-activated) model outputs.
    targets : torch.Tensor  shape [B, C]
        Soft probability targets ∈ [0, 1].
    weights : torch.Tensor or None  shape [B]
        Optional per-sample importance weights.

    Returns
    -------
    torch.Tensor  — scalar loss value
    """
    if weights is None:
        return criterion(logits, targets)

    # Element-wise loss matrix: [B, C]
    loss_matrix = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    # Scale each sample row by its weight and take the global mean
    w = weights.view(-1, 1)              # [B, 1] → broadcasts over C
    return (loss_matrix * w).mean()
