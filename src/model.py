"""
src/model.py
============
2.5D CNN architecture for multi-label knee MRI classification.

Architecture overview
---------------------
KneeClassifier
  └── SliceEncoder  (pretrained ResNet backbone, 1-channel patched conv1)
      applied independently across all D depth slices in a single GPU pass
  └── Depth aggregation  (mean- or max-pool over the D dimension)
  └── Dropout → Linear head → num_classes logits

The "2.5D" design folds the depth dimension into the batch axis
(shape [B×D, 1, H, W]) so standard 2D convolutions process every slice in
one efficient GPU call, then the resulting embeddings are pooled back to
produce a single study-level feature vector [B, embed_dim].
"""

from typing import Optional

import torch
import torch.nn as nn
from torchvision import models


class SliceEncoder(nn.Module):
    """
    Single-channel 2D backbone for encoding individual MRI slices.

    Adapts the first convolutional layer of the chosen torchvision backbone
    from 3-channel RGB to 1-channel greyscale.  When using pretrained weights
    the RGB channels are averaged to initialise the new single-channel kernel,
    which preserves the feature statistics learned on ImageNet.

    Parameters
    ----------
    backbone_name : str
        Any ``torchvision.models`` attribute name, e.g. ``'resnet34'``.
    pretrained : bool
        If True, loads ``weights='DEFAULT'`` (ImageNet-1k pre-training).

    Attributes
    ----------
    embed_dim : int
        Feature dimension output by the backbone (512 for ResNet-34).
    """

    def __init__(
        self,
        backbone_name: str  = "resnet34",
        pretrained:    bool = True,
    ) -> None:
        super().__init__()
        weights  = "DEFAULT" if pretrained else None
        backbone = getattr(models, backbone_name)(weights=weights)

        # ── Patch first conv: 3-ch RGB → 1-ch greyscale ─────────
        orig_conv  = backbone.conv1
        self.conv1 = nn.Conv2d(
            in_channels  = 1,
            out_channels = orig_conv.out_channels,
            kernel_size  = orig_conv.kernel_size,
            stride       = orig_conv.stride,
            padding      = orig_conv.padding,
            bias         = False,
        )
        with torch.no_grad():
            if pretrained:
                # Average RGB filters → single greyscale filter
                self.conv1.weight.copy_(orig_conv.weight.mean(dim=1, keepdim=True))
            else:
                nn.init.kaiming_normal_(
                    self.conv1.weight, mode="fan_out", nonlinearity="relu"
                )

        # ── Remaining backbone layers ────────────────────────────
        self.bn1     = backbone.bn1
        self.relu    = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1  = backbone.layer1
        self.layer2  = backbone.layer2
        self.layer3  = backbone.layer3
        self.layer4  = backbone.layer4
        self.avgpool = backbone.avgpool

        self.embed_dim: int = backbone.fc.in_features   # 512 for ResNet-34

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  shape [B, 1, H, W]

        Returns
        -------
        torch.Tensor  shape [B, embed_dim]
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)   # [B, embed_dim]


class KneeClassifier(nn.Module):
    """
    2.5D CNN for multi-label knee abnormality detection.

    Forward pass
    ------------
    1. Reshape  [B, 1, D, H, W] → [B×D, 1, H, W]  (fold depth into batch)
    2. Encode   [B×D, 1, H, W]  → [B×D, E]          via SliceEncoder
    3. Reshape  [B×D, E]        → [B, D, E]
    4. Pool     [B, D, E]       → [B, E]              (mean or max over D)
    5. Classify [B, E]          → [B, num_classes]    (Dropout + Linear)

    Parameters
    ----------
    num_classes   : int    Number of output logits (12 for this task).
    backbone_name : str    torchvision backbone identifier.
    pretrained    : bool   Load ImageNet weights.
    pool_mode     : str    ``'mean'`` (default) or ``'max'`` depth aggregation.
    dropout       : float  Dropout probability before the linear head.
    """

    def __init__(
        self,
        num_classes:   int   = 12,
        backbone_name: str   = "resnet34",
        pretrained:    bool  = True,
        pool_mode:     str   = "mean",
        dropout:       float = 0.3,
    ) -> None:
        super().__init__()
        self.pool_mode = pool_mode
        self.encoder   = SliceEncoder(backbone_name, pretrained)
        embed_dim      = self.encoder.embed_dim

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  shape [B, 1, D, H, W]

        Returns
        -------
        torch.Tensor  shape [B, num_classes]  — raw logits (no sigmoid)
        """
        B, C, D, H, W = x.shape

        # Fold depth into batch for a single efficient 2D conv pass
        x = x.permute(0, 2, 1, 3, 4)    # [B, D, 1, H, W]
        x = x.reshape(B * D, C, H, W)   # [B×D, 1, H, W]

        feats = self.encoder(x)          # [B×D, embed_dim]
        feats = feats.view(B, D, -1)     # [B, D, embed_dim]

        # Depth aggregation
        if self.pool_mode == "max":
            feats, _ = feats.max(dim=1)
        else:
            feats = feats.mean(dim=1)    # [B, embed_dim]

        return self.classifier(feats)    # [B, num_classes]
