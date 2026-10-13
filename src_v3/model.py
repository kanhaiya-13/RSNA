"""
src_v3/model.py
===============
Series-level backbone classifier for RSNA Knee MRI v3.

Architecture
------------
  1. Backbone: one of densenet121 / inception_v3 / efficientnet_b3 /
               convnext_tiny / resnet34 (ImageNet pretrained, 3-channel).
  2. All B*24 slices fed through the backbone => per-slice features [B,24,F].
  3. Series feature = concat(mean_over_slices, max_over_slices)  -> [B, 2F].
  4. Metadata: plane embedding (3->16) + fluid embedding (2->8).
  5. Head: Dropout(0.3) -> Linear(2F+24, 512) -> GELU -> Dropout(0.2)
           -> Linear(512, 12).

Config flags (implemented, not trained by default):
  - head_per_plane    : separate final linear per plane (0/1/2).
  - pos_weight_mode   : 'none' | 'sqrt'.
  - pooling           : 'mean' | 'max' | 'meanmax' (default).
  - no_metadata       : drop plane/fluid embeddings.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torchvision.models as tvm

# ── Feature dimensions produced by each backbone ─────────────────────────────
_BACKBONE_FEAT_DIM: Dict[str, int] = {
    "densenet121":    1024,
    "inception_v3":   2048,
    "efficientnet_b3": 1536,
    "convnext_tiny":   768,
    "resnet34":        512,
}


class SeriesClassifier(nn.Module):
    """
    Single-series classifier.

    Parameters
    ----------
    backbone_name : str
    num_classes   : int   (12)
    pretrained    : bool
    pooling       : 'mean' | 'max' | 'meanmax'
    no_metadata   : bool  – if True, skip plane/fluid embeddings
    head_per_plane: bool  – if True, use 3 separate final linear layers
    """

    def __init__(
        self,
        backbone_name: str = "densenet121",
        num_classes: int = 12,
        pretrained: bool = True,
        pooling: str = "meanmax",
        no_metadata: bool = False,
        head_per_plane: bool = False,
    ) -> None:
        super().__init__()
        assert backbone_name in _BACKBONE_FEAT_DIM, \
            f"Unknown backbone: {backbone_name}. Choose from {list(_BACKBONE_FEAT_DIM)}"

        self.backbone_name = backbone_name
        self.pooling = pooling
        self.no_metadata = no_metadata
        self.head_per_plane = head_per_plane
        self.num_classes = num_classes

        F = _BACKBONE_FEAT_DIM[backbone_name]   # scalar feature dim
        self.feat_dim = F

        # ── Backbone ─────────────────────────────────────────────────────────
        self.backbone = self._build_backbone(backbone_name, pretrained)

        # ── Pooling output dim ────────────────────────────────────────────────
        if pooling == "meanmax":
            pool_dim = 2 * F
        else:
            pool_dim = F

        # ── Metadata embeddings ───────────────────────────────────────────────
        if not no_metadata:
            self.plane_emb = nn.Embedding(3, 16)
            self.fluid_emb = nn.Embedding(2, 8)
            meta_dim = 16 + 8   # 24
        else:
            meta_dim = 0

        head_in = pool_dim + meta_dim

        # ── Classification head ───────────────────────────────────────────────
        if head_per_plane:
            self.head_shared = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(head_in, 512),
                nn.GELU(),
                nn.Dropout(0.2),
            )
            self.heads = nn.ModuleList([nn.Linear(512, num_classes) for _ in range(3)])
        else:
            self.head = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(head_in, 512),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(512, num_classes),
            )

    # ── Backbone builders ─────────────────────────────────────────────────────
    def _build_backbone(self, name: str, pretrained: bool) -> nn.Module:
        weights_arg = "DEFAULT" if pretrained else None

        if name == "densenet121":
            m = tvm.densenet121(weights=weights_arg)
            # Replace classifier with identity; use features -> ReLU -> GAP
            m.classifier = nn.Identity()
            # Wrap so forward returns F-dim vector
            return _DenseNetEncoder(m)

        elif name == "inception_v3":
            m = tvm.inception_v3(weights=weights_arg, aux_logits=False)
            m.aux_logits = False
            m.fc = nn.Identity()
            # inception_v3 with transform_input=True scales to [-1,1] range
            # but our input is already ImageNet-normalised => set False
            m.transform_input = False
            return _InceptionEncoder(m)

        elif name == "efficientnet_b3":
            m = tvm.efficientnet_b3(weights=weights_arg)
            m.classifier = nn.Identity()
            return _EfficientNetEncoder(m)

        elif name == "convnext_tiny":
            m = tvm.convnext_tiny(weights=weights_arg)
            m.classifier = nn.Identity()
            return _ConvNextEncoder(m)

        elif name == "resnet34":
            m = tvm.resnet34(weights=weights_arg)
            m.fc = nn.Identity()
            return _ResNetEncoder(m)

        raise ValueError(f"Unknown backbone: {name}")

    def _pool_slices(self, slice_feats: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        slice_feats : [B, 24, F]

        Returns
        -------
        [B, pool_dim]
        """
        if self.pooling == "mean":
            return slice_feats.mean(dim=1)
        elif self.pooling == "max":
            return slice_feats.max(dim=1).values
        else:  # meanmax
            return torch.cat([slice_feats.mean(dim=1), slice_feats.max(dim=1).values], dim=1)

    def forward(
        self,
        slices_25d: torch.Tensor,   # [B*24, 3, H, W]  float32 on GPU
        planes: torch.Tensor,       # [B] long
        fluids: torch.Tensor,       # [B] long
    ) -> torch.Tensor:
        """
        Returns
        -------
        logits : [B, 12]
        """
        BD, C, H, W = slices_25d.shape
        B = planes.shape[0]
        D = BD // B   # should be 24

        # ── Backbone: all slices at once ──────────────────────────────────────
        feats_flat = self.backbone(slices_25d)   # [B*24, F]
        feats = feats_flat.view(B, D, -1)        # [B, 24, F]

        # ── Pooling ───────────────────────────────────────────────────────────
        series_feat = self._pool_slices(feats)   # [B, pool_dim]

        # ── Metadata ──────────────────────────────────────────────────────────
        if not self.no_metadata:
            pe = self.plane_emb(planes)           # [B, 16]
            fe = self.fluid_emb(fluids)           # [B, 8]
            series_feat = torch.cat([series_feat, pe, fe], dim=1)  # [B, pool_dim+24]

        # ── Head ──────────────────────────────────────────────────────────────
        if self.head_per_plane:
            h = self.head_shared(series_feat)      # [B, 512]
            # Gather per-plane logits; planes values are 0/1/2
            logits = torch.zeros(B, self.num_classes, device=slices_25d.device,
                                  dtype=slices_25d.dtype)
            for p in range(3):
                mask = planes == p
                if mask.any():
                    logits[mask] = self.heads[p](h[mask])
        else:
            logits = self.head(series_feat)        # [B, 12]

        return logits


# ─────────────────────────────────────────────────────────────────────────────
# Thin backbone wrapper classes
# Each wraps a torchvision model and exposes a forward(x) -> [N, F] interface.
# ─────────────────────────────────────────────────────────────────────────────

class _DenseNetEncoder(nn.Module):
    """DenseNet121: features -> ReLU -> global avg pool -> [N, 1024]."""
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.features = model.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)            # [N, 1024, h, w]
        f = torch.relu(f)
        f = self.pool(f).flatten(1)     # [N, 1024]
        return f


class _InceptionEncoder(nn.Module):
    """Inception-v3: forward with aux_logits=False => [N, 2048]."""
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # torchvision inception_v3 with aux_logits=False returns tensor directly
        out = self.model(x)
        if isinstance(out, torch.Tensor):
            return out      # [N, 2048] after fc=Identity
        # InceptionOutputs named tuple
        return out.logits


class _EfficientNetEncoder(nn.Module):
    """EfficientNet-B3: features + avgpool -> [N, 1536]."""
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.features = model.features
        self.avgpool = model.avgpool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        f = self.avgpool(f).flatten(1)  # [N, 1536]
        return f


class _ConvNextEncoder(nn.Module):
    """ConvNeXt-Tiny: features -> avgpool -> [N, 768]."""
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.features = model.features
        self.avgpool = model.avgpool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        f = self.avgpool(f).flatten(1)  # [N, 768]
        return f


class _ResNetEncoder(nn.Module):
    """ResNet-34: standard forward with fc=Identity -> [N, 512]."""
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)    # [N, 512]
