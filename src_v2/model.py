"""
src_v2/model.py
===============
Study-level Multi-Label Classifier for RSNA Knee MRI Abnormality Detection.

Architecture:
  1. 2D Slice Backbone (3-channel 2.5D slices -> embed_dim 512) with gradient checkpointing support.
  2. Slice Tokenizer:
       LayerNorm(Linear(embed_dim -> 256)) +
       Depth Embedding (24) +
       Plane Embedding (3) +
       Fluid Sensitive Embedding (2) [+ Fat Suppression Embedding (2)]
  3. Study-level Transformer Encoder:
       2-layer Pre-LN Transformer (d=256, 8 heads, ff=512, dropout=0.1)
  4. Label-wise Cross-Attention Head:
       12 Learned Label Queries cross-attend over all study slice tokens -> Dropout(0.2) -> 12 Logits.

Ablations supported via configuration:
  - no_metadata: bool
  - mean_pool_instead_of_label_attention: bool
  - backbone: 'resnet34' | 'efficientnet_v2_s' | 'convnext_tiny'
  - use_grad_checkpointing: bool
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torchvision import models

from src_v2.dataset import build_25d_slices, gpu_normalize_volumes


class SliceBackbone(nn.Module):
    """
    Standard 3-channel 2D convolutional or vision backbone applied to 2.5D slices.
    """

    def __init__(
        self,
        backbone_name: str = "resnet34",
        pretrained: bool = True,
        use_grad_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.use_grad_checkpointing = use_grad_checkpointing
        weights = "DEFAULT" if pretrained else None

        if backbone_name == "resnet34":
            base = models.resnet34(weights=weights)
            # Retain unmodified 3-channel conv1
            self.features = nn.Sequential(
                base.conv1,
                base.bn1,
                base.relu,
                base.maxpool,
                base.layer1,
                base.layer2,
                base.layer3,
                base.layer4,
                base.avgpool,
            )
            self.embed_dim = base.fc.in_features  # 512

        elif backbone_name == "efficientnet_v2_s":
            base = models.efficientnet_v2_s(weights=weights)
            self.features = nn.Sequential(
                base.features,
                base.avgpool,
            )
            self.embed_dim = base.classifier[1].in_features  # 1280

        elif backbone_name == "convnext_tiny":
            base = models.convnext_tiny(weights=weights)
            self.features = nn.Sequential(
                base.features,
                base.avgpool,
            )
            self.embed_dim = base.classifier[2].in_features  # 768

        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [N, 3, H, W]
        Returns : [N, embed_dim]
        """
        if self.use_grad_checkpointing and self.training:
            out = checkpoint.checkpoint(self.features, x, use_reentrant=False)
        else:
            out = self.features(x)
        return torch.flatten(out, 1)


class StudyKneeClassifier(nn.Module):
    """
    End-to-End Study-Level Multi-Label Classifier with Label-Wise Cross Attention.
    """

    def __init__(
        self,
        num_classes: int = 12,
        backbone_name: str = "resnet34",
        pretrained: bool = True,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        head_dropout: float = 0.2,
        fixed_depth: int = 24,
        use_grad_checkpointing: bool = False,
        no_metadata: bool = False,
        mean_pool_instead_of_label_attention: bool = False,
        drop_fat_embedding: bool = True,  # Fat suppression is 100% identical to fluid sensitivity
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.d_model = d_model
        self.fixed_depth = fixed_depth
        self.no_metadata = no_metadata
        self.mean_pool_instead_of_label_attention = mean_pool_instead_of_label_attention
        self.drop_fat_embedding = drop_fat_embedding

        # ── 1. 2D Slice Backbone ──────────────────────────────────
        self.backbone = SliceBackbone(
            backbone_name=backbone_name,
            pretrained=pretrained,
            use_grad_checkpointing=use_grad_checkpointing,
        )
        backbone_dim = self.backbone.embed_dim

        # ── 2. Slice Token Projection & Embeddings ───────────────
        self.proj = nn.Sequential(
            nn.Linear(backbone_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # Positional & Metadata Embeddings
        self.depth_embed = nn.Embedding(fixed_depth, d_model)  # 24 depth slices
        if not no_metadata:
            self.plane_embed = nn.Embedding(3, d_model)        # 0=Sagittal, 1=Coronal, 2=Axial
            self.fluid_embed = nn.Embedding(2, d_model)        # 0=No, 1=Yes
            if not drop_fat_embedding:
                self.fat_embed = nn.Embedding(2, d_model)

        # ── 3. Study-level Transformer Encoder ────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            enable_nested_tensor=False,
        )

        # ── 4. Classification Head ───────────────────────────────
        if self.mean_pool_instead_of_label_attention:
            # Ablation: Global mean pool over slice tokens
            self.head = nn.Sequential(
                nn.Dropout(p=head_dropout),
                nn.Linear(d_model, num_classes),
            )
        else:
            # Label-wise Cross-Attention Head
            self.label_queries = nn.Parameter(
                torch.randn(1, num_classes, d_model) * (1.0 / (d_model ** 0.5))
            )
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout,
                batch_first=True,
            )
            self.head_dropout = nn.Dropout(p=head_dropout)
            # Per-label linear weight [12, d_model] and bias [12]
            self.label_fc_w = nn.Parameter(torch.randn(num_classes, d_model) * (1.0 / (d_model ** 0.5)))
            self.label_fc_b = nn.Parameter(torch.zeros(num_classes))

    def forward(
        self,
        volumes: torch.Tensor,
        planes: torch.Tensor,
        fluid: torch.Tensor,
        fat: torch.Tensor,
        series_mask: torch.Tensor,
        token_dropout_p: float = 0.0,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        volumes : torch.Tensor  shape [B, S, 24, H, W]  dtype=uint8 (from DataLoader)
            Raw uint8 volumes; GPU normalization is applied inside this method.
        planes : torch.Tensor   shape [B, S]
        fluid : torch.Tensor    shape [B, S]
        fat : torch.Tensor      shape [B, S]
        series_mask : torch.Tensor shape [B, S] (bool)
        token_dropout_p : float (fraction of slice tokens to randomly drop in training)
        return_attention : bool (if True, returns attention weights)

        Returns
        -------
        logits : torch.Tensor shape [B, 12]
        [attn_weights : torch.Tensor shape [B, 12, Total_Tokens]] (optional)
        """
        B, S, D, H, W = volumes.shape
        device = volumes.device

        # ── 0. GPU Normalization (uint8 -> float32, per-volume percentile + ImageNet) ──
        # If volumes is already float32 (training path: normalized before augmentation),
        # skip normalization. If uint8 (validation path: raw from DataLoader), normalize here.
        if volumes.dtype == torch.uint8:
            volumes = gpu_normalize_volumes(volumes, series_mask)  # [B, S, 24, H, W] float32
        # else: already float32 normalized (training path after gpu_normalize + GPUAugmentor)

        # Reshape valid series to compute 2.5D convolutions
        # [B, S, 24, H, W] -> [B*S, 24, H, W]
        vols_flat = volumes.view(B * S, D, H, W)
        tri_slices = build_25d_slices(vols_flat)  # [B*S, 24, 3, H, W]

        # Fold depth into batch for single efficient 2D backbone pass: [B*S*24, 3, H, W]
        tri_flat = tri_slices.view(B * S * D, 3, H, W)
        feats_flat = self.backbone(tri_flat)      # [B*S*24, embed_dim]

        # Project to d_model: [B, S, 24, d_model]
        tokens = self.proj(feats_flat).view(B, S, D, self.d_model)

        # ── Add Positional & Metadata Embeddings ─────────────────
        # Depth embedding: [1, 1, 24, d_model]
        depth_indices = torch.arange(D, device=device).view(1, 1, D)
        tokens = tokens + self.depth_embed(depth_indices)

        if not self.no_metadata:
            # Plane embedding: [B, S, 1, d_model]
            plane_emb = self.plane_embed(planes.clamp(0, 2)).unsqueeze(2)
            tokens = tokens + plane_emb

            # Fluid embedding: [B, S, 1, d_model]
            fluid_emb = self.fluid_embed(fluid.clamp(0, 1)).unsqueeze(2)
            tokens = tokens + fluid_emb

            if not self.drop_fat_embedding:
                fat_emb = self.fat_embed(fat.clamp(0, 1)).unsqueeze(2)
                tokens = tokens + fat_emb

        # Flatten series and depth dimensions into a single sequence of tokens per study:
        # [B, S*24, d_model]
        seq_len = S * D
        tokens_seq = tokens.view(B, seq_len, self.d_model)

        # ── Construct Attention Mask (Series Padding + Token Dropout) ──
        # series_mask: [B, S] -> expand to [B, S, 24] -> [B, S*24]
        # In PyTorch Transformer: key_padding_mask is True for PAD positions that must be IGNORED.
        valid_mask = series_mask.unsqueeze(2).expand(B, S, D).reshape(B, seq_len)  # True = valid, False = pad

        if self.training and token_dropout_p > 0.0:
            # Randomly drop a fraction of valid slice tokens
            drop_mask = torch.rand(B, seq_len, device=device) < token_dropout_p
            # Keep at least 1 token per study even if dropout hits everything
            valid_mask = valid_mask & (~drop_mask)
            # If all tokens in a study row got dropped, restore valid_mask for that row
            all_dropped = ~valid_mask.any(dim=-1, keepdim=True)
            valid_mask = torch.where(all_dropped, series_mask.unsqueeze(2).expand(B, S, D).reshape(B, seq_len), valid_mask)

        key_padding_mask = ~valid_mask  # True = ignore

        # ── Transformer Encoding ─────────────────────────────────
        encoded_tokens = self.encoder(tokens_seq, src_key_padding_mask=key_padding_mask)  # [B, S*24, d_model]

        # ── Classification Head ──────────────────────────────────
        if self.mean_pool_instead_of_label_attention:
            # Mask out invalid tokens before mean pool
            mask_expanded = valid_mask.unsqueeze(-1).float()  # [B, S*24, 1]
            sum_tokens = (encoded_tokens * mask_expanded).sum(dim=1)  # [B, d_model]
            denom = mask_expanded.sum(dim=1).clamp(min=1.0)
            study_rep = sum_tokens / denom
            logits = self.head(study_rep)  # [B, 12]
            return logits

        # Label-wise Cross Attention: Queries [B, 12, d_model] cross-attend to Encoded Tokens [B, S*24, d_model]
        queries = self.label_queries.expand(B, -1, -1)  # [B, 12, d_model]
        label_reps, attn_weights = self.cross_attn(
            query=queries,
            key=encoded_tokens,
            value=encoded_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=return_attention,
        )  # [B, 12, d_model], [B, 12, S*24]

        label_reps = self.head_dropout(label_reps)  # [B, 12, d_model]

        # Per-label linear layer: (label_reps * label_fc_w).sum(dim=-1) + label_fc_b -> [B, 12]
        logits = (label_reps * self.label_fc_w.unsqueeze(0)).sum(dim=-1) + self.label_fc_b.unsqueeze(0)

        if return_attention:
            return logits, attn_weights
        return logits
