"""
Swin Transformer backbone for ultrasound analysis.

Uses MONAI's implementation or timm for pretrained weights,
optimized for medical imaging tasks.
"""

from __future__ import annotations

import logging
from typing import Optional, Literal

import torch
import torch.nn as nn
import timm
from timm.models.swin_transformer import SwinTransformer

from uterus_scope.config import get_config, ModelBackbone, PretrainedWeights

logger = logging.getLogger(__name__)


class SwinBackbone(nn.Module):
    """
    Swin Transformer backbone optimized for medical imaging.
    
    Extracts multi-scale features suitable for segmentation,
    classification, and detection tasks.
    
    Attributes:
        backbone: Underlying Swin Transformer model
        feature_dims: Dimensions of features at each stage
        pretrained: Whether using pretrained weights
    """
    
    # Feature dimensions for different Swin variants
    FEATURE_DIMS = {
        ModelBackbone.SWIN_TINY: [96, 192, 384, 768],
        ModelBackbone.SWIN_SMALL: [96, 192, 384, 768],
        ModelBackbone.SWIN_BASE: [128, 256, 512, 1024],
    }
    
    MODEL_NAMES = {
        ModelBackbone.SWIN_TINY: "swin_tiny_patch4_window7_224",
        ModelBackbone.SWIN_SMALL: "swin_small_patch4_window7_224",
        ModelBackbone.SWIN_BASE: "swin_base_patch4_window7_224",
    }
    
    def __init__(
        self,
        backbone_type: Optional[ModelBackbone] = None,
        pretrained_weights: Optional[PretrainedWeights] = None,
        in_channels: int = 1,
        features_only: bool = True,
        out_indices: tuple[int, ...] = (0, 1, 2, 3),
    ):
        """
        Initialize the Swin Transformer backbone.
        
        Args:
            backbone_type: Type of Swin backbone
            pretrained_weights: Pretrained weight configuration
            in_channels: Number of input channels (1 for grayscale)
            features_only: Return intermediate features
            out_indices: Which stage features to return
        """
        super().__init__()
        
        config = get_config()
        self.backbone_type = backbone_type or config.model.backbone
        self.pretrained_weights = pretrained_weights or config.model.pretrained_weights
        self.in_channels = in_channels
        self.out_indices = out_indices
        
        # Get model configuration
        model_name = self.MODEL_NAMES[self.backbone_type]
        self.feature_dims = self.FEATURE_DIMS[self.backbone_type]
        
        # Determine if using pretrained weights
        pretrained = self.pretrained_weights != PretrainedWeights.NONE
        
        logger.info(
            f"Initializing {self.backbone_type.value} backbone "
            f"(pretrained: {pretrained})"
        )
        
        if features_only:
            # Use timm's features_only mode
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                features_only=True,
                out_indices=out_indices,
                in_chans=in_channels,
            )
        else:
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,  # Remove classifier
                in_chans=in_channels,
            )
        
        # Adapt first layer for grayscale if pretrained on RGB
        if in_channels == 1 and pretrained:
            self._adapt_first_layer()
    
    def _adapt_first_layer(self) -> None:
        """
        Adapt first convolutional layer for single-channel input.
        
        Averages RGB weights to create grayscale weights.
        """
        # Get the patch embedding layer
        if hasattr(self.backbone, 'patch_embed'):
            patch_embed = self.backbone.patch_embed
            if hasattr(patch_embed, 'proj'):
                old_weight = patch_embed.proj.weight.data
                # Average RGB channels
                new_weight = old_weight.mean(dim=1, keepdim=True)
                patch_embed.proj.weight.data = new_weight
                logger.info("Adapted patch embedding for grayscale input")
    
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Forward pass extracting multi-scale features.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            List of feature tensors at different scales
        """
        return self.backbone(x)
    
    def get_feature_dims(self) -> list[int]:
        """Get feature dimensions at each output stage."""
        return [self.feature_dims[i] for i in self.out_indices]
    
    @property
    def num_features(self) -> int:
        """Get the number of features in the final stage."""
        return self.feature_dims[-1]


class FeaturePyramidNeck(nn.Module):
    """
    Feature Pyramid Network neck for multi-scale feature fusion.
    
    Combines features from different Swin stages for
    segmentation and detection tasks.
    """
    
    def __init__(
        self,
        in_channels: list[int],
        out_channels: int = 256,
    ):
        """
        Initialize FPN neck.
        
        Args:
            in_channels: Feature dimensions from backbone stages
            out_channels: Output channel dimension
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Lateral connections
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1)
            for in_ch in in_channels
        ])
        
        # Output convolutions
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels
        ])
    
    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Forward pass with top-down pathway.
        
        Args:
            features: Multi-scale features from backbone
            
        Returns:
            Fused features at each scale
        """
        # Lateral connections
        laterals = [
            conv(f) for conv, f in zip(self.lateral_convs, features)
        ]
        
        # Top-down pathway
        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] = laterals[i] + nn.functional.interpolate(
                laterals[i + 1],
                size=laterals[i].shape[-2:],
                mode='bilinear',
                align_corners=False,
            )
        
        # Output convolutions
        outputs = [
            conv(lat) for conv, lat in zip(self.output_convs, laterals)
        ]
        
        return outputs


def create_swin_backbone(
    backbone_type: Optional[ModelBackbone] = None,
    pretrained: bool = True,
    in_channels: int = 1,
    **kwargs,
) -> SwinBackbone:
    """
    Factory function to create Swin backbone.
    
    Args:
        backbone_type: Type of Swin backbone
        pretrained: Whether to use pretrained weights
        in_channels: Number of input channels
        **kwargs: Additional arguments for SwinBackbone
        
    Returns:
        Configured SwinBackbone instance
    """
    config = get_config()
    
    weights = (
        PretrainedWeights.IMAGENET if pretrained
        else PretrainedWeights.NONE
    )
    
    return SwinBackbone(
        backbone_type=backbone_type or config.model.backbone,
        pretrained_weights=weights,
        in_channels=in_channels,
        **kwargs,
    )
