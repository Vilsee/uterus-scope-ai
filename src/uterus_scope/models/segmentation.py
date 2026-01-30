"""
Endometrial segmentation head.

U-Net style decoder attached to Swin backbone for
pixel-wise endometrial region segmentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from uterus_scope.config import get_config


@dataclass
class SegmentationOutput:
    """Output from endometrial segmentation head."""
    
    # Segmentation mask (B, 1, H, W) - probabilities
    mask: torch.Tensor
    
    # Thickness estimation in mm
    thickness_mm: torch.Tensor
    
    # Boundary uncertainty map
    boundary_uncertainty: Optional[torch.Tensor] = None
    
    # Raw logits before sigmoid
    logits: Optional[torch.Tensor] = None


class ConvBlock(nn.Module):
    """Convolutional block with batch norm and ReLU."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DecoderBlock(nn.Module):
    """Decoder block with upsampling and skip connection."""
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels // 2,
            kernel_size=2, stride=2,
        )
        self.conv = ConvBlock(
            in_channels // 2 + skip_channels,
            out_channels,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        skip: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.upsample(x)
        
        if skip is not None:
            # Ensure matching spatial dimensions
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(
                    x, size=skip.shape[-2:],
                    mode='bilinear', align_corners=False,
                )
            x = torch.cat([x, skip], dim=1)
        
        return self.conv(x)


class EndometrialSegmentationHead(nn.Module):
    """
    Segmentation head for endometrial region detection.
    
    Uses a U-Net style decoder attached to Swin backbone
    features for pixel-wise segmentation.
    
    Also estimates endometrial thickness from the segmentation mask.
    
    Attributes:
        feature_dims: Input feature dimensions from backbone
        decoder_dims: Decoder channel dimensions
        pixels_per_mm: Conversion factor for thickness calculation
    """
    
    def __init__(
        self,
        feature_dims: list[int] = [96, 192, 384, 768],
        decoder_dims: list[int] = [256, 128, 64, 32],
        output_size: int = 224,
        pixels_per_mm: float = 10.0,
    ):
        """
        Initialize the segmentation head.
        
        Args:
            feature_dims: Feature dimensions from backbone stages
            decoder_dims: Channel dimensions for decoder stages
            output_size: Target output resolution
            pixels_per_mm: Pixel to mm conversion factor
        """
        super().__init__()
        
        self.feature_dims = feature_dims
        self.decoder_dims = decoder_dims
        self.output_size = output_size
        self.pixels_per_mm = pixels_per_mm
        
        # Bottleneck
        self.bottleneck = ConvBlock(feature_dims[-1], decoder_dims[0])
        
        # Decoder path
        self.decoder_blocks = nn.ModuleList()
        
        in_ch = decoder_dims[0]
        for i, out_ch in enumerate(decoder_dims[1:]):
            skip_ch = feature_dims[-(i + 2)] if i < len(feature_dims) - 1 else 0
            self.decoder_blocks.append(
                DecoderBlock(in_ch, skip_ch, out_ch)
            )
            in_ch = out_ch
        
        # Final upsampling to target size
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(decoder_dims[-1], 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        
        # Segmentation output
        self.seg_head = nn.Conv2d(16, 1, kernel_size=1)
        
        # Boundary detection head (for uncertainty)
        self.boundary_head = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )
        
        # Thickness regression from features
        self.thickness_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(decoder_dims[0], 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.ReLU(),  # Thickness is always positive
        )
    
    def forward(
        self,
        features: list[torch.Tensor],
        return_uncertainty: bool = True,
    ) -> SegmentationOutput:
        """
        Forward pass for segmentation.
        
        Args:
            features: Multi-scale features from backbone [stage1, stage2, ...]
            return_uncertainty: Whether to compute boundary uncertainty
            
        Returns:
            SegmentationOutput with mask, thickness, and optional uncertainty
        """
        # Start from deepest features
        x = self.bottleneck(features[-1])
        
        # Store for thickness estimation
        bottleneck_features = x
        
        # Decoder path with skip connections
        for i, decoder_block in enumerate(self.decoder_blocks):
            skip_idx = -(i + 2)
            skip = features[skip_idx] if abs(skip_idx) <= len(features) else None
            x = decoder_block(x, skip)
        
        # Final upsampling
        x = self.final_upsample(x)
        
        # Resize to target size if needed
        if x.shape[-2:] != (self.output_size, self.output_size):
            x = F.interpolate(
                x, size=(self.output_size, self.output_size),
                mode='bilinear', align_corners=False,
            )
        
        # Segmentation output
        logits = self.seg_head(x)
        mask = torch.sigmoid(logits)
        
        # Thickness estimation (from bottleneck features and mask)
        thickness = self.thickness_head(bottleneck_features)
        
        # Also estimate from mask geometry as backup
        mask_thickness = self._estimate_thickness_from_mask(mask)
        
        # Combine estimates (learned + geometric)
        final_thickness = thickness.squeeze(-1) * 0.7 + mask_thickness * 0.3
        
        # Boundary uncertainty
        boundary_uncertainty = None
        if return_uncertainty:
            boundary_logits = self.boundary_head(x)
            boundary_uncertainty = torch.sigmoid(boundary_logits)
        
        return SegmentationOutput(
            mask=mask,
            thickness_mm=final_thickness,
            boundary_uncertainty=boundary_uncertainty,
            logits=logits,
        )
    
    def _estimate_thickness_from_mask(
        self,
        mask: torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """
        Estimate thickness from segmentation mask geometry.
        
        Measures the vertical extent of the segmented region.
        
        Args:
            mask: Segmentation mask (B, 1, H, W)
            threshold: Binarization threshold
            
        Returns:
            Thickness estimates (B,)
        """
        batch_size = mask.shape[0]
        thicknesses = []
        
        for b in range(batch_size):
            binary_mask = (mask[b, 0] > threshold).float()
            
            # Find vertical extent
            row_sums = binary_mask.sum(dim=1)
            nonzero_rows = (row_sums > 0).float()
            
            if nonzero_rows.sum() > 0:
                # Find extent of non-zero rows
                indices = torch.nonzero(nonzero_rows).squeeze()
                if indices.numel() > 1:
                    thickness_px = (indices[-1] - indices[0]).float()
                else:
                    thickness_px = torch.tensor(0.0, device=mask.device)
            else:
                thickness_px = torch.tensor(0.0, device=mask.device)
            
            thickness_mm = thickness_px / self.pixels_per_mm
            thicknesses.append(thickness_mm)
        
        return torch.stack(thicknesses)
    
    def compute_loss(
        self,
        output: SegmentationOutput,
        target_mask: torch.Tensor,
        target_thickness: torch.Tensor,
        dice_weight: float = 0.5,
        bce_weight: float = 0.5,
        thickness_weight: float = 0.1,
    ) -> dict[str, torch.Tensor]:
        """
        Compute combined segmentation loss.
        
        Args:
            output: Model output
            target_mask: Ground truth mask
            target_thickness: Ground truth thickness
            dice_weight: Weight for Dice loss
            bce_weight: Weight for BCE loss
            thickness_weight: Weight for thickness regression loss
            
        Returns:
            Dictionary with individual losses and total
        """
        # Binary cross-entropy
        bce_loss = F.binary_cross_entropy_with_logits(
            output.logits, target_mask,
        )
        
        # Dice loss
        dice_loss = self._dice_loss(output.mask, target_mask)
        
        # Thickness regression
        thickness_loss = F.mse_loss(output.thickness_mm, target_thickness)
        
        # Combined loss
        total_loss = (
            bce_weight * bce_loss +
            dice_weight * dice_loss +
            thickness_weight * thickness_loss
        )
        
        return {
            "total": total_loss,
            "bce": bce_loss,
            "dice": dice_loss,
            "thickness": thickness_loss,
        }
    
    def _dice_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        smooth: float = 1.0,
    ) -> torch.Tensor:
        """Compute soft Dice loss."""
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        
        return 1 - (2 * intersection + smooth) / (
            pred_flat.sum() + target_flat.sum() + smooth
        )
