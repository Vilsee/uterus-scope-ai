"""
Fibrosis/scar pattern detection head.

Detects and localizes fibrotic scar patterns in ultrasound images.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FibrosisOutput:
    """Output from fibrosis detector."""
    
    # Fibrosis probability map (B, 1, H, W)
    probability_map: torch.Tensor
    
    # Overall severity score 0-1 (B,)
    severity_score: torch.Tensor
    
    # Detected fibrosis regions (list of bounding boxes per sample)
    regions: Optional[list[list[dict]]] = None
    
    # Binary detection (has fibrosis or not)
    has_fibrosis: Optional[torch.Tensor] = None
    
    def get_severity_level(self, idx: int = 0) -> str:
        """Get human-readable severity level."""
        score = self.severity_score[idx].item()
        if score < 0.2:
            return "None/Minimal"
        elif score < 0.4:
            return "Mild"
        elif score < 0.6:
            return "Moderate"
        elif score < 0.8:
            return "Significant"
        else:
            return "Severe"


class SpatialAttention(nn.Module):
    """Spatial attention module for focusing on fibrotic regions."""
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, kernel_size=1),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply spatial attention.
        
        Returns:
            Tuple of (attended features, attention map)
        """
        attention = self.conv(x)
        attended = x * attention
        return attended, attention


class FibrosisDetector(nn.Module):
    """
    Fibrosis/scar pattern detector.
    
    Detects and localizes fibrotic tissue patterns that appear
    as hyperechoic (bright) regions in ultrasound.
    
    Outputs both a probability map for localization and
    an overall severity score.
    
    Attributes:
        feature_dim: Input feature dimension
        threshold: Detection threshold for binary output
    """
    
    def __init__(
        self,
        feature_dims: list[int] = [96, 192, 384, 768],
        hidden_dim: int = 128,
        output_size: int = 224,
        threshold: float = 0.5,
    ):
        """
        Initialize the detector.
        
        Args:
            feature_dims: Feature dimensions from backbone stages
            hidden_dim: Hidden dimension for processing
            output_size: Output spatial resolution
            threshold: Detection threshold
        """
        super().__init__()
        
        self.feature_dims = feature_dims
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self.threshold = threshold
        
        # Process deepest features for detection
        self.feature_processor = nn.Sequential(
            nn.Conv2d(feature_dims[-1], hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # Spatial attention for focusing on fibrotic regions
        self.spatial_attention = SpatialAttention(hidden_dim)
        
        # Upsampling path for probability map
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, kernel_size=2, stride=2),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim // 4, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        
        # Probability map output
        self.prob_head = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1),
        )
        
        # Severity score regression
        self.severity_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        
        # Binary detection head
        self.detection_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )
    
    def forward(
        self,
        features: list[torch.Tensor],
        extract_regions: bool = False,
    ) -> FibrosisOutput:
        """
        Forward pass for fibrosis detection.
        
        Args:
            features: Multi-scale features from backbone
            extract_regions: Whether to extract bounding boxes
            
        Returns:
            FibrosisOutput with probability map and severity
        """
        # Process deepest features
        x = self.feature_processor(features[-1])
        
        # Apply spatial attention
        attended, attention_map = self.spatial_attention(x)
        
        # Severity score from attended features
        severity = self.severity_head(attended).squeeze(-1)
        
        # Binary detection
        detection_logits = self.detection_head(attended)
        has_fibrosis = torch.sigmoid(detection_logits).squeeze(-1)
        
        # Upsample for probability map
        upsampled = self.upsample(attended)
        
        # Resize to output size
        if upsampled.shape[-2:] != (self.output_size, self.output_size):
            upsampled = F.interpolate(
                upsampled,
                size=(self.output_size, self.output_size),
                mode='bilinear',
                align_corners=False,
            )
        
        # Probability map
        prob_logits = self.prob_head(upsampled)
        probability_map = torch.sigmoid(prob_logits)
        
        # Extract regions if requested
        regions = None
        if extract_regions:
            regions = self._extract_regions(probability_map)
        
        return FibrosisOutput(
            probability_map=probability_map,
            severity_score=severity,
            regions=regions,
            has_fibrosis=has_fibrosis,
        )
    
    def _extract_regions(
        self,
        probability_map: torch.Tensor,
        min_area: int = 100,
    ) -> list[list[dict]]:
        """
        Extract bounding boxes for detected fibrotic regions.
        
        Args:
            probability_map: Fibrosis probability map (B, 1, H, W)
            min_area: Minimum region area in pixels
            
        Returns:
            List of region dictionaries per sample
        """
        import cv2
        
        batch_regions = []
        
        for b in range(probability_map.shape[0]):
            prob = probability_map[b, 0].detach().cpu().numpy()
            
            # Threshold to binary
            binary = (prob > self.threshold).astype('uint8') * 255
            
            # Find contours
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            regions = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area >= min_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Calculate mean probability in region
                    mask = prob[y:y+h, x:x+w]
                    mean_prob = float(mask.mean())
                    
                    regions.append({
                        'x': int(x),
                        'y': int(y),
                        'width': int(w),
                        'height': int(h),
                        'area': int(area),
                        'probability': mean_prob,
                    })
            
            batch_regions.append(regions)
        
        return batch_regions
    
    def compute_loss(
        self,
        output: FibrosisOutput,
        target_map: Optional[torch.Tensor] = None,
        target_score: Optional[torch.Tensor] = None,
        target_has_fibrosis: Optional[torch.Tensor] = None,
        map_weight: float = 1.0,
        score_weight: float = 0.5,
        detection_weight: float = 0.3,
    ) -> dict[str, torch.Tensor]:
        """
        Compute fibrosis detection loss.
        
        Args:
            output: Model output
            target_map: Ground truth probability map
            target_score: Ground truth severity score
            target_has_fibrosis: Ground truth binary detection
            map_weight: Weight for map loss
            score_weight: Weight for severity loss
            detection_weight: Weight for detection loss
            
        Returns:
            Dictionary with loss values
        """
        losses = {}
        total_loss = torch.tensor(0.0, device=output.severity_score.device)
        
        # Probability map loss (BCE)
        if target_map is not None:
            map_loss = F.binary_cross_entropy(
                output.probability_map, target_map,
            )
            losses['map'] = map_loss
            total_loss = total_loss + map_weight * map_loss
        
        # Severity score loss (MSE)
        if target_score is not None:
            score_loss = F.mse_loss(output.severity_score, target_score)
            losses['severity'] = score_loss
            total_loss = total_loss + score_weight * score_loss
        
        # Binary detection loss (BCE)
        if target_has_fibrosis is not None and output.has_fibrosis is not None:
            detection_loss = F.binary_cross_entropy(
                output.has_fibrosis, target_has_fibrosis.float(),
            )
            losses['detection'] = detection_loss
            total_loss = total_loss + detection_weight * detection_loss
        
        losses['total'] = total_loss
        
        return losses
