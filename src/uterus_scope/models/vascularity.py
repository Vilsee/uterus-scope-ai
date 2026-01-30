"""
Vascularity classification head.

Classifies blood flow patterns into clinical types (0-III).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from uterus_scope.config import VascularityType


@dataclass
class VascularityOutput:
    """Output from vascularity classifier."""
    
    # Class probabilities (B, 4) for Types 0-III
    probabilities: torch.Tensor
    
    # Predicted class (B,)
    predicted_type: torch.Tensor
    
    # Confidence score (B,)
    confidence: torch.Tensor
    
    # Raw logits
    logits: Optional[torch.Tensor] = None
    
    def get_type_name(self, idx: int = 0) -> str:
        """Get human-readable type name for sample."""
        type_idx = self.predicted_type[idx].item()
        names = {
            0: "Type 0 (Avascular)",
            1: "Type I (Minimal flow)",
            2: "Type II (Moderate flow)",
            3: "Type III (High vascularity)",
        }
        return names.get(type_idx, f"Unknown ({type_idx})")


class AttentionPool(nn.Module):
    """Attention-based pooling for feature aggregation."""
    
    def __init__(self, in_features: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, in_features // 4),
            nn.Tanh(),
            nn.Linear(in_features // 4, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply attention pooling.
        
        Args:
            x: Features (B, N, C) where N is spatial positions
            
        Returns:
            Pooled features (B, C)
        """
        # Compute attention weights
        weights = self.attention(x)  # (B, N, 1)
        weights = F.softmax(weights, dim=1)
        
        # Weighted sum
        pooled = (x * weights).sum(dim=1)  # (B, C)
        
        return pooled


class VascularityClassifier(nn.Module):
    """
    Vascularity pattern classifier.
    
    Classifies blood flow patterns visible in Doppler ultrasound
    into clinical vascularity types:
    - Type 0: Avascular - no detectable flow
    - Type I: Minimal flow - occasional spots
    - Type II: Moderate flow - consistent patterns
    - Type III: High vascularity - extensive flow
    
    Attributes:
        num_classes: Number of vascularity types (4)
        feature_dim: Input feature dimension
    """
    
    def __init__(
        self,
        feature_dim: int = 768,
        hidden_dim: int = 256,
        num_classes: int = 4,
        dropout: float = 0.3,
    ):
        """
        Initialize the classifier.
        
        Args:
            feature_dim: Input feature dimension from backbone
            hidden_dim: Hidden layer dimension
            num_classes: Number of classification classes
            dropout: Dropout probability
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        
        # Feature processing
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        # Attention pooling for spatial features
        self.attention_pool = AttentionPool(hidden_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        
        # Confidence estimation head
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        features: torch.Tensor,
        return_features: bool = False,
    ) -> VascularityOutput:
        """
        Forward pass for vascularity classification.
        
        Args:
            features: Backbone features (B, C, H, W) or (B, C)
            return_features: Whether to return intermediate features
            
        Returns:
            VascularityOutput with predictions and confidence
        """
        # Handle different input shapes
        if features.dim() == 4:
            # (B, C, H, W) -> (B, H*W, C)
            B, C, H, W = features.shape
            features = features.permute(0, 2, 3, 1).reshape(B, H * W, C)
        elif features.dim() == 2:
            # (B, C) -> (B, 1, C)
            features = features.unsqueeze(1)
        
        # Project features
        projected = self.feature_proj(features)  # (B, N, hidden_dim)
        
        # Attention pooling
        pooled = self.attention_pool(projected)  # (B, hidden_dim)
        
        # Classification
        logits = self.classifier(pooled)  # (B, num_classes)
        probabilities = F.softmax(logits, dim=-1)
        
        # Predictions
        predicted_type = torch.argmax(probabilities, dim=-1)
        
        # Confidence estimation
        confidence = self.confidence_head(pooled).squeeze(-1)
        
        # Adjust confidence based on prediction certainty
        max_prob = probabilities.max(dim=-1).values
        confidence = confidence * max_prob  # Combine learned and softmax confidence
        
        output = VascularityOutput(
            probabilities=probabilities,
            predicted_type=predicted_type,
            confidence=confidence,
            logits=logits,
        )
        
        return output
    
    def compute_loss(
        self,
        output: VascularityOutput,
        target: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.1,
    ) -> dict[str, torch.Tensor]:
        """
        Compute classification loss.
        
        Args:
            output: Model output
            target: Ground truth class labels (B,)
            class_weights: Optional class weights for imbalanced data
            label_smoothing: Label smoothing factor
            
        Returns:
            Dictionary with loss values
        """
        # Cross-entropy with label smoothing
        ce_loss = F.cross_entropy(
            output.logits,
            target,
            weight=class_weights,
            label_smoothing=label_smoothing,
        )
        
        # Accuracy for monitoring
        correct = (output.predicted_type == target).float().mean()
        
        return {
            "total": ce_loss,
            "ce": ce_loss,
            "accuracy": correct,
        }


class MultiScaleVascularityClassifier(nn.Module):
    """
    Multi-scale vascularity classifier.
    
    Uses features from multiple backbone stages for
    more robust classification.
    """
    
    def __init__(
        self,
        feature_dims: list[int] = [96, 192, 384, 768],
        hidden_dim: int = 256,
        num_classes: int = 4,
        dropout: float = 0.3,
    ):
        """
        Initialize multi-scale classifier.
        
        Args:
            feature_dims: Feature dimensions from each backbone stage
            hidden_dim: Hidden layer dimension
            num_classes: Number of classes
            dropout: Dropout probability
        """
        super().__init__()
        
        self.num_classes = num_classes
        
        # Project each scale to same dimension
        self.scale_projections = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
            )
            for dim in feature_dims
        ])
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * len(feature_dims), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
    
    def forward(self, features: list[torch.Tensor]) -> VascularityOutput:
        """
        Forward pass with multi-scale features.
        
        Args:
            features: List of features from backbone stages
            
        Returns:
            VascularityOutput
        """
        # Project each scale
        projected = [
            proj(feat) for proj, feat in zip(self.scale_projections, features)
        ]
        
        # Concatenate and fuse
        fused = torch.cat(projected, dim=-1)
        fused = self.fusion(fused)
        
        # Classify
        logits = self.classifier(fused)
        probabilities = F.softmax(logits, dim=-1)
        predicted_type = torch.argmax(probabilities, dim=-1)
        confidence = probabilities.max(dim=-1).values
        
        return VascularityOutput(
            probabilities=probabilities,
            predicted_type=predicted_type,
            confidence=confidence,
            logits=logits,
        )
