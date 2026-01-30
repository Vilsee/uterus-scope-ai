"""
Unified UterusScope model combining all analysis heads.

This is the main model that integrates backbone, segmentation,
vascularity, fibrosis, and temporal modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

import torch
import torch.nn as nn

from uterus_scope.config import get_config, ModelBackbone
from uterus_scope.models.backbone import SwinBackbone, create_swin_backbone
from uterus_scope.models.segmentation import EndometrialSegmentationHead, SegmentationOutput
from uterus_scope.models.vascularity import VascularityClassifier, VascularityOutput
from uterus_scope.models.fibrosis import FibrosisDetector, FibrosisOutput
from uterus_scope.models.temporal import TemporalAggregator, TemporalOutput


@dataclass
class ModelOutput:
    """Complete output from UterusScope model."""
    
    # Segmentation results
    segmentation: SegmentationOutput
    
    # Vascularity classification
    vascularity: VascularityOutput
    
    # Fibrosis detection
    fibrosis: FibrosisOutput
    
    # Temporal aggregation (if video input)
    temporal: Optional[TemporalOutput] = None
    
    # Backbone features for explainability
    features: Optional[list[torch.Tensor]] = None
    
    # Combined confidence score
    overall_confidence: Optional[torch.Tensor] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            'endometrial_thickness_mm': self.segmentation.thickness_mm.tolist(),
            'vascularity_type': self.vascularity.predicted_type.tolist(),
            'vascularity_probabilities': self.vascularity.probabilities.tolist(),
            'vascularity_confidence': self.vascularity.confidence.tolist(),
            'fibrosis_score': self.fibrosis.severity_score.tolist(),
            'has_fibrosis': self.fibrosis.has_fibrosis.tolist() if self.fibrosis.has_fibrosis is not None else None,
            'fibrosis_regions': self.fibrosis.regions,
            'overall_confidence': self.overall_confidence.tolist() if self.overall_confidence is not None else None,
        }


class UterusScopeModel(nn.Module):
    """
    Unified model for ultrasound analysis.
    
    Combines Swin Transformer backbone with specialized heads
    for endometrial segmentation, vascularity classification,
    and fibrosis detection.
    
    Supports both single-frame and video input modes.
    
    Attributes:
        backbone: Swin Transformer feature extractor
        segmentation_head: Endometrial segmentation
        vascularity_head: Vascularity classifier
        fibrosis_head: Fibrosis detector
        temporal_aggregator: Video frame aggregation
    """
    
    def __init__(
        self,
        backbone_type: Optional[ModelBackbone] = None,
        pretrained: bool = True,
        in_channels: int = 1,
        output_size: int = 224,
        temporal_method: str = 'attention',
        freeze_backbone: bool = False,
    ):
        """
        Initialize the unified model.
        
        Args:
            backbone_type: Swin backbone variant
            pretrained: Use pretrained weights
            in_channels: Input channels (1 for grayscale)
            output_size: Output spatial resolution
            temporal_method: Temporal aggregation method
            freeze_backbone: Freeze backbone weights
        """
        super().__init__()
        
        config = get_config()
        backbone_type = backbone_type or config.model.backbone
        
        # Backbone
        self.backbone = create_swin_backbone(
            backbone_type=backbone_type,
            pretrained=pretrained,
            in_channels=in_channels,
        )
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Get feature dimensions
        feature_dims = self.backbone.get_feature_dims()
        
        # Segmentation head
        self.segmentation_head = EndometrialSegmentationHead(
            feature_dims=feature_dims,
            output_size=output_size,
        )
        
        # Vascularity classifier
        self.vascularity_head = VascularityClassifier(
            feature_dim=feature_dims[-1],
        )
        
        # Fibrosis detector
        self.fibrosis_head = FibrosisDetector(
            feature_dims=feature_dims,
            output_size=output_size,
        )
        
        # Temporal aggregator for video
        self.temporal_aggregator = TemporalAggregator(
            feature_dim=feature_dims[-1],
            method=temporal_method,
        )
        
        # Overall confidence estimation
        self.confidence_head = nn.Sequential(
            nn.Linear(feature_dims[-1], 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        extract_fibrosis_regions: bool = True,
    ) -> ModelOutput:
        """
        Forward pass for single frame or batch.
        
        Args:
            x: Input tensor (B, C, H, W)
            return_features: Include backbone features in output
            extract_fibrosis_regions: Extract fibrosis bounding boxes
            
        Returns:
            ModelOutput with all predictions
        """
        # Extract multi-scale features
        features = self.backbone(x)
        
        # Segmentation
        seg_output = self.segmentation_head(features)
        
        # Vascularity classification
        vasc_output = self.vascularity_head(features[-1])
        
        # Fibrosis detection
        fibrosis_output = self.fibrosis_head(
            features,
            extract_regions=extract_fibrosis_regions,
        )
        
        # Overall confidence from deepest features
        pooled = features[-1].mean(dim=[-2, -1])  # Global average pool
        overall_conf = self.confidence_head(pooled).squeeze(-1)
        
        # Combine with individual confidences
        overall_conf = overall_conf * vasc_output.confidence
        
        return ModelOutput(
            segmentation=seg_output,
            vascularity=vasc_output,
            fibrosis=fibrosis_output,
            temporal=None,
            features=features if return_features else None,
            overall_confidence=overall_conf,
        )
    
    def forward_video(
        self,
        frames: torch.Tensor,
        return_features: bool = False,
        aggregate_predictions: bool = True,
    ) -> ModelOutput:
        """
        Forward pass for video sequence.
        
        Args:
            frames: Video frames (B, T, C, H, W)
            return_features: Include features in output
            aggregate_predictions: Aggregate frame predictions
            
        Returns:
            ModelOutput with temporally aggregated predictions
        """
        B, T, C, H, W = frames.shape
        
        # Process each frame
        frame_features = []
        frame_outputs = []
        
        for t in range(T):
            frame = frames[:, t]
            features = self.backbone(frame)
            frame_features.append(features[-1])
            
            if not aggregate_predictions:
                output = self.forward(frame, return_features=False)
                frame_outputs.append(output)
        
        # Stack frame features: (B, T, C, H, W)
        stacked_features = torch.stack(frame_features, dim=1)
        
        # Temporal aggregation
        temporal_output = self.temporal_aggregator(
            stacked_features, return_frame_features=True,
        )
        
        if aggregate_predictions:
            # Run heads on aggregated features
            # We need to reconstruct spatial features for some heads
            # Use attention-weighted combination
            
            if temporal_output.attention_weights is not None:
                # (B, T) weights
                weights = temporal_output.attention_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                # Weighted combination of spatial features
                aggregated_spatial = (stacked_features * weights).sum(dim=1)
            else:
                aggregated_spatial = stacked_features.mean(dim=1)
            
            # Get full feature pyramid for the "best" frame
            # Use the frame with highest attention weight
            if temporal_output.attention_weights is not None:
                best_frame_idx = temporal_output.attention_weights.argmax(dim=1)
            else:
                best_frame_idx = torch.zeros(B, dtype=torch.long)
            
            # Process best frame for full features
            best_frames = frames[torch.arange(B), best_frame_idx]
            all_features = self.backbone(best_frames)
            
            # Replace deepest features with aggregated
            all_features[-1] = aggregated_spatial
            
            # Run heads
            seg_output = self.segmentation_head(all_features)
            vasc_output = self.vascularity_head(aggregated_spatial)
            fibrosis_output = self.fibrosis_head(all_features)
            
            # Overall confidence
            pooled = temporal_output.features
            overall_conf = self.confidence_head(pooled).squeeze(-1)
            overall_conf = overall_conf * vasc_output.confidence * temporal_output.confidence
            
            return ModelOutput(
                segmentation=seg_output,
                vascularity=vasc_output,
                fibrosis=fibrosis_output,
                temporal=temporal_output,
                features=all_features if return_features else None,
                overall_confidence=overall_conf,
            )
        else:
            # Return per-frame outputs
            return frame_outputs, temporal_output
    
    def get_attention_maps(
        self,
        x: torch.Tensor,
        layer_idx: int = -1,
    ) -> torch.Tensor:
        """
        Extract attention maps from backbone for visualization.
        
        Args:
            x: Input tensor (B, C, H, W)
            layer_idx: Which transformer layer to extract from
            
        Returns:
            Attention maps
        """
        # This would require modifying the backbone to capture attention
        # For now, return the deepest feature map as proxy
        features = self.backbone(x)
        return features[layer_idx]
    
    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        device: str = 'cuda',
        **kwargs,
    ) -> 'UterusScopeModel':
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
            device: Target device
            **kwargs: Additional model arguments
            
        Returns:
            Loaded model
        """
        model = cls(**kwargs)
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        return model.to(device)
    
    def save_checkpoint(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Save model checkpoint.
        
        Args:
            path: Save path
            optimizer: Optional optimizer state
            epoch: Optional epoch number
            **kwargs: Additional items to save
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': {
                'backbone_type': self.backbone.backbone_type.value,
            },
            **kwargs,
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if epoch is not None:
            checkpoint['epoch'] = epoch
        
        torch.save(checkpoint, path)
