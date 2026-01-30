"""
Temporal frame aggregator for video analysis.

Aggregates predictions across video frames using various
temporal modeling strategies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TemporalOutput:
    """Output from temporal aggregation."""
    
    # Aggregated features (B, C)
    features: torch.Tensor
    
    # Per-frame attention weights (B, T)
    attention_weights: Optional[torch.Tensor] = None
    
    # Aggregation confidence (B,)
    confidence: Optional[torch.Tensor] = None
    
    # Individual frame features before aggregation (B, T, C)
    frame_features: Optional[torch.Tensor] = None


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences."""
    
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return x + self.pe[:, :x.size(1)]


class AttentionAggregator(nn.Module):
    """Attention-based temporal aggregation."""
    
    def __init__(
        self,
        feature_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # Query for aggregation (learnable)
        self.query = nn.Parameter(torch.randn(1, 1, feature_dim))
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
        )
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Aggregate temporal features using attention.
        
        Args:
            x: Features (B, T, C)
            
        Returns:
            Tuple of (aggregated features, attention weights)
        """
        B = x.size(0)
        
        # Expand query for batch
        query = self.query.expand(B, -1, -1)
        
        # Attention
        aggregated, weights = self.attention(query, x, x)
        
        # Project
        output = self.output_proj(aggregated.squeeze(1))
        
        return output, weights.squeeze(1)


class LSTMAggregator(nn.Module):
    """LSTM-based temporal aggregation."""
    
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        
        output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.output_proj = nn.Sequential(
            nn.Linear(output_dim, feature_dim),
            nn.LayerNorm(feature_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aggregate using LSTM.
        
        Args:
            x: Features (B, T, C)
            
        Returns:
            Aggregated features (B, C)
        """
        # LSTM forward
        output, (hidden, _) = self.lstm(x)
        
        # Use last hidden state
        if self.lstm.bidirectional:
            # Concatenate forward and backward final states
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        else:
            hidden = hidden[-1]
        
        # Project back to feature dimension
        return self.output_proj(hidden)


class ConfidenceWeightedAggregator(nn.Module):
    """Aggregation weighted by per-frame confidence scores."""
    
    def __init__(self, feature_dim: int):
        super().__init__()
        
        self.confidence_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 4, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Aggregate weighted by predicted confidence.
        
        Args:
            x: Features (B, T, C)
            
        Returns:
            Tuple of (aggregated features, confidence weights)
        """
        # Predict confidence for each frame
        confidence = self.confidence_net(x).squeeze(-1)  # (B, T)
        
        # Normalize to sum to 1
        weights = confidence / (confidence.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Weighted sum
        aggregated = (x * weights.unsqueeze(-1)).sum(dim=1)
        
        return aggregated, confidence


class TemporalAggregator(nn.Module):
    """
    Temporal frame aggregator for video ultrasound analysis.
    
    Aggregates predictions across video frames using configurable
    strategies: attention, LSTM, or confidence-weighted averaging.
    
    Attributes:
        feature_dim: Input feature dimension
        method: Aggregation method
    """
    
    METHODS = ['attention', 'lstm', 'confidence', 'mean']
    
    def __init__(
        self,
        feature_dim: int = 768,
        method: Literal['attention', 'lstm', 'confidence', 'mean'] = 'attention',
        num_heads: int = 4,
        lstm_hidden: int = 256,
        dropout: float = 0.1,
        use_positional_encoding: bool = True,
    ):
        """
        Initialize temporal aggregator.
        
        Args:
            feature_dim: Feature dimension from backbone
            method: Aggregation method
            num_heads: Number of attention heads (if using attention)
            lstm_hidden: LSTM hidden dimension (if using LSTM)
            dropout: Dropout probability
            use_positional_encoding: Add positional encoding
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.method = method
        self.use_positional_encoding = use_positional_encoding
        
        # Positional encoding
        if use_positional_encoding:
            self.pos_encoding = PositionalEncoding(feature_dim)
        
        # Aggregation module based on method
        if method == 'attention':
            self.aggregator = AttentionAggregator(
                feature_dim, num_heads, dropout,
            )
        elif method == 'lstm':
            self.aggregator = LSTMAggregator(
                feature_dim, lstm_hidden, dropout=dropout,
            )
        elif method == 'confidence':
            self.aggregator = ConfidenceWeightedAggregator(feature_dim)
        elif method == 'mean':
            self.aggregator = None
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
        
        # Aggregation confidence estimation
        self.confidence_head = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        frame_features: torch.Tensor,
        return_frame_features: bool = False,
    ) -> TemporalOutput:
        """
        Aggregate temporal frame features.
        
        Args:
            frame_features: Per-frame features (B, T, C) or (B, T, C, H, W)
            return_frame_features: Include original frame features in output
            
        Returns:
            TemporalOutput with aggregated features
        """
        # Handle spatial features
        if frame_features.dim() == 5:
            # (B, T, C, H, W) -> (B, T, C) via global average pooling
            B, T, C, H, W = frame_features.shape
            frame_features = frame_features.mean(dim=[-2, -1])
        
        # Add positional encoding
        if self.use_positional_encoding:
            x = self.pos_encoding(frame_features)
        else:
            x = frame_features
        
        # Aggregate
        attention_weights = None
        
        if self.method == 'mean':
            aggregated = x.mean(dim=1)
        elif self.method in ['attention', 'confidence']:
            aggregated, attention_weights = self.aggregator(x)
        else:  # lstm
            aggregated = self.aggregator(x)
        
        # Estimate aggregation confidence
        confidence = self.confidence_head(aggregated).squeeze(-1)
        
        return TemporalOutput(
            features=aggregated,
            attention_weights=attention_weights,
            confidence=confidence,
            frame_features=frame_features if return_frame_features else None,
        )
    
    def aggregate_outputs(
        self,
        outputs: list[dict],
        weights: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Aggregate multiple per-frame outputs.
        
        Useful for combining predictions from individual frames
        after running through the model.
        
        Args:
            outputs: List of output dictionaries per frame
            weights: Optional per-frame weights
            
        Returns:
            Aggregated output dictionary
        """
        if not outputs:
            return {}
        
        if weights is None:
            weights = torch.ones(len(outputs)) / len(outputs)
        else:
            weights = weights / weights.sum()
        
        aggregated = {}
        
        # Get keys from first output
        for key in outputs[0]:
            values = [out[key] for out in outputs]
            
            if isinstance(values[0], torch.Tensor):
                # Weighted average of tensors
                stacked = torch.stack(values)
                aggregated[key] = (stacked * weights.view(-1, *([1] * (stacked.dim() - 1)))).sum(dim=0)
            elif isinstance(values[0], (int, float)):
                # Weighted average of scalars
                aggregated[key] = sum(v * w.item() for v, w in zip(values, weights))
            else:
                # Take from highest weighted frame
                max_idx = weights.argmax().item()
                aggregated[key] = values[max_idx]
        
        return aggregated
