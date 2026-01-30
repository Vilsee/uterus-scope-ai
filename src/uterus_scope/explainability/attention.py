"""
Attention visualization for Vision Transformers.

Extracts and visualizes attention patterns from
Swin Transformer and other ViT architectures.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import cv2
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def extract_attention_weights(
    model: nn.Module,
    input_tensor: torch.Tensor,
    layer_indices: Optional[list[int]] = None,
) -> list[torch.Tensor]:
    """
    Extract attention weights from transformer layers.
    
    Args:
        model: Vision Transformer model
        input_tensor: Input image tensor
        layer_indices: Which layers to extract (all if None)
        
    Returns:
        List of attention weight tensors
    """
    attention_weights = []
    hooks = []
    
    def get_attention_hook(store_list):
        def hook(module, input, output):
            # Handle different attention output formats
            if isinstance(output, tuple):
                # (attn_output, attn_weights)
                if len(output) > 1 and output[1] is not None:
                    store_list.append(output[1].detach())
            elif hasattr(module, 'attn_weights'):
                store_list.append(module.attn_weights.detach())
        return hook
    
    # Find attention layers
    attention_layers = []
    for name, module in model.named_modules():
        if 'attention' in name.lower() or 'attn' in name.lower():
            if isinstance(module, nn.MultiheadAttention):
                attention_layers.append(module)
    
    # Filter by indices
    if layer_indices is not None:
        attention_layers = [attention_layers[i] for i in layer_indices if i < len(attention_layers)]
    
    # Register hooks
    for layer in attention_layers:
        hook = layer.register_forward_hook(get_attention_hook(attention_weights))
        hooks.append(hook)
    
    try:
        # Forward pass
        model.eval()
        with torch.no_grad():
            _ = model(input_tensor)
    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()
    
    return attention_weights


class AttentionVisualizer:
    """
    Visualize attention patterns from Vision Transformers.
    
    Provides multiple visualization methods:
    - Raw attention maps
    - Attention rollout (aggregated across layers)
    - Attention flow (gradient weighted)
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
    ):
        """
        Initialize visualizer.
        
        Args:
            model: Vision Transformer model
            device: Computation device
        """
        self.model = model
        self.device = device
        self.attention_maps = []
    
    def visualize_attention(
        self,
        input_tensor: torch.Tensor,
        head_idx: Optional[int] = None,
        layer_idx: int = -1,
    ) -> np.ndarray:
        """
        Visualize attention from a specific layer and head.
        
        Args:
            input_tensor: Input image tensor
            head_idx: Attention head index (average if None)
            layer_idx: Layer index (-1 for last)
            
        Returns:
            Attention map as numpy array
        """
        input_tensor = input_tensor.to(self.device)
        
        # Extract attention weights
        attention_weights = extract_attention_weights(
            self.model, input_tensor,
        )
        
        if not attention_weights:
            logger.warning("No attention weights extracted")
            return np.zeros((input_tensor.shape[-2], input_tensor.shape[-1]))
        
        # Get specified layer
        attn = attention_weights[layer_idx]  # (B, H, N, N)
        
        # Average or select head
        if head_idx is None:
            attn = attn.mean(dim=1)  # (B, N, N)
        else:
            attn = attn[:, head_idx]  # (B, N, N)
        
        # Get attention to CLS token (or average)
        # Assuming CLS token is first
        if attn.shape[-1] > 1:
            attn_map = attn[0, 0, 1:]  # Attention from CLS to patches
        else:
            attn_map = attn[0].mean(dim=0)
        
        # Reshape to spatial
        attn_map = attn_map.cpu().numpy()
        n_patches = len(attn_map)
        h = w = int(np.sqrt(n_patches))
        
        if h * w == n_patches:
            attn_map = attn_map.reshape(h, w)
        else:
            # Pad if necessary
            attn_map = attn_map[:h*w].reshape(h, w)
        
        # Normalize
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
        
        # Resize to input size
        attn_map = cv2.resize(
            attn_map,
            (input_tensor.shape[-1], input_tensor.shape[-2]),
            interpolation=cv2.INTER_LINEAR,
        )
        
        return attn_map
    
    def attention_rollout(
        self,
        input_tensor: torch.Tensor,
        discard_ratio: float = 0.9,
    ) -> np.ndarray:
        """
        Compute attention rollout across all layers.
        
        Attention rollout aggregates attention from all layers
        to show cumulative attention flow.
        
        Args:
            input_tensor: Input image tensor
            discard_ratio: Fraction of lowest attention to discard
            
        Returns:
            Rollout attention map
        """
        input_tensor = input_tensor.to(self.device)
        
        attention_weights = extract_attention_weights(
            self.model, input_tensor,
        )
        
        if not attention_weights:
            return np.zeros((input_tensor.shape[-2], input_tensor.shape[-1]))
        
        # Initialize with identity
        result = None
        
        for attn in attention_weights:
            # Average across heads
            attn = attn.mean(dim=1)  # (B, N, N)
            
            # Add identity for residual
            attn = attn + torch.eye(attn.shape[-1], device=attn.device)
            
            # Normalize
            attn = attn / attn.sum(dim=-1, keepdim=True)
            
            # Rollout
            if result is None:
                result = attn
            else:
                result = torch.bmm(attn, result)
        
        # Get attention to CLS token
        if result is not None:
            mask = result[0, 0, 1:]  # Exclude CLS
            mask = mask.cpu().numpy()
            
            # Reshape to spatial
            n = len(mask)
            h = w = int(np.sqrt(n))
            if h * w == n:
                mask = mask.reshape(h, w)
            else:
                mask = mask[:h*w].reshape(h, w)
            
            # Normalize
            mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
            
            # Resize
            mask = cv2.resize(
                mask,
                (input_tensor.shape[-1], input_tensor.shape[-2]),
            )
            
            return mask
        
        return np.zeros((input_tensor.shape[-2], input_tensor.shape[-1]))
    
    def visualize_all_heads(
        self,
        input_tensor: torch.Tensor,
        layer_idx: int = -1,
        grid_size: Optional[tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Visualize attention from all heads in a grid.
        
        Args:
            input_tensor: Input image tensor
            layer_idx: Layer to visualize
            grid_size: Grid dimensions (auto-compute if None)
            
        Returns:
            Grid image of all attention heads
        """
        input_tensor = input_tensor.to(self.device)
        
        attention_weights = extract_attention_weights(
            self.model, input_tensor,
        )
        
        if not attention_weights:
            return np.zeros((224, 224))
        
        attn = attention_weights[layer_idx]  # (B, H, N, N)
        num_heads = attn.shape[1]
        
        # Compute grid size
        if grid_size is None:
            cols = int(np.ceil(np.sqrt(num_heads)))
            rows = int(np.ceil(num_heads / cols))
            grid_size = (rows, cols)
        
        head_maps = []
        for h in range(num_heads):
            head_attn = attn[0, h]
            
            # Get spatial attention
            if head_attn.shape[0] > 1:
                attn_map = head_attn[0, 1:]  # CLS to patches
            else:
                attn_map = head_attn.mean(dim=0)
            
            attn_map = attn_map.cpu().numpy()
            n = len(attn_map)
            hw = int(np.sqrt(n))
            
            if hw * hw == n:
                attn_map = attn_map.reshape(hw, hw)
            else:
                attn_map = attn_map[:hw*hw].reshape(hw, hw)
            
            # Normalize and resize
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
            attn_map = cv2.resize(attn_map, (64, 64))
            head_maps.append(attn_map)
        
        # Create grid
        rows, cols = grid_size
        grid = np.zeros((rows * 64, cols * 64))
        
        for i, head_map in enumerate(head_maps):
            if i >= rows * cols:
                break
            r, c = i // cols, i % cols
            grid[r*64:(r+1)*64, c*64:(c+1)*64] = head_map
        
        return grid
    
    def create_attention_overlay(
        self,
        image: np.ndarray,
        attention_map: np.ndarray,
        colormap: int = cv2.COLORMAP_VIRIDIS,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """
        Create overlay of attention on original image.
        
        Args:
            image: Original image
            attention_map: Attention map
            colormap: Color mapping
            alpha: Blend factor
            
        Returns:
            Overlay image in RGB
        """
        # Ensure proper format
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Resize attention map
        attention_map = cv2.resize(
            attention_map,
            (image.shape[1], image.shape[0]),
        )
        
        # Apply colormap
        attention_colored = cv2.applyColorMap(
            (attention_map * 255).astype(np.uint8),
            colormap,
        )
        
        # Blend
        overlay = cv2.addWeighted(image, 1 - alpha, attention_colored, alpha, 0)
        
        return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
