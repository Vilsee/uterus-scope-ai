"""
GradCAM++ implementation for Vision Transformers.

Generates attention heatmaps showing which image regions
contributed most to model predictions.
"""

from __future__ import annotations

from typing import Optional, Union, Callable
import logging

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class VisionTransformerGradCAM:
    """
    GradCAM++ for Vision Transformer models.
    
    Generates class activation maps by computing gradients
    of the target class with respect to feature maps.
    
    Supports Swin Transformer and other ViT variants.
    
    Attributes:
        model: The model to explain
        target_layer: Layer to compute gradients for
        device: Computation device
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
        device: str = 'cuda',
    ):
        """
        Initialize GradCAM.
        
        Args:
            model: Model to generate explanations for
            target_layer: Specific layer to target (auto-detect if None)
            device: Computation device
        """
        self.model = model
        self.device = device
        
        # Auto-detect target layer if not provided
        if target_layer is None:
            target_layer = self._find_target_layer()
        
        self.target_layer = target_layer
        
        # Storage for gradients and activations
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _find_target_layer(self) -> nn.Module:
        """Find the last convolutional or attention layer."""
        target = None
        
        # Look for the last norm or attention layer in backbone
        if hasattr(self.model, 'backbone'):
            backbone = self.model.backbone
            if hasattr(backbone, 'backbone'):
                backbone = backbone.backbone
            
            # Try to find layers attribute
            for name, module in backbone.named_modules():
                if isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
                    target = module
        
        if target is None:
            # Fallback: use last module
            modules = list(self.model.modules())
            for m in reversed(modules):
                if isinstance(m, (nn.Conv2d, nn.Linear, nn.LayerNorm)):
                    target = m
                    break
        
        logger.info(f"GradCAM target layer: {type(target).__name__}")
        return target
    
    def _register_hooks(self) -> None:
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_category: Optional[int] = None,
        target_output: str = 'vascularity',
    ) -> np.ndarray:
        """
        Generate GradCAM heatmap.
        
        Args:
            input_tensor: Input image tensor (B, C, H, W)
            target_category: Target class index (uses predicted if None)
            target_output: Which model output to use for gradients
            
        Returns:
            Heatmap array (H, W) normalized to [0, 1]
        """
        self.model.eval()
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad = True
        
        # Forward pass
        output = self.model(input_tensor, return_features=True)
        
        # Get target output for gradients
        if target_output == 'vascularity':
            logits = output.vascularity.logits
        elif target_output == 'fibrosis':
            logits = output.fibrosis.severity_score
        elif target_output == 'segmentation':
            logits = output.segmentation.thickness_mm
        else:
            logits = output.vascularity.logits
        
        # Determine target class
        if target_category is None and logits.dim() > 1:
            target_category = logits.argmax(dim=-1)[0].item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        if logits.dim() > 1:
            # Classification output
            one_hot = torch.zeros_like(logits)
            one_hot[0, target_category] = 1
            logits.backward(gradient=one_hot, retain_graph=True)
        else:
            # Regression output
            logits[0].backward(retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients
        activations = self.activations
        
        if gradients is None or activations is None:
            logger.warning("No gradients captured, returning empty heatmap")
            return np.zeros((input_tensor.shape[-2], input_tensor.shape[-1]))
        
        # Handle different tensor shapes
        if gradients.dim() == 3:
            # (B, N, C) -> compute weights via global average
            weights = gradients.mean(dim=1, keepdim=True)  # (B, 1, C)
            cam = (weights * activations).sum(dim=-1)  # (B, N)
            
            # Reshape to spatial
            N = cam.shape[1]
            h = w = int(np.sqrt(N))
            if h * w != N:
                # Handle CLS token
                cam = cam[:, 1:]  # Remove CLS
                N = cam.shape[1]
                h = w = int(np.sqrt(N))
            
            cam = cam.view(-1, h, w)
        
        elif gradients.dim() == 4:
            # (B, C, H, W) - standard CNN format
            weights = gradients.mean(dim=[2, 3], keepdim=True)
            cam = (weights * activations).sum(dim=1)
        
        else:
            logger.warning(f"Unexpected gradient shape: {gradients.shape}")
            return np.zeros((input_tensor.shape[-2], input_tensor.shape[-1]))
        
        # Apply ReLU (keep positive contributions only)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam[0].cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        # Resize to input size
        cam = cv2.resize(
            cam,
            (input_tensor.shape[-1], input_tensor.shape[-2]),
            interpolation=cv2.INTER_LINEAR,
        )
        
        return cam
    
    def generate_multi_target(
        self,
        input_tensor: torch.Tensor,
    ) -> dict[str, np.ndarray]:
        """
        Generate heatmaps for all output types.
        
        Args:
            input_tensor: Input image tensor
            
        Returns:
            Dictionary of heatmaps for each output type
        """
        heatmaps = {}
        
        for target in ['vascularity', 'fibrosis', 'segmentation']:
            try:
                heatmaps[target] = self.generate(
                    input_tensor.clone(),
                    target_output=target,
                )
            except Exception as e:
                logger.warning(f"Failed to generate {target} heatmap: {e}")
                heatmaps[target] = np.zeros(
                    (input_tensor.shape[-2], input_tensor.shape[-1])
                )
        
        return heatmaps


def generate_heatmap(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_output: str = 'vascularity',
    target_category: Optional[int] = None,
) -> np.ndarray:
    """
    Convenience function to generate a single heatmap.
    
    Args:
        model: Model to explain
        input_tensor: Input image
        target_output: Output type to explain
        target_category: Target class (for classification)
        
    Returns:
        Heatmap array
    """
    device = next(model.parameters()).device
    gradcam = VisionTransformerGradCAM(model, device=str(device))
    
    return gradcam.generate(
        input_tensor,
        target_category=target_category,
        target_output=target_output,
    )


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Overlay heatmap on original image.
    
    Args:
        image: Original image (H, W) or (H, W, 3)
        heatmap: Heatmap array (H, W) normalized to [0, 1]
        alpha: Blending factor
        colormap: OpenCV colormap
        
    Returns:
        RGB overlay image
    """
    # Ensure image is uint8
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    # Convert grayscale to BGR
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[-1] == 1:
        image = cv2.cvtColor(image.squeeze(-1), cv2.COLOR_GRAY2BGR)
    
    # Resize heatmap if needed
    if heatmap.shape[:2] != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Apply colormap to heatmap
    heatmap_colored = cv2.applyColorMap(
        (heatmap * 255).astype(np.uint8),
        colormap,
    )
    
    # Blend
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    
    # Convert to RGB
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


class GradCAMPlusPlus(VisionTransformerGradCAM):
    """
    GradCAM++ variant with improved weighting.
    
    Uses second-order gradients for better localization.
    """
    
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_category: Optional[int] = None,
        target_output: str = 'vascularity',
    ) -> np.ndarray:
        """Generate GradCAM++ heatmap with improved weighting."""
        self.model.eval()
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad = True
        
        # Forward pass
        output = self.model(input_tensor, return_features=True)
        
        # Get target output
        if target_output == 'vascularity':
            logits = output.vascularity.logits
        else:
            logits = output.fibrosis.severity_score
        
        if target_category is None and logits.dim() > 1:
            target_category = logits.argmax(dim=-1)[0].item()
        
        self.model.zero_grad()
        
        # Backward pass
        if logits.dim() > 1:
            one_hot = torch.zeros_like(logits)
            one_hot[0, target_category] = 1
            logits.backward(gradient=one_hot, retain_graph=True)
        else:
            logits[0].backward(retain_graph=True)
        
        gradients = self.gradients
        activations = self.activations
        
        if gradients is None or activations is None:
            return np.zeros((input_tensor.shape[-2], input_tensor.shape[-1]))
        
        # GradCAM++ weighting
        if gradients.dim() == 4:
            # Alpha computation for GradCAM++
            grad_2 = gradients ** 2
            grad_3 = grad_2 * gradients
            
            sum_activations = activations.sum(dim=[2, 3], keepdim=True)
            alpha_num = grad_2
            alpha_denom = 2 * grad_2 + sum_activations * grad_3 + 1e-7
            alpha = alpha_num / alpha_denom
            
            weights = (alpha * F.relu(gradients)).sum(dim=[2, 3], keepdim=True)
            cam = (weights * activations).sum(dim=1)
        else:
            # Fallback to standard GradCAM
            return super().generate(input_tensor, target_category, target_output)
        
        cam = F.relu(cam)
        cam = cam[0].cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        cam = cv2.resize(
            cam,
            (input_tensor.shape[-1], input_tensor.shape[-2]),
        )
        
        return cam
