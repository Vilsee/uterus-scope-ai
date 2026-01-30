"""Explainability module for UterusScope-AI."""

from uterus_scope.explainability.gradcam import (
    VisionTransformerGradCAM,
    generate_heatmap,
    overlay_heatmap,
)
from uterus_scope.explainability.attention import (
    AttentionVisualizer,
    extract_attention_weights,
)

__all__ = [
    "VisionTransformerGradCAM",
    "generate_heatmap",
    "overlay_heatmap",
    "AttentionVisualizer",
    "extract_attention_weights",
]
