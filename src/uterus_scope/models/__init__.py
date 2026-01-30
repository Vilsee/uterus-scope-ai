"""AI Models module for UterusScope-AI."""

from uterus_scope.models.backbone import (
    SwinBackbone,
    create_swin_backbone,
)
from uterus_scope.models.segmentation import (
    EndometrialSegmentationHead,
    SegmentationOutput,
)
from uterus_scope.models.vascularity import (
    VascularityClassifier,
    VascularityOutput,
)
from uterus_scope.models.fibrosis import (
    FibrosisDetector,
    FibrosisOutput,
)
from uterus_scope.models.temporal import (
    TemporalAggregator,
    TemporalOutput,
)
from uterus_scope.models.unified import (
    UterusScopeModel,
    ModelOutput,
)

__all__ = [
    # Backbone
    "SwinBackbone",
    "create_swin_backbone",
    # Heads
    "EndometrialSegmentationHead",
    "SegmentationOutput",
    "VascularityClassifier",
    "VascularityOutput",
    "FibrosisDetector",
    "FibrosisOutput",
    "TemporalAggregator",
    "TemporalOutput",
    # Unified model
    "UterusScopeModel",
    "ModelOutput",
]
