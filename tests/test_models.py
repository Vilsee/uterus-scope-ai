"""Tests for AI models."""

import pytest
import torch

from uterus_scope.models.backbone import SwinBackbone, create_swin_backbone
from uterus_scope.models.segmentation import EndometrialSegmentationHead
from uterus_scope.models.vascularity import VascularityClassifier
from uterus_scope.models.fibrosis import FibrosisDetector
from uterus_scope.models.unified import UterusScopeModel
from uterus_scope.config import ModelBackbone


class TestSwinBackbone:
    """Tests for Swin Transformer backbone."""
    
    @pytest.fixture
    def backbone(self):
        return create_swin_backbone(
            backbone_type=ModelBackbone.SWIN_TINY,
            pretrained=False,
            in_channels=1,
        )
    
    def test_forward_shape(self, backbone):
        x = torch.randn(2, 1, 224, 224)
        features = backbone(x)
        assert len(features) == 4
        assert features[0].shape[0] == 2
    
    def test_feature_dims(self, backbone):
        dims = backbone.get_feature_dims()
        assert len(dims) == 4
        assert dims == [96, 192, 384, 768]


class TestSegmentationHead:
    """Tests for segmentation head."""
    
    @pytest.fixture
    def head(self):
        return EndometrialSegmentationHead(
            feature_dims=[96, 192, 384, 768],
            output_size=224,
        )
    
    def test_forward(self, head):
        features = [
            torch.randn(2, 96, 56, 56),
            torch.randn(2, 192, 28, 28),
            torch.randn(2, 384, 14, 14),
            torch.randn(2, 768, 7, 7),
        ]
        output = head(features)
        assert output.mask.shape == (2, 1, 224, 224)
        assert output.thickness_mm.shape == (2,)


class TestVascularityClassifier:
    """Tests for vascularity classifier."""
    
    @pytest.fixture
    def classifier(self):
        return VascularityClassifier(feature_dim=768, num_classes=4)
    
    def test_forward(self, classifier):
        features = torch.randn(2, 768, 7, 7)
        output = classifier(features)
        assert output.probabilities.shape == (2, 4)
        assert output.predicted_type.shape == (2,)
        assert output.confidence.shape == (2,)


class TestFibrosisDetector:
    """Tests for fibrosis detector."""
    
    @pytest.fixture
    def detector(self):
        return FibrosisDetector(
            feature_dims=[96, 192, 384, 768],
            output_size=224,
        )
    
    def test_forward(self, detector):
        features = [
            torch.randn(2, 96, 56, 56),
            torch.randn(2, 192, 28, 28),
            torch.randn(2, 384, 14, 14),
            torch.randn(2, 768, 7, 7),
        ]
        output = detector(features)
        assert output.probability_map.shape == (2, 1, 224, 224)
        assert output.severity_score.shape == (2,)


class TestUnifiedModel:
    """Tests for unified model."""
    
    @pytest.fixture
    def model(self):
        return UterusScopeModel(pretrained=False)
    
    def test_forward(self, model):
        x = torch.randn(1, 1, 224, 224)
        output = model(x)
        
        assert output.segmentation is not None
        assert output.vascularity is not None
        assert output.fibrosis is not None
        assert output.overall_confidence is not None
    
    def test_forward_video(self, model):
        frames = torch.randn(1, 8, 1, 224, 224)
        output = model.forward_video(frames)
        
        assert output.temporal is not None
        assert output.segmentation is not None
