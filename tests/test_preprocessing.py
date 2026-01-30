"""Tests for data preprocessing module."""

import numpy as np
import pytest

from uterus_scope.data.preprocessing import (
    apply_clahe,
    reduce_speckle_noise,
    preprocess_frame,
    UltrasoundPreprocessor,
)


class TestCLAHE:
    """Tests for CLAHE enhancement."""
    
    def test_grayscale_input(self):
        image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result = apply_clahe(image)
        assert result.shape == image.shape
        assert result.dtype == np.uint8
    
    def test_bgr_input(self):
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result = apply_clahe(image)
        assert result.shape == (100, 100)
        assert result.dtype == np.uint8


class TestSpeckleReduction:
    """Tests for speckle noise reduction."""
    
    def test_denoising(self):
        image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result = reduce_speckle_noise(image)
        assert result.shape == image.shape


class TestPreprocessFrame:
    """Tests for frame preprocessing."""
    
    def test_output_shape(self):
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        result = preprocess_frame(frame, target_size=(224, 224))
        assert result.shape == (224, 224)
        assert result.dtype == np.float32
    
    def test_normalization_range(self):
        frame = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result = preprocess_frame(frame)
        assert result.min() >= 0.0
        assert result.max() <= 1.0


class TestUltrasoundPreprocessor:
    """Tests for UltrasoundPreprocessor class."""
    
    @pytest.fixture
    def preprocessor(self):
        return UltrasoundPreprocessor(target_size=(224, 224), device='cpu')
    
    def test_preprocess_frame(self, preprocessor):
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        tensor = preprocessor.preprocess_frame(frame)
        assert tensor.shape == (1, 224, 224)
    
    def test_preprocess_batch(self, preprocessor):
        frames = [np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8) for _ in range(5)]
        batch = preprocessor.preprocess_batch(frames)
        assert batch.shape == (5, 1, 224, 224)
