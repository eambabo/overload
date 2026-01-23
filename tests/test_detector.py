"""Tests for garbage can detector."""

import numpy as np
import pytest
from PIL import Image

from overload.detection.detector import GarbageCanDetector


@pytest.fixture
def detector():
    """Create a detector instance."""
    return GarbageCanDetector()


def test_detector_initialization(detector):
    """Test that detector initializes correctly."""
    assert detector.model is not None
    assert detector.confidence == 0.3
    assert len(detector.DEFAULT_CLASSES) > 0


def test_detector_custom_confidence():
    """Test detector with custom confidence threshold."""
    detector = GarbageCanDetector(confidence=0.5)
    assert detector.confidence == 0.5


def test_detect_image_returns_list(detector, tmp_path):
    """Test that detect_image returns a list of detections."""
    # Create a blank test image
    img = Image.new("RGB", (640, 480), color="white")
    img_path = tmp_path / "test.jpg"
    img.save(img_path)

    detections = detector.detect_image(img_path)

    assert isinstance(detections, list)
    # May or may not find anything in blank image, but should return list


def test_detection_format(detector, tmp_path):
    """Test that detections have the expected format."""
    # Create a test image
    img = Image.new("RGB", (640, 480), color="gray")
    img_path = tmp_path / "test.jpg"
    img.save(img_path)

    detections = detector.detect_image(img_path)

    # If any detections found, verify format
    for det in detections:
        assert "bbox" in det
        assert "confidence" in det
        assert "class_name" in det
        assert len(det["bbox"]) == 4
        assert 0 <= det["confidence"] <= 1
        assert det["class_name"] in detector.DEFAULT_CLASSES
