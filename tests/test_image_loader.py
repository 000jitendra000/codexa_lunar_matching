import os
import cv2
import numpy as np
import pytest
from src.preprocessing.image_loader import ImageLoader
from src.preprocessing.image_pair import ImagePair

# Fixtures for tiny dummy images in tests
@pytest.fixture
def dummy_image_path(tmp_path):
    # Create a tiny 10x10 dummy RGB image
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img_path = tmp_path / "dummy.png"
    cv2.imwrite(str(img_path), img)
    return str(img_path)

def test_valid_image_loading(dummy_image_path):
    info = ImageLoader.load_image(dummy_image_path)
    assert info["width"] == 10
    assert info["height"] == 10
    assert info["channels"] == 3
    assert info["image"] is not None

def test_missing_image_path():
    with pytest.raises(FileNotFoundError):
        ImageLoader.load_image("non_existent_file_absolutely_missing.png")
        
def test_image_pair_creation(dummy_image_path):
    info1 = ImageLoader.load_image(dummy_image_path)
    info2 = ImageLoader.load_image(dummy_image_path)
    
    pair = ImagePair(info1, info2, label="positive")
    assert pair.label == "positive"
    assert pair.source_image is not None
    assert pair.reference_image is not None
    assert "10x10x3" in pair.summary()

def test_metadata_extraction(dummy_image_path):
    info = ImageLoader.load_image(dummy_image_path)
    # Testing that loader properly abstracts dict format
    assert "width" in info
    assert "height" in info
    assert "channels" in info
    assert type(info["image"]) == np.ndarray
