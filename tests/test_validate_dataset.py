"""
tests/test_validate_dataset.py

Pytest tests for src/preprocessing/validate_dataset.py.
Uses only temporary directories and tiny synthetic images as fixtures.
Does NOT place any synthetic images in the project's real dataset directories.
"""

import os
import sys
import json
import pytest
import cv2
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.validate_dataset import validate_dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_tiny_image(path: str) -> None:
    """Write a 10x10 three-channel image to the given absolute path."""
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    cv2.imwrite(path, img)


def _write_meta(path: str, meta: dict) -> None:
    with open(path, "w") as f:
        json.dump(meta, f)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def two_image_dir(tmp_path):
    """
    tmp_path/
        img_a.png   (valid 10x10 RGB)
        img_b.png   (valid 10x10 RGB)
    Returns (img_a_path, img_b_path)
    """
    img_a = str(tmp_path / "img_a.png")
    img_b = str(tmp_path / "img_b.png")
    _write_tiny_image(img_a)
    _write_tiny_image(img_b)
    return img_a, img_b


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestValidDataset:
    def test_valid_metadata_with_real_images_passes(self, two_image_dir, tmp_path):
        img_a, img_b = two_image_dir
        meta = {
            "images": [
                {"image_id": "img_a", "file_path": img_a},
                {"image_id": "img_b", "file_path": img_b},
            ],
            "pairs": [
                {"source_id": "img_a", "reference_id": "img_b", "label": "positive"}
            ],
        }
        meta_path = str(tmp_path / "valid_meta.json")
        _write_meta(meta_path, meta)

        assert validate_dataset(meta_path) is True

    def test_empty_images_list_still_passes(self, tmp_path):
        """An empty dataset (no images, no pairs) contains no errors."""
        meta = {"images": [], "pairs": []}
        meta_path = str(tmp_path / "empty_meta.json")
        _write_meta(meta_path, meta)

        assert validate_dataset(meta_path) is True


class TestMissingFiles:
    def test_missing_image_file_returns_false(self, tmp_path):
        meta = {
            "images": [
                {"image_id": "ghost", "file_path": str(tmp_path / "does_not_exist.png")}
            ],
            "pairs": [],
        }
        meta_path = str(tmp_path / "missing_img.json")
        _write_meta(meta_path, meta)

        assert validate_dataset(meta_path) is False

    def test_missing_metadata_file_returns_false(self, tmp_path):
        missing_path = str(tmp_path / "nowhere.json")
        assert validate_dataset(missing_path) is False


class TestDuplicateImageIDs:
    def test_duplicate_image_ids_returns_false(self, two_image_dir, tmp_path):
        img_a, img_b = two_image_dir
        meta = {
            "images": [
                {"image_id": "same_id", "file_path": img_a},
                {"image_id": "same_id", "file_path": img_b},   # duplicate ID
            ],
            "pairs": [],
        }
        meta_path = str(tmp_path / "dup_meta.json")
        _write_meta(meta_path, meta)

        assert validate_dataset(meta_path) is False


class TestInvalidPairs:
    def test_pair_source_not_in_images_returns_false(self, two_image_dir, tmp_path):
        img_a, _ = two_image_dir
        meta = {
            "images": [
                {"image_id": "img_a", "file_path": img_a},
            ],
            "pairs": [
                {"source_id": "nonexistent_id", "reference_id": "img_a"}
            ],
        }
        meta_path = str(tmp_path / "bad_src.json")
        _write_meta(meta_path, meta)

        assert validate_dataset(meta_path) is False

    def test_pair_reference_not_in_images_returns_false(self, two_image_dir, tmp_path):
        img_a, _ = two_image_dir
        meta = {
            "images": [
                {"image_id": "img_a", "file_path": img_a},
            ],
            "pairs": [
                {"source_id": "img_a", "reference_id": "nonexistent_id"}
            ],
        }
        meta_path = str(tmp_path / "bad_ref.json")
        _write_meta(meta_path, meta)

        assert validate_dataset(meta_path) is False


class TestMalformedMetadata:
    def test_malformed_json_returns_false(self, tmp_path):
        meta_path = str(tmp_path / "malformed.json")
        with open(meta_path, "w") as f:
            f.write("{ this is not valid json ...")

        assert validate_dataset(meta_path) is False
