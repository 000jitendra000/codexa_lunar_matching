"""
tests/test_preprocessing.py

Comprehensive unit tests for Phase 3 preprocessing modules.
All test images are generated synthetically within this file.
No synthetic images are placed in data/raw/ or any project dataset directory.
"""

import os
import sys
import pytest
import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.normalize import normalize, minmax_normalize
from src.preprocessing.clahe import apply_clahe
from src.preprocessing.resize import resize_image, resize_by_scale_factor
from src.preprocessing.denoise import denoise
from src.preprocessing.pyramid import build_pyramid
from src.preprocessing.pipeline import PreprocessingPipeline, _to_grayscale
from configs.default import Config


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_gray(h=64, w=64, dtype=np.uint8):
    rng = np.random.default_rng(42)
    img = (rng.integers(50, 200, (h, w))).astype(dtype)
    return img

def make_color(h=64, w=64, dtype=np.uint8):
    rng = np.random.default_rng(42)
    return (rng.integers(50, 200, (h, w, 3))).astype(dtype)

def make_constant(h=64, w=64, value=128, dtype=np.uint8):
    return np.full((h, w), value, dtype=dtype)


# ── 1. Grayscale conversion ───────────────────────────────────────────────────

class TestGrayscale:
    def test_color_to_grayscale_reduces_dims(self):
        color = make_color()
        gray = _to_grayscale(color)
        assert gray.ndim == 2

    def test_already_gray_unchanged_shape(self):
        gray = make_gray()
        result = _to_grayscale(gray)
        assert result.shape == gray.shape
        assert result.ndim == 2

    def test_single_channel_3d_accepted(self):
        img = make_gray()[:, :, np.newaxis]  # (H, W, 1)
        result = _to_grayscale(img)
        assert result.ndim == 2

    def test_invalid_input_raises(self):
        with pytest.raises(ValueError):
            _to_grayscale(np.zeros((64, 64, 64, 64)))  # 4-D


# ── 2. Intensity normalization ────────────────────────────────────────────────

class TestNormalization:
    def test_minmax_range(self):
        img = make_gray()
        result = minmax_normalize(img)
        assert result.dtype == np.float32
        assert result.min() >= 0.0 - 1e-6
        assert result.max() <= 1.0 + 1e-6

    def test_minmax_custom_range(self):
        img = make_gray()
        result = minmax_normalize(img, out_range=(0.0, 255.0))
        assert result.min() >= 0.0 - 1e-4
        assert result.max() <= 255.0 + 1e-4

    def test_constant_image_no_divide_by_zero(self):
        img = make_constant()
        result = minmax_normalize(img)
        assert np.all(np.isfinite(result))

    def test_normalize_dispatch_minmax(self):
        img = make_gray()
        result = normalize(img, method="minmax")
        assert result.dtype == np.float32

    def test_normalize_unknown_method_raises(self):
        with pytest.raises(ValueError):
            normalize(make_gray(), method="nonexistent_method")

    def test_normalize_none_image_raises(self):
        with pytest.raises(ValueError):
            minmax_normalize(None)


# ── 3. CLAHE ──────────────────────────────────────────────────────────────────

class TestCLAHE:
    def test_clahe_returns_uint8(self):
        img = make_gray()
        result = apply_clahe(img)
        assert result.dtype == np.uint8

    def test_clahe_same_spatial_shape(self):
        img = make_gray(64, 64)
        result = apply_clahe(img)
        assert result.shape == (64, 64)

    def test_clahe_on_float32_input(self):
        img = make_gray().astype(np.float32) / 255.0
        result = apply_clahe(img)
        assert result.dtype == np.uint8

    def test_clahe_rejects_color_image(self):
        with pytest.raises(ValueError):
            apply_clahe(make_color())

    def test_clahe_clip_limit_param(self):
        img = make_gray()
        result = apply_clahe(img, clip_limit=4.0, tile_grid_size=(4, 4))
        assert result.shape == img.shape

    def test_clahe_single_channel_3d(self):
        img = make_gray()[:, :, np.newaxis]
        result = apply_clahe(img)
        assert result.ndim == 3
        assert result.shape[2] == 1


# ── 4. Resize / scale handling ────────────────────────────────────────────────

class TestResize:
    def test_resize_to_target(self):
        img = make_gray(128, 128)
        result = resize_image(img, target_size=(64, 64), keep_aspect_ratio=False)
        assert result.shape == (64, 64)

    def test_resize_keep_aspect_ratio(self):
        img = make_gray(128, 64)          # 2:1 portrait
        result = resize_image(img, target_size=(32, 32), keep_aspect_ratio=True)
        h, w = result.shape[:2]
        # Shorter axis = 32; longer axis ≤ 32
        assert max(h, w) <= 32

    def test_resize_same_size_returns_copy(self):
        img = make_gray(64, 64)
        result = resize_image(img, target_size=(64, 64))
        assert result.shape == img.shape

    def test_resize_invalid_target_raises(self):
        with pytest.raises(ValueError):
            resize_image(make_gray(), target_size=(0, 64))

    def test_scale_factor_half(self):
        img = make_gray(64, 64)
        result = resize_by_scale_factor(img, scale=0.5)
        assert result.shape == (32, 32)

    def test_scale_factor_negative_raises(self):
        with pytest.raises(ValueError):
            resize_by_scale_factor(make_gray(), scale=-1.0)


# ── 5. Denoising ──────────────────────────────────────────────────────────────

class TestDenoise:
    def test_none_returns_copy(self):
        img = make_gray()
        result = denoise(img, method="none")
        assert result.shape == img.shape
        np.testing.assert_array_equal(result, img)

    def test_gaussian_denoising(self):
        img = make_gray()
        result = denoise(img, method="gaussian", ksize=3)
        assert result.shape == img.shape

    def test_median_denoising(self):
        img = make_gray()
        result = denoise(img, method="median", ksize=3)
        assert result.shape == img.shape

    def test_bilateral_denoising(self):
        img = make_gray()
        result = denoise(img, method="bilateral", d=5, sigma_color=50, sigma_space=50)
        assert result.shape == img.shape

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            denoise(make_gray(), method="magic_filter")

    def test_none_image_raises(self):
        with pytest.raises(ValueError):
            denoise(None, method="gaussian")


# ── 6. Multi-scale pyramid ────────────────────────────────────────────────────

class TestPyramid:
    def test_level_0_matches_input_shape(self):
        img = make_gray(64, 64)
        pyr = build_pyramid(img, num_levels=3)
        assert pyr[0].shape == img.shape

    def test_pyramid_decreasing_sizes(self):
        img = make_gray(128, 128)
        pyr = build_pyramid(img, num_levels=4)
        for i in range(1, len(pyr)):
            assert pyr[i].shape[0] < pyr[i - 1].shape[0]

    def test_pyramid_stops_before_min_dim(self):
        # 20x20 image should not produce many levels before hitting MIN_DIM=16
        img = make_gray(20, 20)
        pyr = build_pyramid(img, num_levels=10)
        assert len(pyr) < 10

    def test_pyramid_num_levels_1(self):
        img = make_gray(64, 64)
        pyr = build_pyramid(img, num_levels=1)
        assert len(pyr) == 1

    def test_pyramid_invalid_scale_factor_raises(self):
        with pytest.raises(ValueError):
            build_pyramid(make_gray(), scale_factor=1.5)

    def test_pyramid_invalid_num_levels_raises(self):
        with pytest.raises(ValueError):
            build_pyramid(make_gray(), num_levels=0)


# ── 7. Full pipeline ──────────────────────────────────────────────────────────

class TestPipeline:
    def _default_cfg(self, **overrides):
        cfg = dict(Config.PREPROCESSING)
        cfg.update(overrides)
        return cfg

    def test_pipeline_runs_on_color_image(self):
        pipeline = PreprocessingPipeline(self._default_cfg())
        result = pipeline.process(make_color())
        assert "image" in result
        assert "pyramid" in result
        assert "applied_ops" in result

    def test_pipeline_runs_on_gray_image(self):
        pipeline = PreprocessingPipeline(self._default_cfg())
        result = pipeline.process(make_gray())
        assert result["image"] is not None

    def test_applied_ops_recorded(self):
        pipeline = PreprocessingPipeline(self._default_cfg())
        result = pipeline.process(make_color())
        ops = result["applied_ops"]
        assert "grayscale" in ops
        assert any("normalize" in op for op in ops)
        assert "clahe" in ops

    def test_pyramid_in_result(self):
        cfg = self._default_cfg(pyramid_enabled=True, pyramid_num_levels=3)
        pipeline = PreprocessingPipeline(cfg)
        result = pipeline.process(make_gray(64, 64))
        assert len(result["pyramid"]) >= 1

    def test_pipeline_with_denoise(self):
        cfg = self._default_cfg(denoise_method="gaussian", denoise_kwargs={"ksize": 3})
        pipeline = PreprocessingPipeline(cfg)
        result = pipeline.process(make_gray())
        assert any("denoise" in op for op in result["applied_ops"])

    def test_pipeline_with_resize(self):
        cfg = self._default_cfg(resize_target_size=[32, 32], resize_keep_aspect_ratio=False)
        pipeline = PreprocessingPipeline(cfg)
        result = pipeline.process(make_gray(64, 64))
        assert result["processed_shape"][0] == 32

    def test_original_shape_preserved(self):
        img = make_color(80, 80)
        pipeline = PreprocessingPipeline(self._default_cfg())
        result = pipeline.process(img)
        assert result["original_shape"] == img.shape

    def test_none_image_raises(self):
        pipeline = PreprocessingPipeline(self._default_cfg())
        with pytest.raises(ValueError):
            pipeline.process(None)

    def test_grayscale_disabled(self):
        cfg = self._default_cfg(grayscale=False, clahe_enabled=False, pyramid_enabled=False,
                                 normalization_method="none")
        pipeline = PreprocessingPipeline(cfg)
        result = pipeline.process(make_color())
        assert result["image"].ndim == 3   # still colour

    def test_constant_image_no_crash(self):
        pipeline = PreprocessingPipeline(self._default_cfg())
        result = pipeline.process(make_constant())
        assert result["image"] is not None
