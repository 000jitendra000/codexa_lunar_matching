"""
src/preprocessing/pipeline.py

High-level preprocessing pipeline for lunar imagery.

Usage
-----
    from configs.default import Config
    from src.preprocessing.pipeline import PreprocessingPipeline
    from src.preprocessing.image_loader import ImageLoader

    loader_result = ImageLoader.load_image("path/to/image.png")
    image = loader_result["image"]

    pipeline = PreprocessingPipeline(Config.PREPROCESSING)
    result = pipeline.process(image)

    # result keys:
    #   "image"            — final processed image (numpy array)
    #   "pyramid"          — list of numpy arrays (level 0 = finest)
    #   "original_shape"   — (H, W, C) or (H, W) of the raw input
    #   "processed_shape"  — shape after all spatial transforms
    #   "applied_ops"      — ordered list of operation names that ran
    #   "scale_info"       — dict with resize metadata
"""

import cv2
import numpy as np
import logging
from dataclasses import dataclass, field

from src.preprocessing.normalize import normalize
from src.preprocessing.clahe import apply_clahe
from src.preprocessing.resize import resize_image
from src.preprocessing.denoise import denoise
from src.preprocessing.pyramid import build_pyramid

logger = logging.getLogger(__name__)


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert to 2-D grayscale. Already-grayscale images are left untouched."""
    if image.ndim == 2:
        return image.copy()
    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0].copy()
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Unsupported image shape for grayscale conversion: {image.shape}")


class PreprocessingPipeline:
    """
    Executes a configurable sequence of preprocessing operations.

    The pipeline reads its parameters from a config dict (typically
    Config.PREPROCESSING).  Each stage can be enabled or disabled
    independently via that config.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Dict of preprocessing parameters.
                    See Config.PREPROCESSING in configs/default.py for defaults.
        """
        self.config = config

    def process(self, image: np.ndarray) -> dict:
        """
        Run the full preprocessing pipeline on a single image array.

        Args:
            image: Raw numpy array from ImageLoader. NOT modified.

        Returns:
            dict with keys: image, pyramid, original_shape,
                            processed_shape, applied_ops, scale_info.
        """
        if image is None:
            raise ValueError("image must not be None.")

        cfg = self.config
        original_shape = image.shape
        applied_ops = []
        scale_info = {}

        img = image  # working copy; each step returns a NEW array

        # ── 1. Grayscale ──────────────────────────────────────────────────
        if cfg.get("grayscale", True):
            img = _to_grayscale(img)
            applied_ops.append("grayscale")
            logger.debug("Applied grayscale → shape %s", img.shape)

        # ── 2. Normalization ──────────────────────────────────────────────
        norm_method = cfg.get("normalization_method", "minmax")
        if norm_method and norm_method != "none":
            img = normalize(img, method=norm_method)
            applied_ops.append(f"normalize:{norm_method}")
            logger.debug("Applied normalization (%s)", norm_method)

        # ── 3. CLAHE ──────────────────────────────────────────────────────
        if cfg.get("clahe_enabled", True):
            img = apply_clahe(
                img,
                clip_limit=cfg.get("clahe_clip_limit", 2.0),
                tile_grid_size=tuple(cfg.get("clahe_tile_grid_size", [8, 8])),
            )
            applied_ops.append("clahe")
            logger.debug("Applied CLAHE")

        # ── 4. Denoising ──────────────────────────────────────────────────
        denoise_method = cfg.get("denoise_method", "none")
        if denoise_method and denoise_method != "none":
            denoise_kwargs = cfg.get("denoise_kwargs", {})
            img = denoise(img, method=denoise_method, **denoise_kwargs)
            applied_ops.append(f"denoise:{denoise_method}")
            logger.debug("Applied denoising (%s)", denoise_method)

        # ── 5. Resize ─────────────────────────────────────────────────────
        target_size = cfg.get("resize_target_size")  # (width, height) or None
        if target_size:
            orig_hw = img.shape[:2]
            img = resize_image(
                img,
                target_size=tuple(target_size),
                keep_aspect_ratio=cfg.get("resize_keep_aspect_ratio", True),
            )
            new_hw = img.shape[:2]
            scale_info = {
                "original_pixel_hw": orig_hw,
                "resized_pixel_hw": new_hw,
                "target_size": target_size,
                "keep_aspect_ratio": cfg.get("resize_keep_aspect_ratio", True),
                "note": (
                    "Pixel resize only. Physical GSD normalization is a "
                    "separate concern handled in a later phase."
                ),
            }
            applied_ops.append(f"resize:{target_size}")
            logger.debug("Applied resize → %s", img.shape)

        processed_shape = img.shape

        # ── 6. Multi-scale pyramid ────────────────────────────────────────
        pyramid = []
        if cfg.get("pyramid_enabled", True):
            pyramid = build_pyramid(
                img,
                num_levels=cfg.get("pyramid_num_levels", 4),
                scale_factor=cfg.get("pyramid_scale_factor", 0.5),
            )
            applied_ops.append(
                f"pyramid:{len(pyramid)} levels"
            )
            logger.debug("Built %d-level pyramid", len(pyramid))

        return {
            "image": img,
            "pyramid": pyramid,
            "original_shape": original_shape,
            "processed_shape": processed_shape,
            "applied_ops": applied_ops,
            "scale_info": scale_info,
        }
