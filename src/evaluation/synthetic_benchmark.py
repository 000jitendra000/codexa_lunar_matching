"""
src/evaluation/synthetic_benchmark.py

Synthetic benchmark generator for controlled location matching and registration
experiments in Milestone C (Phases 18 + 19).

ARCHITECTURAL RULES:
- Deterministic synthetic lunar terrain generation (low-frequency base + crater circular structures + sensor noise).
- Generates known ground-truth SimilarityTransform2D instances.
- Applies mathematically correct forward warping via inverse pull-warp in OpenCV.
- Provides controlled perturbations (noise, blur, contrast, brightness, gamma, resolution mismatch).
- Explicitly designated as 'synthetic registration benchmark'; does NOT masquerade as real lunar imagery.
"""

import math
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np

from src.matching.transformation import SimilarityTransform2D
from src.evaluation.evaluation_types import EvaluationCase


class SyntheticBenchmark:
    """
    Generator for controlled synthetic image pairs with known similarity ground truth.
    """

    @staticmethod
    def generate_base_terrain(
        size: int = 512,
        seed: int = 42,
        density_mode: str = "normal",
    ) -> Tuple[np.ndarray, List[Tuple[float, float, float]]]:
        """
        Generate deterministic lunar-like synthetic terrain.

        Args:
            size: Image height and width in pixels.
            seed: PRNG seed for reproducibility.
            density_mode: 'normal' (12 craters), 'sparse' (2 craters),
                          'dense' (35 craters), or 'texture_poor' (flat plain, subtle craters).

        Returns:
            Tuple of (image: np.ndarray [uint8], crater_centers: List of (x, y, radius)).
        """
        rng = np.random.default_rng(seed)

        # 1. Low-frequency background variation simulating lunar mare/highlands
        x = np.linspace(0, 4 * np.pi, size)
        y = np.linspace(0, 4 * np.pi, size)
        xx, yy = np.meshgrid(x, y)

        if density_mode == "texture_poor":
            # Very flat, low-contrast mare plain
            bg = 120.0 + 5.0 * np.sin(xx) * np.cos(yy)
        else:
            bg = 110.0 + 25.0 * np.sin(xx) * np.cos(yy)

        img = bg.astype(np.float32)

        # 2. Crater definitions (cx, cy, radius, rim_val, floor_val)
        craters: List[Tuple[int, int, int, float, float]] = []

        if density_mode == "sparse":
            # Only 2 craters
            craters = [
                (int(size * 0.35), int(size * 0.40), int(size * 0.08), 195.0, 40.0),
                (int(size * 0.65), int(size * 0.60), int(size * 0.07), 190.0, 45.0),
            ]
        elif density_mode == "dense":
            # 35 craters tightly clustered with similar sizes
            grid_steps = np.linspace(int(size * 0.15), int(size * 0.85), 6)
            for gx in grid_steps:
                for gy in grid_steps:
                    jitter_x = rng.uniform(-10.0, 10.0)
                    jitter_y = rng.uniform(-10.0, 10.0)
                    rad = int(rng.uniform(14.0, 22.0))
                    craters.append((int(gx + jitter_x), int(gy + jitter_y), rad, 200.0, 40.0))
        elif density_mode == "texture_poor":
            # 5 subtle craters with low rim-to-floor contrast
            craters = [
                (140, 160, 25, 135.0, 105.0),
                (360, 180, 30, 138.0, 102.0),
                (250, 280, 35, 140.0, 100.0),
                (160, 390, 22, 136.0, 104.0),
                (380, 370, 28, 137.0, 103.0),
            ]
        else:
            # Standard 'normal' 12 crater constellation
            scale_ratio = size / 512.0
            base_coords = [
                (120, 140, 28, 190.0, 40.0),
                (220, 100, 20, 180.0, 50.0),
                (350, 160, 35, 200.0, 35.0),
                (160, 260, 24, 185.0, 45.0),
                (280, 240, 40, 210.0, 30.0),
                (400, 280, 22, 175.0, 55.0),
                (100, 380, 30, 195.0, 40.0),
                (240, 380, 26, 180.0, 50.0),
                (360, 400, 32, 205.0, 35.0),
                (450, 120, 18, 170.0, 60.0),
                (80, 200, 22, 185.0, 45.0),
                (300, 450, 25, 190.0, 40.0),
            ]
            craters = [
                (int(cx * scale_ratio), int(cy * scale_ratio), int(r * scale_ratio), rim, flr)
                for cx, cy, r, rim, flr in base_coords
            ]

        crater_centers: List[Tuple[float, float, float]] = []
        for cx, cy, rad, rim_val, floor_val in craters:
            # Outer rim ring
            cv2.circle(img, (cx, cy), rad, float(rim_val), -1)
            # Inner shadowed floor
            cv2.circle(img, (cx, cy), max(int(rad * 0.65), 1), float(floor_val), -1)
            # Northwest illumination specular highlight
            hl_rad = max(int(rad * 0.4), 1)
            hl_x = cx - int(rad * 0.2)
            hl_y = cy - int(rad * 0.2)
            cv2.circle(img, (hl_x, hl_y), hl_rad, float(min(rim_val + 20.0, 255.0)), -1)
            crater_centers.append((float(cx), float(cy), float(rad)))

        # 3. Micro-texture / fine surface roughness
        texture_sigma = 1.0 if density_mode == "texture_poor" else 3.5
        noise = rng.normal(0, texture_sigma, (size, size)).astype(np.float32)
        out = np.clip(img + noise, 0.0, 255.0).astype(np.uint8)

        return out, crater_centers

    @staticmethod
    def warp_image_forward(
        image: np.ndarray,
        transform: SimilarityTransform2D,
        output_shape: Optional[Tuple[int, int]] = None,
        border_value: float = 0.0,
    ) -> np.ndarray:
        """
        Synthesize Image B by applying forward transformation p_B = s * R * p_A + t to Image A.

        In OpenCV, cv2.warpAffine(src, M, dsize) samples each output pixel p_B from
        source coordinate M * [p_B, 1]^T. Therefore, M is the inverse transformation
        matrix T^(-1) = T_(B -> A).

        Args:
            image: Source Image A [uint8 or float].
            transform: Forward SimilarityTransform2D mapping A -> B.
            output_shape: (height, width) of output Image B. Defaults to input image shape.
            border_value: Intensity for pixels outside Image A's boundaries.

        Returns:
            Transformed Image B.
        """
        if output_shape is None:
            h, w = image.shape[:2]
        else:
            h, w = output_shape

        # Warping Image A into Image B applies the forward transform A -> B
        m_affine = transform.to_matrix()[:2, :]

        warped = cv2.warpAffine(
            image,
            m_affine,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=float(border_value),
        )
        return warped

    @staticmethod
    def apply_perturbations(
        image: np.ndarray,
        noise_sigma: float = 0.0,
        contrast_scale: float = 1.0,
        brightness_shift: float = 0.0,
        gamma: float = 1.0,
        blur_ksize: int = 0,
        downsample_factor: float = 1.0,
        seed: int = 42,
    ) -> np.ndarray:
        """
        Apply controlled physical/sensor perturbations to an image.

        Args:
            image: Input image uint8.
            noise_sigma: Std dev of zero-mean Gaussian additive noise.
            contrast_scale: Multiplicative contrast factor.
            brightness_shift: Additive brightness offset.
            gamma: Non-linear gamma factor (x^gamma).
            blur_ksize: Gaussian blur kernel size (must be positive odd, 0 = disabled).
            downsample_factor: Downsample factor > 1.0 (resampled back to original size).
            seed: PRNG seed for reproducible noise.

        Returns:
            Perturbed image uint8.
        """
        img = image.copy()
        h, w = img.shape[:2]

        # 1. Resolution degradation (downsample and upsample back)
        if downsample_factor > 1.0:
            dw = max(int(w / downsample_factor), 16)
            dh = max(int(h / downsample_factor), 16)
            small = cv2.resize(img, (dw, dh), interpolation=cv2.INTER_AREA)
            img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

        # 2. Optical blur
        if blur_ksize > 0:
            k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
            img = cv2.GaussianBlur(img, (k, k), 0)

        # 3. Photometric / contrast & brightness changes
        img_f = img.astype(np.float32)
        if contrast_scale != 1.0 or brightness_shift != 0.0:
            img_f = img_f * contrast_scale + brightness_shift

        # 4. Gamma correction
        if gamma != 1.0 and gamma > 0.0:
            norm = np.clip(img_f / 255.0, 0.0, 1.0)
            img_f = (norm ** gamma) * 255.0

        # 5. Additive sensor noise
        if noise_sigma > 0.0:
            rng = np.random.default_rng(seed)
            noise = rng.normal(0.0, noise_sigma, img_f.shape).astype(np.float32)
            img_f = img_f + noise

        return np.clip(img_f, 0.0, 255.0).astype(np.uint8)

    # ──────────────────────────────────────────────────────────────────────────
    # Canonical Benchmark Generators
    # ──────────────────────────────────────────────────────────────────────────

    @classmethod
    def generate_identity_case(cls, size: int = 512, seed: int = 42) -> EvaluationCase:
        """Benchmark Case: Identity transformation (s=1.0, theta=0, t=(0, 0))."""
        img_a, _ = cls.generate_base_terrain(size=size, seed=seed)
        t_gt = SimilarityTransform2D(scale=1.0, rotation_rad=0.0, translation_x=0.0, translation_y=0.0)
        img_b = img_a.copy()
        return EvaluationCase(
            case_name="Identity",
            image_a=img_a,
            image_b=img_b,
            ground_truth=t_gt,
            parameters={"scale": 1.0, "rotation_deg": 0.0, "translation": (0.0, 0.0)},
            expected_behavior="Zero error, perfect identity registration.",
        )

    @classmethod
    def generate_translation_case(
        cls, tx: float = 45.0, ty: float = -30.0, size: int = 512, seed: int = 42
    ) -> EvaluationCase:
        """Benchmark Case: Pure translation."""
        img_a, _ = cls.generate_base_terrain(size=size, seed=seed)
        t_gt = SimilarityTransform2D(scale=1.0, rotation_rad=0.0, translation_x=tx, translation_y=ty)
        img_b = cls.warp_image_forward(img_a, t_gt)
        return EvaluationCase(
            case_name="Pure Translation",
            image_a=img_a,
            image_b=img_b,
            ground_truth=t_gt,
            parameters={"scale": 1.0, "rotation_deg": 0.0, "translation": (tx, ty)},
            expected_behavior="Accurate recovery of horizontal and vertical shift.",
        )

    @classmethod
    def generate_rotation_case(
        cls, angle_deg: float = 25.0, size: int = 512, seed: int = 42
    ) -> EvaluationCase:
        """Benchmark Case: Pure in-plane rotation."""
        img_a, _ = cls.generate_base_terrain(size=size, seed=seed)
        rad = math.radians(angle_deg)
        t_gt = SimilarityTransform2D(scale=1.0, rotation_rad=rad, translation_x=0.0, translation_y=0.0)
        img_b = cls.warp_image_forward(img_a, t_gt)
        return EvaluationCase(
            case_name="Pure Rotation",
            image_a=img_a,
            image_b=img_b,
            ground_truth=t_gt,
            parameters={"scale": 1.0, "rotation_deg": angle_deg, "translation": (0.0, 0.0)},
            expected_behavior="Invariant descriptors and learned matcher recover rotation accurately.",
        )

    @classmethod
    def generate_scale_case(
        cls, scale: float = 1.25, size: int = 512, seed: int = 42
    ) -> EvaluationCase:
        """Benchmark Case: Pure uniform scale change."""
        img_a, _ = cls.generate_base_terrain(size=size, seed=seed)
        t_gt = SimilarityTransform2D(scale=scale, rotation_rad=0.0, translation_x=0.0, translation_y=0.0)
        img_b = cls.warp_image_forward(img_a, t_gt)
        return EvaluationCase(
            case_name="Pure Scale",
            image_a=img_a,
            image_b=img_b,
            ground_truth=t_gt,
            parameters={"scale": scale, "rotation_deg": 0.0, "translation": (0.0, 0.0)},
            expected_behavior="Scale-invariant descriptors resolve scale difference.",
        )

    @classmethod
    def generate_combined_case(
        cls,
        scale: float = 1.15,
        angle_deg: float = 20.0,
        tx: float = 40.0,
        ty: float = 30.0,
        size: int = 512,
        seed: int = 42,
    ) -> EvaluationCase:
        """Benchmark Case: Full combined similarity transformation."""
        img_a, _ = cls.generate_base_terrain(size=size, seed=seed)
        rad = math.radians(angle_deg)
        t_gt = SimilarityTransform2D(scale=scale, rotation_rad=rad, translation_x=tx, translation_y=ty)
        img_b = cls.warp_image_forward(img_a, t_gt)
        return EvaluationCase(
            case_name="Combined Similarity",
            image_a=img_a,
            image_b=img_b,
            ground_truth=t_gt,
            parameters={"scale": scale, "rotation_deg": angle_deg, "translation": (tx, ty)},
            expected_behavior="Complete recovery of scale, rotation, and translation.",
        )
