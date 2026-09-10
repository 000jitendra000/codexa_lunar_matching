"""
src/evaluation/difficult_cases.py

Comprehensive difficult-case stress suite for Milestone C (Phase 19).

Constructs 10 controlled stress conditions targeting specific failure modes:
- Case A: Resolution mismatch (cross-sensor GSD simulation)
- Case B: Illumination / contrast mismatch (solar elevation / incidence differences)
- Case C: Large rotation (in-plane orientation ambiguity)
- Case D: Significant scale mismatch (scale factor divergence)
- Case E: Partial overlap (limited spatial intersection)
- Case F: Distractor structures (non-corresponding geometric features)
- Case G: Sparse crater scene (insufficient geometric anchors)
- Case H: Dense crater scene (repetitive geometric ambiguity)
- Case I: Texture-poor plain (low gradients / flat lunar mare)
- Case J: Severe noise & optical blur (sensor noise / transmission artifacts)

ARCHITECTURAL RULES:
- Strictly model-side Python; deterministic with configurable seeds.
- Does NOT force all cases to succeed: realistic stress testing produces meaningful diagnostic failures.
- Each case encapsulates expected behavior and ground-truth metadata.
"""

import math
from typing import Dict, Any, List
import cv2
import numpy as np

from src.matching.transformation import SimilarityTransform2D
from src.evaluation.evaluation_types import EvaluationCase
from src.evaluation.synthetic_benchmark import SyntheticBenchmark


def build_case_resolution_mismatch(size: int = 512, seed: int = 42) -> EvaluationCase:
    """Case A: Resolution mismatch (Image B downsampled 2.5x to simulate GSD difference)."""
    img_a, _ = SyntheticBenchmark.generate_base_terrain(size=size, seed=seed)
    rad = math.radians(12.0)
    t_gt = SimilarityTransform2D(scale=1.0, rotation_rad=rad, translation_x=20.0, translation_y=15.0)
    img_b_clean = SyntheticBenchmark.warp_image_forward(img_a, t_gt)
    img_b = SyntheticBenchmark.apply_perturbations(img_b_clean, downsample_factor=2.5, seed=seed)

    return EvaluationCase(
        case_name="Resolution Mismatch",
        image_a=img_a,
        image_b=img_b,
        ground_truth=t_gt,
        parameters={"downsample_factor": 2.5, "scale": 1.0, "rotation_deg": 12.0, "translation": (20.0, 15.0)},
        expected_behavior="Learned / crater features should maintain matchability despite spatial blur and loss of high frequencies.",
    )


def build_case_illumination_mismatch(size: int = 512, seed: int = 42) -> EvaluationCase:
    """Case B: Severe illumination and contrast mismatch (simulates low solar elevation)."""
    img_a, _ = SyntheticBenchmark.generate_base_terrain(size=size, seed=seed)
    rad = math.radians(15.0)
    t_gt = SimilarityTransform2D(scale=1.1, rotation_rad=rad, translation_x=25.0, translation_y=20.0)
    img_b_clean = SyntheticBenchmark.warp_image_forward(img_a, t_gt)
    # Severe contrast attenuation + brightness boost + non-linear gamma
    img_b = SyntheticBenchmark.apply_perturbations(
        img_b_clean,
        contrast_scale=0.6,
        brightness_shift=35.0,
        gamma=1.5,
        seed=seed,
    )

    return EvaluationCase(
        case_name="Illumination Mismatch",
        image_a=img_a,
        image_b=img_b,
        ground_truth=t_gt,
        parameters={"contrast_scale": 0.6, "brightness_shift": 35.0, "gamma": 1.5, "rotation_deg": 15.0},
        expected_behavior="Phase 3 CLAHE preprocessing should normalize intensity distributions to preserve matchability.",
    )


def build_case_large_rotation(size: int = 512, seed: int = 42) -> EvaluationCase:
    """Case C: Large rotation (90 degrees)."""
    img_a, _ = SyntheticBenchmark.generate_base_terrain(size=size, seed=seed)
    rad = math.radians(90.0)
    t_gt = SimilarityTransform2D(scale=1.0, rotation_rad=rad, translation_x=30.0, translation_y=-20.0)
    img_b = SyntheticBenchmark.warp_image_forward(img_a, t_gt)

    return EvaluationCase(
        case_name="Large Rotation",
        image_a=img_a,
        image_b=img_b,
        ground_truth=t_gt,
        parameters={"scale": 1.0, "rotation_deg": 90.0, "translation": (30.0, -20.0)},
        expected_behavior="Crater constellation invariant descriptors provide rotation-invariant anchors.",
    )


def build_case_scale_mismatch(size: int = 512, seed: int = 42) -> EvaluationCase:
    """Case D: Substantial scale mismatch (1.65x scale change)."""
    img_a, _ = SyntheticBenchmark.generate_base_terrain(size=size, seed=seed)
    rad = math.radians(10.0)
    t_gt = SimilarityTransform2D(scale=1.65, rotation_rad=rad, translation_x=35.0, translation_y=25.0)
    img_b = SyntheticBenchmark.warp_image_forward(img_a, t_gt)

    return EvaluationCase(
        case_name="Scale Mismatch",
        image_a=img_a,
        image_b=img_b,
        ground_truth=t_gt,
        parameters={"scale": 1.65, "rotation_deg": 10.0, "translation": (35.0, 25.0)},
        expected_behavior="Scale-invariant crater triangle descriptors enable recovery across significant zoom difference.",
    )


def build_case_partial_overlap(size: int = 512, seed: int = 42) -> EvaluationCase:
    """Case E: Limited spatial overlap (~35% overlapping area)."""
    img_a, _ = SyntheticBenchmark.generate_base_terrain(size=size, seed=seed)
    rad = math.radians(8.0)
    # Substantial translation pushes most of Image A out of frame
    t_gt = SimilarityTransform2D(scale=1.05, rotation_rad=rad, translation_x=180.0, translation_y=160.0)
    img_b = SyntheticBenchmark.warp_image_forward(img_a, t_gt)

    return EvaluationCase(
        case_name="Partial Overlap",
        image_a=img_a,
        image_b=img_b,
        ground_truth=t_gt,
        parameters={"scale": 1.05, "rotation_deg": 8.0, "translation": (180.0, 160.0)},
        expected_behavior="Inlier selection isolates the shared overlapping sub-region; registration mask bounds valid overlap.",
    )


def build_case_distractors(size: int = 512, seed: int = 42) -> EvaluationCase:
    """Case F: Distractor features (spurious crater-like structures inserted into Image B)."""
    img_a, _ = SyntheticBenchmark.generate_base_terrain(size=size, seed=seed)
    rad = math.radians(15.0)
    t_gt = SimilarityTransform2D(scale=1.1, rotation_rad=rad, translation_x=30.0, translation_y=20.0)
    img_b = SyntheticBenchmark.warp_image_forward(img_a, t_gt)

    # Inject 4 prominent distractor craters into Image B that do NOT correspond to Image A
    distractors = [
        (80, 80, 26, 210.0, 30.0),
        (440, 80, 30, 215.0, 25.0),
        (80, 440, 28, 205.0, 35.0),
        (250, 450, 32, 220.0, 20.0),
    ]
    for cx, cy, r, rim, flr in distractors:
        cv2.circle(img_b, (cx, cy), r, float(rim), -1)
        cv2.circle(img_b, (cx, cy), int(r * 0.6), float(flr), -1)

    return EvaluationCase(
        case_name="Distractors",
        image_a=img_a,
        image_b=img_b,
        ground_truth=t_gt,
        parameters={"num_distractors": 4, "scale": 1.1, "rotation_deg": 15.0, "translation": (30.0, 20.0)},
        expected_behavior="RANSAC geometric verification must reject distractor-induced false correspondences.",
    )


def build_case_sparse_craters(size: int = 512, seed: int = 42) -> EvaluationCase:
    """Case G: Sparse crater scene (only 2 craters present)."""
    img_a, _ = SyntheticBenchmark.generate_base_terrain(size=size, seed=seed, density_mode="sparse")
    rad = math.radians(10.0)
    t_gt = SimilarityTransform2D(scale=1.05, rotation_rad=rad, translation_x=25.0, translation_y=20.0)
    img_b = SyntheticBenchmark.warp_image_forward(img_a, t_gt)

    return EvaluationCase(
        case_name="Sparse Craters",
        image_a=img_a,
        image_b=img_b,
        ground_truth=t_gt,
        parameters={"density_mode": "sparse", "scale": 1.05, "rotation_deg": 10.0, "translation": (25.0, 20.0)},
        expected_behavior="With few crater anchors, hybrid fusion relies heavily on learned/dense features.",
    )


def build_case_dense_craters(size: int = 512, seed: int = 42) -> EvaluationCase:
    """Case H: Dense repetitive crater field (35+ clustered craters of similar size)."""
    img_a, _ = SyntheticBenchmark.generate_base_terrain(size=size, seed=seed, density_mode="dense")
    rad = math.radians(8.0)
    t_gt = SimilarityTransform2D(scale=1.0, rotation_rad=rad, translation_x=20.0, translation_y=15.0)
    img_b = SyntheticBenchmark.warp_image_forward(img_a, t_gt)

    return EvaluationCase(
        case_name="Dense Craters",
        image_a=img_a,
        image_b=img_b,
        ground_truth=t_gt,
        parameters={"density_mode": "dense", "scale": 1.0, "rotation_deg": 8.0, "translation": (20.0, 15.0)},
        expected_behavior="Tests descriptor discriminability and mutual consistency against repetitive geometric ambiguity.",
    )


def build_case_texture_poor(size: int = 512, seed: int = 42) -> EvaluationCase:
    """Case I: Texture-poor lunar mare plain (flat intensity, subtle low-contrast craters)."""
    img_a, _ = SyntheticBenchmark.generate_base_terrain(size=size, seed=seed, density_mode="texture_poor")
    rad = math.radians(10.0)
    t_gt = SimilarityTransform2D(scale=1.0, rotation_rad=rad, translation_x=20.0, translation_y=15.0)
    img_b = SyntheticBenchmark.warp_image_forward(img_a, t_gt)

    return EvaluationCase(
        case_name="Texture Poor",
        image_a=img_a,
        image_b=img_b,
        ground_truth=t_gt,
        parameters={"density_mode": "texture_poor", "scale": 1.0, "rotation_deg": 10.0, "translation": (20.0, 15.0)},
        expected_behavior="Evaluates robustness when both gradient magnitude and crater edge definition are low.",
    )


def build_case_noise_and_blur(size: int = 512, seed: int = 42) -> EvaluationCase:
    """Case J: Sensor noise and optical blur (sigma=15.0, blur k=5)."""
    img_a, _ = SyntheticBenchmark.generate_base_terrain(size=size, seed=seed)
    rad = math.radians(12.0)
    t_gt = SimilarityTransform2D(scale=1.05, rotation_rad=rad, translation_x=25.0, translation_y=20.0)
    img_b_clean = SyntheticBenchmark.warp_image_forward(img_a, t_gt)
    img_b = SyntheticBenchmark.apply_perturbations(img_b_clean, noise_sigma=15.0, blur_ksize=5, seed=seed)

    return EvaluationCase(
        case_name="Noise and Blur",
        image_a=img_a,
        image_b=img_b,
        ground_truth=t_gt,
        parameters={"noise_sigma": 15.0, "blur_ksize": 5, "scale": 1.05, "rotation_deg": 12.0, "translation": (25.0, 20.0)},
        expected_behavior="Tests pipeline degradation under high sensor noise and defocused camera optics.",
    )


def build_difficult_cases_suite(size: int = 512, seed: int = 42) -> List[EvaluationCase]:
    """
    Construct the full 10-case difficult stress suite.

    Returns:
        Ordered list of 10 EvaluationCase instances.
    """
    return [
        build_case_resolution_mismatch(size=size, seed=seed),
        build_case_illumination_mismatch(size=size, seed=seed),
        build_case_large_rotation(size=size, seed=seed),
        build_case_scale_mismatch(size=size, seed=seed),
        build_case_partial_overlap(size=size, seed=seed),
        build_case_distractors(size=size, seed=seed),
        build_case_sparse_craters(size=size, seed=seed),
        build_case_dense_craters(size=size, seed=seed),
        build_case_texture_poor(size=size, seed=seed),
        build_case_noise_and_blur(size=size, seed=seed),
    ]
