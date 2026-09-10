"""
experiments/inspect_registration.py

Demonstration and verification of Milestone B: Registration & Quality Engine (Phases 14-17).
Evaluates the downstream pipeline consuming geometrically verified correspondences from Milestone A:
1. Inlier extraction
2. Uniform spatial tie-point selection (regular grid)
3. Optional sub-pixel / local refinement
4. Final transformation refit (Phase 9 Umeyama)
5. Image registration / pull-based backward warping
6. Quantitative quality evaluation (RMSE, coverage, bounded confidence, classification)

Evaluates three synthetic scenarios:
- Scenario A (Clean Inliers): Evaluates baseline registration and quality metrics.
- Scenario B (Outliers + Clustering): Evaluates RANSAC outlier rejection followed by uniform tie-point selection.
- Scenario C (Noisy / Sub-pixel): Evaluates sub-pixel template refinement on fractional-pixel perturbations.

Generates a 5-panel visualization saved to:
data/processed/registration/synthetic_registration_demo.png
- Panel 1: Reference Image A with tie points
- Panel 2: Transformed Query Image B with corresponding points
- Panel 3: Registered Image B (aligned into Image A's frame)
- Panel 4: Overlap Difference Map (|Image A - Registered B|)
- Panel 5: Valid Overlap Mask

NOTE: Strictly software verification on synthetic data.
Real OHRC/LROC imagery is not yet available.
Do NOT claim real lunar matching or scientific registration accuracy.
"""

import os
import sys
import math
import logging
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.matching.transformation import SimilarityTransform2D, estimate_similarity_transform
from src.matching.correspondence_fusion import Correspondence
from src.matching.hybrid_matcher import HybridMatchResult, HybridMatcher
from src.registration.registration_engine import RegistrationEngine, RegistrationResult
from src.registration.register import register_image
from src.registration.refinement import refine_tie_points

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def generate_synthetic_crater_terrain(size: int = 512, seed: int = 42) -> np.ndarray:
    """Generate deterministic synthetic terrain with crater-like circular features."""
    rng = np.random.default_rng(seed)
    # Low-frequency background variation
    x = np.linspace(0, 4 * np.pi, size)
    y = np.linspace(0, 4 * np.pi, size)
    xx, yy = np.meshgrid(x, y)
    bg = 100.0 + 25.0 * np.sin(xx) * np.cos(yy)

    img = bg.astype(np.float32)

    # Add 12 distinct crater structures
    craters = [
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

    for cx, cy, rad, rim_val, floor_val in craters:
        # Rim (bright ring)
        cv2.circle(img, (cx, cy), rad, float(rim_val), -1)
        # Floor (shadowed inner basin)
        cv2.circle(img, (cx, cy), int(rad * 0.65), float(floor_val), -1)
        # Subtle rim highlight on northwest edge
        cv2.circle(img, (cx - int(rad * 0.2), cy - int(rad * 0.2)), int(rad * 0.4), float(rim_val + 20), -1)

    # Subtle gaussian texture
    noise = rng.normal(0, 3.0, (size, size)).astype(np.float32)
    img = np.clip(img + noise, 0.0, 255.0).astype(np.uint8)
    return img


def main():
    logger.info("============================================================")
    logger.info("Milestone B: Registration & Quality Engine Inspection")
    logger.info("SYNTHETIC REGISTRATION & QUALITY VERIFICATION")
    logger.info("============================================================")

    out_dir = os.path.join(PROJECT_ROOT, "data", "processed", "registration")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Ground Truth Transformation
    gt_scale = 1.35
    gt_angle_deg = 38.0
    gt_angle_rad = math.radians(gt_angle_deg)
    gt_tx = 45.0
    gt_ty = -30.0

    gt_r_mat = np.array([
        [math.cos(gt_angle_rad), -math.sin(gt_angle_rad)],
        [math.sin(gt_angle_rad), math.cos(gt_angle_rad)],
    ], dtype=np.float64)

    gt_transform = SimilarityTransform2D(
        scale=gt_scale,
        rotation_rad=gt_angle_rad,
        translation_x=gt_tx,
        translation_y=gt_ty,
        rotation_matrix=gt_r_mat,
    )

    # 2. Synthetic Terrain Images
    img_size = 512
    img_a = generate_synthetic_crater_terrain(size=img_size, seed=42)

    # Generate Image B by warping Image A with forward transform T (A -> B)
    # Forward matrix maps Image A (src) to Image B (dst):
    # p_B = s * R * p_A + t
    m_fwd = np.zeros((2, 3), dtype=np.float64)
    m_fwd[:2, :2] = gt_scale * gt_r_mat
    m_fwd[:2, 2] = [gt_tx, gt_ty]

    # Warp into slightly larger canvas to accommodate scale expansion
    canvas_size = 700
    img_b = cv2.warpAffine(
        img_a,
        m_fwd,
        (canvas_size, canvas_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # 3. Ground Truth Inlier Coordinates
    pts_a = np.array([
        [120.0, 140.0],
        [220.0, 100.0],
        [350.0, 160.0],
        [160.0, 260.0],
        [280.0, 240.0],
        [400.0, 280.0],
        [100.0, 380.0],
        [240.0, 380.0],
        [360.0, 400.0],
        [80.0, 200.0],
    ], dtype=np.float64)

    pts_b = gt_transform.apply(pts_a)

    corrs_clean = [
        Correspondence(
            point_a=pts_a[i],
            point_b=pts_b[i],
            confidence=round(0.95 - 0.02 * i, 3),
            source="crater" if i % 2 == 0 else "learned",
            source_index=i,
        )
        for i in range(len(pts_a))
    ]

    engine = RegistrationEngine()

    # =========================================================================
    # SCENARIO A — Clean Inliers
    # =========================================================================
    logger.info("\n--- SCENARIO A: Clean Inliers ---")
    hybrid_res_a = HybridMatchResult(
        matched=True,
        correspondences=corrs_clean,
        inlier_indices=list(range(len(corrs_clean))),
        outlier_indices=[],
        transform=gt_transform,
        confidence=0.95,
        rmse=0.0,
        inlier_ratio=1.0,
        metadata={"scenario": "clean"},
    )

    reg_res_a = engine.register(img_a, img_b, hybrid_res_a)

    logger.info("Registration Success       : %s", reg_res_a.success)
    logger.info("Quality Classification     : %s", reg_res_a.quality)
    logger.info("Confidence Score           : %.4f", reg_res_a.confidence)
    logger.info("Input Inliers              : %d", reg_res_a.num_inliers)
    logger.info("Selected Tie Points        : %d", reg_res_a.num_tie_points)
    logger.info("Spatial Coverage           : %.2f%%", reg_res_a.coverage * 100)
    logger.info("Final RMSE                 : %.4f px", reg_res_a.rmse)
    if reg_res_a.transform is not None:
        s_err = abs(reg_res_a.transform.scale - gt_scale)
        r_err = abs(reg_res_a.transform.rotation_deg - gt_angle_deg)
        t_err = np.linalg.norm(np.array(reg_res_a.transform.translation) - np.array([gt_tx, gt_ty]))
        logger.info("Scale Error                : %.2e", s_err)
        logger.info("Rotation Error             : %.2e deg", r_err)
        logger.info("Translation Error          : %.4f px", t_err)

    # =========================================================================
    # SCENARIO B — Inliers + False Outliers
    # =========================================================================
    logger.info("\n--- SCENARIO B: Outliers + Clustered Inliers ---")
    # Add 4 false distractor correspondences
    outlier_pts_a = np.array([[50.0, 50.0], [450.0, 50.0], [50.0, 450.0], [450.0, 450.0]])
    outlier_pts_b = np.array([[600.0, 100.0], [100.0, 600.0], [50.0, 50.0], [300.0, 300.0]])

    all_corrs = list(corrs_clean)
    for j in range(len(outlier_pts_a)):
        all_corrs.append(
            Correspondence(
                point_a=outlier_pts_a[j],
                point_b=outlier_pts_b[j],
                confidence=0.4,
                source="learned",
                source_index=len(corrs_clean) + j,
            )
        )

    # Pre-verified hybrid match result
    hybrid_res_b = HybridMatchResult(
        matched=True,
        correspondences=all_corrs,
        inlier_indices=list(range(len(corrs_clean))),
        outlier_indices=list(range(len(corrs_clean), len(all_corrs))),
        transform=gt_transform,
        confidence=0.82,
        rmse=0.0,
        inlier_ratio=len(corrs_clean) / len(all_corrs),
        metadata={"scenario": "outliers"},
    )

    reg_res_b = engine.register(img_a, img_b, hybrid_res_b)
    logger.info("Registration Success       : %s", reg_res_b.success)
    logger.info("Quality Classification     : %s", reg_res_b.quality)
    logger.info("Confidence Score           : %.4f", reg_res_b.confidence)
    logger.info("Inlier Ratio               : %.2f%%", (reg_res_b.num_inliers / len(all_corrs)) * 100)
    logger.info("Selected Tie Points        : %d", reg_res_b.num_tie_points)
    logger.info("Spatial Coverage           : %.2f%%", reg_res_b.coverage * 100)

    # =========================================================================
    # SCENARIO C — Sub-Pixel Refinement on Fractional-Pixel Perturbations
    # =========================================================================
    logger.info("\n--- SCENARIO C: Sub-Pixel Refinement on Perturbed Points ---")
    # Add a controlled sub-pixel shift of 0.45 px to detected points in Image B
    shift_vector = np.array([0.45, -0.30])
    noisy_pts_b = pts_b + shift_vector

    rmse_before = float(np.sqrt(np.mean(np.linalg.norm(noisy_pts_b - pts_b, axis=1)**2)))
    logger.info("Coarse Perturbation RMSE   : %.4f px", rmse_before)

    # Run refinement
    refine_cfg = {"enabled": True, "patch_radius": 15, "search_radius": 3, "min_correlation": 0.5, "subpixel_interpolation": True}
    ref_res = refine_tie_points(
        image_a=img_a,
        image_b=img_b,
        points_a=pts_a,
        points_b=noisy_pts_b,
        transform=gt_transform,
        config=refine_cfg,
    )

    rmse_after = float(np.sqrt(np.mean(np.linalg.norm(ref_res.refined_points_b - pts_b, axis=1)**2)))
    logger.info("Points Attempted           : %d", ref_res.num_attempted)
    logger.info("Points Successfully Refined: %d", ref_res.num_refined)
    logger.info("Mean Refinement Shift      : %.4f px", ref_res.mean_shift_px)
    logger.info("Refined Pointset RMSE      : %.4f px", rmse_after)
    logger.info("RMSE Reduction             : %.2f%%", ((rmse_before - rmse_after) / rmse_before) * 100)

    # 4. Generate Comprehensive 5-Panel Diagnostic Visualization
    logger.info("\nGenerating diagnostic 5-panel visualization...")
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))

    reg_img = reg_res_a.registered_image
    vmask = reg_res_a.valid_mask

    # Panel 1: Image A with tie points
    axes[0].imshow(img_a, cmap="gray")
    axes[0].scatter(pts_a[:, 0], pts_a[:, 1], c="lime", s=40, edgecolors="black", label="Tie Points (A)")
    axes[0].set_title("Reference Image A\n(with Tie Points)")
    axes[0].axis("off")
    axes[0].legend(loc="lower right", fontsize=8)

    # Panel 2: Image B with transformed points
    axes[1].imshow(img_b, cmap="gray")
    axes[1].scatter(pts_b[:, 0], pts_b[:, 1], c="cyan", s=40, edgecolors="black", label="Target Points (B)")
    axes[1].set_title("Query Image B\n(Transformed + Expanded)")
    axes[1].axis("off")
    axes[1].legend(loc="lower right", fontsize=8)

    # Panel 3: Registered Image B in Image A frame
    axes[2].imshow(reg_img, cmap="gray")
    axes[2].set_title(f"Registered Image B\n(Warped to Frame A, Quality: {reg_res_a.quality})")
    axes[2].axis("off")

    # Panel 4: Overlap Difference Map (|Image A - Registered B|)
    diff = np.abs(img_a.astype(np.float32) - reg_img.astype(np.float32))
    diff_overlap = np.where(vmask > 0, diff, 0.0)
    im_diff = axes[3].imshow(diff_overlap, cmap="inferno")
    axes[3].set_title(f"Overlap Absolute Difference\n(RMSE: {reg_res_a.rmse:.2f} px)")
    axes[3].axis("off")
    plt.colorbar(im_diff, ax=axes[3], fraction=0.046, pad=0.04)

    # Panel 5: Valid Overlap Mask
    axes[4].imshow(vmask, cmap="Blues")
    axes[4].set_title(f"Valid Overlap Mask\n(Coverage: {reg_res_a.coverage * 100:.1f}%)")
    axes[4].axis("off")

    plt.suptitle("SYNTHETIC REGISTRATION & QUALITY VERIFICATION (Milestone B)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()

    out_plot_path = os.path.join(out_dir, "synthetic_registration_demo.png")
    fig.savefig(out_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Diagnostic visualization saved successfully to:\n%s", out_plot_path)
    logger.info("Milestone B inspection completed successfully.")


if __name__ == "__main__":
    main()
