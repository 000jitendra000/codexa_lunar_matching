"""
experiments/inspect_ransac.py

Demonstration and verification of Phase 10: Robust Similarity Transformation Estimation (RANSAC).
Evaluates RANSAC robustness across three distinct synthetic scenarios:
1. Scenario A (Clean):
   8 correct inlier correspondences, 0 outliers.
   Compares Phase 9 Least-Squares vs Phase 10 RANSAC on pure data.
2. Scenario B (Single Outlier):
   7 inliers, 1 severe outlier (the exact Phase 9 corrupted correspondence offset).
   Demonstrates how RANSAC rejects the outlier and recovers the true transform,
   while plain least-squares suffers severe parameter and residual degradation.
3. Scenario C (Multiple Outliers):
   8 inliers, 3 arbitrary/distractor outliers (27% outlier ratio).
   Demonstrates dominant consensus recovery and inlier/outlier classification.

Generates a diagnostic 3-panel visualization:
- Target crater centers (Image B)
- Transformed source crater centers (Image A -> B)
- Inliers (green) vs rejected outliers (red) with alignment residual vectors.

NOTE: Strictly software verification on synthetic data.
Do NOT imply this is real OHRC/LROC data.
"""

import os
import sys
import math
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.matching.transformation import SimilarityTransform2D, estimate_similarity_transform
from src.matching.ransac import estimate_robust_similarity_transform

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("============================================================")
    logger.info("Phase 10 Robust Transformation Estimation Inspection")
    logger.info("SYNTHETIC RANSAC VERIFICATION")
    logger.info("============================================================")

    # 1. Ground Truth Transform Parameters
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

    # 2. Source Crater Constellation (8 craters)
    pts_a_base = np.array([
        [100.0, 150.0],
        [180.0, 120.0],
        [220.0, 260.0],
        [130.0, 310.0],
        [300.0, 180.0],
        [260.0, 340.0],
        [350.0, 280.0],
        [170.0, 210.0],
    ], dtype=np.float64)

    pts_b_base = gt_transform.apply(pts_a_base)

    # -----------------------------------------------------------------------
    # Scenario A: Clean Data (8 inliers, 0 outliers)
    # -----------------------------------------------------------------------
    logger.info("Running Scenario A (Clean: 8 inliers, 0 outliers)...")
    pts_a_a = pts_a_base.copy()
    pts_b_a = pts_b_base.copy()

    res_ls_a = estimate_similarity_transform(pts_a_a, pts_b_a)
    res_ransac_a = estimate_robust_similarity_transform(
        pts_a_a, pts_b_a, reprojection_threshold=3.0, random_seed=42
    )

    # -----------------------------------------------------------------------
    # Scenario B: Single Outlier (7 inliers, 1 outlier)
    # -----------------------------------------------------------------------
    logger.info("Running Scenario B (One Outlier: 7 inliers, 1 outlier)...")
    pts_a_b = pts_a_base.copy()
    pts_b_b = pts_b_base.copy()
    pts_b_b[0] += np.array([120.0, -100.0])  # Corrupted correspondence

    res_ls_b = estimate_similarity_transform(pts_a_b, pts_b_b)
    res_ransac_b = estimate_robust_similarity_transform(
        pts_a_b, pts_b_b, reprojection_threshold=3.0, random_seed=42
    )

    # -----------------------------------------------------------------------
    # Scenario C: Multiple Outliers (8 inliers, 3 outliers)
    # -----------------------------------------------------------------------
    logger.info("Running Scenario C (Multiple Outliers: 8 inliers, 3 outliers)...")
    outlier_src = np.array([
        [400.0, 120.0],
        [160.0, 420.0],
        [330.0, 380.0],
    ])
    # 3 severe outlier offsets (80-100px errors)
    outlier_tgt = gt_transform.apply(outlier_src) + np.array([
        [80.0, 60.0],
        [-70.0, 90.0],
        [100.0, -80.0],
    ])

    pts_a_c = np.vstack([pts_a_base, outlier_src])
    pts_b_c = np.vstack([pts_b_base, outlier_tgt])

    try:
        res_ls_c = estimate_similarity_transform(pts_a_c, pts_b_c)
    except ValueError as e:
        res_ls_c = None
        logger.warning("Phase 9 Least-Squares failed on Scenario C: %s", e)

    res_ransac_c = estimate_robust_similarity_transform(
        pts_a_c, pts_b_c, reprojection_threshold=3.0, random_seed=42
    )

    # -----------------------------------------------------------------------
    # Console Output: Comparative Metrics Tables
    # -----------------------------------------------------------------------
    def print_scenario_report(name: str, n_pts: int, n_true_inliers: int, res_ls, res_ransac):
        tf_ran = res_ransac.transform
        ran_s_err = abs(tf_ran.scale - gt_scale)
        ran_rot_err = abs(tf_ran.rotation_deg - gt_angle_deg)
        ran_t_err = math.hypot(tf_ran.translation_x - gt_tx, tf_ran.translation_y - gt_ty)

        print(f"\n{'=' * 80}")
        print(f"SCENARIO {name}: Total={n_pts} | True Inliers={n_true_inliers} | Outliers={n_pts - n_true_inliers}")
        print(f"{'=' * 80}")
        print(f"{'Metric':<30} | {'Phase 9 Least-Squares':<22} | {'Phase 10 RANSAC':<22}")
        print(f"{'-' * 80}")
        print(f"{'Inliers Detected':<30} | {f'{n_pts} / {n_pts}' if res_ls else 'FAILED':<22} | {f'{res_ransac.num_inliers} / {n_pts}':<22}")
        print(f"{'Inlier Ratio':<30} | {'100.0%' if res_ls else 'N/A':<22} | {f'{res_ransac.inlier_ratio * 100:.1f}%':<22}")
        print(f"{'Iterations':<30} | {'1 (closed-form)' if res_ls else 'N/A':<22} | {f'{res_ransac.num_iterations}':<22}")

        if res_ls is not None:
            tf_ls = res_ls.transform
            ls_s_err = abs(tf_ls.scale - gt_scale)
            ls_rot_err = abs(tf_ls.rotation_deg - gt_angle_deg)
            ls_t_err = math.hypot(tf_ls.translation_x - gt_tx, tf_ls.translation_y - gt_ty)
            print(f"{'Estimated Scale':<30} | {f'{tf_ls.scale:.4f} (err {ls_s_err:.2e})':<22} | {f'{tf_ran.scale:.4f} (err {ran_s_err:.2e})':<22}")
            print(f"{'Estimated Rotation (deg)':<30} | {f'{tf_ls.rotation_deg:.2f}° (err {ls_rot_err:.2e}°)':<22} | {f'{tf_ran.rotation_deg:.2f}° (err {ran_rot_err:.2e}°)':<22}")
            print(f"{'Translation Error (px)':<30} | {f'{ls_t_err:.4f}':<22} | {f'{ran_t_err:.4f}':<22}")
            print(f"{'Inlier RMSE (px)':<30} | {f'{res_ls.rmse:.4f}':<22} | {f'{res_ransac.rmse:.4f}':<22}")
            print(f"{'Max Inlier Error (px)':<30} | {f'{res_ls.max_error:.4f}':<22} | {f'{res_ransac.max_inlier_error:.4f}':<22}")
        else:
            print(f"{'Estimated Scale':<30} | {'FAILED (orientation)':<22} | {f'{tf_ran.scale:.4f} (err {ran_s_err:.2e})':<22}")
            print(f"{'Estimated Rotation (deg)':<30} | {'FAILED':<22} | {f'{tf_ran.rotation_deg:.2f}° (err {ran_rot_err:.2e}°)':<22}")
            print(f"{'Translation Error (px)':<30} | {'FAILED':<22} | {f'{ran_t_err:.4f}':<22}")
            print(f"{'Inlier RMSE (px)':<30} | {'FAILED':<22} | {f'{res_ransac.rmse:.4f}':<22}")
            print(f"{'Max Inlier Error (px)':<30} | {'FAILED':<22} | {f'{res_ransac.max_inlier_error:.4f}':<22}")
        print(f"{'=' * 80}")

    print_scenario_report("A (CLEAN)", len(pts_a_a), 8, res_ls_a, res_ransac_a)
    print_scenario_report("B (ONE OUTLIER)", len(pts_a_b), 7, res_ls_b, res_ransac_b)
    print_scenario_report("C (MULTIPLE OUTLIERS)", len(pts_a_c), 8, res_ls_c, res_ransac_c)

    # -----------------------------------------------------------------------
    # Diagnostic Visualization: 3-Panel Figure
    # -----------------------------------------------------------------------
    out_dir = os.path.join(PROJECT_ROOT, "data", "processed", "ransac")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "synthetic_ransac_demo.png")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "SYNTHETIC RANSAC INLIER/OUTLIER DEMO (Software Verification Only)\n"
        f"Ground Truth: Scale={gt_scale}, Rot={gt_angle_deg}°, Translation=({gt_tx}, {gt_ty}) px",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    scenarios = [
        ("Scenario A: Clean (8 Inliers)", pts_a_a, pts_b_a, res_ransac_a, axes[0]),
        ("Scenario B: 1 Outlier (7 Inliers)", pts_a_b, pts_b_b, res_ransac_b, axes[1]),
        ("Scenario C: 3 Outliers (8 Inliers)", pts_a_c, pts_b_c, res_ransac_c, axes[2]),
    ]

    for title, pa, pb, ransac_res, ax in scenarios:
        tf = ransac_res.transform
        pa_trans = tf.apply(pa)

        inliers = set(ransac_res.inlier_indices)
        outliers = set(ransac_res.outlier_indices)

        # Plot Target Points (p_B)
        ax.scatter(pb[:, 0], pb[:, 1], edgecolors="black", facecolors="none", marker="o", s=70,
                   linewidths=1.5, label="Target Centers ($p_B$)", zorder=3)

        # Plot Transformed Source Points (T(p_A))
        # Inliers
        inlier_idx = [i for i in range(len(pa)) if i in inliers]
        if inlier_idx:
            ax.scatter(pa_trans[inlier_idx, 0], pa_trans[inlier_idx, 1],
                       c="#2ca02c", marker="o", s=45, label=f"Inliers ({len(inlier_idx)})", zorder=4)
            # Alignment error vectors for inliers
            for i in inlier_idx:
                ax.plot([pa_trans[i, 0], pb[i, 0]], [pa_trans[i, 1], pb[i, 1]],
                        color="#2ca02c", linestyle="-", linewidth=1.0, alpha=0.7)

        # Outliers
        outlier_idx = [i for i in range(len(pa)) if i in outliers]
        if outlier_idx:
            ax.scatter(pa_trans[outlier_idx, 0], pa_trans[outlier_idx, 1],
                       c="#d62728", marker="x", s=80, linewidths=2.0, label=f"Outliers ({len(outlier_idx)})", zorder=5)
            # Residual vectors for outliers
            for i in outlier_idx:
                ax.plot([pa_trans[i, 0], pb[i, 0]], [pa_trans[i, 1], pb[i, 1]],
                        color="#d62728", linestyle="--", linewidth=1.2, alpha=0.8)

        ax.set_title(
            f"{title}\n"
            f"RANSAC s={tf.scale:.3f}, θ={tf.rotation_deg:.1f}°, RMSE={ransac_res.rmse:.2f}px",
            fontsize=10,
        )
        ax.set_xlabel("X coordinate (pixels)", fontsize=9)
        ax.set_ylabel("Y coordinate (pixels)", fontsize=9)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.set_aspect("equal", "datalim")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_path, dpi=150)
    plt.close()

    logger.info("Visual diagnostic plot saved to: %s", out_path)
    logger.info("Phase 10 RANSAC Inspection complete.")


if __name__ == "__main__":
    main()
