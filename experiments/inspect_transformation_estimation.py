"""
experiments/inspect_transformation_estimation.py

Demonstration and verification of Phase 9: Initial Similarity Transformation Estimation.
Evaluates closed-form 2D similarity estimation (Umeyama):
1. Clean Correspondences:
   Estimates scale, rotation, and translation without prior knowledge of transform parameters.
   Evaluates parameter errors against ground truth and residual diagnostics.
2. Outlier Sensitivity Demonstration:
   Demonstrates unweighted least-squares sensitivity to a corrupted correspondence,
   showing parameter shifts and elevated RMSE (documenting why RANSAC is deferred to later phases).

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

from src.matching.transformation import estimate_similarity_transform

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("============================================================")
    logger.info("Phase 9 Transformation Estimation Inspection")
    logger.info("SYNTHETIC TRANSFORMATION ESTIMATION VERIFICATION")
    logger.info("============================================================")

    # 1. Source crater center coordinates (6 craters)
    pts_a = np.array([
        [100.0, 100.0],
        [240.0, 110.0],
        [280.0, 240.0],
        [190.0, 310.0],
        [80.0, 230.0],
        [175.0, 190.0],
    ], dtype=np.float64)

    # 2. Known ground truth transform parameters
    gt_scale = 1.6
    gt_angle_deg = 55.0
    gt_angle_rad = math.radians(gt_angle_deg)
    gt_tx = 220.0
    gt_ty = 140.0

    gt_r_mat = np.array([
        [math.cos(gt_angle_rad), -math.sin(gt_angle_rad)],
        [math.sin(gt_angle_rad), math.cos(gt_angle_rad)],
    ], dtype=np.float64)

    # Ground-truth transformed target points
    pts_b_clean = gt_scale * (pts_a @ gt_r_mat.T) + np.array([gt_tx, gt_ty])

    # 3. Experiment 1: Clean Closed-Form Estimation
    logger.info("Running clean closed-form similarity estimation...")
    res_clean = estimate_similarity_transform(pts_a, pts_b_clean)
    tf_clean = res_clean.transform

    scale_err = abs(tf_clean.scale - gt_scale)
    rot_err_deg = abs(tf_clean.rotation_deg - gt_angle_deg)
    tx_err = abs(tf_clean.translation_x - gt_tx)
    ty_err = abs(tf_clean.translation_y - gt_ty)

    print("\n" + "=" * 80)
    print("EXPERIMENT 1: CLEAN ESTIMATION VS GROUND TRUTH")
    print("=" * 80)
    print(f"Correspondences     : {res_clean.num_correspondences}")
    print(f"{'Parameter':<18} | {'Ground Truth':<15} | {'Estimated':<15} | {'Absolute Error':<15}")
    print("-" * 80)
    print(f"{'Scale (s)':<18} | {gt_scale:<15.6f} | {tf_clean.scale:<15.6f} | {scale_err:<15.6e}")
    print(f"{'Rotation (deg)':<18} | {gt_angle_deg:<15.4f} | {tf_clean.rotation_deg:<15.4f} | {rot_err_deg:<15.6e}")
    print(f"{'Translation X':<18} | {gt_tx:<15.4f} | {tf_clean.translation_x:<15.4f} | {tx_err:<15.6e}")
    print(f"{'Translation Y':<18} | {gt_ty:<15.4f} | {tf_clean.translation_y:<15.4f} | {ty_err:<15.6e}")
    print("-" * 80)
    print(f"RMSE (pixels)       : {res_clean.rmse:.6f}")
    print(f"Mean Residual       : {res_clean.mean_error:.6f}")
    print(f"Median Residual     : {res_clean.median_error:.6f}")
    print(f"Max Residual        : {res_clean.max_error:.6f}")
    print("=" * 80)

    # 4. Experiment 2: Outlier Sensitivity Demonstration
    logger.info("Running outlier sensitivity demonstration...")
    pts_b_outlier = pts_b_clean.copy()
    # Deliberately corrupt point 0 by a gross +80px, -60px displacement
    pts_b_outlier[0] += np.array([80.0, -60.0])

    res_outlier = estimate_similarity_transform(pts_a, pts_b_outlier)
    tf_outlier = res_outlier.transform

    print("\n" + "=" * 80)
    print("EXPERIMENT 2: SYNTHETIC OUTLIER SENSITIVITY DEMO")
    print("=" * 80)
    print("Note: Demonstrates why unweighted least-squares requires RANSAC (Phase 10+)")
    print(f"{'Metric':<25} | {'Clean Fit':<20} | {'With 1 Outlier':<20}")
    print("-" * 80)
    print(f"{'Estimated Scale':<25} | {tf_clean.scale:<20.4f} | {tf_outlier.scale:<20.4f}")
    print(f"{'Estimated Rotation (deg)':<25} | {tf_clean.rotation_deg:<20.4f} | {tf_outlier.rotation_deg:<20.4f}")
    print(f"{'Estimated Translation (tx, ty)':<25} | {f'({tf_clean.translation_x:.1f}, {tf_clean.translation_y:.1f})':<20} | {f'({tf_outlier.translation_x:.1f}, {tf_outlier.translation_y:.1f})':<20}")
    print(f"{'RMSE (pixels)':<25} | {res_clean.rmse:<20.4f} | {res_outlier.rmse:<20.4f}")
    print(f"{'Max Residual (pixels)':<25} | {res_clean.max_error:<20.4f} | {res_outlier.max_error:<20.4f}")
    print("=" * 80)

    # 5. Visual Diagnostic Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), facecolor="#0a0e17")
    fig.suptitle(
        "SYNTHETIC TRANSFORMATION ESTIMATION VERIFICATION\n"
        "[Software Verification: Closed-Form Umeyama 2D Similarity Estimation & Outlier Sensitivity]",
        color="white", fontsize=13, fontweight="bold", y=0.98
    )

    pts_a_trans_clean = tf_clean.apply(pts_a)
    pts_a_trans_outlier = tf_outlier.apply(pts_a)

    for ax, tf_pts, target_pts, title, rmse_val, has_outlier in [
        (ax1, pts_a_trans_clean, pts_b_clean, "Clean Least-Squares Alignment", res_clean.rmse, False),
        (ax2, pts_a_trans_outlier, pts_b_outlier, "Outlier Sensitivity Demo (1 Corrupted Pair)", res_outlier.rmse, True),
    ]:
        ax.set_facecolor("#121720")

        # Plot original source points
        ax.scatter(pts_a[:, 0], pts_a[:, 1], color="#718096", s=60, marker="o", alpha=0.6, label="Source Points (p_A)")
        for i, (x, y) in enumerate(pts_a):
            ax.text(x + 5, y + 5, f"A{i}", color="#a0aec0", fontsize=8)

        # Plot target points
        tgt_colors = ["#e53e3e" if (has_outlier and i == 0) else "#48bb78" for i in range(len(target_pts))]
        ax.scatter(target_pts[:, 0], target_pts[:, 1], color=tgt_colors, s=120, marker="s", edgecolors="white", linewidth=1.5, label="Target Points (p_B)")
        for i, (x, y) in enumerate(target_pts):
            lbl = f"B{i} (OUTLIER)" if (has_outlier and i == 0) else f"B{i}"
            ax.text(x + 6, y + 6, lbl, color="#48bb78" if not (has_outlier and i == 0) else "#fc8181", fontsize=9, fontweight="bold")

        # Plot transformed source points
        ax.scatter(tf_pts[:, 0], tf_pts[:, 1], color="#4fd1c5", s=100, marker="x", linewidth=2.2, label="Estimated T(p_A)")

        # Residual error lines
        for i in range(len(pts_a)):
            ax.plot(
                [tf_pts[i, 0], target_pts[i, 0]],
                [tf_pts[i, 1], target_pts[i, 1]],
                color="#e53e3e" if has_outlier else "#4fd1c5",
                linestyle=":" if has_outlier else "-",
                linewidth=2.0 if (has_outlier and i == 0) else 1.2,
                alpha=0.8,
            )

        ax.set_title(f"{title}\nRMSE = {rmse_val:.4f} px", color="white", fontsize=11, fontweight="bold", pad=8)
        ax.set_aspect("equal", adjustable="datalim")
        ax.tick_params(colors="gray")
        ax.legend(loc="upper left", facecolor="#1a202c", edgecolor="#4a5568", labelcolor="white", fontsize=8)
        for spine in ax.spines.values():
            spine.set_color("#2d3748")

    # Global synthetic labeling banner
    fig.text(
        0.5, 0.02,
        "SYNTHETIC TRANSFORMATION ESTIMATION  |  NO REAL OHRC/LROC DATA USED  |  PHASE 9 COMPLETE",
        ha="center", fontsize=10, color="#a0aec0", fontweight="bold"
    )

    out_dir = os.path.join("data", "processed", "transformation_estimation")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "synthetic_similarity_transformation_demo.png")
    plt.tight_layout(rect=[0, 0.04, 1, 0.94])
    plt.savefig(out_path, dpi=160, facecolor=fig.get_facecolor())
    plt.close()

    logger.info("Visual diagnostic plot saved to: %s", out_path)
    logger.info("Phase 9 Transformation Estimation Inspection complete.")


if __name__ == "__main__":
    main()
