"""
experiments/inspect_hybrid_matching.py

Demonstration and verification of Milestone A: Hybrid Matching Engine (Phases 11-13).
Evaluates the fusion of classical crater geometry and learned matching across three scenarios:
1. Scenario A (Classical Crater Only):
   Tests pipeline execution when only crater correspondences are available.
2. Scenario B (Learned Only):
   Tests pipeline execution when only learned correspondences are available
   (synthetic learned match data representing the same geometric transform).
3. Scenario C (Hybrid Fusion & Robust Verification):
   Fuses crater correspondences + learned correspondences + distractor false matches,
   demonstrating source balancing, deduplication, and final Phase 10 RANSAC outlier rejection.

Generates a diagnostic 3-panel visualization:
- Target coordinates (Image B)
- Transformed source coordinates (Image A -> B)
- Crater inliers (green), learned inliers (cyan), and rejected outliers (red).

NOTE: Strictly software verification on synthetic data.
Do NOT imply this demonstrates real LoFTR performance on lunar imagery.
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

from src.matching.transformation import SimilarityTransform2D
from src.matching.correspondence_fusion import Correspondence
from src.matching.hybrid_matcher import HybridMatcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("============================================================")
    logger.info("Milestone A: Hybrid Matching Engine Inspection")
    logger.info("SYNTHETIC HYBRID MATCHING VERIFICATION")
    logger.info("============================================================")

    # 1. Known Ground Truth Transform Parameters
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

    # 2. Base Ground Truth Points in Image A
    # Group 1: Crater locations (6 craters)
    crater_pts_a = np.array([
        [100.0, 150.0],
        [180.0, 120.0],
        [220.0, 260.0],
        [130.0, 310.0],
        [300.0, 180.0],
        [260.0, 340.0],
    ], dtype=np.float64)
    crater_pts_b = gt_transform.apply(crater_pts_a)

    # Group 2: Learned feature points (distinct fine-grained keypoints, 6 points)
    learned_pts_a = np.array([
        [150.0, 190.0],
        [240.0, 140.0],
        [190.0, 300.0],
        [320.0, 230.0],
        [280.0, 310.0],
        [350.0, 280.0],
    ], dtype=np.float64)
    learned_pts_b = gt_transform.apply(learned_pts_a)

    # Convert to Correspondence objects
    crater_corrs = [
        Correspondence(
            point_a=crater_pts_a[i],
            point_b=crater_pts_b[i],
            confidence=round(0.95 - 0.03 * i, 3),
            source="crater",
            source_index=i,
            metadata={"crater_id": i},
        )
        for i in range(len(crater_pts_a))
    ]

    learned_corrs = [
        Correspondence(
            point_a=learned_pts_a[i],
            point_b=learned_pts_b[i],
            confidence=round(0.88 - 0.04 * i, 3),
            source="learned",
            source_index=i,
            metadata={"keypoint_id": i},
        )
        for i in range(len(learned_pts_a))
    ]

    matcher = HybridMatcher()

    # -----------------------------------------------------------------------
    # Scenario A: Classical Crater Only
    # -----------------------------------------------------------------------
    logger.info("Executing Scenario A (Classical Crater Only)...")
    res_a = matcher.match_from_correspondences(
        crater_correspondences=crater_corrs,
        learned_correspondences=[],
    )

    # -----------------------------------------------------------------------
    # Scenario B: Learned Only (Synthetic Learned Correspondences)
    # -----------------------------------------------------------------------
    logger.info("Executing Scenario B (Learned Only - Synthetic Matches)...")
    res_b = matcher.match_from_correspondences(
        crater_correspondences=[],
        learned_correspondences=learned_corrs,
    )

    # -----------------------------------------------------------------------
    # Scenario C: Hybrid (Crater + Learned + Distractors)
    # -----------------------------------------------------------------------
    logger.info("Executing Scenario C (Hybrid: Crater + Learned + Distractors)...")
    # Add 2 false crater distractors and 2 false learned distractors
    distractor_crater = [
        Correspondence(
            point_a=np.array([80.0, 90.0]),
            point_b=np.array([450.0, 50.0]),  # False match
            confidence=0.65,
            source="crater",
            source_index=10,
        ),
        Correspondence(
            point_a=np.array([340.0, 100.0]),
            point_b=np.array([80.0, 480.0]),  # False match
            confidence=0.55,
            source="crater",
            source_index=11,
        ),
    ]

    distractor_learned = [
        Correspondence(
            point_a=np.array([120.0, 280.0]),
            point_b=np.array([380.0, 80.0]),  # False match
            confidence=0.58,
            source="learned",
            source_index=12,
        ),
        Correspondence(
            point_a=np.array([290.0, 150.0]),
            point_b=np.array([100.0, 390.0]),  # False match
            confidence=0.50,
            source="learned",
            source_index=13,
        ),
    ]

    hybrid_crater = crater_corrs + distractor_crater
    hybrid_learned = learned_corrs + distractor_learned

    res_c = matcher.match_from_correspondences(
        crater_correspondences=hybrid_crater,
        learned_correspondences=hybrid_learned,
        reprojection_threshold=3.0,
    )

    # -----------------------------------------------------------------------
    # Console Reporting Table
    # -----------------------------------------------------------------------
    def print_scenario_metrics(name: str, res, n_crater: int, n_learned: int):
        tf = res.transform
        s_err = abs(tf.scale - gt_scale) if tf else 0.0
        rot_err = abs(tf.rotation_deg - gt_angle_deg) if tf else 0.0
        t_err = math.hypot(tf.translation_x - gt_tx, tf.translation_y - gt_ty) if tf else 0.0

        crater_inl = res.metadata.get("crater_inliers", 0)
        learned_inl = res.metadata.get("learned_inliers", 0)

        print(f"\n{'=' * 80}")
        print(f"SCENARIO {name}")
        print(f"{'=' * 80}")
        print(f"Match Status               : {'SUCCESS (Consensus achieved)' if res.matched else 'FAILED'}")
        print(f"Crater Candidates Input    : {n_crater}")
        print(f"Learned Candidates Input   : {n_learned}")
        print(f"Fused Candidates           : {res.num_correspondences}")
        print(f"Final Inliers Total        : {res.num_inliers} (Crater: {crater_inl}, Learned: {learned_inl})")
        print(f"Final Outliers Total       : {res.num_outliers}")
        print(f"Inlier Ratio               : {res.inlier_ratio * 100:.1f}%")
        print(f"RANSAC Iterations          : {res.metadata.get('ransac_iterations', 0)}")
        print(f"Estimated Scale            : {tf.scale:.4f} (error: {s_err:.2e})")
        print(f"Estimated Rotation (deg)   : {tf.rotation_deg:.2f}° (error: {rot_err:.2e}°)")
        print(f"Translation Error (pixels) : {t_err:.4f} px")
        print(f"Inlier RMSE (pixels)       : {res.rmse:.6f} px" if res.rmse is not None else "N/A")
        print(f"Hybrid Confidence Score    : {res.confidence:.4f}")
        print(f"{'=' * 80}")

    print_scenario_metrics("A — Classical Crater Only", res_a, len(crater_corrs), 0)
    print_scenario_metrics("B — Learned Only (Synthetic LoFTR Matches)", res_b, 0, len(learned_corrs))
    print_scenario_metrics("C — Hybrid (Crater + Learned + Distractors)", res_c, len(hybrid_crater), len(hybrid_learned))

    # -----------------------------------------------------------------------
    # 3-Panel Diagnostic Visualization
    # -----------------------------------------------------------------------
    out_dir = os.path.join(PROJECT_ROOT, "data", "processed", "hybrid_matching")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "synthetic_hybrid_matching_demo.png")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "SYNTHETIC HYBRID MATCHING VERIFICATION (Software Verification Only)\n"
        f"Ground Truth: Scale={gt_scale}, Rot={gt_angle_deg}°, Translation=({gt_tx}, {gt_ty}) px",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    scenarios = [
        ("Scenario A: Crater Only (6 Inliers)", res_a, axes[0]),
        ("Scenario B: Learned Only (6 Inliers)", res_b, axes[1]),
        ("Scenario C: Hybrid (12 Inliers, 4 Outliers)", res_c, axes[2]),
    ]

    for title, res, ax in scenarios:
        tf = res.transform
        corrs = res.correspondences
        inlier_set = set(res.inlier_indices)

        pts_a = np.array([c.point_a for c in corrs])
        pts_b = np.array([c.point_b for c in corrs])
        pts_a_trans = tf.apply(pts_a)

        # Plot Target Points (p_B)
        ax.scatter(pts_b[:, 0], pts_b[:, 1], edgecolors="black", facecolors="none",
                   marker="o", s=70, linewidths=1.5, label="Target Centers ($p_B$)", zorder=3)

        # Plot Crater Inliers (Green circles)
        crater_inl_idx = [i for i in range(len(corrs)) if i in inlier_set and corrs[i].source == "crater"]
        if crater_inl_idx:
            ax.scatter(pts_a_trans[crater_inl_idx, 0], pts_a_trans[crater_inl_idx, 1],
                       c="#2ca02c", marker="o", s=50, label=f"Crater Inliers ({len(crater_inl_idx)})", zorder=4)
            for i in crater_inl_idx:
                ax.plot([pts_a_trans[i, 0], pts_b[i, 0]], [pts_a_trans[i, 1], pts_b[i, 1]],
                        color="#2ca02c", linestyle="-", linewidth=1.0, alpha=0.7)

        # Plot Learned Inliers (Cyan triangles)
        learned_inl_idx = [i for i in range(len(corrs)) if i in inlier_set and corrs[i].source == "learned"]
        if learned_inl_idx:
            ax.scatter(pts_a_trans[learned_inl_idx, 0], pts_a_trans[learned_inl_idx, 1],
                       c="#17becf", marker="^", s=60, label=f"Learned Inliers ({len(learned_inl_idx)})", zorder=4)
            for i in learned_inl_idx:
                ax.plot([pts_a_trans[i, 0], pts_b[i, 0]], [pts_a_trans[i, 1], pts_b[i, 1]],
                        color="#17becf", linestyle="-", linewidth=1.0, alpha=0.7)

        # Plot Outliers (Red cross)
        outl_idx = [i for i in range(len(corrs)) if i not in inlier_set]
        if outl_idx:
            ax.scatter(pts_a_trans[outl_idx, 0], pts_a_trans[outl_idx, 1],
                       c="#d62728", marker="x", s=80, linewidths=2.0, label=f"Outliers ({len(outl_idx)})", zorder=5)
            for i in outl_idx:
                ax.plot([pts_a_trans[i, 0], pts_b[i, 0]], [pts_a_trans[i, 1], pts_b[i, 1]],
                        color="#d62728", linestyle="--", linewidth=1.2, alpha=0.8)

        ax.set_title(
            f"{title}\n"
            f"s={tf.scale:.3f}, θ={tf.rotation_deg:.1f}°, RMSE={res.rmse:.2f}px, Conf={res.confidence:.2f}",
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
    logger.info("Milestone A Inspection complete.")


if __name__ == "__main__":
    main()
