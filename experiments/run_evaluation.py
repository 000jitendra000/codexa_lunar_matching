"""
experiments/run_evaluation.py

Milestone C (Phases 18 + 19) — Evaluation & Robustness Demonstration.

Executes a comprehensive benchmark across controlled stress conditions:
1. Clean baseline (Combined similarity)
2. Case A: Resolution mismatch (simulates cross-sensor GSD difference)
3. Case B: Illumination / contrast mismatch (solar elevation / incidence differences)
4. Case C: Large rotation (in-plane orientation)
5. Case D: Scale mismatch (scale divergence)
6. Case E: Partial overlap (limited spatial intersection)
7. Case F: Distractors (non-corresponding geometric features)
8. Case G: Noise and blur (sensor noise and optical degradation)

Outputs formatted ASCII tables:
- Evaluation & Robustness Results
- Ground-Truth Transformation Accuracy
- Failure Taxonomy Distribution
- Hyperparameter Sensitivity Analysis

Saves multi-panel visual artifact to:
data/processed/evaluation/synthetic_evaluation_demo.png

NOTE: Strictly software verification on synthetic data.
Real OHRC/LROC imagery is not yet available in data/raw/.
Do NOT claim real lunar matching or scientific registration accuracy.
"""

import os
import sys
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.evaluation.evaluator import Evaluator
from src.evaluation.synthetic_benchmark import SyntheticBenchmark
from src.evaluation.difficult_cases import (
    build_case_resolution_mismatch,
    build_case_illumination_mismatch,
    build_case_large_rotation,
    build_case_scale_mismatch,
    build_case_partial_overlap,
    build_case_distractors,
    build_case_noise_and_blur,
)
from src.evaluation.evaluation_types import EvaluationCase

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_benchmark():
    logger.info("============================================================")
    logger.info("Milestone C: Evaluation & Robustness Framework Inspection")
    logger.info("SYNTHETIC BENCHMARK & STRESS-CASE VERIFICATION")
    logger.info("============================================================")

    evaluator = Evaluator()

    # 1. Define benchmark suite
    size = 512
    seed = 42

    clean_case = SyntheticBenchmark.generate_combined_case(
        scale=1.1,
        angle_deg=10.0,
        tx=25.0,
        ty=20.0,
        size=size,
        seed=seed,
    )
    clean_case.case_name = "Clean Baseline"

    suite = [
        clean_case,
        build_case_resolution_mismatch(size=size, seed=seed),
        build_case_illumination_mismatch(size=size, seed=seed),
        build_case_large_rotation(size=size, seed=seed),
        build_case_scale_mismatch(size=size, seed=seed),
        build_case_partial_overlap(size=size, seed=seed),
        build_case_distractors(size=size, seed=seed),
        build_case_noise_and_blur(size=size, seed=seed),
    ]

    # 2. Execute benchmark suite
    logger.info("Executing benchmark suite across %d controlled scenarios...", len(suite))
    summary = evaluator.evaluate_dataset(suite)

    # ──────────────────────────────────────────────────────────────────────────
    # TABLE 1: MAIN EVALUATION / ROBUSTNESS TABLE
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 105)
    print("EVALUATION / ROBUSTNESS RESULTS")
    print("=" * 105)
    header = (
        f"{'Case':<23} | {'Matched':<7} | {'Registered':<10} | {'Correct':<7} | "
        f"{'RMSE (px)':<9} | {'Inlier%':<7} | {'Coverage':<8} | {'Conf':<6} | {'Quality':<9} | {'Failure Reason'}"
    )
    print(header)
    print("-" * 105)

    for cr in summary.case_results:
        m = cr.metrics
        matched_str = "YES" if m.matched else "NO"
        reg_str = "YES" if m.registered else "NO"
        corr_str = "PASS" if m.correct_registration else "FAIL"
        rmse_str = f"{m.rmse:.4f}" if m.rmse is not None else "N/A"
        inlier_str = f"{m.inlier_ratio * 100.0:.1f}%"
        cov_str = f"{m.coverage * 100.0:.1f}%"
        conf_str = f"{m.confidence:.2f}"
        reason_str = m.failure_reason if not m.correct_registration else "NONE"

        line = (
            f"{cr.case_name:<23} | {matched_str:<7} | {reg_str:<10} | {corr_str:<7} | "
            f"{rmse_str:<9} | {inlier_str:<7} | {cov_str:<8} | {conf_str:<6} | {m.quality:<9} | {reason_str}"
        )
        print(line)

    print("=" * 105)

    # ──────────────────────────────────────────────────────────────────────────
    # TABLE 2: GROUND-TRUTH ERROR TABLE
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("GROUND-TRUTH TRANSFORMATION ACCURACY")
    print("=" * 80)
    gt_header = (
        f"{'Case':<23} | {'Scale Error':<13} | {'Rel Scale Err':<13} | "
        f"{'Rot Error (deg)':<15} | {'Trans Error (px)':<15}"
    )
    print(gt_header)
    print("-" * 80)

    for cr in summary.case_results:
        m = cr.metrics
        scale_err = f"{m.scale_error:.6f}" if m.scale_error is not None else "N/A"
        rel_scale = f"{m.relative_scale_error * 100.0:.3f}%" if m.relative_scale_error is not None else "N/A"
        rot_err = f"{m.rotation_error_deg:.4f}" if m.rotation_error_deg is not None else "N/A"
        trans_err = f"{m.translation_error_px:.4f}" if m.translation_error_px is not None else "N/A"

        gt_line = f"{cr.case_name:<23} | {scale_err:<13} | {rel_scale:<13} | {rot_err:<15} | {trans_err:<15}"
        print(gt_line)

    print("=" * 80)

    # ──────────────────────────────────────────────────────────────────────────
    # TABLE 3: FAILURE SUMMARY
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FAILURE TAXONOMY BREAKDOWN")
    print("=" * 60)
    print(f"{'Failure Reason / Status':<35} | {'Count':<10} | {'Percentage'}")
    print("-" * 60)
    for reason, count in sorted(summary.failure_distribution.items(), key=lambda x: -x[1]):
        pct = (count / float(summary.total_cases)) * 100.0
        print(f"{reason:<35} | {count:<10} | {pct:.1f}%")
    print("=" * 60)
    print(f"Overall Success Rate (Correct Registrations): {summary.correct_registrations}/{summary.total_cases} ({(summary.correct_registrations/summary.total_cases)*100.0:.1f}%)")

    # ──────────────────────────────────────────────────────────────────────────
    # TABLE 4: PARAMETER SENSITIVITY SUMMARY
    # ──────────────────────────────────────────────────────────────────────────
    logger.info("Executing parameter sensitivity analysis...")
    sensitivity_results = evaluator.evaluate_sensitivity(clean_case)

    print("\n" + "=" * 95)
    print("HIGH-IMPACT HYPERPARAMETER SENSITIVITY")
    print("=" * 95)
    sens_header = (
        f"{'Parameter':<25} | {'Setting':<18} | {'Correct':<7} | "
        f"{'Inlier%':<7} | {'RMSE (px)':<9} | {'Coverage':<8} | {'Conf':<6} | {'Quality'}"
    )
    print(sens_header)
    print("-" * 95)

    for sr in sensitivity_results:
        corr_str = "PASS" if sr["correct"] else "FAIL"
        inlier_str = f"{sr['inlier_ratio'] * 100.0:.1f}%"
        rmse_str = f"{sr['rmse']:.4f}" if sr["rmse"] is not None else "N/A"
        cov_str = f"{sr['coverage'] * 100.0:.1f}%"
        conf_str = f"{sr['confidence']:.2f}"

        line = (
            f"{sr['parameter']:<25} | {sr['setting']:<18} | {corr_str:<7} | "
            f"{inlier_str:<7} | {rmse_str:<9} | {cov_str:<8} | {conf_str:<6} | {sr['quality']}"
        )
        print(line)

    print("=" * 95)

    # ──────────────────────────────────────────────────────────────────────────
    # 5. GENERATE DIAGNOSTIC VISUALIZATION
    # ──────────────────────────────────────────────────────────────────────────
    out_dir = os.path.join(PROJECT_ROOT, "data", "processed", "evaluation")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "synthetic_evaluation_demo.png")

    logger.info("Generating multi-scenario diagnostic visual artifact...")
    # Select 4 representative scenarios: Clean, Resolution Mismatch, Illumination Mismatch, Noise/Blur
    display_cases = [summary.case_results[0], summary.case_results[1], summary.case_results[2], summary.case_results[7]]

    fig, axes = plt.subplots(len(display_cases), 3, figsize=(14, 4.0 * len(display_cases)))
    if len(display_cases) == 1:
        axes = np.array([axes])

    for row_idx, cr in enumerate(display_cases):
        ax_a = axes[row_idx, 0]
        ax_b = axes[row_idx, 1]
        ax_reg = axes[row_idx, 2]

        m = cr.metrics
        status = "PASSED" if m.correct_registration else f"FAILED ({m.failure_reason})"

        # Image A
        ax_a.imshow(clean_case.image_a, cmap="gray")
        ax_a.set_title(f"{cr.case_name}\nReference Image A", fontsize=10, fontweight="bold")
        ax_a.axis("off")

        # Query Image B
        # Fetch corresponding query image from case suite
        query_img = suite[row_idx if row_idx < 3 else 7].image_b
        ax_b.imshow(query_img, cmap="gray")
        ax_b.set_title(f"Query Image B\nStatus: {status}", fontsize=10)
        ax_b.axis("off")

        # Registered Image B
        if cr.registration_result is not None and cr.registration_result.registered_image is not None:
            ax_reg.imshow(cr.registration_result.registered_image, cmap="gray")
            rmse_info = f"RMSE: {m.rmse:.3f} px" if m.rmse is not None else ""
            ax_reg.set_title(f"Registered Image B\n{m.quality} ({rmse_info})", fontsize=10)
        else:
            ax_reg.text(0.5, 0.5, "Registration Failed", ha="center", va="center", color="red", fontsize=12)
            ax_reg.set_title(f"Registered Image B\n{m.failure_reason}", fontsize=10, color="red")
        ax_reg.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Visual diagnostic plot successfully saved to: {out_path}")
    logger.info("Milestone C Evaluation Inspection complete.")


if __name__ == "__main__":
    run_benchmark()
