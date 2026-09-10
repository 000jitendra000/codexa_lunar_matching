"""
tests/test_evaluation.py

Unit and integration tests for Milestone C: Evaluation & Robustness Engine
(Phases 18 + 19).

Covers:
- Ground-truth transformation comparisons and angular wraparound
- Synthetic benchmark generation and deterministic terrain synthesis
- Difficult stress cases suite (all 10 conditions)
- Registration evaluation and standardized FailureReason classification
- RobustnessRunner execution and summary aggregation
- High-impact parameter sensitivity evaluation
- JSON serialization round-tripping (EvaluationMetrics, EvaluationCaseResult, EvaluationSummary)
- End-to-end Evaluator integration
"""

import math
import numpy as np
import pytest

from src.matching.transformation import SimilarityTransform2D
from src.matching.correspondence_fusion import Correspondence
from src.matching.hybrid_matcher import HybridMatchResult
from src.registration.registration_engine import RegistrationResult
from src.evaluation.evaluation_types import (
    FailureReason,
    EvaluationCase,
    EvaluationMetrics,
    EvaluationCaseResult,
    EvaluationSummary,
)
from src.evaluation.ground_truth import (
    compute_ground_truth_errors,
    check_ground_truth_compliance,
)
from src.evaluation.registration_evaluator import (
    evaluate_registration,
)
from src.evaluation.synthetic_benchmark import (
    SyntheticBenchmark,
)
from src.evaluation.difficult_cases import (
    build_difficult_cases_suite,
    build_case_resolution_mismatch,
    build_case_illumination_mismatch,
    build_case_large_rotation,
    build_case_scale_mismatch,
    build_case_partial_overlap,
    build_case_distractors,
    build_case_sparse_craters,
    build_case_dense_craters,
    build_case_texture_poor,
    build_case_noise_and_blur,
)
from src.evaluation.robustness import RobustnessRunner
from src.evaluation.sensitivity import evaluate_parameter_sensitivity
from src.evaluation.evaluator import Evaluator


# ──────────────────────────────────────────────────────────────────────────────
# 1. Ground Truth Metrics Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_gt_identity_comparison():
    """Identical transforms must produce zero errors across all metrics."""
    t1 = SimilarityTransform2D(scale=1.0, rotation_rad=0.0, translation_x=0.0, translation_y=0.0)
    t2 = SimilarityTransform2D(scale=1.0, rotation_rad=0.0, translation_x=0.0, translation_y=0.0)

    errs = compute_ground_truth_errors(t1, t2)
    assert errs["scale_error"] == 0.0
    assert errs["relative_scale_error"] == 0.0
    assert errs["rotation_error_deg"] == 0.0
    assert errs["translation_error_px"] == 0.0


def test_gt_scale_error():
    """Scale error should accurately report absolute and relative discrepancy."""
    t1 = SimilarityTransform2D(scale=1.0, rotation_rad=0.0, translation_x=0.0, translation_y=0.0)
    t2 = SimilarityTransform2D(scale=1.25, rotation_rad=0.0, translation_x=0.0, translation_y=0.0)

    errs = compute_ground_truth_errors(t1, t2)
    assert pytest.approx(errs["scale_error"], abs=1e-6) == 0.25
    assert pytest.approx(errs["relative_scale_error"], abs=1e-6) == 0.25


def test_gt_rotation_error():
    """Rotation error should compute shortest angular distance in degrees."""
    t1 = SimilarityTransform2D(scale=1.0, rotation_rad=math.radians(10.0), translation_x=0.0, translation_y=0.0)
    t2 = SimilarityTransform2D(scale=1.0, rotation_rad=math.radians(25.0), translation_x=0.0, translation_y=0.0)

    errs = compute_ground_truth_errors(t1, t2)
    assert pytest.approx(errs["rotation_error_deg"], abs=1e-4) == 15.0


def test_gt_angular_wraparound():
    """Crucial: Angles of +179 deg and -179 deg have an error of 2 deg, not 358 deg."""
    t1 = SimilarityTransform2D(scale=1.0, rotation_rad=math.radians(179.0), translation_x=0.0, translation_y=0.0)
    t2 = SimilarityTransform2D(scale=1.0, rotation_rad=math.radians(-179.0), translation_x=0.0, translation_y=0.0)

    errs = compute_ground_truth_errors(t1, t2)
    assert pytest.approx(errs["rotation_error_deg"], abs=1e-4) == 2.0


def test_gt_translation_error():
    """Translation error should be standard 2D Euclidean distance."""
    t1 = SimilarityTransform2D(scale=1.0, rotation_rad=0.0, translation_x=10.0, translation_y=20.0)
    t2 = SimilarityTransform2D(scale=1.0, rotation_rad=0.0, translation_x=13.0, translation_y=24.0)

    errs = compute_ground_truth_errors(t1, t2)
    assert pytest.approx(errs["translation_error_px"], abs=1e-6) == 5.0  # 3-4-5 right triangle


def test_gt_combined_error():
    """Simultaneous discrepancy across scale, rotation, and translation."""
    t1 = SimilarityTransform2D(scale=1.0, rotation_rad=0.0, translation_x=0.0, translation_y=0.0)
    t2 = SimilarityTransform2D(
        scale=1.1,
        rotation_rad=math.radians(10.0),
        translation_x=3.0,
        translation_y=4.0,
    )

    errs = compute_ground_truth_errors(t1, t2)
    assert pytest.approx(errs["scale_error"], abs=1e-4) == 0.1
    assert pytest.approx(errs["rotation_error_deg"], abs=1e-4) == 10.0
    assert pytest.approx(errs["translation_error_px"], abs=1e-4) == 5.0


def test_gt_compliance_passes_within_tolerances():
    """check_ground_truth_compliance returns True when within tolerances."""
    t_gt = SimilarityTransform2D(scale=1.0, rotation_rad=0.0, translation_x=0.0, translation_y=0.0)
    t_est = SimilarityTransform2D(scale=1.01, rotation_rad=math.radians(0.5), translation_x=1.0, translation_y=1.0)

    ok, reason, _ = check_ground_truth_compliance(t_gt, t_est)
    assert ok is True
    assert reason is None


def test_gt_compliance_fails_when_exceeding_tolerances():
    """check_ground_truth_compliance flags TRANSFORM_ERROR_EXCEEDED when breached."""
    t_gt = SimilarityTransform2D(scale=1.0, rotation_rad=0.0, translation_x=0.0, translation_y=0.0)
    t_est = SimilarityTransform2D(scale=1.20, rotation_rad=math.radians(10.0), translation_x=20.0, translation_y=20.0)

    ok, reason, errs = check_ground_truth_compliance(t_gt, t_est)
    assert ok is False
    assert reason == FailureReason.TRANSFORM_ERROR_EXCEEDED
    assert errs["scale_error"] > 0.05


# ──────────────────────────────────────────────────────────────────────────────
# 2. Synthetic Benchmark Generator Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_synthetic_benchmark_deterministic():
    """Same seed must produce identical pixel arrays and crater centers."""
    img1, craters1 = SyntheticBenchmark.generate_base_terrain(size=256, seed=42)
    img2, craters2 = SyntheticBenchmark.generate_base_terrain(size=256, seed=42)

    np.testing.assert_array_equal(img1, img2)
    assert craters1 == craters2


def test_synthetic_benchmark_density_modes():
    """Different density modes produce appropriate crater populations."""
    _, sparse = SyntheticBenchmark.generate_base_terrain(size=256, seed=42, density_mode="sparse")
    _, normal = SyntheticBenchmark.generate_base_terrain(size=256, seed=42, density_mode="normal")
    _, dense = SyntheticBenchmark.generate_base_terrain(size=256, seed=42, density_mode="dense")
    _, flat = SyntheticBenchmark.generate_base_terrain(size=256, seed=42, density_mode="texture_poor")

    assert len(sparse) == 2
    assert len(normal) == 12
    assert len(dense) >= 30
    assert len(flat) == 5


def test_warp_image_forward_shape_and_dtype():
    """Warping forward must preserve image dimensions and uint8 dtype."""
    img, _ = SyntheticBenchmark.generate_base_terrain(size=256, seed=42)
    t_fwd = SimilarityTransform2D(scale=1.2, rotation_rad=math.radians(15.0), translation_x=10.0, translation_y=-5.0)

    warped = SyntheticBenchmark.warp_image_forward(img, t_fwd, output_shape=(256, 256))
    assert warped.shape == (256, 256)
    assert warped.dtype == np.uint8


def test_benchmark_canonical_generators():
    """Verify standard canonical benchmark case generators instantiate correctly."""
    c_ident = SyntheticBenchmark.generate_identity_case(size=256, seed=1)
    c_trans = SyntheticBenchmark.generate_translation_case(tx=20.0, ty=10.0, size=256, seed=1)
    c_rot = SyntheticBenchmark.generate_rotation_case(angle_deg=15.0, size=256, seed=1)
    c_scale = SyntheticBenchmark.generate_scale_case(scale=1.1, size=256, seed=1)
    c_comb = SyntheticBenchmark.generate_combined_case(scale=1.1, angle_deg=10.0, tx=15.0, ty=15.0, size=256, seed=1)

    assert c_ident.case_name == "Identity"
    assert c_ident.ground_truth.scale == 1.0

    assert c_trans.case_name == "Pure Translation"
    assert c_trans.ground_truth.translation == (20.0, 10.0)

    assert c_rot.case_name == "Pure Rotation"
    assert pytest.approx(c_rot.ground_truth.rotation_deg, abs=1e-4) == 15.0

    assert c_scale.case_name == "Pure Scale"
    assert c_scale.ground_truth.scale == 1.1

    assert c_comb.case_name == "Combined Similarity"
    assert c_comb.ground_truth.scale == 1.1


def test_apply_perturbations_properties():
    """Perturbations must alter values within valid [0, 255] uint8 range."""
    img, _ = SyntheticBenchmark.generate_base_terrain(size=256, seed=42)
    perturbed = SyntheticBenchmark.apply_perturbations(
        img,
        noise_sigma=10.0,
        contrast_scale=0.7,
        brightness_shift=20.0,
        gamma=1.3,
        blur_ksize=3,
        downsample_factor=2.0,
        seed=42,
    )
    assert perturbed.shape == img.shape
    assert perturbed.dtype == np.uint8
    assert not np.array_equal(img, perturbed)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Difficult Stress Cases Suite Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_difficult_cases_suite_contains_all_ten_cases():
    """Verify that build_difficult_cases_suite constructs all 10 required conditions."""
    suite = build_difficult_cases_suite(size=256, seed=42)
    assert len(suite) == 10
    names = [c.case_name for c in suite]

    expected = [
        "Resolution Mismatch",
        "Illumination Mismatch",
        "Large Rotation",
        "Scale Mismatch",
        "Partial Overlap",
        "Distractors",
        "Sparse Craters",
        "Dense Craters",
        "Texture Poor",
        "Noise and Blur",
    ]
    assert names == expected


def test_case_resolution_mismatch_properties():
    case = build_case_resolution_mismatch(size=256, seed=42)
    assert case.parameters["downsample_factor"] == 2.5
    assert case.ground_truth is not None


def test_case_illumination_mismatch_properties():
    case = build_case_illumination_mismatch(size=256, seed=42)
    assert case.parameters["contrast_scale"] == 0.6
    assert case.parameters["gamma"] == 1.5


def test_case_large_rotation_properties():
    case = build_case_large_rotation(size=256, seed=42)
    assert pytest.approx(case.parameters["rotation_deg"], abs=1e-4) == 90.0


def test_case_scale_mismatch_properties():
    case = build_case_scale_mismatch(size=256, seed=42)
    assert case.parameters["scale"] == 1.65


def test_case_partial_overlap_properties():
    case = build_case_partial_overlap(size=256, seed=42)
    assert case.parameters["translation"] == (180.0, 160.0)


def test_case_distractors_properties():
    case = build_case_distractors(size=256, seed=42)
    assert case.parameters["num_distractors"] == 4


def test_case_sparse_craters_properties():
    case = build_case_sparse_craters(size=256, seed=42)
    assert case.parameters["density_mode"] == "sparse"


def test_case_dense_craters_properties():
    case = build_case_dense_craters(size=256, seed=42)
    assert case.parameters["density_mode"] == "dense"


def test_case_texture_poor_properties():
    case = build_case_texture_poor(size=256, seed=42)
    assert case.parameters["density_mode"] == "texture_poor"


def test_case_noise_and_blur_properties():
    case = build_case_noise_and_blur(size=256, seed=42)
    assert case.parameters["noise_sigma"] == 15.0
    assert case.parameters["blur_ksize"] == 5


# ──────────────────────────────────────────────────────────────────────────────
# 4. Registration Evaluation & Failure Classification Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_evaluate_registration_unmatched():
    """Unmatched hybrid result must classify as NO_VALID_MATCHES or INSUFFICIENT_INLIERS."""
    hybrid_res = HybridMatchResult(
        matched=False,
        correspondences=[],
        inlier_indices=[],
        outlier_indices=[],
        transform=None,
        confidence=0.0,
        rmse=None,
        inlier_ratio=0.0,
    )
    reg_res = RegistrationResult(
        success=False,
        registered_image=None,
        valid_mask=None,
        transform=None,
        selected_tie_points=[],
        num_inliers=0,
        num_tie_points=0,
        rmse=None,
        mean_error=None,
        median_error=None,
        max_error=None,
        coverage=0.0,
        confidence=0.0,
        quality="FAILED",
    )

    metrics = evaluate_registration(reg_res, hybrid_res)
    assert metrics.matched is False
    assert metrics.registered is False
    assert metrics.correct_registration is False
    assert metrics.failure_reason == FailureReason.NO_VALID_MATCHES


def test_evaluate_registration_low_coverage():
    """Registration with inadequate spatial coverage must be classified as LOW_SPATIAL_COVERAGE."""
    reg_res = RegistrationResult(
        success=False,
        registered_image=None,
        valid_mask=None,
        transform=None,
        selected_tie_points=[],
        num_inliers=5,
        num_tie_points=3,
        rmse=1.0,
        mean_error=0.8,
        median_error=0.8,
        max_error=1.2,
        coverage=0.05,  # Below min_coverage (0.15)
        confidence=0.8,
        quality="FAILED",
    )
    hybrid_res = HybridMatchResult(
        matched=True,
        correspondences=[Correspondence(point_a=np.array([0.0, 0.0]), point_b=np.array([0.0, 0.0]), confidence=0.8, source="crater", source_index=i) for i in range(5)],
        inlier_indices=[0, 1, 2, 3, 4],
        outlier_indices=[],
        transform=SimilarityTransform2D(1.0, 0.0, 0.0, 0.0),
        confidence=0.8,
        rmse=1.0,
        inlier_ratio=1.0,
    )

    metrics = evaluate_registration(reg_res, hybrid_res)
    assert metrics.correct_registration is False
    assert metrics.failure_reason == FailureReason.LOW_SPATIAL_COVERAGE


def test_evaluate_registration_gt_exceeded():
    """Successful registration exceeding ground-truth tolerance must flag TRANSFORM_ERROR_EXCEEDED."""
    t_gt = SimilarityTransform2D(1.0, 0.0, 0.0, 0.0)
    t_est = SimilarityTransform2D(1.25, 0.0, 0.0, 0.0)  # Scale error 0.25 > 0.05

    reg_res = RegistrationResult(
        success=True,
        registered_image=np.zeros((100, 100), dtype=np.uint8),
        valid_mask=np.ones((100, 100), dtype=np.uint8) * 255,
        transform=t_est,
        selected_tie_points=[],
        num_inliers=10,
        num_tie_points=8,
        rmse=0.5,
        mean_error=0.4,
        median_error=0.4,
        max_error=0.8,
        coverage=0.5,
        confidence=0.9,
        quality="EXCELLENT",
    )

    metrics = evaluate_registration(reg_res, ground_truth=t_gt)
    assert metrics.registered is True
    assert metrics.correct_registration is False
    assert metrics.failure_reason == FailureReason.TRANSFORM_ERROR_EXCEEDED


def test_evaluate_registration_blind_high_quality():
    """Blind (no ground truth) pair with excellent metrics must be evaluated as correct."""
    t_est = SimilarityTransform2D(1.0, 0.0, 0.0, 0.0)
    reg_res = RegistrationResult(
        success=True,
        registered_image=np.zeros((100, 100), dtype=np.uint8),
        valid_mask=np.ones((100, 100), dtype=np.uint8) * 255,
        transform=t_est,
        selected_tie_points=[],
        num_inliers=10,
        num_tie_points=8,
        rmse=0.8,
        mean_error=0.6,
        median_error=0.6,
        max_error=1.2,
        coverage=0.4,
        confidence=0.88,
        quality="EXCELLENT",
    )

    metrics = evaluate_registration(reg_res, ground_truth=None)
    assert metrics.registered is True
    assert metrics.correct_registration is True
    assert metrics.failure_reason == FailureReason.NONE


# ──────────────────────────────────────────────────────────────────────────────
# 5. Robustness Runner & Summary Aggregation Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_robustness_runner_invalid_input():
    """Invalid or empty images must gracefully produce INVALID_INPUT result."""
    runner = RobustnessRunner()
    case = EvaluationCase(
        case_name="Corrupt Input",
        image_a=np.array([], dtype=np.uint8),
        image_b=np.zeros((100, 100), dtype=np.uint8),
    )
    matcher = pytest.importorskip("src.matching.hybrid_matcher").HybridMatcher()
    engine = pytest.importorskip("src.registration.registration_engine").RegistrationEngine()

    result = runner.run_case(case, matcher, engine)
    assert result.metrics.matched is False
    assert result.metrics.failure_reason == FailureReason.INVALID_INPUT


def test_robustness_runner_aggregates_summary():
    """Aggregate summary properly computes distribution counts and averages."""
    c1 = EvaluationCaseResult(
        case_name="Case 1",
        metrics=EvaluationMetrics(
            matched=True,
            registered=True,
            correct_registration=True,
            rmse=0.5,
            inlier_ratio=0.8,
            coverage=0.4,
            confidence=0.9,
            quality="EXCELLENT",
            scale_error=0.01,
            rotation_error_deg=0.5,
            translation_error_px=1.0,
            failure_reason=FailureReason.NONE,
        ),
    )
    c2 = EvaluationCaseResult(
        case_name="Case 2",
        metrics=EvaluationMetrics(
            matched=False,
            registered=False,
            correct_registration=False,
            rmse=None,
            inlier_ratio=0.0,
            coverage=0.0,
            confidence=0.0,
            quality="FAILED",
            failure_reason=FailureReason.NO_VALID_MATCHES,
        ),
    )

    summary = RobustnessRunner.aggregate_summary([c1, c2])
    assert summary.total_cases == 2
    assert summary.successful_matches == 1
    assert summary.successful_registrations == 1
    assert summary.correct_registrations == 1
    assert summary.failure_count == 1
    assert pytest.approx(summary.mean_rmse, abs=1e-4) == 0.5
    assert summary.quality_distribution["EXCELLENT"] == 1
    assert summary.quality_distribution["FAILED"] == 1
    assert summary.failure_distribution[FailureReason.NONE] == 1
    assert summary.failure_distribution[FailureReason.NO_VALID_MATCHES] == 1


def test_parameter_sensitivity_executes_all_variations():
    """evaluate_parameter_sensitivity probes high-impact parameters across expected variations."""
    case = SyntheticBenchmark.generate_combined_case(scale=1.05, angle_deg=5.0, tx=10.0, ty=10.0, size=256)
    results = evaluate_parameter_sensitivity(case)

    # 3 RANSAC + 3 LoFTR + 3 Dup + 3 TiePoints + 2 Subpixel = 14 probes
    assert len(results) == 14
    param_names = set(r["parameter"] for r in results)
    assert "RANSAC Threshold (px)" in param_names
    assert "LoFTR Min Confidence" in param_names
    assert "Duplicate Tolerance (px)" in param_names
    assert "Tie-Point Max Count" in param_names
    assert "Sub-Pixel Refinement" in param_names


# ──────────────────────────────────────────────────────────────────────────────
# 6. Serialization Round-Trip Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_evaluation_metrics_round_trip():
    """EvaluationMetrics serializes and deserializes losslessly."""
    m = EvaluationMetrics(
        num_candidates=25,
        num_inliers=18,
        inlier_ratio=0.72,
        scale_error=0.002,
        relative_scale_error=0.002,
        rotation_error_deg=0.15,
        translation_error_px=0.45,
        rmse=0.25,
        mean_error=0.2,
        median_error=0.2,
        max_error=0.5,
        coverage=0.38,
        confidence=0.91,
        quality="EXCELLENT",
        num_tie_points=12,
        valid_overlap_pixels=85000,
        matched=True,
        registered=True,
        correct_registration=True,
        failure_reason=FailureReason.NONE,
    )
    d = m.to_dict()
    m2 = EvaluationMetrics.from_dict(d)

    assert m2.matched == m.matched
    assert m2.correct_registration == m.correct_registration
    assert m2.quality == m.quality
    assert pytest.approx(m2.rmse, abs=1e-4) == m.rmse
    assert pytest.approx(m2.scale_error, abs=1e-6) == m.scale_error


def test_evaluation_case_result_round_trip():
    """EvaluationCaseResult round-trip serialization."""
    m = EvaluationMetrics(matched=True, registered=True, correct_registration=True, quality="GOOD")
    res = EvaluationCaseResult(
        case_name="Test Case",
        parameters={"scale": 1.1},
        expected_behavior="Accurate alignment",
        metrics=m,
        metadata={"mission": "synthetic"},
    )
    d = res.to_dict()
    res2 = EvaluationCaseResult.from_dict(d)

    assert res2.case_name == "Test Case"
    assert res2.parameters == {"scale": 1.1}
    assert res2.metrics.quality == "GOOD"
    assert res2.metrics.matched is True


def test_evaluation_summary_round_trip():
    """EvaluationSummary round-trip serialization."""
    m = EvaluationMetrics(matched=True, registered=True, correct_registration=True, quality="EXCELLENT", rmse=0.3)
    c = EvaluationCaseResult(case_name="C1", metrics=m)
    summary = RobustnessRunner.aggregate_summary([c])

    d = summary.to_dict()
    s2 = EvaluationSummary.from_dict(d)

    assert s2.total_cases == 1
    assert s2.successful_matches == 1
    assert s2.correct_registrations == 1
    assert pytest.approx(s2.mean_rmse, abs=1e-4) == 0.3
    assert len(s2.case_results) == 1


# ──────────────────────────────────────────────────────────────────────────────
# 7. High-Level Evaluator End-to-End Integration Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_evaluator_evaluate_case_clean_synthetic():
    """End-to-end integration: Evaluator correctly evaluates a clean synthetic pair."""
    evaluator = Evaluator()
    case = SyntheticBenchmark.generate_combined_case(
        scale=1.1,
        angle_deg=12.0,
        tx=25.0,
        ty=20.0,
        size=256,
        seed=101,
    )

    result = evaluator.evaluate_case(
        image_a=case.image_a,
        image_b=case.image_b,
        ground_truth=case.ground_truth,
        case_name=case.case_name,
        parameters=case.parameters,
        expected_behavior=case.expected_behavior,
    )

    assert result.metrics.matched is True
    assert result.metrics.registered is True
    assert result.metrics.correct_registration is True
    assert result.metrics.failure_reason == FailureReason.NONE
    assert result.metrics.scale_error is not None
    assert result.metrics.scale_error < 0.05
    assert result.metrics.rotation_error_deg < 3.0
    assert result.metrics.translation_error_px < 5.0
