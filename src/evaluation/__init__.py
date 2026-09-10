"""
src/evaluation/__init__.py

Milestone C (Phases 18 + 19) — Evaluation & Robustness Engine.

Public Exports:
- FailureReason: Standard failure taxonomy constants.
- EvaluationCase: Encapsulates image pairs, ground truth, and sensor metadata.
- EvaluationMetrics: Structured matching, transformation, and registration quality metrics.
- EvaluationCaseResult: Result container for an individual evaluated case.
- EvaluationSummary: Aggregated benchmark dataset evaluation report.
- compute_ground_truth_errors: Canonical transformation discrepancy calculation.
- check_ground_truth_compliance: Ground-truth tolerance compliance verification.
- evaluate_registration: Core accuracy, residual, and failure classifier.
- SyntheticBenchmark: Generator for controlled synthetic registration benchmarks.
- build_difficult_cases_suite: 10-case stress suite builder.
- RobustnessRunner: Test suite orchestrator.
- evaluate_parameter_sensitivity: Hyperparameter sensitivity analyzer.
- Evaluator: Unified high-level evaluation engine.
"""

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
from src.evaluation.robustness import (
    RobustnessRunner,
)
from src.evaluation.sensitivity import (
    evaluate_parameter_sensitivity,
)
from src.evaluation.evaluator import (
    Evaluator,
)

__all__ = [
    "FailureReason",
    "EvaluationCase",
    "EvaluationMetrics",
    "EvaluationCaseResult",
    "EvaluationSummary",
    "compute_ground_truth_errors",
    "check_ground_truth_compliance",
    "evaluate_registration",
    "SyntheticBenchmark",
    "build_difficult_cases_suite",
    "build_case_resolution_mismatch",
    "build_case_illumination_mismatch",
    "build_case_large_rotation",
    "build_case_scale_mismatch",
    "build_case_partial_overlap",
    "build_case_distractors",
    "build_case_sparse_craters",
    "build_case_dense_craters",
    "build_case_texture_poor",
    "build_case_noise_and_blur",
    "RobustnessRunner",
    "evaluate_parameter_sensitivity",
    "Evaluator",
]
