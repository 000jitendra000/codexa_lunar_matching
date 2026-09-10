"""
src/evaluation/robustness.py

Robustness Runner executing evaluation suites across the end-to-end
matching & registration pipeline for Milestone C (Phases 18 + 19).

ARCHITECTURAL RULES:
- Strictly model-side Python.
- Coordinates HybridMatcher and RegistrationEngine cleanly.
- Catches runtime exceptions gracefully without terminating the batch runner,
  logging them accurately under FailureReason.UNKNOWN_FAILURE or INVALID_INPUT.
- Deterministic execution with aggregation into EvaluationSummary.
"""

import logging
import traceback
from typing import Dict, Any, List, Optional
import numpy as np

from src.matching.hybrid_matcher import HybridMatcher, HybridMatchResult
from src.registration.registration_engine import RegistrationEngine, RegistrationResult
from src.evaluation.evaluation_types import (
    EvaluationCase,
    EvaluationCaseResult,
    EvaluationSummary,
    EvaluationMetrics,
    FailureReason,
)
from src.evaluation.registration_evaluator import evaluate_registration

logger = logging.getLogger(__name__)


class RobustnessRunner:
    """
    Executes controlled evaluation suites and aggregates performance diagnostics.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config: Dict[str, Any] = {}
        if config is not None:
            self.config.update(config)

        # Extract evaluation settings
        try:
            from configs.default import Config
            default_eval = getattr(Config, "EVALUATION", {})
        except (ImportError, AttributeError):
            default_eval = {}

        self.eval_cfg = dict(default_eval)
        self.eval_cfg.update(self.config.get("evaluation", {}))

        self.tolerances = {
            "max_scale_error": self.eval_cfg.get("max_scale_error", 0.05),
            "max_rotation_error_deg": self.eval_cfg.get("max_rotation_error_deg", 3.0),
            "max_translation_error_px": self.eval_cfg.get("max_translation_error_px", 5.0),
            "max_rmse_px": self.eval_cfg.get("max_rmse_px", 3.0),
            "min_coverage": self.eval_cfg.get("min_coverage", 0.15),
            "min_confidence": self.eval_cfg.get("min_confidence", 0.5),
        }

    def run_case(
        self,
        case: EvaluationCase,
        matcher: HybridMatcher,
        registration_engine: RegistrationEngine,
    ) -> EvaluationCaseResult:
        """
        Execute the full pipeline on a single evaluation case.

        Args:
            case: EvaluationCase containing images and optional ground truth.
            matcher: Instantiated HybridMatcher.
            registration_engine: Instantiated RegistrationEngine.

        Returns:
            EvaluationCaseResult with metrics and failure analysis.
        """
        logger.info(f"Evaluating case: '{case.case_name}'")

        # Validate input images
        if (
            not isinstance(case.image_a, np.ndarray)
            or not isinstance(case.image_b, np.ndarray)
            or case.image_a.size == 0
            or case.image_b.size == 0
        ):
            metrics = EvaluationMetrics(
                matched=False,
                registered=False,
                correct_registration=False,
                failure_reason=FailureReason.INVALID_INPUT,
            )
            return EvaluationCaseResult(
                case_name=case.case_name,
                parameters=case.parameters,
                expected_behavior=case.expected_behavior,
                metrics=metrics,
                metadata={"error": "Input image array empty or invalid type."},
            )

        hybrid_res: Optional[HybridMatchResult] = None
        reg_res: Optional[RegistrationResult] = None

        try:
            # 1. Matching Stage
            hybrid_res = matcher.match(case.image_a, case.image_b)

            # 2. Registration Stage
            reg_res = registration_engine.register(case.image_a, case.image_b, hybrid_res)

            # 3. Evaluation & Failure Classification
            metrics = evaluate_registration(
                registration_result=reg_res,
                hybrid_result=hybrid_res,
                ground_truth=case.ground_truth,
                tolerances=self.tolerances,
            )

            return EvaluationCaseResult(
                case_name=case.case_name,
                parameters=case.parameters,
                expected_behavior=case.expected_behavior,
                metrics=metrics,
                metadata=case.metadata,
                hybrid_result=hybrid_res,
                registration_result=reg_res,
            )

        except Exception as exc:
            tb = traceback.format_exc()
            logger.error(f"Unhandled exception during evaluation of case '{case.case_name}': {exc}\n{tb}")
            metrics = EvaluationMetrics(
                matched=False,
                registered=False,
                correct_registration=False,
                failure_reason=FailureReason.UNKNOWN_FAILURE,
            )
            return EvaluationCaseResult(
                case_name=case.case_name,
                parameters=case.parameters,
                expected_behavior=case.expected_behavior,
                metrics=metrics,
                metadata={"exception": str(exc), "traceback": tb},
                hybrid_result=hybrid_res,
                registration_result=reg_res,
            )

    def run_suite(
        self,
        cases: List[EvaluationCase],
        matcher: Optional[HybridMatcher] = None,
        registration_engine: Optional[RegistrationEngine] = None,
    ) -> EvaluationSummary:
        """
        Execute evaluation across an entire suite of test cases and aggregate results.

        Args:
            cases: List of EvaluationCase instances.
            matcher: Optional HybridMatcher (creates default if None).
            registration_engine: Optional RegistrationEngine (creates default if None).

        Returns:
            EvaluationSummary dataclass.
        """
        if matcher is None:
            matcher = HybridMatcher(self.config)
        if registration_engine is None:
            registration_engine = RegistrationEngine(self.config)

        case_results: List[EvaluationCaseResult] = []

        for case in cases:
            res = self.run_case(case, matcher, registration_engine)
            case_results.append(res)

        return self.aggregate_summary(case_results)

    @staticmethod
    def aggregate_summary(case_results: List[EvaluationCaseResult]) -> EvaluationSummary:
        """
        Aggregate a collection of case results into an EvaluationSummary.
        """
        total = len(case_results)
        if total == 0:
            return EvaluationSummary()

        matches = sum(1 for c in case_results if c.metrics.matched)
        regs = sum(1 for c in case_results if c.metrics.registered)
        corrects = sum(1 for c in case_results if c.metrics.correct_registration)
        failures = total - corrects

        # Filter valid numerical metrics
        rmses = [c.metrics.rmse for c in case_results if c.metrics.rmse is not None]
        inlier_ratios = [c.metrics.inlier_ratio for c in case_results]
        coverages = [c.metrics.coverage for c in case_results]
        confidences = [c.metrics.confidence for c in case_results]

        scale_errs = [c.metrics.scale_error for c in case_results if c.metrics.scale_error is not None]
        rot_errs = [c.metrics.rotation_error_deg for c in case_results if c.metrics.rotation_error_deg is not None]
        trans_errs = [c.metrics.translation_error_px for c in case_results if c.metrics.translation_error_px is not None]

        # Distributions
        quality_dist: Dict[str, int] = {}
        failure_dist: Dict[str, int] = {}

        for c in case_results:
            q = c.metrics.quality
            quality_dist[q] = quality_dist.get(q, 0) + 1

            r = c.metrics.failure_reason
            failure_dist[r] = failure_dist.get(r, 0) + 1

        return EvaluationSummary(
            total_cases=total,
            successful_matches=matches,
            successful_registrations=regs,
            correct_registrations=corrects,
            failure_count=failures,
            mean_rmse=float(np.mean(rmses)) if rmses else None,
            median_rmse=float(np.median(rmses)) if rmses else None,
            mean_inlier_ratio=float(np.mean(inlier_ratios)) if inlier_ratios else 0.0,
            mean_coverage=float(np.mean(coverages)) if coverages else 0.0,
            mean_confidence=float(np.mean(confidences)) if confidences else 0.0,
            quality_distribution=quality_dist,
            failure_distribution=failure_dist,
            mean_scale_error=float(np.mean(scale_errs)) if scale_errs else None,
            median_scale_error=float(np.median(scale_errs)) if scale_errs else None,
            mean_rotation_error=float(np.mean(rot_errs)) if rot_errs else None,
            median_rotation_error=float(np.median(rot_errs)) if rot_errs else None,
            mean_translation_error=float(np.mean(trans_errs)) if trans_errs else None,
            median_translation_error=float(np.median(trans_errs)) if trans_errs else None,
            case_results=case_results,
        )
