"""
src/evaluation/evaluator.py

High-level Evaluator orchestrator providing a unified API for Milestone C
(Phases 18 + 19) Evaluation & Robustness Engine.

API Example:
    evaluator = Evaluator(config)
    result = evaluator.evaluate_case(image_a, image_b, ground_truth=t_gt)
    summary = evaluator.evaluate_dataset(cases)

ARCHITECTURAL RULES:
- High-level Python facade coordinating matcher, registration, and evaluation.
- Cleanly decoupled from any UI/frontend framework.
- Fully compatible with future real lunar imagery metadata.
"""

from typing import Dict, Any, List, Optional
import numpy as np

from src.matching.transformation import SimilarityTransform2D
from src.matching.hybrid_matcher import HybridMatcher
from src.registration.registration_engine import RegistrationEngine
from src.evaluation.evaluation_types import EvaluationCase, EvaluationCaseResult, EvaluationSummary
from src.evaluation.robustness import RobustnessRunner
from src.evaluation.difficult_cases import build_difficult_cases_suite
from src.evaluation.synthetic_benchmark import SyntheticBenchmark
from src.evaluation.sensitivity import evaluate_parameter_sensitivity


class Evaluator:
    """
    High-level evaluation facade for the Cross-Sensor Lunar Location Matching pipeline.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config: Dict[str, Any] = {}
        if config is not None:
            self.config.update(config)

        self.matcher = HybridMatcher(self.config)
        self.registration_engine = RegistrationEngine(self.config)
        self.runner = RobustnessRunner(self.config)

    def evaluate_case(
        self,
        image_a: np.ndarray,
        image_b: np.ndarray,
        ground_truth: Optional[SimilarityTransform2D] = None,
        case_name: str = "custom_case",
        parameters: Optional[Dict[str, Any]] = None,
        expected_behavior: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EvaluationCaseResult:
        """
        Evaluate a single image pair.

        Args:
            image_a: Reference Image A.
            image_b: Query Image B.
            ground_truth: Optional known SimilarityTransform2D (A -> B).
            case_name: Human-readable identifier for reporting.
            parameters: Optional parameter dictionary.
            expected_behavior: Expected qualitative/quantitative outcome.
            metadata: Optional sensor/mission provenance (e.g. mission_a="Chandrayaan-2").

        Returns:
            EvaluationCaseResult containing all metrics, intermediate outputs, and failure classification.
        """
        case = EvaluationCase(
            case_name=case_name,
            image_a=image_a,
            image_b=image_b,
            ground_truth=ground_truth,
            parameters=parameters or {},
            expected_behavior=expected_behavior,
            metadata=metadata or {},
        )
        return self.runner.run_case(case, self.matcher, self.registration_engine)

    def evaluate_dataset(self, cases: List[EvaluationCase]) -> EvaluationSummary:
        """
        Evaluate a list of EvaluationCase instances and return an aggregated summary.

        Args:
            cases: List of EvaluationCase instances.

        Returns:
            EvaluationSummary dataclass.
        """
        return self.runner.run_suite(cases, self.matcher, self.registration_engine)

    def evaluate_difficult_cases(self, size: int = 512, seed: int = 42) -> EvaluationSummary:
        """
        Build and evaluate the full 10-case difficult stress suite.

        Returns:
            EvaluationSummary dataclass covering all 10 stress scenarios.
        """
        cases = build_difficult_cases_suite(size=size, seed=seed)
        return self.evaluate_dataset(cases)

    def evaluate_sensitivity(self, case: Optional[EvaluationCase] = None) -> List[Dict[str, Any]]:
        """
        Run high-impact parameter sensitivity analysis.

        Args:
            case: Optional EvaluationCase (defaults to combined similarity case).

        Returns:
            List of dictionaries detailing parameter variations and performance.
        """
        if case is None:
            case = SyntheticBenchmark.generate_combined_case(scale=1.15, angle_deg=20.0, tx=40.0, ty=30.0)
        return evaluate_parameter_sensitivity(case, self.config)
