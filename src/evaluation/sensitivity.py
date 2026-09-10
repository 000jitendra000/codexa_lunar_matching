"""
src/evaluation/sensitivity.py

Lightweight parameter-sensitivity evaluation for Milestone C (Phase 18).

Evaluates high-impact hyperparameters:
- RANSAC reprojection threshold (1.5, 3.0 [base], 5.0 px)
- LoFTR minimum confidence (0.1, 0.2 [base], 0.4)
- Correspondence duplicate tolerance (2.0, 4.0 [base], 8.0 px)
- Maximum uniform tie points (10, 30 [base], 50)
- Sub-pixel refinement (Disabled, Enabled [base])

ARCHITECTURAL RULES:
- Strictly model-side Python.
- Does NOT perform massive brute-force grid searches.
- Measures stability: does a slight parameter perturbation trigger pipeline failure?
- Reports findings without automatically modifying project defaults.
"""

from typing import Dict, Any, List, Optional
import copy

from src.matching.hybrid_matcher import HybridMatcher
from src.registration.registration_engine import RegistrationEngine
from src.evaluation.evaluation_types import EvaluationCase, EvaluationCaseResult
from src.evaluation.robustness import RobustnessRunner


def evaluate_parameter_sensitivity(
    base_case: EvaluationCase,
    base_config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Evaluate pipeline sensitivity across high-impact parameter perturbations.

    Args:
        base_case: Reference EvaluationCase to evaluate under different configurations.
        base_config: Optional baseline configuration dictionary.

    Returns:
        List of result dictionaries summarizing parameter variation and performance.
    """
    runner = RobustnessRunner(base_config)

    # Define high-impact parameters to probe
    # Structure: (parameter_label, config_path, [lower_val, baseline_val, higher_val])
    variations = [
        {
            "parameter": "RANSAC Threshold (px)",
            "section": "hybrid_matching",
            "key": "reprojection_threshold",
            "values": [1.5, 3.0, 5.0],
            "labels": ["Lower (1.5 px)", "Baseline (3.0 px)", "Higher (5.0 px)"],
        },
        {
            "parameter": "LoFTR Min Confidence",
            "section": "learned_matching",
            "key": "min_confidence",
            "values": [0.1, 0.2, 0.4],
            "labels": ["Lower (0.1)", "Baseline (0.2)", "Higher (0.4)"],
        },
        {
            "parameter": "Duplicate Tolerance (px)",
            "section": "correspondence_fusion",
            "key": "duplicate_tolerance_px",
            "values": [2.0, 4.0, 8.0],
            "labels": ["Lower (2.0 px)", "Baseline (4.0 px)", "Higher (8.0 px)"],
        },
        {
            "parameter": "Tie-Point Max Count",
            "section": "tie_point_selection",
            "key": "max_points",
            "values": [10, 30, 50],
            "labels": ["Lower (10)", "Baseline (30)", "Higher (50)"],
        },
        {
            "parameter": "Sub-Pixel Refinement",
            "section": "subpixel_refinement",
            "key": "enabled",
            "values": [False, True],
            "labels": ["Disabled (False)", "Baseline (True)"],
        },
    ]

    results: List[Dict[str, Any]] = []

    for var in variations:
        param_name = var["parameter"]
        section = var["section"]
        key = var["key"]

        for val, label in zip(var["values"], var["labels"]):
            # Build perturbed configuration
            cfg = copy.deepcopy(base_config) if base_config is not None else {}
            if section not in cfg:
                cfg[section] = {}
            cfg[section][key] = val

            # Also ensure sub-configs in RegistrationEngine receive registration-specific params
            if section == "tie_point_selection":
                cfg.setdefault("registration", {})["tie_point_selection"] = {key: val}
            if section == "subpixel_refinement":
                cfg.setdefault("registration", {})["subpixel_refinement"] = {key: val}

            # Run case with perturbed config
            matcher = HybridMatcher(cfg)
            engine = RegistrationEngine(cfg)
            case_res: EvaluationCaseResult = runner.run_case(base_case, matcher, engine)

            results.append({
                "parameter": param_name,
                "setting": label,
                "value": val,
                "matched": bool(case_res.metrics.matched),
                "registered": bool(case_res.metrics.registered),
                "correct": bool(case_res.metrics.correct_registration),
                "inlier_ratio": round(float(case_res.metrics.inlier_ratio), 4),
                "rmse": round(float(case_res.metrics.rmse), 4) if case_res.metrics.rmse is not None else None,
                "coverage": round(float(case_res.metrics.coverage), 4),
                "confidence": round(float(case_res.metrics.confidence), 4),
                "quality": case_res.metrics.quality,
                "failure_reason": case_res.metrics.failure_reason,
            })

    return results
