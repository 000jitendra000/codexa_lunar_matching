"""
src/matching/ransac.py

Robust similarity transformation estimation using RANSAC:
- RANSACResult: Dataclass encapsulating robust estimation results, inlier/outlier
  classifications, residuals, error statistics, and diagnostic metadata.
- evaluate_hypothesis(): Helper evaluating residual errors and inliers for a given candidate transform.
- estimate_robust_similarity_transform(): Core RANSAC loop estimating planar similarity
  transform with minimal sample size, deterministic local PRNG, adaptive iteration count,
  geometric consensus scoring, and final inlier least-squares refit.
- estimate_robust_transform_from_matches(): High-level entry point consuming Phase 8
  ConstellationMatch correspondences.

ARCHITECTURAL RULES:
- Strictly estimates 2D planar similarity: p_B = s * R(theta) * p_A + t.
- Generates hypotheses using Phase 9 estimate_similarity_transform().
- Default minimal sample size is 2 (rejecting degenerate/coincident pairs).
- Strictly uses reprojection residuals (r_i <= reprojection_threshold) for inlier determination.
- Deterministic local NumPy PRNG (no global random state).
- Supports single final least-squares refit on consensus inliers.
- Does NOT implement LoFTR, SuperGlue, deep feature matching, correspondence fusion,
  or sub-pixel refinement (strictly reserved for future phases).
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import math
import numpy as np

from src.crater_graph.graph import CraterGraph
from src.crater_graph.constellation_matcher import ConstellationMatch
from src.matching.transformation import (
    SimilarityTransform2D,
    TransformationEstimationResult,
    estimate_similarity_transform,
)


@dataclass
class RANSACResult:
    """
    Container for robust transformation estimation results.

    Attributes:
        transform: Final SimilarityTransform2D (refitted or best hypothesis).
        inlier_indices: List of 0-indexed correspondence indices identified as inliers.
        outlier_indices: List of 0-indexed correspondence indices identified as outliers.
        num_iterations: Total number of RANSAC sampling iterations performed.
        num_inliers: Count of verified consensus inliers.
        inlier_ratio: Fraction of correspondences classified as inliers (num_inliers / N).
        residuals: (N,) array of Euclidean reprojection errors under the final transform.
        rmse: Root-Mean-Square Error across all final inliers.
        mean_inlier_error: Mean residual error among final inliers.
        median_inlier_error: Median residual error among final inliers.
        max_inlier_error: Maximum residual error among final inliers.
        consensus_score: Consensus metric achieved by the winning hypothesis.
        metadata: Configuration, seed, adaptive stop status, and runtime diagnostics.
    """
    transform: SimilarityTransform2D
    inlier_indices: List[int]
    outlier_indices: List[int]
    num_iterations: int
    num_inliers: int
    inlier_ratio: float
    residuals: np.ndarray
    rmse: float
    mean_inlier_error: float
    median_inlier_error: float
    max_inlier_error: float
    consensus_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize robust estimation result to primitive dictionary."""
        return {
            "transform": self.transform.to_dict(),
            "inlier_indices": [int(idx) for idx in self.inlier_indices],
            "outlier_indices": [int(idx) for idx in self.outlier_indices],
            "num_iterations": int(self.num_iterations),
            "num_inliers": int(self.num_inliers),
            "inlier_ratio": round(float(self.inlier_ratio), 4),
            "residuals": [round(float(r), 4) for r in self.residuals.tolist()],
            "rmse": round(float(self.rmse), 4),
            "mean_inlier_error": round(float(self.mean_inlier_error), 4),
            "median_inlier_error": round(float(self.median_inlier_error), 4),
            "max_inlier_error": round(float(self.max_inlier_error), 4),
            "consensus_score": round(float(self.consensus_score), 4),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RANSACResult":
        """Deserialize robust estimation result from primitive dictionary."""
        tf = SimilarityTransform2D.from_dict(data["transform"])
        res = np.array(data.get("residuals", []), dtype=np.float64)
        return cls(
            transform=tf,
            inlier_indices=[int(i) for i in data.get("inlier_indices", [])],
            outlier_indices=[int(i) for i in data.get("outlier_indices", [])],
            num_iterations=int(data.get("num_iterations", 0)),
            num_inliers=int(data.get("num_inliers", 0)),
            inlier_ratio=float(data.get("inlier_ratio", 0.0)),
            residuals=res,
            rmse=float(data.get("rmse", 0.0)),
            mean_inlier_error=float(data.get("mean_inlier_error", 0.0)),
            median_inlier_error=float(data.get("median_inlier_error", 0.0)),
            max_inlier_error=float(data.get("max_inlier_error", 0.0)),
            consensus_score=float(data.get("consensus_score", 0.0)),
            metadata=dict(data.get("metadata", {})),
        )


def evaluate_hypothesis(
    transform: SimilarityTransform2D,
    points_a: np.ndarray,
    points_b: np.ndarray,
    reprojection_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Evaluate candidate similarity transformation on all correspondences.

    Args:
        transform: Candidate SimilarityTransform2D hypothesis.
        points_a: (N, 2) array of source points.
        points_b: (N, 2) array of target points.
        reprojection_threshold: Distance threshold (pixels) for inlier classification.

    Returns:
        Tuple of:
            residuals: (N,) array of Euclidean point alignment errors.
            inlier_mask: (N,) boolean mask where residual <= threshold.
            inlier_rmse: Root-Mean-Square Error across inliers (or inf if no inliers).
            consensus_score: Primary consensus metric (number of inliers).
    """
    pts_a_trans = transform.apply(points_a)
    diff = pts_a_trans - points_b
    residuals = np.sqrt(np.sum(diff * diff, axis=1))

    inlier_mask = (residuals <= reprojection_threshold)
    num_inliers = int(np.sum(inlier_mask))

    if num_inliers > 0:
        inlier_res = residuals[inlier_mask]
        inlier_rmse = float(np.sqrt(np.mean(inlier_res * inlier_res)))
    else:
        inlier_rmse = float("inf")

    consensus_score = float(num_inliers)
    return residuals, inlier_mask, inlier_rmse, consensus_score


def estimate_robust_similarity_transform(
    points_a: np.ndarray,
    points_b: np.ndarray,
    weights: Optional[np.ndarray] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> RANSACResult:
    """
    Robustly estimate 2D planar similarity transformation using RANSAC.

    p_B ≈ s * R(theta) * p_A + t

    Algorithm:
    1. Validate input shapes, lengths, and numerical finiteness.
    2. Initialize local NumPy PRNG with deterministic seed.
    3. Iteratively draw minimal sample subsets (default size = 2).
    4. Reject degenerate subsets (coincident points or zero variance).
    5. Fit candidate hypothesis via Phase 9 estimate_similarity_transform().
    6. Compute reprojection residuals and identify inliers (r_i <= threshold).
    7. Rank hypotheses lexicographically: (num_inliers, -inlier_rmse, -total_inlier_err).
    8. Adaptively update iteration count N(p, w, s) if enabled.
    9. Verify minimum inlier constraint (num_inliers >= min_inliers).
    10. Perform single least-squares refit on final consensus inliers if enabled.

    Args:
        points_a: (N, 2) source coordinates.
        points_b: (N, 2) target coordinates.
        weights: Optional (N,) confidence weights (e.g. from Phase 8).
        config: Optional configuration dictionary (falls back to Config.RANSAC).
        **kwargs: Overrides for configuration parameters.

    Returns:
        RANSACResult containing optimal transform, inlier/outlier indices, and error statistics.

    Raises:
        ValueError: On insufficient points, degenerate data, NaN/Inf, or failure to find consensus.
    """
    pts_a = np.asarray(points_a, dtype=np.float64)
    pts_b = np.asarray(points_b, dtype=np.float64)

    # 1. Input Validation
    if pts_a.ndim != 2 or pts_a.shape[1] != 2:
        raise ValueError(f"points_a must have shape (N, 2), got {pts_a.shape}.")
    if pts_b.ndim != 2 or pts_b.shape[1] != 2:
        raise ValueError(f"points_b must have shape (N, 2), got {pts_b.shape}.")
    if len(pts_a) != len(pts_b):
        raise ValueError(f"points_a and points_b must have identical length: {len(pts_a)} vs {len(pts_b)}.")

    if np.any(np.isnan(pts_a)) or np.any(np.isinf(pts_a)):
        raise ValueError("points_a contains NaN or Inf coordinates.")
    if np.any(np.isnan(pts_b)) or np.any(np.isinf(pts_b)):
        raise ValueError("points_b contains NaN or Inf coordinates.")

    n = len(pts_a)

    # Load configuration
    cfg: Dict[str, Any] = {}
    if config is not None:
        cfg.update(config)
    else:
        try:
            from configs.default import Config
            cfg.update(getattr(Config, "RANSAC", {}))
        except (ImportError, AttributeError):
            pass
    cfg.update(kwargs)

    sample_size = int(cfg.get("sample_size", 2))
    if sample_size < 2:
        raise ValueError(f"RANSAC sample_size must be at least 2 for similarity estimation, got {sample_size}.")
    if n < sample_size:
        raise ValueError(f"RANSAC requires at least {sample_size} correspondences, but got {n}.")

    max_iterations = int(cfg.get("max_iterations", 1000))
    reprojection_threshold = float(cfg.get("reprojection_threshold", 3.0))
    min_inliers = int(cfg.get("min_inliers", 3))
    confidence = float(cfg.get("confidence", 0.99))
    random_seed = int(cfg.get("random_seed", 42))
    adaptive_iterations = bool(cfg.get("adaptive_iterations", True))
    use_match_confidence = bool(cfg.get("use_match_confidence", False))
    allow_reflection = bool(cfg.get("allow_reflection", False))
    refine_inliers = bool(cfg.get("refine_inliers", True))
    degeneracy_epsilon = float(cfg.get("degeneracy_epsilon", 1e-8))

    if reprojection_threshold <= 0.0:
        raise ValueError(f"reprojection_threshold must be positive, got {reprojection_threshold}.")

    # Sampling probability distribution
    sample_probs = None
    if use_match_confidence and weights is not None:
        w_arr = np.asarray(weights, dtype=np.float64).flatten()
        if len(w_arr) != n:
            raise ValueError(f"weights must match points length {n}, got {len(w_arr)}.")
        if np.any(np.isnan(w_arr)) or np.any(w_arr < 0.0):
            raise ValueError("weights must be non-negative finite values.")
        total_w = float(np.sum(w_arr))
        if total_w > degeneracy_epsilon:
            sample_probs = w_arr / total_w

    # Deterministic local PRNG
    rng = np.random.default_rng(random_seed)

    best_hypothesis: Optional[SimilarityTransform2D] = None
    best_inlier_mask: Optional[np.ndarray] = None
    best_inlier_count: int = 0
    best_inlier_rmse: float = float("inf")
    best_total_err: float = float("inf")
    best_consensus_score: float = 0.0

    current_max_iters = max_iterations
    actual_iterations = 0
    degenerate_samples_count = 0

    # Main RANSAC sampling loop
    for it in range(max_iterations):
        actual_iterations += 1

        # 1. Sample minimal subset
        try:
            sample_indices = rng.choice(n, size=sample_size, replace=False, p=sample_probs)
        except Exception:
            sample_indices = rng.choice(n, size=sample_size, replace=False)

        pts_a_sample = pts_a[sample_indices]
        pts_b_sample = pts_b[sample_indices]

        # 2. Check sample degeneracy
        if sample_size == 2:
            d_src = float(np.linalg.norm(pts_a_sample[0] - pts_a_sample[1]))
            d_tgt = float(np.linalg.norm(pts_b_sample[0] - pts_b_sample[1]))
            if d_src < degeneracy_epsilon or d_tgt < degeneracy_epsilon:
                degenerate_samples_count += 1
                continue
        else:
            var_src = float(np.var(pts_a_sample, axis=0).sum())
            if var_src < degeneracy_epsilon:
                degenerate_samples_count += 1
                continue

        # 3. Estimate candidate hypothesis via Phase 9 closed-form estimator
        try:
            cand_result = estimate_similarity_transform(
                points_a=pts_a_sample,
                points_b=pts_b_sample,
                allow_reflection=allow_reflection,
                degeneracy_epsilon=degeneracy_epsilon,
            )
        except ValueError:
            degenerate_samples_count += 1
            continue

        cand_tf = cand_result.transform

        # 4. Evaluate hypothesis on all points
        residuals, inlier_mask, inlier_rmse, score = evaluate_hypothesis(
            transform=cand_tf,
            points_a=pts_a,
            points_b=pts_b,
            reprojection_threshold=reprojection_threshold,
        )
        inlier_count = int(np.sum(inlier_mask))
        total_inlier_err = float(np.sum(residuals[inlier_mask])) if inlier_count > 0 else float("inf")

        # 5. Deterministic lexicographical consensus ranking:
        #    1. Maximize number of inliers
        #    2. Minimize inlier RMSE
        #    3. Minimize total inlier residual
        is_better = False
        if inlier_count > best_inlier_count:
            is_better = True
        elif inlier_count == best_inlier_count and inlier_count > 0:
            if inlier_rmse < best_inlier_rmse - 1e-9:
                is_better = True
            elif abs(inlier_rmse - best_inlier_rmse) <= 1e-9:
                if total_inlier_err < best_total_err:
                    is_better = True

        if is_better:
            best_hypothesis = cand_tf
            best_inlier_mask = inlier_mask
            best_inlier_count = inlier_count
            best_inlier_rmse = inlier_rmse
            best_total_err = total_inlier_err
            best_consensus_score = score

            # 6. Adaptive stopping rule: N = log(1 - p) / log(1 - w^s)
            if adaptive_iterations and inlier_count >= min_inliers:
                inlier_ratio_est = float(inlier_count) / float(n)
                prob_all_inliers = inlier_ratio_est ** sample_size
                denom = 1.0 - prob_all_inliers

                if denom <= 1e-12:
                    n_est = 1
                elif denom >= 1.0 - 1e-12:
                    n_est = max_iterations
                else:
                    log_denom = math.log(denom)
                    log_p = math.log(1.0 - confidence)
                    n_est = int(math.ceil(log_p / log_denom))

                current_max_iters = min(current_max_iters, max(1, n_est))

        if actual_iterations >= current_max_iters:
            break

    # 7. Consensus validation check
    if best_hypothesis is None or best_inlier_count < min_inliers:
        found = best_inlier_count
        raise ValueError(
            f"RANSAC failed to find consensus: found {found} inliers, "
            f"minimum required is {min_inliers} (performed {actual_iterations} iterations, "
            f"{degenerate_samples_count} degenerate samples)."
        )

    best_inlier_indices = np.where(best_inlier_mask)[0]

    # 8. Final Inlier Refinement
    if refine_inliers and len(best_inlier_indices) >= 2:
        refine_weights = None
        if use_match_confidence and weights is not None:
            refine_weights = weights[best_inlier_indices]

        try:
            refit_result = estimate_similarity_transform(
                points_a=pts_a[best_inlier_indices],
                points_b=pts_b[best_inlier_indices],
                weights=refine_weights,
                allow_reflection=allow_reflection,
                degeneracy_epsilon=degeneracy_epsilon,
            )
            final_transform = refit_result.transform
        except ValueError:
            final_transform = best_hypothesis
    else:
        final_transform = best_hypothesis

    # 9. Recompute residuals for ALL correspondences under final transform
    final_pts_a_trans = final_transform.apply(pts_a)
    final_errors = final_pts_a_trans - pts_b
    final_residuals = np.sqrt(np.sum(final_errors * final_errors, axis=1))

    # Reclassify inliers under refined model
    final_inlier_mask = (final_residuals <= reprojection_threshold)
    final_inlier_indices = [int(i) for i in np.where(final_inlier_mask)[0]]

    # Fallback to initial best inliers if threshold condition is on border
    if len(final_inlier_indices) < min_inliers:
        final_inlier_indices = [int(i) for i in best_inlier_indices]

    final_inlier_set = set(final_inlier_indices)
    final_outlier_indices = [i for i in range(n) if i not in final_inlier_set]

    # 10. Inlier Error Diagnostics
    inlier_residuals = final_residuals[final_inlier_indices]
    final_rmse = float(np.sqrt(np.mean(inlier_residuals * inlier_residuals)))
    mean_err = float(np.mean(inlier_residuals))
    med_err = float(np.median(inlier_residuals))
    max_err = float(np.max(inlier_residuals))
    inlier_ratio = float(len(final_inlier_indices)) / float(n)

    metadata = {
        "random_seed": random_seed,
        "sample_size": sample_size,
        "reprojection_threshold": reprojection_threshold,
        "min_inliers": min_inliers,
        "max_iterations_configured": max_iterations,
        "actual_iterations": actual_iterations,
        "adaptive_iterations_enabled": adaptive_iterations,
        "refine_inliers": refine_inliers,
        "degenerate_samples_count": degenerate_samples_count,
        "use_match_confidence": use_match_confidence,
        "allow_reflection": allow_reflection,
    }

    return RANSACResult(
        transform=final_transform,
        inlier_indices=final_inlier_indices,
        outlier_indices=final_outlier_indices,
        num_iterations=actual_iterations,
        num_inliers=len(final_inlier_indices),
        inlier_ratio=inlier_ratio,
        residuals=final_residuals,
        rmse=final_rmse,
        mean_inlier_error=mean_err,
        median_inlier_error=med_err,
        max_inlier_error=max_err,
        consensus_score=best_consensus_score,
        metadata=metadata,
    )


def estimate_robust_transform_from_matches(
    graph_a: CraterGraph,
    graph_b: CraterGraph,
    matches: List[ConstellationMatch],
    config: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> RANSACResult:
    """
    Robustly estimate 2D planar similarity transformation directly from Phase 8 ConstellationMatch correspondences.

    Args:
        graph_a: Source CraterGraph.
        graph_b: Target CraterGraph.
        matches: List of ConstellationMatch correspondences from Phase 8.
        config: Optional configuration dictionary (defaults to Config.RANSAC).
        **kwargs: Overrides for configuration parameters.

    Returns:
        RANSACResult.

    Raises:
        ValueError: On insufficient matches or RANSAC consensus failure.
    """
    if len(matches) == 0:
        raise ValueError("Cannot estimate robust transformation: matches list is empty.")

    pts_a_list = []
    pts_b_list = []
    weights_list = []

    for m in matches:
        ca = graph_a.get_crater(m.source_node_id)
        cb = graph_b.get_crater(m.target_node_id)
        pts_a_list.append([ca.x, ca.y])
        pts_b_list.append([cb.x, cb.y])
        weights_list.append(max(float(m.confidence), 1e-4))

    pts_a = np.array(pts_a_list, dtype=np.float64)
    pts_b = np.array(pts_b_list, dtype=np.float64)
    weights = np.array(weights_list, dtype=np.float64)

    return estimate_robust_similarity_transform(
        points_a=pts_a,
        points_b=pts_b,
        weights=weights,
        config=config,
        **kwargs,
    )
