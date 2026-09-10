"""
src/matching/transformation.py

Initial planar similarity transformation estimation:
- SimilarityTransform2D: Dataclass encapsulating 2D similarity transform:
    p_B = s * R(theta) * p_A + t
- TransformationEstimationResult: Container for estimation output, residuals, and metrics.
- estimate_similarity_transform(): Closed-form least-squares 2D similarity estimator
  (Umeyama formulation) from corresponding point sets.
- estimate_transform_from_matches(): High-level entry point consuming Phase 8 ConstellationMatch objects.

ARCHITECTURAL RULES:
- Closed-form least-squares estimation only.
- Does NOT perform RANSAC or robust outlier rejection (deferred to subsequent phases).
- Rejects reflections by default (allow_reflection=False).
- Requires at least 2 non-degenerate correspondences.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import math
import numpy as np

from src.crater_graph.graph import CraterGraph
from src.crater_graph.constellation_matcher import ConstellationMatch


@dataclass
class SimilarityTransform2D:
    """
    Planar similarity transformation: p' = s * R(theta) * p + t.

    Attributes:
        scale: Uniform positive scale factor s.
        rotation_rad: Rotation angle theta in radians in (-pi, pi].
        translation_x: Horizontal translation t_x.
        translation_y: Vertical translation t_y.
        rotation_matrix: 2x2 orthogonal rotation matrix R.
        metadata: Optional dictionary with diagnostic flags (e.g. is_reflection).
    """
    scale: float
    rotation_rad: float
    translation_x: float
    translation_y: float
    rotation_matrix: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not math.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError(f"SimilarityTransform2D scale must be positive finite, got {self.scale}.")
        for name, val in [
            ("rotation_rad", self.rotation_rad),
            ("translation_x", self.translation_x),
            ("translation_y", self.translation_y),
        ]:
            if not math.isfinite(val):
                raise ValueError(f"SimilarityTransform2D {name} must be finite, got {val}.")

        if self.rotation_matrix is None or not isinstance(self.rotation_matrix, np.ndarray) or self.rotation_matrix.shape != (2, 2):
            self.rotation_matrix = np.array([
                [math.cos(self.rotation_rad), -math.sin(self.rotation_rad)],
                [math.sin(self.rotation_rad), math.cos(self.rotation_rad)],
            ], dtype=np.float64)

    @property
    def rotation_deg(self) -> float:
        """Rotation angle in degrees."""
        return math.degrees(self.rotation_rad)

    @property
    def translation(self) -> Tuple[float, float]:
        """Translation tuple (tx, ty)."""
        return (self.translation_x, self.translation_y)

    def apply(self, points: np.ndarray) -> np.ndarray:
        """
        Apply transformation to 2D points: p' = s * p @ R.T + t.

        Args:
            points: (N, 2) or (2,) array of 2D coordinates.

        Returns:
            Transformed points array matching input shape.
        """
        pts = np.asarray(points, dtype=np.float64)
        is_single = (pts.ndim == 1 and pts.shape == (2,))
        if is_single:
            pts = pts.reshape(1, 2)
        elif pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(f"Expected points array of shape (N, 2) or (2,), got {pts.shape}.")

        # p' = s * R * p + t = s * (p @ R.T) + t
        t_vec = np.array([self.translation_x, self.translation_y], dtype=np.float64)
        transformed = self.scale * (pts @ self.rotation_matrix.T) + t_vec

        return transformed[0] if is_single else transformed

    def inverse(self) -> "SimilarityTransform2D":
        """
        Compute exact analytical inverse transformation:
            p_A = s^(-1) * R^T * (p_B - t) = s^(-1) * R^T * p_B - s^(-1) * R^T * t.

        Returns:
            Inverse SimilarityTransform2D instance.
        """
        inv_scale = 1.0 / self.scale
        inv_r_mat = self.rotation_matrix.T
        t_vec = np.array([self.translation_x, self.translation_y], dtype=np.float64)
        inv_t = -inv_scale * (inv_r_mat @ t_vec)
        inv_theta = math.atan2(inv_r_mat[1, 0], inv_r_mat[0, 0])

        meta = dict(self.metadata)
        meta["is_inverse"] = True
        return SimilarityTransform2D(
            scale=inv_scale,
            rotation_rad=inv_theta,
            translation_x=float(inv_t[0]),
            translation_y=float(inv_t[1]),
            rotation_matrix=inv_r_mat,
            metadata=meta,
        )

    def to_matrix(self) -> np.ndarray:
        """
        Return 3x3 homogeneous transformation matrix:
            [[s * R00, s * R01, tx],
             [s * R10, s * R11, ty],
             [0,       0,       1 ]]
        """
        m = np.eye(3, dtype=np.float64)
        m[0:2, 0:2] = self.scale * self.rotation_matrix
        m[0, 2] = self.translation_x
        m[1, 2] = self.translation_y
        return m

    def to_dict(self) -> Dict[str, Any]:
        """Serialize transformation to primitive dictionary."""
        return {
            "scale": round(float(self.scale), 6),
            "rotation_rad": round(float(self.rotation_rad), 6),
            "rotation_deg": round(float(self.rotation_deg), 4),
            "translation_x": round(float(self.translation_x), 4),
            "translation_y": round(float(self.translation_y), 4),
            "matrix_3x3": [[round(float(v), 6) for v in row] for row in self.to_matrix().tolist()],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimilarityTransform2D":
        """Deserialize transformation from primitive dictionary."""
        theta = float(data["rotation_rad"])
        r_mat = np.array([
            [math.cos(theta), -math.sin(theta)],
            [math.sin(theta), math.cos(theta)],
        ], dtype=np.float64)
        return cls(
            scale=float(data["scale"]),
            rotation_rad=theta,
            translation_x=float(data["translation_x"]),
            translation_y=float(data["translation_y"]),
            rotation_matrix=r_mat,
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class TransformationEstimationResult:
    """
    Output of similarity transformation estimation including error diagnostics.

    Attributes:
        transform: Estimated SimilarityTransform2D.
        num_correspondences: Number of point correspondences used.
        residuals: (N,) array of Euclidean point alignment errors.
        rmse: Root-Mean-Square Error across all correspondences.
        mean_error: Mean residual error.
        median_error: Median residual error.
        max_error: Maximum residual error.
        is_weighted: Boolean indicating whether confidence weights were applied.
        metadata: Configuration, reflection status, and diagnostic flags.
    """
    transform: SimilarityTransform2D
    num_correspondences: int
    residuals: np.ndarray
    rmse: float
    mean_error: float
    median_error: float
    max_error: float
    is_weighted: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize estimation result to primitive dictionary."""
        return {
            "transform": self.transform.to_dict(),
            "num_correspondences": int(self.num_correspondences),
            "residuals": [round(float(r), 4) for r in self.residuals.tolist()],
            "rmse": round(float(self.rmse), 4),
            "mean_error": round(float(self.mean_error), 4),
            "median_error": round(float(self.median_error), 4),
            "max_error": round(float(self.max_error), 4),
            "is_weighted": bool(self.is_weighted),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TransformationEstimationResult":
        """Deserialize estimation result from primitive dictionary."""
        tf = SimilarityTransform2D.from_dict(data["transform"])
        res = np.array(data.get("residuals", []), dtype=np.float64)
        return cls(
            transform=tf,
            num_correspondences=int(data["num_correspondences"]),
            residuals=res,
            rmse=float(data["rmse"]),
            mean_error=float(data["mean_error"]),
            median_error=float(data["median_error"]),
            max_error=float(data["max_error"]),
            is_weighted=bool(data.get("is_weighted", False)),
            metadata=dict(data.get("metadata", {})),
        )


def estimate_similarity_transform(
    points_a: np.ndarray,
    points_b: np.ndarray,
    weights: Optional[np.ndarray] = None,
    allow_reflection: bool = False,
    degeneracy_epsilon: float = 1e-8,
) -> TransformationEstimationResult:
    """
    Closed-form least-squares estimation of 2D similarity transformation:
        p_B ≈ s * R(theta) * p_A + t

    Uses standard Umeyama / Horn SVD formulation.

    Minimum Correspondence & Degeneracy Rules:
    - Requires at least 2 point pairs.
    - Two-point configurations with coincident source points are rejected.
    - Multi-point configurations with effectively zero spatial variance (sigma_A^2 < eps) are rejected.
    - If unconstrained SVD fit produces det(R) < 0 (reflection):
        - By default (allow_reflection=False), raises ValueError with a clear explanation.
        - If allow_reflection=True, permits reflection and records is_reflection=True in metadata.

    Args:
        points_a: (N, 2) array of source 2D points.
        points_b: (N, 2) array of target 2D points.
        weights: Optional (N,) array of non-negative confidence weights.
        allow_reflection: Whether to accept reflected configurations (default False).
        degeneracy_epsilon: Variance threshold below which points are considered degenerate.

    Returns:
        TransformationEstimationResult containing estimated transform and residual metrics.

    Raises:
        ValueError: On insufficient points, degenerate geometry, or reflection rejection.
    """
    pts_a = np.asarray(points_a, dtype=np.float64)
    pts_b = np.asarray(points_b, dtype=np.float64)

    if pts_a.ndim != 2 or pts_a.shape[1] != 2:
        raise ValueError(f"points_a must have shape (N, 2), got {pts_a.shape}.")
    if pts_b.ndim != 2 or pts_b.shape[1] != 2:
        raise ValueError(f"points_b must have shape (N, 2), got {pts_b.shape}.")
    if len(pts_a) != len(pts_b):
        raise ValueError(f"points_a and points_b must have the same length: {len(pts_a)} vs {len(pts_b)}.")

    n = len(pts_a)
    if n < 2:
        raise ValueError(f"At least 2 point correspondences are required for 2D similarity estimation, got {n}.")

    # Handle weights
    is_weighted = False
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64).flatten()
        if len(w) != n:
            raise ValueError(f"weights must have length {n}, got {len(w)}.")
        if np.any(np.isnan(w)) or np.any(w < 0.0):
            raise ValueError("weights must be non-negative finite values.")
        total_w = float(np.sum(w))
        if total_w < degeneracy_epsilon:
            raise ValueError("Sum of weights is effectively zero.")
        w_norm = w / total_w
        is_weighted = True
    else:
        w_norm = np.full(n, 1.0 / float(n), dtype=np.float64)

    # 1. Centroids
    mu_a = w_norm @ pts_a  # (2,)
    mu_b = w_norm @ pts_b  # (2,)

    # 2. Centered coordinates
    x = pts_a - mu_a  # (N, 2)
    y = pts_b - mu_b  # (N, 2)

    # Two-point direct analytical solution
    if n == 2:
        d_src = float(np.linalg.norm(pts_a[0] - pts_a[1]))
        if d_src < degeneracy_epsilon:
            raise ValueError(f"Degenerate 2-point correspondence: source points coincide (distance={d_src:.2e} < {degeneracy_epsilon}).")
        d_tgt = float(np.linalg.norm(pts_b[0] - pts_b[1]))
        if d_tgt < degeneracy_epsilon:
            raise ValueError(f"Degenerate 2-point correspondence: target points coincide (distance={d_tgt:.2e} < {degeneracy_epsilon}).")

        scale = d_tgt / d_src
        v_a = pts_a[1] - pts_a[0]
        v_b = pts_b[1] - pts_b[0]
        ang_a = math.atan2(v_a[1], v_a[0])
        ang_b = math.atan2(v_b[1], v_b[0])
        theta = ang_b - ang_a
        theta = (theta + math.pi) % (2.0 * math.pi) - math.pi
        r_mat = np.array([
            [math.cos(theta), -math.sin(theta)],
            [math.sin(theta), math.cos(theta)],
        ], dtype=np.float64)

        t_vec = mu_b - scale * (r_mat @ mu_a)

        tf = SimilarityTransform2D(
            scale=scale,
            rotation_rad=theta,
            translation_x=float(t_vec[0]),
            translation_y=float(t_vec[1]),
            rotation_matrix=r_mat,
            metadata={"is_reflection": False, "allow_reflection": allow_reflection},
        )
        pts_a_trans = tf.apply(pts_a)
        errors = pts_a_trans - pts_b
        residuals = np.sqrt(np.sum(errors * errors, axis=1))
        rmse = float(np.sqrt(np.mean(residuals * residuals)))

        return TransformationEstimationResult(
            transform=tf,
            num_correspondences=2,
            residuals=residuals,
            rmse=rmse,
            mean_error=float(np.mean(residuals)),
            median_error=float(np.median(residuals)),
            max_error=float(np.max(residuals)),
            is_weighted=is_weighted,
            metadata={
                "allow_reflection": allow_reflection,
                "is_reflection": False,
                "source_variance": float(np.sum(w_norm * np.sum(x * x, axis=1))),
            },
        )

    # 3. Source variance
    var_a = float(np.sum(w_norm * np.sum(x * x, axis=1)))
    if var_a < degeneracy_epsilon:
        raise ValueError(f"Source points are degenerate: spatial variance ({var_a:.2e}) is effectively zero.")

    # 4. Covariance matrix H = X.T @ W @ Y
    h_cov = (x * w_norm[:, np.newaxis]).T @ y  # (2, 2)

    # 5. SVD of H
    u, s_vals, vh = np.linalg.svd(h_cov)
    v = vh.T

    det_vu = float(np.linalg.det(v @ u.T))
    is_reflection = (det_vu < -1e-7)

    if is_reflection and not allow_reflection:
        raise ValueError(
            f"Reflection detected: estimated transformation contains a reflection (det={det_vu:.4f} < 0), "
            f"which is invalid for 2D similarity (allow_reflection=False)."
        )

    # 6. Orthogonal matrix R = V @ U.T (proper rotation if not reflection, reflection if allowed)
    r_mat = v @ u.T

    # 7. Scale factor s
    scale = float(np.sum(s_vals)) / var_a
    if scale <= 0.0 or not math.isfinite(scale):
        raise ValueError(f"Degenerate scale estimated: {scale}.")


    # 8. Translation vector t = mu_B - s * R * mu_A
    t_vec = mu_b - scale * (r_mat @ mu_a)

    # 9. Rotation angle theta in (-pi, pi]
    theta = math.atan2(r_mat[1, 0], r_mat[0, 0])

    tf = SimilarityTransform2D(
        scale=scale,
        rotation_rad=theta,
        translation_x=float(t_vec[0]),
        translation_y=float(t_vec[1]),
        rotation_matrix=r_mat,
        metadata={"is_reflection": is_reflection, "allow_reflection": allow_reflection},
    )

    # 10. Residual diagnostics: e_i = T(p_A_i) - p_B_i
    pts_a_trans = tf.apply(pts_a)
    errors = pts_a_trans - pts_b
    residuals = np.sqrt(np.sum(errors * errors, axis=1))

    rmse = float(np.sqrt(np.mean(residuals * residuals)))
    mean_err = float(np.mean(residuals))
    med_err = float(np.median(residuals))
    max_err = float(np.max(residuals))

    return TransformationEstimationResult(
        transform=tf,
        num_correspondences=n,
        residuals=residuals,
        rmse=rmse,
        mean_error=mean_err,
        median_error=med_err,
        max_error=max_err,
        is_weighted=is_weighted,
        metadata={
            "allow_reflection": allow_reflection,
            "is_reflection": is_reflection,
            "source_variance": var_a,
        },
    )


def estimate_transform_from_matches(
    graph_a: CraterGraph,
    graph_b: CraterGraph,
    matches: List[ConstellationMatch],
    config: Optional[Dict[str, Any]] = None,
) -> TransformationEstimationResult:
    """
    Estimate initial 2D similarity transformation directly from Phase 8 ConstellationMatch correspondences.

    Args:
        graph_a: Source CraterGraph.
        graph_b: Target CraterGraph.
        matches: List of ConstellationMatch correspondences.
        config: Configuration dictionary (defaults to Config.TRANSFORMATION_ESTIMATION).

    Returns:
        TransformationEstimationResult.
    """
    if config is None:
        try:
            from configs.default import Config
            config = getattr(Config, "TRANSFORMATION_ESTIMATION", {})
        except (ImportError, AttributeError):
            config = {}

    min_pts = int(config.get("min_correspondences", 2))
    use_weights = bool(config.get("use_confidence_weights", False))
    allow_refl = bool(config.get("allow_reflection", False))
    eps = float(config.get("degeneracy_epsilon", 1e-8))

    if len(matches) < min_pts:
        raise ValueError(
            f"Insufficient correspondences for transformation estimation: {len(matches)} provided, "
            f"minimum required is {min_pts}."
        )

    pts_a_list = []
    pts_b_list = []
    weights_list = []

    for m in matches:
        ca = graph_a.get_crater(m.source_node_id)
        cb = graph_b.get_crater(m.target_node_id)
        pts_a_list.append([ca.x, ca.y])
        pts_b_list.append([cb.x, cb.y])
        weights_list.append(max(float(m.confidence), 1e-4) if use_weights else 1.0)

    pts_a = np.array(pts_a_list, dtype=np.float64)
    pts_b = np.array(pts_b_list, dtype=np.float64)
    weights = np.array(weights_list, dtype=np.float64) if use_weights else None

    return estimate_similarity_transform(
        points_a=pts_a,
        points_b=pts_b,
        weights=weights,
        allow_reflection=allow_refl,
        degeneracy_epsilon=eps,
    )
