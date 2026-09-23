import os

# Default configuration

class Config:
    PROJECT_NAME = "Cross-Sensor Lunar Location Matching"
    VERSION = "0.1.0"

    # Data paths
    DATA_DIR = "data"
    RAW_DIR = "data/raw"
    PROCESSED_DIR = "data/processed"
    VISUALIZATION_DIR = "data/processed/visualizations"

    # -- Visualization & Explainability ---------------------------------------
    VISUALIZATION = {
        "enabled": True,
        "production_mode": os.environ.get("LUNAR_PRODUCTION", "true").lower() in ("true", "1", "yes"),
        "retention_enabled": True,
        "retention_hours": 1,
        "max_job_directories": 5,
        "max_job_history": 5,
        "output_dir": "data/processed/visualizations",
        "max_visualized_matches": 100,
        "include_outliers": True,
        "include_checkerboard": True,
        "confidence_sigma": 15.0,
        "overlay_alpha": 0.45,
        "max_display_dim": 1200,
        "colormap": "TURBO",
    }

    # ── Phase 3: Preprocessing Pipeline ──────────────────────────────────────
    PREPROCESSING = {
        # Grayscale conversion
        # All preprocessing is performed in grayscale for sensor-agnostic matching.
        "grayscale": True,

        # Intensity normalization
        # Method: 'minmax' → scales to [0, 1].  Set to 'none' to skip.
        "normalization_method": "minmax",

        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # Improves local contrast visibility of craters and terrain.
        "clahe_enabled": True,
        "clahe_clip_limit": 2.0,       # Higher → more contrast, more noise
        "clahe_tile_grid_size": [8, 8], # Grid tile count (cols, rows)

        # Denoising  (default 'none' — conservative to preserve crater edges)
        # Options: 'none', 'gaussian', 'median', 'bilateral'
        "denoise_method": "none",
        "denoise_kwargs": {},           # E.g. {"ksize": 3} for gaussian/median

        # Pixel-space resize (None = no resize)
        # NOTE: This is pixel-only.  Physical GSD normalization is separate.
        "resize_target_size": None,     # Set to [width, height] to enable
        "resize_keep_aspect_ratio": True,

        # Multi-scale pyramid
        "pyramid_enabled": True,
        "pyramid_num_levels": 4,        # Includes level-0 (original scale)
        "pyramid_scale_factor": 0.5,    # Each level is 50% of the previous
    }

    # -- Phase 4: Classical Feature-Matching Baseline -------------------------
    CLASSICAL_MATCHING = {
        # SIFT detector parameters (OpenCV defaults are good starting points)
        "sift": {
            "n_features": 0,              # 0 = unlimited
            "n_octave_layers": 3,
            "contrast_threshold": 0.04,
            "edge_threshold": 10,
            "sigma": 1.6,
        },

        # AKAZE detector parameters
        "akaze": {
            "threshold": 0.001,
            "n_octaves": 4,
            "n_octave_layers": 4,
        },

        # Lowe ratio test threshold (applies to both SIFT and AKAZE)
        "ratio_threshold": 0.75,

        # RANSAC geometric verification
        "ransac": {
            "model": "affine",     # 'affine' or 'homography'
            "threshold": 5.0,      # Max reprojection error (pixels) for inlier
            "confidence": 0.99,
            "max_iters": 2000,
        },
    }

    # -- Phase 5: Crater Detection --------------------------------------------
    CRATER_DETECTION = {
        # Detector selection: 'mock' (testing), 'hough' (classical baseline), 'yolo' (deep learning)
        "detector_type": "hough",

        # Model weights path (required when detector_type='yolo')
        "model_path": None,

        # Confidence threshold for crater candidate acceptance
        "confidence_threshold": 0.25,

        # Non-Maximum Suppression (NMS) IoU overlap threshold
        "nms_iou_threshold": 0.45,

        # Crater radius limits in pixels
        "min_radius_px": 3.0,
        "max_radius_px": None,          # None = unconstrained

        # Maximum number of crater detections to retain
        "max_detections": 300,

        # Hardware execution device for neural network models ('cpu' or 'cuda')
        "device": "cpu",

        # Expected network input spatial resolution (pixels)
        "input_size": 640,

        # Parameters for the classical Hough baseline detector (detector_type='hough')
        "hough_params": {
            "dp": 1.2,                  # Inverse ratio of accumulator resolution
            "min_dist": 20.0,           # Minimum distance between detected circle centers
            "param1": 50.0,             # Higher Canny edge detection threshold
            "param2": 30.0,             # Accumulator threshold for circle centers
            "min_radius": 5,            # Minimum circle radius in pixels
            "max_radius": 120,          # Maximum circle radius in pixels
        },
    }

    # -- Phase 6: Crater Graph ------------------------------------------------
    CRATER_GRAPH = {
        # Graph construction strategy: 'delaunay', 'knn', 'radius'
        # Delaunay is default: provides sparse spatial neighborhood O(N) edges.
        "method": "delaunay",

        # KNN neighborhood size (used when method='knn')
        "k_neighbors": 5,

        # Radius neighborhood threshold in pixels (used when method='radius')
        "radius_threshold_px": 150.0,

        # Distance normalization strategy: 'median_radius', 'mean_radius', 'pair_mean_radius'
        "normalization_scale": "median_radius",

        # Maximum edge distance in pixels (optional prune threshold, None = unconstrained)
        "max_edge_distance_px": None,

        # Ensure undirected symmetric edge topology
        "symmetric_edges": True,
    }

    # -- Phase 7: Scale/Rotation-Robust Normalized Relationships ---------------
    CRATER_INVARIANTS = {
        # Scale normalization strategy: 'median_radius', 'mean_radius', 'pair_mean_radius'
        "scale_normalization": "median_radius",

        # Maximum number of canonical neighbors in local feature vector (zero-padded)
        "max_neighbors": 8,

        # Minimum triangle area in square pixels (triangles below this are considered degenerate)
        "min_triangle_area": 1.0,

        # Angular encoding: relative to canonical reference neighbor
        "angle_representation": "relative",

        # Enable triangle 3-clique descriptors
        "triangle_enabled": True,
    }

    # -- Phase 8: Local Crater Constellation Matching -------------------------
    CRATER_MATCHING = {
        # Maximum allowed descriptor distance for a candidate match pair
        "max_descriptor_distance": 0.5,

        # Distance metric component weights (must be non-negative)
        "distance_weight": 0.35,              # Weight for normalized distance discrepancy
        "radius_weight": 0.25,                # Weight for radius ratio discrepancy
        "angle_weight": 0.25,                 # Weight for relative angle discrepancy
        "neighbor_count_weight": 0.15,        # Penalty weight for neighborhood size mismatch

        # Lowe's ratio test threshold (best / second_best distance). None to disable.
        "ratio_test_threshold": 0.85,

        # Mutual (bidirectional) consistency requirement (A->B and B->A agree)
        "mutual_consistency": True,

        # Minimum compatible neighbors required to evaluate match
        "min_neighbors_for_match": 2,

        # Enable triangle 3-clique mutual support bonus
        "triangle_support_enabled": True,

        # Max L_inf difference between supporting triangle feature vectors
        "triangle_distance_threshold": 0.15,

        # Confidence decay scale parameter
        "confidence_sigma": 0.3,
    }

    # -- Phase 9: Initial Similarity Transformation Estimation ----------------
    TRANSFORMATION_ESTIMATION = {
        # Whether to weight least-squares by Phase 8 confidence scores (default False = unweighted)
        "use_confidence_weights": False,

        # Minimum number of correspondences required for 2D similarity estimation (>= 2)
        "min_correspondences": 2,

        # Spatial variance threshold below which points are considered degenerate
        "degeneracy_epsilon": 1e-8,

        # Allow reflected configurations (det(R) < 0). Default False rejects reflections.
        "allow_reflection": False,
    }

    # -- Phase 10: Robust Similarity Transformation Estimation (RANSAC) -------
    RANSAC = {
        # Maximum number of RANSAC sampling iterations
        "max_iterations": 1000,

        # Reprojection threshold in pixels for inlier classification (r_i <= threshold)
        "reprojection_threshold": 3.0,

        # Minimum number of consensus inliers required for valid transformation
        "min_inliers": 3,

        # Desired RANSAC confidence level (p) for adaptive stopping rule
        "confidence": 0.99,

        # Deterministic random seed for local NumPy PRNG
        "random_seed": 42,

        # Minimal sample size for 2D similarity model hypothesis generation
        "sample_size": 2,

        # Whether to enable adaptive iteration count based on estimated inlier ratio
        "adaptive_iterations": True,

        # Whether to bias sample selection by Phase 8 correspondence confidence
        "use_match_confidence": False,

        # Allow reflected configurations (det(R) < 0). Default False rejects reflections.
        "allow_reflection": False,

        # Whether to perform final least-squares refit on consensus inliers
        "refine_inliers": True,

        # Spatial variance threshold below which sampled points are considered degenerate
        "degeneracy_epsilon": 1e-8,
    }

    # -- Milestone A (Phase 11): Learned Deep Feature Matching ----------------
    LEARNED_MATCHING = {
        # Backend architecture: 'loftr' (default) or 'mock' (for deterministic testing)
        "backend": "loftr",

        # LoFTR pretrained checkpoint type: 'outdoor' or 'indoor'
        "pretrained": "outdoor",

        # Minimum confidence threshold for retaining learned keypoint correspondences
        "min_confidence": 0.2,

        # Execution hardware device: 'cpu' or 'cuda'
        "device": "cpu",

        # Maximum image dimension for single-pass LoFTR inference
        "max_image_dimension": 640,

        # Real-image input-size handling / Tiling configuration
        # 'auto': enable tiling when image max dimension > max_image_dimension
        # 'always': always tile regardless of image size
        # 'never': single-pass direct matching only
        "tiling_mode": "auto",

        # Tile spatial dimensions (pixels) for memory-bounded inference
        "tile_size": 640,

        # Overlap in pixels between adjacent tiles to avoid edge boundary loss
        "tile_overlap": 160,

        # Maximum total tiles allowed per image pair (safety guard rail)
        "max_tiles": 100,

        # Spatial tolerance in pixels for deduplicating tile overlap correspondences
        "duplicate_tolerance_px": 3.0,
    }

    # -- Milestone A (Phase 12): Correspondence Fusion ------------------------
    CORRESPONDENCE_FUSION = {
        # Spatial radius tolerance (pixels) for merging duplicate correspondences
        "duplicate_tolerance_px": 4.0,

        # Maximum number of correspondences retained per source branch (balancing)
        "max_correspondences_per_source": 300,

        # Per-source confidence multipliers
        "crater_weight_multiplier": 1.0,
        "learned_weight_multiplier": 1.0,
    }

    # -- Milestone A (Phase 13): Hybrid Matching Engine -----------------------
    HYBRID_MATCHING = {
        # Enable/disable branches
        "enable_crater_branch": True,
        "enable_learned_branch": True,

        # RANSAC geometric verification settings for fused correspondences
        "reprojection_threshold": 3.0,
        "min_inliers": 3,
        "max_iterations": 1000,
        "random_seed": 42,
        "confidence": 0.99,
        "allow_reflection": False,
        "refine_inliers": True,
    }

    # -- Match Acceptance Engine Configuration --------------------------------
    MATCH_ACCEPTANCE = {
        # Whether Match Acceptance Engine is enabled
        "enabled": True,

        # Absolute minimum required RANSAC inliers for location match acceptance
        "minimum_inliers": 10,

        # Minimum required inlier ratio (inliers / candidate correspondences)
        "minimum_inlier_ratio": 0.20,

        # Maximum allowed reprojection RMSE in pixels
        "maximum_rmse_px": 3.0,

        # Minimum required spatial grid occupancy coverage in [0.0, 1.0]
        "minimum_coverage": 0.15,

        # Minimum required pipeline confidence score in [0.0, 1.0]
        "minimum_confidence": 0.50,

        # Whether to enforce finite numerical transform parameters (non-NaN, non-Inf)
        "require_finite_transform": True,

        # Whether to enforce positive isotropic scale factor (> 0)
        "require_positive_scale": True,
    }

    # -- Milestone B (Phase 15): Uniform Tie-Point Selection ------------------
    TIE_POINT_SELECTION = {
        # Whether uniform tie-point selection is enabled
        "enabled": True,

        # Spatial grid dimensions partitioning Image A's domain
        "grid_rows": 6,
        "grid_cols": 6,

        # Maximum number of tie points retained after spatial selection
        "max_points": 30,

        # Minimum required tie points for valid registration
        "min_points": 3,
    }

    # -- Milestone B (Phase 14): Sub-Pixel / Local Refinement -----------------
    SUBPIXEL_REFINEMENT = {
        # Optional flag: whether local template/patch refinement is enabled
        "enabled": True,

        # Half-window radius in pixels for reference patch in Image A (patch size = 2*r + 1)
        "patch_radius": 7,

        # Search window radius in pixels around projected point in Image B
        "search_radius": 3,

        # Minimum normalized cross-correlation (NCC) required to accept refinement
        "min_correlation": 0.5,

        # Maximum allowed displacement in pixels from coarse location
        "max_shift_px": 3.0,

        # Enable sub-pixel parabolic peak interpolation
        "subpixel_interpolation": True,
    }

    # -- Milestone B (Phase 16): Image Registration ---------------------------
    REGISTRATION = {
        # Interpolation method for cv2.warpAffine ('linear', 'cubic', 'nearest')
        "interpolation": "linear",

        # Border mode ('constant', 'reflect', 'replicate')
        "border_mode": "constant",

        # Value for pixels outside source image boundaries
        "border_value": 0.0,
    }

    # -- Milestone B (Phase 17): Registration Quality Evaluation --------------
    REGISTRATION_QUALITY = {
        # Thresholds for quality categories
        "excellent_rmse": 1.5,
        "good_rmse": 3.0,
        "fair_rmse": 5.0,

        # Minimum inlier ratio requirements
        "min_inlier_ratio": 0.3,

        # Minimum spatial coverage requirements (fraction of grid cells occupied)
        "min_coverage": 0.1,

        # Minimum tie points for stable quality assessment
        "min_tie_points": 3,

        # Target number of tie points for confidence scoring
        "target_tie_points": 10,

        # Error decay scale for confidence score calculation
        "rmse_sigma": 3.0,
    }

    # -- Milestone C (Phases 18 + 19): Evaluation & Robustness ----------------
    EVALUATION = {
        # Tolerances for classifying registration as correct against ground truth
        "max_scale_error": 0.05,            # Absolute scale discrepancy
        "max_rotation_error_deg": 3.0,      # Absolute angular discrepancy in degrees
        "max_translation_error_px": 5.0,    # Euclidean translation discrepancy in pixels
        "max_rmse_px": 3.0,                 # Maximum allowed geometric RMSE
        "min_coverage": 0.15,               # Minimum required spatial tie-point coverage
        "min_confidence": 0.5,              # Minimum registration confidence

        # Evaluation dataset settings
        "default_image_size": 512,
        "random_seed": 42,

        # Output artifact path
        "output_dir": "data/processed/evaluation",
    }
