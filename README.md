# Cross-Sensor Lunar Location Matching

## Project Purpose
The goal of this project is to determine whether two lunar surface images captured by different sensors/missions (e.g., Chandrayaan-2 OHRC and LRO LROC NAC) represent the same lunar location, despite differences in resolution, scale, orientation, illumination, contrast, viewpoint, and framing.

## Problem Being Solved
Accurately co-registering and matching lunar images from different sensors is challenging due to the dynamic range of illumination, varying perspectives, and disparate sensor characteristics. Finding reliable correspondences ensures that science data and mapping efforts can be accurately layered and analyzed.

## Supported Image Sources
Currently supported / anticipated sensors:
- Chandrayaan-2 OHRC
- Chandrayaan-2 TMC-2
- Chandrayaan-2 IIRS
- LRO LROC NAC

## Dataset Conventions
- **`data/raw/`**: Where original image files (TIFF/PNG/etc) are stored. Keep it small for testing pairs.
- **`data/dataset_meta.json`**: Describes images and creates labeled positive/negative pair relationships.

### Ground Truth vs Matching Inputs
Geographic information (like latitude/longitude) in `dataset_meta.json` is ONLY used for dataset organization, evaluation, and creating positive/negative ground truth pairs. The actual image-matching functions receive only `ImagePair` abstractions and never depend on known coordinates.

### Adding New Images
1. Place the new image file in `data/raw/`.
2. Add its metadata entry in `data/dataset_meta.json` under the `images` list.
3. If it forms a pair with another image, add a relationship object under the `pairs` list.

## Preprocessing Pipeline

Each stage is independently configurable via `configs/default.py -> Config.PREPROCESSING`.

`
Raw Image
  -> Grayscale conversion
  -> Intensity normalization (min-max -> [0, 1])
  -> CLAHE (contrast enhancement)
  -> Optional denoising (gaussian / median / bilateral)
  -> Optional pixel-space resize
  -> Multi-scale image pyramid
  -> Preprocessed result dict
`

### Available Preprocessing Operations

| Module | Operation | Key Config Keys |
|---|---|---|
| normalize.py | Min-max normalization | normalization_method |
| clahe.py | CLAHE | clahe_enabled, clahe_clip_limit, clahe_tile_grid_size |
| resize.py | Pixel resize / scale factor | resize_target_size, resize_keep_aspect_ratio |
| denoise.py | Gaussian / Median / Bilateral | denoise_method, denoise_kwargs |
| pyramid.py | Multi-scale pyramid | pyramid_enabled, pyramid_num_levels, pyramid_scale_factor |
| pipeline.py | Orchestrates all stages | All of the above |

### Pixel Resize vs Physical GSD Normalization
IMPORTANT: resize.py performs pixel-space resizing only. Physical GSD normalization (making OHRC 0.25m/px and LROC NAC 0.5m/px spatially consistent) is a separate concern addressed in a later phase.

### Configuring the Pipeline
Edit `configs/default.py -> Config.PREPROCESSING`. All keys have sensible defaults.

## Classical Feature-Matching Baseline (Phase 4)

Phase 4 establishes a classical computer vision baseline (SIFT and AKAZE) before deep learning or crater-graph methods:

```
Image A + Image B
       ↓
 Preprocessing
       ↓
┌──────────────┐
↓              ↓
SIFT         AKAZE
↓              ↓
Descriptor   Descriptor
matching     matching (Hamming for uint8, L2 for float)
↓              ↓
Ratio test   Ratio test (Lowe's ratio test)
└──────┬───────┘
       ↓
     RANSAC (Affine / Homography)
       ↓
 Geometric verification
       ↓
    Metrics (inliers, ratio, reprojection RMSE)
       ↓
 Visualization (matches, inliers, warp overlay)
```

### Key Modules

| Module | Description |
|---|---|
| `src/matching/feature_detector.py` | SIFT and AKAZE detector wrappers with consistent `detect_and_compute()` API. |
| `src/matching/descriptor_matcher.py` | FLANN / BFMatcher with Lowe's ratio test; automatically uses Hamming for binary and L2 for floating-point descriptors. |
| `src/matching/geometric_verification.py` | RANSAC transformation estimation supporting configurable `affine` (2x3) and `homography` (3x3) models. |
| `src/matching/metrics.py` | Inlier ratio, keypoint counts, candidate count, and reprojection RMSE calculation. |
| `src/matching/classical_baseline.py` | High-level baseline orchestrator: `run_classical_baseline(image_a, image_b, method='sift')`. |

### Limitation Note: Synthetic Verification vs Real Lunar Validation
Software verification tests and experiments run on controlled synthetic test pairs with known affine transformations. Synthetic success validates algorithmic correctness but does **NOT** imply solved cross-sensor matching on real lunar imagery (OHRC/LROC). Real lunar validation will occur once sensor pairs are placed in `data/raw/`.

## Crater Detection (Phase 5)

Phase 5 introduces the first structural matching component of the system, extracting standardized crater candidates from lunar images:

```
Lunar Image
     ↓
Preprocessing (Grayscale, CLAHE enhancement)
     ↓
Crater Detector Interface
┌───────────────────────────────┐
│ • MockCraterDetector          │ (Deterministic software verification)
│ • HoughCraterDetector         │ (Classical circular edge baseline)
│ • YOLOCraterDetector          │ (Deep learning model wrapper)
└───────────────────────────────┘
     ↓
Raw Candidate Detections
     ↓
Post-Processing Layer
  - Confidence filtering
  - Boundary & coordinate validation
  - Non-Maximum Suppression (NMS)
  - Candidate sorting & truncation
     ↓
Standardized Crater Candidates: {x, y, radius, confidence, bbox}
```

### Standardized Representation
Each detected crater is represented by `CraterCandidate` in 2D pixel space:
- `x, y`: Center coordinate in pixels (horizontal/vertical, origin at top-left).
- `radius`: Approximate crater radius in pixels ($r \approx \min(w, h) / 2$ for bounding boxes).
- `confidence`: Confidence score in $[0.0, 1.0]$.
- `bbox`: Optional raw bounding box `(xmin, ymin, xmax, ymax)`.

### Detector Implementations

| Detector | Purpose | Dependencies / Status |
|---|---|---|
| `MockCraterDetector` | Deterministic testing and offline validation without weights. | None (built-in). |
| `HoughCraterDetector` | Classical baseline using `cv2.HoughCircles` + local gradient contrast analysis. | OpenCV (built-in). Configurable `dp`, `min_dist`, `param1`, `param2`, `min_radius`, `max_radius`. |
| `YOLOCraterDetector` | Deep learning detector wrapper isolating neural network details. | Requires `ultralytics` package and trained weights file (`model_path`). Fails cleanly with clear messages when weights/package are absent. |

### Evaluation Infrastructure
Evaluation module (`src/crater_detection/evaluation.py`) provides greedy one-to-one bipartite matching between ground-truth and predicted craters using IoU thresholds to compute precision, recall, F1, and mean IoU.

### Limitation Note: Real Lunar Data vs Synthetic Verification
Synthetic crater fixtures are used solely for software, interface, and visualization verification. **Real lunar crater-detection performance is not yet evaluated** as labeled Chandrayaan-2 OHRC or LRO LROC NAC crater datasets have not yet been placed in `data/raw/`.

## Crater Graph (Phase 6)

Phase 6 constructs a structural graph representation $G = (V, E)$ over detected craters to capture spatial geometry for constellation matching:

```
Crater Candidates (Phase 5)
          ↓
Graph Construction Factory
┌─────────────────────────────────┐
│ • Delaunay Triangulation        │ (Default: sparse planar neighborhood, O(N) edges)
│ • k-Nearest Neighbors (KNN)     │ (k closest spatial neighbors per crater)
│ • Radius-Neighborhood Graph     │ (Euclidean distance threshold d <= R)
└─────────────────────────────────┘
          ↓
Crater Graph G = (V, E)
  - Nodes V: {node_id, x, y, radius, confidence, diameter, area, crater_ref}
  - Edges E: {distance_px, normalized_distance, relative_angle_rad, radius_ratio, weight}
```

### Distinction Between Matching Phases
- **Phase 6 (Current)**: Graph construction and topological representation in image space.
- **Phase 7 (Next)**: Scale/rotation-robust normalized relationships and invariant features.
- **Phase 8**: Crater constellation matching and cross-sensor graph correspondence.

### Node & Edge Attributes
- **Nodes ($V$)**: Image-space coordinates $(x, y)$, crater radius $r$, confidence, area, diameter, and reference to the original `CraterCandidate`.
- **Edges ($E$)**:
  - `distance_px`: Euclidean center distance $\sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}$.
  - `normalized_distance`: Distance scaled by reference crater scale (default: median radius $\tilde{r}$).
  - `relative_angle_rad`: Directed orientation $\text{atan2}(y_j - y_i, x_j - x_i) \in [-\pi, \pi]$.
  - `radius_ratio`: Symmetric scale relationship $\frac{\min(r_i, r_j)}{\max(r_i, r_j)} \in (0.0, 1.0]$.
  - `weight`: Edge weight (defaults to `distance_px`).

### Construction Strategies

| Method | Rationale & Characteristics | Default / Fallback |
|---|---|---|
| `delaunay` | **Default strategy**. Produces a planar, non-crossing spatial neighborhood with $O(N)$ sparse edges. Scale-adaptive to local crater density. | Automatically falls back to 1D ordered spatial edges for collinear or duplicate point configurations without crashing. |
| `knn` | Connects each crater to its $k$ nearest neighbors (symmetric union). | $k$ is automatically capped at $N-1$ when node count is small. |
| `radius` | Connects any two craters within a fixed pixel radius ($d(i, j) \le R_{threshold}$). | Configurable via `radius_threshold_px`. |

### Limitation Note: Graph Topology vs Full Cross-Sensor Invariance
Delaunay triangulation provides a planar spatial neighborhood, but raw image-space graph topology is **not** inherently invariant to sensor GSD differences, perspective skew, or partial occlusions. Phase 7 specifically focuses on establishing scale- and rotation-robust invariant representations before constellation matching (Phase 8).

## Scale/Rotation-Robust Normalized Relationships (Phase 7)

Phase 7 transforms the raw image-space relationships of `CraterGraph` into canonical, approximately scale- and rotation-invariant descriptors:

```
CraterGraph (Phase 6)
        ↓
Planar Similarity Invariance Layer
┌─────────────────────────────────────────────────────────────┐
│ • Scale Normalization: Global Median / Pairwise / Perimeter │
│ • Rotation Invariance: Relative Angles / Internal Angles   │
│ • Local Star-Neighborhood Descriptors                       │
│ • Canonical Permutation-Invariant Triangle Descriptors      │
└─────────────────────────────────────────────────────────────┘
        ↓
Phase 8-Ready Invariant Representation
```

### Why Raw Pixel Distances Are Insufficient
Images acquired across different sensors (e.g. Chandrayaan-2 OHRC at ~0.25 m/pixel vs TMC-2 at ~5 m/pixel or LROC NAC at ~0.5 m/pixel) differ by substantial unknown scale factors, translations, and rotation angles. Raw pixel coordinates and distances cannot be directly compared across sensors.

### Normalization Strategies
1. **Triangle Perimeter Normalization**: For 3-cliques with sides $d_1, d_2, d_3$, normalized sides $\hat{d}_k = d_k / (d_1 + d_2 + d_3)$ are strictly scale-independent and independent of crater radius measurement error.
2. **Global Median Radius Normalization**: $\hat{d}_{ij} = d_{ij} / \operatorname{median}(r_1, \dots, r_N)$.
3. **Pairwise Mean Radius Normalization**: $\hat{d}_{ij} = d_{ij} / ((r_i + r_j)/2)$.

### Descriptors
- **`TriangleDescriptor`**:
  - Encapsulates connected 3-cliques.
  - Features: perimeter-normalized side lengths, internal angles (via Law of Cosines), and max-radius normalized crater radii.
  - **Permutation Invariance**: Evaluates all 6 vertex permutations and chooses the lexicographically smallest representation, strictly maintaining correspondence between vertex $v_i$, opposite side $s_i$, internal angle $\alpha_i$, and crater radius $r_i$.
- **`LocalCraterDescriptor`**:
  - Encapsulates the star-neighborhood around a crater node.
  - Deterministic canonical neighbor ordering: sorted by `(normalized_distance, radius_ratio, node_id)`.
  - Directional angles: computed relative to the canonical reference neighbor ($\Delta\theta_m = \operatorname{wrap}_{(-\pi, \pi]}(\theta_m - \theta_{\text{ref}})$), guaranteeing rotational invariance.
  - Feature vector: fixed-length zero-padded array for fast Phase 8 matching.

### Separation of Responsibilities
- **Phase 6**: Graph topology and spatial neighborhood construction.
- **Phase 7**: Invariant descriptor extraction and similarity-robust relationships.
- **Phase 8 (Current)**: Local crater constellation matching and correspondence candidate generation.
- **Phase 9 (Next)**: Initial similarity-transformation estimation.

### Limitation Note: Real Lunar Validation
Descriptors are verified against synthetic constellations under exact planar similarity transformations ($x' = s R_\theta x + t, r' = s r$). **Real cross-sensor invariant-descriptor performance is not yet evaluated** pending real OHRC/LROC imagery in `data/raw/`.

## Local Crater Constellation Matching (Phase 8)

Phase 8 implements the first geometric correspondence stage: matching crater constellations between Image A and Image B using invariant descriptors:

```
Image A (CraterGraph + Invariants)      Image B (CraterGraph + Invariants)
                  \                    /
                   \                  /
                    ▼                ▼
         ┌───────────────────────────────────────────────┐
         │ Local Crater Constellation Matcher            │
         │ - Pairwise descriptor distance (no padding)   │
         │ - Distance thresholding (D <= D_max)          │
         │ - Lowe's ratio test (best vs second best)     │
         │ - Bidirectional mutual consistency            │
         │ - O(N_tri) supporting triangle verification   │
         │ - Confidence ranking                          │
         └───────────────────────────────────────────────┘
                                 ↓
                     Candidate Correspondences
                                 ↓
                  Phase 9 (Transform Estimation)
```

### Invariant Features Used
The matcher strictly operates on the invariant descriptors produced in Phase 7:
- Normalized neighbor distances
- Neighbor radius ratios
- Relative angular relationships
- Neighborhood size compatibility (excluding zero padding from distance calculations)
- Triangle 3-clique compatibility

> **Strict Architectural Rule**: No image coordinates $(x, y)$, pixel scales, or orientation angles are used for candidate generation. Matching is strictly descriptor-based.

### Variable Neighborhood Size Handling
Descriptors with different neighbor counts (e.g. $K_A \neq K_B$) compare only the $K_{\text{compat}} = \min(K_A, K_B)$ valid entries. Zero-padded entries in `feature_vector` are strictly ignored during distance calculation, and degree discrepancies are penalized via a transparent topological difference term:
$$D = w_{\text{dist}} E_{\text{dist}} + w_{\text{radius}} E_{\text{radius}} + w_{\text{angle}} E_{\text{angle}} + w_{\text{count}} \frac{|K_A - K_B|}{\max(K_A, K_B, 1)}$$

### Candidate Filtering & Confidence Scoring
- **Thresholding**: Candidate pairs with $D > D_{\max}$ are rejected.
- **Lowe's Ratio Test**: Ambiguous matches with $D_{\text{best}} / D_{\text{second\_best}} > \text{threshold}$ are rejected.
- **Mutual Consistency**: Only symmetric correspondences where $A \to B$ and $B \to A$ mutually agree are retained.
- **Triangle Support**: Mutually compatible 3-cliques ($T_A \approx T_B$) increment the supporting triangle count in $O(N_{\text{tri}})$ time via hash lookup.
- **Confidence**: Combines exponential distance decay with a triangle support bonus:
  $$\text{confidence} = \min(1.0, \exp(-D / \sigma) + \text{tri\_bonus})$$

### Limitation Note
Phase 8 performs synthetic geometric correspondence verification only. **Real cross-sensor lunar matching has not yet been evaluated because real OHRC/LROC imagery is not yet present in data/raw/.**

## Initial Similarity Transformation Estimation (Phase 9)

Phase 9 establishes the initial global geometric alignment between Image A and Image B from Phase 8 crater correspondences:

```
Phase 8 Correspondences: {(p_A_i, p_B_i)}
                     ↓
        Closed-Form Similarity Estimator
       ┌─────────────────────────────────┐
       │ • Centroid extraction           │
       │ • Coordinate centering          │
       │ • Covariance SVD: H = U Σ V^T   │
       │ • Reflection check (det(R) > 0) │
       │ • Scale s = tr(Σ)/σ_A^2         │
       │ • Rotation R = V U^T            │
       │ • Translation t = μ_B - s R μ_A │
       └─────────────────────────────────┘
                     ↓
         SimilarityTransform2D: p' = s R p + t
                     ↓
         Residual Error Diagnostics (RMSE)
                     ↓
          Phase 10 (Robust Estimation)
```

### Planar Similarity Model
The transformation model consists of 4 degrees of freedom:
$$p_B \approx s R(\theta) p_A + t$$
- $s > 0$: uniform scale factor
- $\theta \in (-\pi, \pi]$: 2D rotation angle with orthogonal matrix $R(\theta)$
- $t = (t_x, t_y)^T$: 2D translation vector

### Closed-Form Estimation & Reflection Policy
- **Umeyama Formulation**: Computes optimal least-squares scale, rotation, and translation in closed-form without iterative optimization.
- **Reflection Policy**: A physical rigid sensor-to-sensor transform cannot include a reflection ($\det(R) = -1$). By default (`allow_reflection=False`), reflected configurations are rejected with a clear `ValueError` rather than silently converting to an improper rotation.
- **Degeneracy Handling**: Minimum 2 point correspondences required. Coincident 2-point pairs or constellations with effectively zero spatial variance ($\sigma_A^2 < 10^{-8}$) are explicitly rejected.
- **Residual Diagnostics**: Exposes point alignment errors $e_i = T(p_{A, i}) - p_{B, i}$, RMSE, mean error, median error, and max residual.
- **Outlier Sensitivity**: Unweighted least-squares is sensitive to gross correspondence errors, demonstrating why robust estimation (RANSAC) is deferred to subsequent phases.

### Limitation Note
Phase 9 estimates an initial similarity transformation from Phase 8 crater correspondences. **It is not robust to arbitrary outliers; robust estimation is handled in Phase 10.**
**Real cross-sensor lunar transformation accuracy has not yet been evaluated because real OHRC/LROC imagery is not yet present in data/raw/.**

## Robust Similarity Transformation Estimation (Phase 10)

Phase 10 implements robust planar similarity estimation using RANSAC over Phase 8 crater correspondences, effectively rejecting false, ambiguous, or distractor correspondences:

```
Phase 8 Correspondences: {(p_A_i, p_B_i)}
                     ↓
        RANSAC Hypothesize-and-Verify Loop
       ┌────────────────────────────────────────────────────────┐
       │ 1. Minimal Sample (N=2, non-coincident pairs)          │
       │ 2. Candidate Fit via Phase 9 closed-form estimator     │
       │ 3. Reprojection Residuals: r_i = ||T(p_A_i) - p_B_i||  │
       │ 4. Consensus Classification: r_i <= threshold (3.0 px) │
       │ 5. Lexicographical Ranking: (inliers, -rmse, -tot_err) │
       │ 6. Adaptive Stopping: N = log(1-p) / log(1-w^s)        │
       └────────────────────────────────────────────────────────┘
                     ↓
           Best Consensus Inlier Set
                     ↓
       Single Final Inlier Refit (Phase 9)
                     ↓
        RANSACResult: {transform, inliers, outliers, rmse, ...}
```

### Key Components
- **Planar Similarity Model**: Strictly estimates $p_B \approx s R(\theta) p_A + t$ ($s, \theta, t_x, t_y$), preventing unphysical affine or homography shear distortions.
- **Minimal Sample Size**: Default `sample_size = 2` (the theoretical minimum for 2D similarity). Pairs with coincident source points or spatial variance below `degeneracy_epsilon` are discarded during sampling.
- **Deterministic Local PRNG**: Uses `np.random.default_rng(random_seed)` with a configurable seed (`random_seed: 42`). Identical runs produce bit-identical results without touching global random state.
- **Geometric Consensus Scoring**: Inliers are classified strictly by reprojection residuals ($r_i \le \text{reprojection\_threshold}$). Hypotheses are ranked deterministically by `(num_inliers, -inlier_rmse, -total_inlier_error)`.
- **Adaptive Stopping Rule**: Dynamically updates the iteration ceiling based on estimated inlier ratio $w$:
  $$N = \frac{\log(1 - p)}{\log(1 - w^s)}$$
  guarded against $w \le 0$ and $w \ge 1$.
- **Final Inlier Least-Squares Refit**: Refits the winning consensus inliers once via Phase 9 closed-form estimator, providing optimal sub-pixel transformation parameters.
- **Failure Handling**: If detected inliers are fewer than `min_inliers` (default 3), raises a descriptive `ValueError`.

### Limitation Note
Phase 10 provides robust similarity transformation estimation using RANSAC over Phase 8 crater correspondences. **It is still based entirely on synthetic verification because real OHRC/LROC imagery is not yet available.**

## Hybrid Matching Engine (Milestone A: Phases 11 + 12 + 13)

Milestone A unifies the structural **Classical Crater Branch** (Phases 5–8) and the dense **Learned Image Branch** (LoFTR / Phase 11) into a single, cohesive, model-side matching engine:

```
                    IMAGE A
                       │
                    IMAGE B
                       │
                       ▼
                Preprocessing (Phase 3)
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼
   CLASSICAL CRATER           LEARNED IMAGE
       BRANCH                    BRANCH
          │                         │
     Phases 5–8                   LoFTR
          │                         │
          ▼                         ▼
   Crater correspondences   Learned correspondences
          │                         │
          └────────────┬────────────┘
                       ▼
              CORRESPONDENCE FUSION (Phase 12)
                       │
                       ▼
                 FINAL RANSAC (Phase 10 / 13)
                       │
                       ▼
              HYBRID MATCH RESULT
```

### Core Innovations & Architecture
1. **Learned Matcher Abstraction (`BaseLearnedMatcher`)**:
   - Clean interface returning `LearnedMatchResult`.
   - `LoFTRMatcher`: Pretrained transformer matcher (`kornia.feature.LoFTR`) with lazy loading and automated multiple-of-8 spatial padding.
   - **Graceful Fallback**: If weights, CUDA, or dependencies are unavailable, `LoFTRMatcher` reports `available=False` with error metadata. It never fabricates correspondences and never breaks the classical crater pipeline.
   - `MockLearnedMatcher`: Deterministic mock for fast unit tests and headless environments.
2. **Common Correspondence Representation**:
   - Both branches convert to unified `@dataclass Correspondence(point_a, point_b, confidence, source, source_index, metadata)`.
3. **Deterministic Correspondence Fusion (`fuse_correspondences`)**:
   - Rejects invalid coordinates (NaN, Inf, negative, or beyond image boundaries).
   - Spatial duplicate resolution: pairs within `duplicate_tolerance_px` (default 4.0 px) merge into the higher-confidence correspondence.
   - Source balancing: caps correspondences per branch via `max_correspondences_per_source` to prevent dense learned points from overwhelming sparse crater landmarks### Limitation Note
Milestone A provides a clean, model-side matching engine ready for downstream application/backend integration.
- **Real OHRC/LROC imagery is not yet present in `data/raw/`**; all evaluations remain synthetic.
- Does **NOT** include web UI, Streamlit, React frontend, Node backend, REST API, or deployment infrastructure.

## Milestone B — Registration & Quality Engine (Combined Phases 14 + 15 + 16 + 17)

Milestone B consumes the geometrically verified `HybridMatchResult` from Milestone A and turns its consensus correspondences into an aligned/registered image and a quantitative engineering quality report.

```text
                  HybridMatchResult (Milestone A)
                                │
                                ▼
                   1. Verified Inlier Extraction
                      (src/registration/inliers.py)
                                │
                                ▼
                  2. Uniform Tie-Point Selection
                      (src/registration/tie_points.py)
                                │
                                ▼
                  3. Optional Sub-Pixel Refinement
                      (src/registration/refinement.py)
                                │
                                ▼
                 4. Final Transformation Refit
                      (estimate_similarity_transform / Phase 9)
                                │
                                ▼
                     5. Image Registration / Warping
                      (src/registration/register.py)
                                │
                                ▼
                  6. Quality Assessment & Classification
                      (src/registration/quality.py)
                                │
                                ▼
                        RegistrationResult
                      (src/registration/registration_engine.py)
```

### Core Innovations & Architecture
1. **Verified Inlier Extraction (`extract_verified_inliers`)**:
   - Strictly consumes RANSAC inlier indices (`result.inlier_indices`), discarding outliers.
   - Validates finite coordinates and minimum point thresholds.
   - Preserves provenance labels (`'crater'` vs `'learned'`).
2. **Uniform Spatial Tie-Point Selection (`select_uniform_tie_points`)**:
   - Partitions Image A's domain into a regular spatial grid (`grid_rows x grid_cols`, default 6x6).
   - Assigns inliers to cells and retains the strongest candidate per cell (highest confidence with deterministic index tie-breaking).
   - Enforces `max_points` global cap while maximizing spatial coverage across the overlap domain.
   - Computes transparent spatial coverage metric: $\text{coverage} = \frac{\text{num\_occupied\_cells}}{\text{total\_cells}}$.
3. **Sub-Pixel / Local Refinement (`refine_tie_points`)**:
   - Optional local template matching (toggled via `Config.SUBPIXEL_REFINEMENT["enabled"]`).
   - Rotation- and scale-aware: samples patches from Image B using the coarse similarity transform $T$.
   - Continuous 2D parabolic peak interpolation over normalized cross-correlation surface.
   - **Graceful Fallback**: If correlation is weak ($< \rho_{\min}$), near image boundary, or displacement exceeds `max_shift_px`, original coordinates are retained without failing the pipeline.
4. **Image Registration & Warping (`register_image`)**:
   - SimilarityTransform2D represents $p_B \approx T(p_A) = s R(\theta) p_A + \mathbf{t}$.
   - Pull-based image resampling (`cv2.warpAffine`) evaluates source pixel coordinates from destination coordinates. To warp Image B back into Image A's coordinate space, it applies the exact inverse transform $T^{-1}$ mapping B $\to$ A.
   - Verified by dedicated synthetic tests asserting that transformed features align with reference centroids to $< 0.5$ px.
   - Warps source mask to produce pixel-accurate `valid_mask` (255 for valid overlap pixels, 0 for borders).
5. **Quantitative Quality & Classification (`compute_registration_quality`)**:
   - Evaluates residual errors: RMSE, mean, median, max residual error.
   - Calculates bounded overall confidence score in $[0.0, 1.0]$:
     $$\text{confidence} = 0.35 S_{\text{inlier}} + 0.30 S_{\text{rmse}} + 0.20 S_{\text{coverage}} + 0.15 S_{\text{points}}$$
   - Classifies registration into engineering categories: `EXCELLENT`, `GOOD`, `FAIR`, `POOR`, `FAILED`.
6. **Unified `RegistrationResult` & `RegistrationEngine`**:
   - Clean orchestrator: `RegistrationEngine().register(image_a, image_b, hybrid_result)`.
   - Lightweight backend-friendly serialization via `to_dict(include_arrays=False)` avoiding multi-megabyte JSON payloads.

### Model-Side API Invocation Example
```python
from src.matching.hybrid_matcher import HybridMatcher
from src.registration.registration_engine import RegistrationEngine

# 1. Execute hybrid matching
matcher = HybridMatcher()
hybrid_result = matcher.match(image_a, image_b)

# 2. Execute registration and quality assessment
engine = RegistrationEngine()
reg_result = engine.register(image_a, image_b, hybrid_result)

if reg_result.success:
    print(f"Registration Succeeded! Quality: {reg_result.quality}")
    print(f"Final RMSE: {reg_result.rmse:.4f} px")
    print(f"Spatial Coverage: {reg_result.coverage * 100:.1f}%")
    print(f"Registration Confidence: {reg_result.confidence:.4f}")
    print(f"Selected Tie Points: {reg_result.num_tie_points}/{reg_result.num_inliers}")
    
    # Registered image and valid mask
    aligned_b = reg_result.registered_image
    overlap_mask = reg_result.valid_mask
    
    # Lightweight JSON payload for backend/React response
    payload = reg_result.to_dict(include_arrays=False)
else:
    print(f"Registration Failed: {reg_result.metadata.get('reason')}")
```

### Limitation Note
- **Real OHRC/LROC imagery is not yet present in `data/raw/`**; all evaluations remain synthetic.
- Quality categories are engineering heuristics; no scientific planetary ground-truth accuracy has been established.
- Strictly model-side Python; no web frontend, Node backend, REST API, or database is included.

## Usage Commands
- **Environment Check**: `python verify_env.py`
- **Run all tests**: `python -m pytest -q`
- **Dataset validation**: `python src/preprocessing/validate_dataset.py`
- **Dataset inspection**: `python experiments/inspect_dataset.py`
- **Preprocessing visualization**: `python experiments/inspect_preprocessing.py`
- **Classical baseline experiment**: `python experiments/run_classical_baseline.py`
- **Crater detection inspection**: `python experiments/inspect_crater_detection.py`
- **Crater graph inspection**: `python experiments/inspect_crater_graph.py`
- **Crater invariants inspection**: `python experiments/inspect_crater_invariants.py`
- **Constellation matching inspection**: `python experiments/inspect_constellation_matching.py`
- **Transformation estimation inspection**: `python experiments/inspect_transformation_estimation.py`
- **RANSAC robust estimation inspection**: `python experiments/inspect_ransac.py`
- **Hybrid matching inspection**: `python experiments/inspect_hybrid_matching.py`
- **Registration & quality inspection**: `python experiments/inspect_registration.py`

## Repository Structure
```
lunar-image-matching/
  README.md
  requirements.txt
  verify_env.py
  configs/default.py        <- Configuration (PREPROCESSING, CLASSICAL_MATCHING, CRATER_DETECTION, CRATER_GRAPH, CRATER_INVARIANTS, CRATER_MATCHING, TRANSFORMATION_ESTIMATION, RANSAC, LEARNED_MATCHING, CORRESPONDENCE_FUSION, HYBRID_MATCHING, TIE_POINT_SELECTION, SUBPIXEL_REFINEMENT, REGISTRATION, REGISTRATION_QUALITY)
  data/
    raw/                    <- Real lunar images go here (not committed)
    processed/              <- Pipeline and experiment outputs
    dataset_meta.json       <- Image + pair metadata
  src/
    preprocessing/          <- Image loading, normalization, CLAHE, denoising, pyramid
    matching/               <- SIFT/AKAZE, matcher, RANSAC (Phase 10), transformation (Phase 9), learned_matcher (Phase 11), correspondence_fusion (Phase 12), hybrid_matcher (Milestone A)
    registration/           <- inliers (Phase 14/15), tie_points (Phase 15), refinement (Phase 14), register (Phase 16), quality (Phase 17), registration_engine (Milestone B)
    crater_detection/       <- CraterCandidate types, postprocessing, detectors, pipeline, evaluation
    crater_graph/           <- CraterGraph, builder, features, invariants, constellation_matcher, visualization
  experiments/
    inspect_dataset.py                 <- Phase 2 dataset inspection
    inspect_preprocessing.py          <- Phase 3 pipeline visualization
    run_classical_baseline.py         <- Phase 4 SIFT vs AKAZE comparison
    inspect_crater_detection.py       <- Phase 5 Crater detection visualization
    inspect_crater_graph.py           <- Phase 6 Crater graph visualization
    inspect_crater_invariants.py       <- Phase 7 Invariant descriptor inspection
    inspect_constellation_matching.py  <- Phase 8 Crater constellation matching demo
    inspect_transformation_estimation.py <- Phase 9 Similarity transformation demo
    inspect_ransac.py                  <- Phase 10 RANSAC robust estimation demo
    inspect_hybrid_matching.py         <- Milestone A Hybrid matching demo
    inspect_registration.py            <- Milestone B Registration & quality demo
  tests/
    test_image_loader.py
    test_validate_dataset.py
    test_preprocessing.py
    test_classical_matching.py
    test_crater_detection.py
    test_crater_graph.py
    test_crater_invariants.py
    test_constellation_matcher.py
    test_transformation.py
    test_ransac.py
    test_hybrid_matching.py
    test_registration.py
    test_evaluation.py
```

## Development Philosophy
- **Incremental Development**: One verified step at a time.
- **No Placeholders**: No hollow stubs that pretend to work.
- **Verification First**: Do not move to the next phase until the current phase is tested.

## Current Implementation Status
- **Phase 1** - COMPLETE: Project scaffolding, configuration, environment verification.
- **Phase 2** - COMPLETE: Image loader, ImagePair abstraction, dataset metadata validation.
- **Phase 3** - COMPLETE: Full preprocessing pipeline (grayscale, normalization, CLAHE, denoising, resize, pyramid).
- **Phase 4** - COMPLETE: Classical feature-matching baseline (SIFT/AKAZE + RANSAC).
- **Phase 5** - COMPLETE: Crater Detection (standardized representation, Mock/Hough/YOLO abstraction, post-processing, evaluation infrastructure).
- **Phase 6** - COMPLETE: Crater Graph (CraterGraph model, Delaunay/KNN/Radius builders, edge attributes, feature extraction, visualization).
- **Phase 7** - COMPLETE: Scale/Rotation-Robust Normalized Relationships (local & triangle invariant descriptors, canonicalization, similarity-transform invariance).
- **Phase 8** - COMPLETE: Local Crater Constellation Matching (descriptor distance, variable neighborhood support, mutual consistency, triangle support, confidence scoring).
- **Phase 9** - COMPLETE: Initial Similarity Transformation Estimation (closed-form Umeyama 2D similarity, reflection rejection, residual diagnostics, outlier sensitivity demo).
- **Phase 10** - COMPLETE: Robust Transformation Estimation (RANSAC & Inlier Refinement).
- **Milestone A (Phases 11+12+13)** - COMPLETE: Hybrid Matching Engine (LoFTR learned matcher, correspondence fusion, Phase 10 RANSAC integration, unified HybridMatchResult).
- **Milestone B (Phases 14+15+16+17)** - COMPLETE: Registration & Quality Engine (verified inlier extraction, uniform tie-point selection, sub-pixel refinement, pull-based image registration, quantitative quality metrics, and unified RegistrationResult).
- **Milestone C (Phases 18+19)** - COMPLETE: Evaluation & Robustness Engine (synthetic ground-truth error metrics, 10 difficult stress test cases, FailureReason taxonomy, RobustnessRunner, hyperparameter sensitivity analysis, and Evaluator facade).

---

## Milestone C: Evaluation & Robustness Framework

Milestone C provides a model-side framework to evaluate performance, failure modes, and robustness under controlled stress conditions.

### 1. Architectural Pipeline
```
Image A + Image B (Optional Ground Truth)
                  │
                  ▼
          HybridMatcher (Milestone A)
                  │
                  ▼
       RegistrationEngine (Milestone B)
                  │
                  ▼
         RegistrationResult
                  │
     ┌────────────┴────────────┐
     ▼                         ▼
Ground-Truth Comparison   Failure Classification (Taxonomy)
     │                         │
     └────────────┬────────────┘
                  ▼
        EvaluationMetrics & EvaluationCaseResult
                  │
                  ▼
          EvaluationSummary
```

### 2. Ground-Truth Transformation Metrics
- **Scale Discrepancy**: Absolute $|s_{\text{est}} - s_{\text{gt}}|$ and relative $|s_{\text{est}} - s_{\text{gt}}| / s_{\text{gt}}$.
- **Rotation Error**: Shortest angular distance on the circle with strict wraparound handling in $(-180^\circ, 180^\circ]$:
  $$\Delta\theta = |(\theta_{\text{est}} - \theta_{\text{gt}} + 180^\circ) \pmod{360^\circ} - 180^\circ|$$
- **Translation Error**: 2D Euclidean distance:
  $$\Delta t = \sqrt{(t_{x,\text{est}} - t_{x,\text{gt}})^2 + (t_{y,\text{est}} - t_{y,\text{gt}})^2}$$

### 3. Failure Taxonomy
Structured failure reasons without generic catch-all errors:
- `NONE`: Successful alignment meeting all criteria.
- `NO_VALID_MATCHES`: Zero initial correspondences generated.
- `INSUFFICIENT_INLIERS`: Geometric consensus failed to reach the consensus threshold.
- `LOW_SPATIAL_COVERAGE`: Inliers clustered in an inadequate fraction of grid cells.
- `LOW_CONFIDENCE`: Aggregate alignment confidence fell below threshold.
- `HIGH_REPROJECTION_ERROR`: Residual RMSE exceeded acceptable bounds.
- `TRANSFORM_ERROR_EXCEEDED`: Estimated similarity transform diverged from ground truth.
- `REGISTRATION_FAILURE`: Image warping or matrix refit failure.
- `INVALID_INPUT`: Malformed image buffers.
- `UNKNOWN_FAILURE`: Unhandled execution exceptions.

### 4. Controlled Difficult Cases Suite
- **Case A — Resolution Mismatch**: 2.5x downsampling simulating cross-sensor GSD divergence (OHRC vs LROC/TMC).
- **Case B — Illumination Mismatch**: Contrast scaling (0.6), brightness shift (+35), and non-linear gamma (1.5).
- **Case C — Large Rotation**: $90^\circ$ orientation shift without rotational priors.
- **Case D — Scale Mismatch**: Substantial scale difference ($s=1.65$).
- **Case E — Partial Overlap**: Large spatial translation leaving only $\sim 35\%$ scene overlap.
- **Case F — Distractors**: Prominent spurious crater-like structures inserted into Image B.
- **Case G — Sparse Craters**: Only 2 crater structures present.
- **Case H — Dense Craters**: 35+ tightly clustered craters with similar radii.
- **Case I — Texture-Poor Region**: Low-gradient lunar mare plain with subtle features.
- **Case J — Noise and Blur**: Severe sensor noise ($\sigma=15.0$) + optical blur ($k=5$).

### 5. Hyperparameter Sensitivity
- Sub-pixel parabolic refinement reduces alignment RMSE by over $60\%$ (from 0.381 px to 0.149 px).
- RANSAC reprojection thresholds (1.5 px to 5.0 px) demonstrate wide stability basins on consensus inliers.
- LoFTR confidence thresholds (0.1 to 0.4) show robust filtering behavior without inlier starvation.

### 6. Real Data Integration & Boundary Statement
> [!IMPORTANT]
> **Data Boundary Statement**:
> Real Chandrayaan-2 OHRC / TMC-2 and LRO LROC NAC mission imagery has not yet been staged in `data/raw/`.
> All current evaluation results reflect controlled software verification on synthetic benchmarks.
> Scientific lunar registration accuracy on real planetary flight data will be evaluated when real orbital imagery is ingested.
