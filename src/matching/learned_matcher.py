"""
src/matching/learned_matcher.py

Learned deep feature matching abstraction and pretrained LoFTR integration.
- LearnedCorrespondence: Dataclass encapsulating point_a, point_b, and confidence.
- LearnedMatchResult: Container for learned matches, availability status, and metadata.
- BaseLearnedMatcher: Abstract interface for deep matching backends.
- MockLearnedMatcher: Deterministic mock matcher for fast unit testing and offline environments.
- LoFTRMatcher: Pretrained Local Feature TRansformer (LoFTR) via Kornia with graceful degradation.

ARCHITECTURAL RULES:
- LoFTR is wrapped with lazy loading and robust try-except guards.
- If dependencies or weights are unavailable, cleanly returns available=False with diagnostics.
- Never fabricates correspondences when the learned backend fails.
- Operates on grayscale/normalized images.
- Independent of the classical crater branch.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import logging
import numpy as np

from src.matching.progress import ProgressEvent

logger = logging.getLogger(__name__)


@dataclass
class LearnedCorrespondence:
    """
    Candidate correspondence predicted by a learned deep feature matcher.

    Attributes:
        point_a: (2,) coordinate array [x, y] in Image A.
        point_b: (2,) coordinate array [x, y] in Image B.
        confidence: Confidence score in [0.0, 1.0].
    """
    point_a: np.ndarray
    point_b: np.ndarray
    confidence: float

    def __post_init__(self):
        self.point_a = np.asarray(self.point_a, dtype=np.float64).flatten()
        self.point_b = np.asarray(self.point_b, dtype=np.float64).flatten()
        if self.point_a.shape != (2,):
            raise ValueError(f"point_a must have shape (2,), got {self.point_a.shape}.")
        if self.point_b.shape != (2,):
            raise ValueError(f"point_b must have shape (2,), got {self.point_b.shape}.")
        self.confidence = float(self.confidence)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize learned correspondence to primitive dictionary."""
        return {
            "point_a": [round(float(v), 4) for v in self.point_a],
            "point_b": [round(float(v), 4) for v in self.point_b],
            "confidence": round(float(self.confidence), 4),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LearnedCorrespondence":
        """Deserialize learned correspondence from dictionary."""
        return cls(
            point_a=np.array(data["point_a"], dtype=np.float64),
            point_b=np.array(data["point_b"], dtype=np.float64),
            confidence=float(data["confidence"]),
        )


@dataclass
class LearnedMatchResult:
    """
    Result container for learned matching.

    Attributes:
        correspondences: List of detected LearnedCorrespondence objects.
        available: Boolean indicating whether the learned model executed successfully.
        backend: Name of the backend ('loftr', 'mock', etc.).
        metadata: Diagnostic information, device, execution time, error reasons if unavailable.
    """
    correspondences: List[LearnedCorrespondence]
    available: bool
    backend: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_correspondences(self) -> int:
        """Number of correspondences found."""
        return len(self.correspondences)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize result to primitive dictionary."""
        return {
            "correspondences": [c.to_dict() for c in self.correspondences],
            "num_correspondences": len(self.correspondences),
            "available": bool(self.available),
            "backend": str(self.backend),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LearnedMatchResult":
        """Deserialize result from dictionary."""
        corrs = [LearnedCorrespondence.from_dict(c) for c in data.get("correspondences", [])]
        return cls(
            correspondences=corrs,
            available=bool(data.get("available", False)),
            backend=str(data.get("backend", "unknown")),
            metadata=dict(data.get("metadata", {})),
        )


def generate_tile_ranges(length: int, tile_size: int, tile_overlap: int) -> List[Tuple[int, int]]:
    """
    Deterministically generate tile index intervals [start, end) for a given dimension length.
    Ensures complete coverage from 0 to length without empty padding tiles.
    """
    if length <= 0:
        return []
    if length <= tile_size:
        return [(0, length)]

    stride = max(1, tile_size - tile_overlap)
    ranges: List[Tuple[int, int]] = []
    start = 0
    while start + tile_size < length:
        ranges.append((start, start + tile_size))
        start += stride

    last_start = max(0, length - tile_size)
    if not ranges or ranges[-1][0] != last_start:
        ranges.append((last_start, length))

    return ranges


def deduplicate_correspondences(
    correspondences: List[LearnedCorrespondence],
    tolerance_px: float = 3.0,
) -> List[LearnedCorrespondence]:
    """
    Deterministically deduplicate keypoint correspondences in original image space.
    Uses spatial grid hashing for O(N) performance on large correspondence sets.
    """
    if not correspondences:
        return []

    def sort_key(c: LearnedCorrespondence):
        return (
            -c.confidence,
            float(c.point_a[0]),
            float(c.point_a[1]),
            float(c.point_b[0]),
            float(c.point_b[1]),
        )

    sorted_corrs = sorted(correspondences, key=sort_key)
    deduped: List[LearnedCorrespondence] = []
    
    grid: Dict[Tuple[int, int], List[LearnedCorrespondence]] = {}
    cell_size = max(1.0, float(tolerance_px))
    tol_sq = tolerance_px * tolerance_px

    for cand in sorted_corrs:
        pt_a = cand.point_a
        pt_b = cand.point_b
        cell_x = int(pt_a[0] // cell_size)
        cell_y = int(pt_a[1] // cell_size)

        is_dup = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cell_key = (cell_x + dx, cell_y + dy)
                if cell_key in grid:
                    for acc in grid[cell_key]:
                        da_sq = (pt_a[0] - acc.point_a[0]) ** 2 + (pt_a[1] - acc.point_a[1]) ** 2
                        db_sq = (pt_b[0] - acc.point_b[0]) ** 2 + (pt_b[1] - acc.point_b[1]) ** 2
                        if da_sq <= tol_sq and db_sq <= tol_sq:
                            is_dup = True
                            break
                if is_dup:
                    break
            if is_dup:
                break

        if not is_dup:
            deduped.append(cand)
            cell_key = (cell_x, cell_y)
            if cell_key not in grid:
                grid[cell_key] = []
            grid[cell_key].append(cand)

    return deduped


class BaseLearnedMatcher(ABC):
    """Abstract base class for learned deep feature matchers."""

    @abstractmethod
    def match(
        self,
        image_a: np.ndarray,
        image_b: np.ndarray,
        progress_callback: Optional[Any] = None,
    ) -> LearnedMatchResult:
        """
        Estimate dense/semi-dense keypoint correspondences between Image A and Image B.

        Args:
            image_a: 2D grayscale image array (H, W).
            image_b: 2D grayscale image array (H, W).
            progress_callback: Optional callback receiving ProgressEvent objects.

        Returns:
            LearnedMatchResult containing correspondences and status flags.
        """
        pass


class MockLearnedMatcher(BaseLearnedMatcher):
    """
    Deterministic mock learned matcher for unit testing and offline evaluation.
    Returns predefined or parametrically transformed correspondences without neural network overhead.
    """

    def __init__(
        self,
        mock_correspondences: Optional[List[LearnedCorrespondence]] = None,
        available: bool = True,
        fail_reason: Optional[str] = None,
    ):
        self.mock_correspondences = mock_correspondences or []
        self.available = available
        self.fail_reason = fail_reason

    def match(
        self,
        image_a: np.ndarray,
        image_b: np.ndarray,
        progress_callback: Optional[Any] = None,
    ) -> LearnedMatchResult:
        if not self.available:
            return LearnedMatchResult(
                correspondences=[],
                available=False,
                backend="mock",
                metadata={"reason": self.fail_reason or "Mock learned matcher configured as unavailable."},
            )

        return LearnedMatchResult(
            correspondences=list(self.mock_correspondences),
            available=True,
            backend="mock",
            metadata={"source": "mock_generator"},
        )


class LoFTRMatcher(BaseLearnedMatcher):
    """
    Pretrained Local Feature TRansformer (LoFTR) matcher using Kornia.
    
    Features:
    - Lazy model loading on first inference call.
    - Graceful fallback reporting available=False on missing packages, SSL errors, or runtime exceptions.
    - Automatic input shape validation, normalization, and multiple-of-8 padding.
    - Memory-bounded spatial tiling for large real-world lunar images.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config: Dict[str, Any] = {}
        if config is not None:
            self.config.update(config)
        else:
            try:
                from configs.default import Config
                self.config.update(getattr(Config, "LEARNED_MATCHING", {}))
            except (ImportError, AttributeError):
                pass

        self.pretrained: str = self.config.get("pretrained", "outdoor")
        self.min_confidence: float = float(self.config.get("min_confidence", 0.2))
        self.device_str: str = self.config.get("device", "cpu")
        self.max_image_dimension: Optional[int] = self.config.get("max_image_dimension", 640)
        self.tiling_mode: str = str(self.config.get("tiling_mode", "auto")).lower()
        self.tile_size: int = int(self.config.get("tile_size", 640))
        self.tile_overlap: int = int(self.config.get("tile_overlap", 160))
        self.max_tiles: int = int(self.config.get("max_tiles", 100))
        self.duplicate_tolerance_px: float = float(self.config.get("duplicate_tolerance_px", 3.0))

        self._model = None
        self._init_failed: bool = False
        self._init_error_msg: Optional[str] = None

    def _ensure_initialized(self) -> bool:
        """
        Lazily initialize the LoFTR model.
        Returns True if model is successfully loaded, False otherwise.
        """
        if self._model is not None:
            return True
        if self._init_failed:
            return False

        try:
            import torch
            # Configure default SSL context with certifi if available
            try:
                import certifi
                import ssl
                ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
            except Exception:
                pass

            from kornia.feature import LoFTR

            device = torch.device(self.device_str if torch.cuda.is_available() and self.device_str == "cuda" else "cpu")
            logger.info("Initializing LoFTR matcher (pretrained='%s', device=%s)...", self.pretrained, device)
            model = LoFTR(pretrained=self.pretrained)
            model = model.to(device)
            model.eval()

            self._model = model
            self._device = device
            return True

        except Exception as e:
            self._init_failed = True
            self._init_error_msg = f"Failed to initialize LoFTR: {type(e).__name__} - {str(e)}"
            logger.warning("LoFTRMatcher unavailable: %s", self._init_error_msg)
            return False

    def _preprocess_image(self, img: np.ndarray):
        """
        Convert input image array to a normalized PyTorch tensor with dimensions divisible by 8.
        """
        import torch

        arr = np.asarray(img, dtype=np.float32)
        if arr.ndim == 3:
            if arr.shape[2] == 1:
                arr = arr[:, :, 0]
            elif arr.shape[2] in (3, 4):
                # Simple luminance conversion if RGB/RGBA
                arr = 0.2989 * arr[:, :, 0] + 0.5870 * arr[:, :, 1] + 0.1140 * arr[:, :, 2]
            else:
                raise ValueError(f"Unsupported image shape {arr.shape}.")

        if arr.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape {arr.shape}.")

        orig_h, orig_w = arr.shape
        if orig_h == 0 or orig_w == 0:
            raise ValueError("Input image has zero dimension.")

        # Normalize to [0.0, 1.0]
        max_val = float(np.max(arr))
        if max_val > 1.0:
            arr = arr / 255.0
        arr = np.clip(arr, 0.0, 1.0)

        # Pad right and bottom to multiple of 8
        pad_h = (8 - (orig_h % 8)) % 8
        pad_w = (8 - (orig_w % 8)) % 8

        if pad_h > 0 or pad_w > 0:
            arr = np.pad(arr, ((0, pad_h), (0, pad_w)), mode="reflect")

        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(self._device)
        return tensor, orig_h, orig_w

    def _match_single_pass(
        self,
        image_a: np.ndarray,
        image_b: np.ndarray,
    ) -> List[LearnedCorrespondence]:
        """
        Run single-pass LoFTR feature matching on Image A and Image B arrays.
        Operates on sub-tiles or whole small images.
        """
        import torch

        t_a, h_a, w_a = self._preprocess_image(image_a)
        t_b, h_b, w_b = self._preprocess_image(image_b)

        input_dict = {"image0": t_a, "image1": t_b}

        with torch.no_grad():
            output = self._model(input_dict)

        kpts0 = output["keypoints0"].cpu().numpy()
        kpts1 = output["keypoints1"].cpu().numpy()
        conf = output["confidence"].cpu().numpy()

        correspondences: List[LearnedCorrespondence] = []
        for p0, p1, c in zip(kpts0, kpts1, conf):
            c_val = float(c)
            if c_val < self.min_confidence:
                continue
            # Discard coordinates falling outside original unpadded image
            if p0[0] >= w_a or p0[1] >= h_a or p1[0] >= w_b or p1[1] >= h_b:
                continue
            correspondences.append(
                LearnedCorrespondence(
                    point_a=np.array([float(p0[0]), float(p0[1])], dtype=np.float64),
                    point_b=np.array([float(p1[0]), float(p1[1])], dtype=np.float64),
                    confidence=c_val,
                )
            )
        return correspondences

    def match(
        self,
        image_a: np.ndarray,
        image_b: np.ndarray,
        progress_callback: Optional[Any] = None,
    ) -> LearnedMatchResult:
        """Run LoFTR feature matching on Image A and Image B with automatic memory-bounded tiling."""
        if not self._ensure_initialized():
            return LearnedMatchResult(
                correspondences=[],
                available=False,
                backend="loftr",
                metadata={"reason": self._init_error_msg or "Model initialization failed."},
            )

        try:
            arr_a = np.asarray(image_a)
            arr_b = np.asarray(image_b)
            h_a, w_a = arr_a.shape[:2]
            h_b, w_b = arr_b.shape[:2]
            shape_a = (h_a, w_a)
            shape_b = (h_b, w_b)
            max_dim = max(h_a, w_a, h_b, w_b)

            if self.tiling_mode == "always":
                tiling_enabled = True
            elif self.tiling_mode == "never":
                tiling_enabled = False
            else:  # "auto"
                tiling_enabled = (self.max_image_dimension is not None) and (max_dim > self.max_image_dimension)

            if not tiling_enabled:
                raw_corrs = self._match_single_pass(arr_a, arr_b)
                deduped_corrs = deduplicate_correspondences(raw_corrs, tolerance_px=self.duplicate_tolerance_px)

                logger.info(
                    "LoFTR matching summary: image_a_shape=%s, image_b_shape=%s, tiling_enabled=False, "
                    "num_tiles_a=1, num_tiles_b=1, tile_size=%d, tile_overlap=%d, tile_pairs_processed=1, "
                    "raw_tile_correspondences=%d, deduplicated_correspondences=%d",
                    shape_a, shape_b, self.tile_size, self.tile_overlap, len(raw_corrs), len(deduped_corrs)
                )

                return LearnedMatchResult(
                    correspondences=deduped_corrs,
                    available=True,
                    backend="loftr",
                    metadata={
                        "pretrained": self.pretrained,
                        "device": str(self._device),
                        "tiling_enabled": False,
                        "image_a_shape": list(shape_a),
                        "image_b_shape": list(shape_b),
                        "num_tiles_a": 1,
                        "num_tiles_b": 1,
                        "tile_size": self.tile_size,
                        "tile_overlap": self.tile_overlap,
                        "tile_pairs_processed": 1,
                        "raw_tile_correspondences_count": len(raw_corrs),
                        "deduplicated_correspondences_count": len(deduped_corrs),
                        "min_confidence": self.min_confidence,
                    },
                )

            rows_a = generate_tile_ranges(h_a, self.tile_size, self.tile_overlap)
            cols_a = generate_tile_ranges(w_a, self.tile_size, self.tile_overlap)
            rows_b = generate_tile_ranges(h_b, self.tile_size, self.tile_overlap)
            cols_b = generate_tile_ranges(w_b, self.tile_size, self.tile_overlap)

            num_tiles_a = len(rows_a) * len(cols_a)
            num_tiles_b = len(rows_b) * len(cols_b)

            if len(rows_a) != len(rows_b) or len(cols_a) != len(cols_b):
                logger.warning(
                    "Tiled matching incompatible spatial grid shapes: Image A tiles (%d, %d) vs Image B tiles (%d, %d)",
                    len(rows_a), len(cols_a), len(rows_b), len(cols_b)
                )
                return LearnedMatchResult(
                    correspondences=[],
                    available=False,
                    backend="loftr",
                    metadata={
                        "reason": f"Incompatible tile grids: Image A ({len(rows_a)}x{len(cols_a)}) vs Image B ({len(rows_b)}x{len(cols_b)})."
                    },
                )

            tile_pairs_count = len(rows_a) * len(cols_a)
            if tile_pairs_count > self.max_tiles:
                logger.warning(
                    "Tiled matching requested %d tile pairs, exceeding max_tiles limit of %d.",
                    tile_pairs_count, self.max_tiles
                )
                return LearnedMatchResult(
                    correspondences=[],
                    available=False,
                    backend="loftr",
                    metadata={
                        "reason": f"Tile pair count ({tile_pairs_count}) exceeds max_tiles limit ({self.max_tiles})."
                    },
                )

            raw_corrs: List[LearnedCorrespondence] = []
            tile_pairs_processed = 0

            for r_idx in range(len(rows_a)):
                y_a0, y_a1 = rows_a[r_idx]
                y_b0, y_b1 = rows_b[r_idx]
                for c_idx in range(len(cols_a)):
                    x_a0, x_a1 = cols_a[c_idx]
                    x_b0, x_b1 = cols_b[c_idx]

                    tile_a = arr_a[y_a0:y_a1, x_a0:x_a1]
                    tile_b = arr_b[y_b0:y_b1, x_b0:x_b1]

                    tile_corrs = self._match_single_pass(tile_a, tile_b)
                    tile_pairs_processed += 1

                    if progress_callback is not None:
                        try:
                            progress_callback(
                                ProgressEvent(
                                    stage="learned_matching",
                                    current=tile_pairs_processed,
                                    total=tile_pairs_count,
                                    progress=float(tile_pairs_processed) / float(tile_pairs_count),
                                    message=f"Processed LoFTR tile pair {tile_pairs_processed}/{tile_pairs_count}",
                                )
                            )
                        except Exception as p_err:
                            logger.debug("Progress callback exception: %s", p_err)

                    for tc in tile_corrs:
                        mapped_pt_a = np.array([tc.point_a[0] + x_a0, tc.point_a[1] + y_a0], dtype=np.float64)
                        mapped_pt_b = np.array([tc.point_b[0] + x_b0, tc.point_b[1] + y_b0], dtype=np.float64)
                        raw_corrs.append(
                            LearnedCorrespondence(
                                point_a=mapped_pt_a,
                                point_b=mapped_pt_b,
                                confidence=tc.confidence,
                            )
                        )

            deduped_corrs = deduplicate_correspondences(raw_corrs, tolerance_px=self.duplicate_tolerance_px)

            logger.info(
                "LoFTR matching summary: image_a_shape=%s, image_b_shape=%s, tiling_enabled=True, "
                "num_tiles_a=%d, num_tiles_b=%d, tile_size=%d, tile_overlap=%d, tile_pairs_processed=%d, "
                "raw_tile_correspondences=%d, deduplicated_correspondences=%d",
                shape_a, shape_b, num_tiles_a, num_tiles_b, self.tile_size, self.tile_overlap,
                tile_pairs_processed, len(raw_corrs), len(deduped_corrs)
            )

            return LearnedMatchResult(
                correspondences=deduped_corrs,
                available=True,
                backend="loftr",
                metadata={
                    "pretrained": self.pretrained,
                    "device": str(self._device),
                    "tiling_enabled": True,
                    "image_a_shape": list(shape_a),
                    "image_b_shape": list(shape_b),
                    "num_tiles_a": num_tiles_a,
                    "num_tiles_b": num_tiles_b,
                    "tile_size": self.tile_size,
                    "tile_overlap": self.tile_overlap,
                    "tile_pairs_processed": tile_pairs_processed,
                    "raw_tile_correspondences_count": len(raw_corrs),
                    "deduplicated_correspondences_count": len(deduped_corrs),
                    "min_confidence": self.min_confidence,
                },
            )

        except Exception as e:
            logger.warning("LoFTR match execution failed: %s", e)
            return LearnedMatchResult(
                correspondences=[],
                available=False,
                backend="loftr",
                metadata={"reason": f"Execution error: {type(e).__name__} - {str(e)}"},
            )

