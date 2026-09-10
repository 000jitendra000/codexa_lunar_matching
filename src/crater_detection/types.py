"""
src/crater_detection/types.py

Standardized crater candidate representation and detection results.
Coordinates are in 2D image pixel space (origin at top-left, x=horizontal, y=vertical).
No geographic coordinates are used here.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
import math


@dataclass(frozen=True)
class CraterCandidate:
    """
    Standardized crater representation in image pixel coordinates.

    Attributes:
        x: Center X coordinate in pixels (horizontal, origin at top-left).
        y: Center Y coordinate in pixels (vertical, origin at top-left).
        radius: Approximate crater radius in pixels (>= 0).
        confidence: Detection confidence score in range [0.0, 1.0].
        bbox: Optional raw bounding box (xmin, ymin, xmax, ymax) in pixels.
    """
    x: float
    y: float
    radius: float
    confidence: float
    bbox: Optional[Tuple[float, float, float, float]] = None

    def __post_init__(self):
        # Validate numbers are finite
        for name, val in [("x", self.x), ("y", self.y), ("radius", self.radius), ("confidence", self.confidence)]:
            if val is None or math.isnan(val) or math.isinf(val):
                raise ValueError(f"CraterCandidate '{name}' must be a finite number, got {val}.")

        if self.radius < 0.0:
            raise ValueError(f"CraterCandidate radius must be non-negative, got {self.radius}.")

        if not (0.0 <= self.confidence <= 1.0001):
            raise ValueError(f"CraterCandidate confidence must be in [0.0, 1.0], got {self.confidence}.")

        if self.bbox is not None:
            if len(self.bbox) != 4:
                raise ValueError(f"CraterCandidate bbox must have 4 elements (xmin, ymin, xmax, ymax), got {self.bbox}.")
            xmin, ymin, xmax, ymax = self.bbox
            for bname, bval in [("xmin", xmin), ("ymin", ymin), ("xmax", xmax), ("ymax", ymax)]:
                if bval is None or math.isnan(bval) or math.isinf(bval):
                    raise ValueError(f"CraterCandidate bbox '{bname}' must be finite, got {bval}.")
            if xmax < xmin or ymax < ymin:
                raise ValueError(f"CraterCandidate bbox invalid: xmax ({xmax}) < xmin ({xmin}) or ymax ({ymax}) < ymin ({ymin}).")

    @property
    def diameter(self) -> float:
        """Crater diameter in pixels."""
        return 2.0 * self.radius

    @property
    def area(self) -> float:
        """Crater area in square pixels (circular approximation)."""
        return math.pi * (self.radius ** 2)

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """
        Bounding box as (xmin, ymin, xmax, ymax).
        Uses self.bbox if available; otherwise computes from center and radius.
        """
        if self.bbox is not None:
            return self.bbox
        return (self.x - self.radius, self.y - self.radius,
                self.x + self.radius, self.y + self.radius)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize candidate to a dictionary."""
        d: Dict[str, Any] = {
            "x": round(float(self.x), 3),
            "y": round(float(self.y), 3),
            "radius": round(float(self.radius), 3),
            "confidence": round(float(self.confidence), 4),
        }
        if self.bbox is not None:
            d["bbox"] = [round(float(v), 3) for v in self.bbox]
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CraterCandidate":
        """Deserialize from dictionary."""
        bbox = tuple(data["bbox"]) if "bbox" in data and data["bbox"] is not None else None
        return cls(
            x=float(data["x"]),
            y=float(data["y"]),
            radius=float(data["radius"]),
            confidence=float(data["confidence"]),
            bbox=bbox,
        )

    @classmethod
    def from_bbox(cls, xmin: float, ymin: float, xmax: float, ymax: float,
                  confidence: float) -> "CraterCandidate":
        """
        Convert bounding box to circular crater approximation.
        Center: midpoint of box.
        Radius: half of min(width, height).
        """
        w = max(0.0, float(xmax - xmin))
        h = max(0.0, float(ymax - ymin))
        cx = float(xmin) + w / 2.0
        cy = float(ymin) + h / 2.0
        radius = min(w, h) / 2.0
        return cls(
            x=cx,
            y=cy,
            radius=radius,
            confidence=float(confidence),
            bbox=(float(xmin), float(ymin), float(xmax), float(ymax)),
        )


@dataclass
class CraterDetectionResult:
    """
    Standardized container returned by crater detectors and pipelines.
    """
    craters: List[CraterCandidate] = field(default_factory=list)
    detector_name: str = "unknown"
    image_shape: Tuple[int, int] = (0, 0)   # (height, width)
    elapsed_sec: float = 0.0
    raw_detections: Any = None

    @property
    def count(self) -> int:
        return len(self.craters)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detector_name": self.detector_name,
            "crater_count": len(self.craters),
            "image_shape": list(self.image_shape),
            "elapsed_sec": round(self.elapsed_sec, 4),
            "craters": [c.to_dict() for c in self.craters],
        }
