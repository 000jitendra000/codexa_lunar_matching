"""
src/matching/progress.py

Generic progress event data model for the lunar matching engine.
Keeps model progress tracking independent from FastAPI and network protocols.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable


@dataclass
class ProgressEvent:
    """
    Model progress event containing stage details and numeric progress.

    Attributes:
        stage: Pipeline stage identifier (e.g. 'loading', 'crater_detection',
               'learned_matching', 'correspondence_fusion', 'geometric_verification',
               'registration', 'quality_assessment', 'completed', 'failed').
        current: Number of completed items in the stage.
        total: Total number of items in the stage.
        progress: Fractional progress in [0.0, 1.0].
        message: Human-readable progress message.
    """
    stage: str
    current: int
    total: int
    progress: float
    message: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize progress event to primitive dictionary."""
        return {
            "stage": str(self.stage),
            "current": int(self.current),
            "total": int(self.total),
            "progress": round(float(self.progress), 4),
            "message": str(self.message),
        }


# Optional progress callback type alias
ProgressCallback = Callable[[ProgressEvent], None]
