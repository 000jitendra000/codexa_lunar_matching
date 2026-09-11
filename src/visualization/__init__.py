"""
src/visualization/__init__.py

Visualization & Explainable Outputs Module for Lunar Location Matching Engine.
Provides spatial match evidence confidence maps, correspondence line plots,
and registration overlay visualizers.
"""

from src.visualization.confidence_map import generate_confidence_map
from src.visualization.correspondence_plot import plot_correspondences
from src.visualization.registration_overlay import generate_registration_overlay
from src.visualization.match_visualizer import MatchVisualizer, VisualizationResult

__all__ = [
    "generate_confidence_map",
    "plot_correspondences",
    "generate_registration_overlay",
    "MatchVisualizer",
    "VisualizationResult",
]
