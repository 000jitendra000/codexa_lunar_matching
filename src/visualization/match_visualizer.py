"""
src/visualization/match_visualizer.py

Orchestrator for match explainability visualizations.
Consumes HybridMatchResult and RegistrationResult to save PNG assets and return primitive VisualizationResult.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import cv2
import numpy as np

from src.matching.hybrid_matcher import HybridMatchResult
from src.registration.registration_engine import RegistrationResult
from src.visualization.confidence_map import generate_confidence_map
from src.visualization.correspondence_plot import plot_correspondences
from src.visualization.registration_overlay import generate_registration_overlay

logger = logging.getLogger(__name__)


@dataclass
class VisualizationResult:
    """
    Lightweight, primitive container for visualization output file paths.
    Contains no NumPy arrays, PyTorch tensors, or complex models for clean serialization.
    """

    confidence_map_a: Optional[str] = None
    confidence_map_b: Optional[str] = None
    correspondence_image: Optional[str] = None
    registration_overlay: Optional[str] = None
    checkerboard: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize result to JSON-compatible dictionary."""
        return {
            "confidence_map_a": self.confidence_map_a,
            "confidence_map_b": self.confidence_map_b,
            "correspondence_image": self.correspondence_image,
            "registration_overlay": self.registration_overlay,
            "checkerboard": self.checkerboard,
            "metadata": self.metadata,
        }


class MatchVisualizer:
    """
    High-level match visualizer generating explainable evidence maps, correspondence line plots,
    and registration overlays. Keeps model results valid even if an individual visualization fails.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def generate(
        self,
        image_a: np.ndarray,
        image_b: np.ndarray,
        match_result: HybridMatchResult,
        registration_result: Optional[RegistrationResult] = None,
        job_id: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> VisualizationResult:
        """
        Generate and save visualization assets under output_dir/{job_id}/.

        Args:
            image_a: Reference lunar image array.
            image_b: Query lunar image array.
            match_result: HybridMatchResult produced by HybridMatcher.
            registration_result: Optional RegistrationResult produced by RegistrationEngine.
            job_id: Unique string identifier for directory organization.
            output_dir: Override directory path for saving PNGs.

        Returns:
            VisualizationResult containing asset relative/absolute file paths.
        """
        # Resolve output directory
        base_dir = output_dir or self.config.get("output_dir", "data/processed/visualizations")
        sub_dir = job_id or "default"
        target_dir = os.path.abspath(os.path.join(base_dir, sub_dir))
        os.makedirs(target_dir, exist_ok=True)

        res = VisualizationResult()

        corrs = getattr(match_result, "correspondences", [])
        inliers = getattr(match_result, "inlier_indices", [])

        # Extract available transform
        transform = None
        if registration_result is not None and getattr(registration_result, "transform", None) is not None:
            transform = registration_result.transform
        elif getattr(match_result, "transform", None) is not None:
            transform = match_result.transform

        # 1. Generate Match Confidence Map for Image A
        try:
            cmap_a = generate_confidence_map(
                image=image_a,
                correspondences=corrs,
                inlier_indices=inliers,
                is_target_b=False,
                config=self.config,
            )
            path_a = os.path.join(target_dir, "confidence_map_a.png")
            if cv2.imwrite(path_a, cv2.cvtColor(cmap_a, cv2.COLOR_RGB2BGR)):
                res.confidence_map_a = path_a
        except Exception as exc:
            logger.warning("Failed to generate confidence_map_a: %s", exc, exc_info=True)

        # 2. Generate Match Confidence Map for Image B
        try:
            cmap_b = generate_confidence_map(
                image=image_b,
                correspondences=corrs,
                inlier_indices=inliers,
                is_target_b=True,
                config=self.config,
            )
            path_b = os.path.join(target_dir, "confidence_map_b.png")
            if cv2.imwrite(path_b, cv2.cvtColor(cmap_b, cv2.COLOR_RGB2BGR)):
                res.confidence_map_b = path_b
        except Exception as exc:
            logger.warning("Failed to generate confidence_map_b: %s", exc, exc_info=True)

        # 3. Generate Correspondence Line Plot
        try:
            corr_plot = plot_correspondences(
                image_a=image_a,
                image_b=image_b,
                correspondences=corrs,
                inlier_indices=inliers,
                config=self.config,
            )
            path_corr = os.path.join(target_dir, "correspondence_image.png")
            if cv2.imwrite(path_corr, cv2.cvtColor(corr_plot, cv2.COLOR_RGB2BGR)):
                res.correspondence_image = path_corr
        except Exception as exc:
            logger.warning("Failed to generate correspondence_image: %s", exc, exc_info=True)

        # 4. Generate Registration Overlay & Checkerboard (if transform is available)
        if transform is not None:
            try:
                overlay_rgb, checker_rgb = generate_registration_overlay(
                    image_a=image_a,
                    image_b=image_b,
                    transform=transform,
                    config=self.config,
                )
                path_ov = os.path.join(target_dir, "registration_overlay.png")
                if cv2.imwrite(path_ov, cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)):
                    res.registration_overlay = path_ov

                if checker_rgb is not None:
                    path_ck = os.path.join(target_dir, "checkerboard.png")
                    if cv2.imwrite(path_ck, cv2.cvtColor(checker_rgb, cv2.COLOR_RGB2BGR)):
                        res.checkerboard = path_ck
            except Exception as exc:
                logger.warning("Failed to generate registration_overlay: %s", exc, exc_info=True)

        res.metadata = {
            "num_correspondences": len(corrs),
            "num_inliers": len(inliers),
            "has_transform": (transform is not None),
        }

        return res
