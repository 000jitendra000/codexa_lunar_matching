"""
experiments/inspect_crater_detection.py

Crater detection inspection and visualization script.
Executes crater candidate extraction and generates annotated visual overlays
showing crater center, radius, and confidence score.

NOTE: If real lunar images (data/raw/) are unavailable, this experiment runs
on a SYNTHETIC test image for software verification only.
"""

import os
import sys
import json
import logging
from typing import List, Tuple
import numpy as np
import cv2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.default import Config
from src.preprocessing.image_loader import ImageLoader
from src.crater_detection.types import CraterCandidate
from src.crater_detection.detector import HoughCraterDetector, MockCraterDetector
from src.crater_detection.pipeline import CraterDetectionPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "crater_detection")
META_PATH = os.path.join(PROJECT_ROOT, "data", "dataset_meta.json")


def _create_synthetic_crater_scene(h: int = 512, w: int = 512) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
    """
    Generate synthetic lunar terrain with multiple craters of varying sizes.
    Used ONLY for software and visualization verification.
    """
    rng = np.random.default_rng(42)
    base = np.full((h, w), 130, dtype=np.uint8)
    noise = rng.integers(-20, 20, (h, w), dtype=np.int16)
    img = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Coordinates of synthetic craters (cx, cy, radius)
    craters_def = [
        (130, 140, 45),
        (370, 160, 35),
        (260, 320, 60),
        (100, 390, 28),
        (420, 390, 40),
    ]

    for cx, cy, r in craters_def:
        # Dark inner floor
        cv2.circle(img, (cx, cy), r, 50, thickness=-1)
        # Bright sunlit rim
        cv2.circle(img, (cx, cy), r, 220, thickness=3)
        # Inner shadow crescent
        cv2.ellipse(img, (cx, cy), (r, r), 0, 160, 350, 30, thickness=4)

    return img, craters_def


def draw_crater_annotations(image: np.ndarray,
                            craters: List[CraterCandidate],
                            detector_name: str,
                            data_label: str) -> np.ndarray:
    """
    Draw crater detections with centers, circular outlines, and confidence labels.
    """
    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    h, w = vis.shape[:2]

    # Draw semi-transparent header bar
    header_h = 42
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (w, header_h), (30, 30, 30), thickness=-1)
    cv2.addWeighted(overlay, 0.85, vis, 0.15, 0, vis)

    header_text = f"Detector: {detector_name} | Image: {data_label} | Craters: {len(craters)}"
    cv2.putText(vis, header_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (240, 240, 240), 2)

    # Draw each detected crater
    for idx, c in enumerate(craters):
        cx, cy, r = int(round(c.x)), int(round(c.y)), int(round(c.radius))

        # Circle outline (bright green)
        cv2.circle(vis, (cx, cy), r, (0, 230, 50), thickness=2)

        # Center marker (red dot)
        cv2.circle(vis, (cx, cy), 3, (0, 0, 255), thickness=-1)

        # Confidence label
        lbl = f"#{idx+1} r={c.radius:.0f} c={c.confidence:.2f}"
        lbl_x = max(5, cx - r)
        lbl_y = max(header_h + 15, cy - r - 6)
        cv2.putText(vis, lbl, (lbl_x, lbl_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)

    return vis


def run_detection_and_visualize(image: np.ndarray,
                                data_label: str,
                                is_synthetic: bool = True):
    """Run detectors and save visual overlays."""
    os.makedirs(OUT_DIR, exist_ok=True)
    logger.info("=" * 60)
    logger.info("Crater Detection Inspection: %s (synthetic=%s)", data_label, is_synthetic)

    detectors = [
        ("hough", HoughCraterDetector(Config.CRATER_DETECTION)),
        ("mock", MockCraterDetector(cfg=Config.CRATER_DETECTION)),
    ]

    for name, det in detectors:
        pipeline = CraterDetectionPipeline(config=Config.CRATER_DETECTION, detector=det)
        result = pipeline.process(image)
        craters = result.craters

        logger.info("  [%s] Found %d craters in %.3fs", name.upper(), len(craters), result.elapsed_sec)
        for i, c in enumerate(craters[:5]):
            logger.info("    #%d: center=(%.1f, %.1f), r=%.1f px, conf=%.2f",
                        i + 1, c.x, c.y, c.radius, c.confidence)
        if len(craters) > 5:
            logger.info("    ... and %d more candidates", len(craters) - 5)

        vis = draw_crater_annotations(
            image=image,
            craters=craters,
            detector_name=name.upper(),
            data_label="SYNTHETIC TEST IMAGE" if is_synthetic else data_label,
        )

        filename = f"{'synthetic' if is_synthetic else data_label}_{name}_crater_detections.png"
        out_path = os.path.join(OUT_DIR, filename)
        cv2.imwrite(out_path, vis)
        logger.info("  Saved visualization: %s", out_path)

    logger.info("Inspection complete. Visual outputs stored in: %s", OUT_DIR)


def main():
    used_real = False
    if os.path.exists(META_PATH):
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        for img_meta in meta.get("images", []):
            fpath = img_meta.get("file_path", "")
            full_path = fpath if os.path.isabs(fpath) else os.path.join(PROJECT_ROOT, fpath)
            try:
                loaded = ImageLoader.load_image(full_path)
                logger.info("Loaded real lunar image: %s (%s)", img_meta.get("image_id"), full_path)
                run_detection_and_visualize(loaded["image"], data_label=img_meta.get("image_id"), is_synthetic=False)
                used_real = True
                break
            except FileNotFoundError as exc:
                logger.warning("[MISSING] %s -- Place real lunar imagery in data/raw/.", exc)

    if not used_real:
        logger.info("No real lunar imagery found. Generating SYNTHETIC crater test scene.")
        canvas, _ = _create_synthetic_crater_scene(512, 512)
        run_detection_and_visualize(canvas, data_label="synthetic_crater_scene", is_synthetic=True)


if __name__ == "__main__":
    main()
