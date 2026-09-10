"""
experiments/inspect_crater_graph.py

Crater graph construction and visualization experiment.
Constructs spatial graphs across detected craters using Delaunay, KNN, and Radius strategies,
and saves annotated visual overlays with nodes, crater circles, and connection edges.

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
from src.crater_detection.pipeline import detect_craters
from src.crater_detection.types import CraterCandidate
from src.crater_graph.builder import build_crater_graph
from src.crater_graph.features import compute_graph_summary
from src.crater_graph.visualization import draw_crater_graph
from experiments.inspect_crater_detection import _create_synthetic_crater_scene

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "crater_graph")
META_PATH = os.path.join(PROJECT_ROOT, "data", "dataset_meta.json")


def run_graph_experiment(image: np.ndarray,
                         craters: List[CraterCandidate],
                         data_label: str,
                         is_synthetic: bool = True):
    """Build and visualize graphs using different strategies."""
    os.makedirs(OUT_DIR, exist_ok=True)
    logger.info("=" * 60)
    logger.info("Crater Graph Inspection: %s (craters=%d, synthetic=%s)",
                data_label, len(craters), is_synthetic)

    strategies = [
        ("delaunay", {"method": "delaunay"}),
        ("knn", {"method": "knn", "k_neighbors": 4}),
        ("radius", {"method": "radius", "radius_threshold_px": 180.0}),
    ]

    for name, cfg in strategies:
        graph = build_crater_graph(craters, config=cfg, image_shape=image.shape[:2])
        summary = compute_graph_summary(graph)

        logger.info("  [%s] Nodes: %d | Edges: %d | Density: %.3f | Mean Degree: %.2f | Connected: %s",
                    name.upper(), summary["node_count"], summary["edge_count"],
                    summary["density"], summary["mean_degree"], summary["is_connected"])
        if summary["mean_distance_px"] is not None:
            logger.info("       Mean Dist: %.1f px | Median Radius: %.1f px",
                        summary["mean_distance_px"], summary.get("median_radius_px") or 0.0)

        label_str = "SYNTHETIC CRATER GRAPH" if is_synthetic else data_label
        vis = draw_crater_graph(image=image, graph=graph, data_label=label_str)

        prefix = "synthetic" if is_synthetic else data_label
        out_path = os.path.join(OUT_DIR, f"{prefix}_{name}_crater_graph.png")
        cv2.imwrite(out_path, vis)
        logger.info("  Saved visualization: %s", out_path)

    logger.info("Graph inspection complete. Outputs in: %s", OUT_DIR)


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
                img = loaded["image"]
                det_res = detect_craters(img, config=Config.CRATER_DETECTION)
                run_graph_experiment(img, det_res.craters, data_label=img_meta.get("image_id"), is_synthetic=False)
                used_real = True
                break
            except FileNotFoundError as exc:
                logger.warning("[MISSING] %s -- Place real lunar imagery in data/raw/.", exc)

    if not used_real:
        logger.info("No real lunar imagery found. Generating SYNTHETIC crater test scene.")
        canvas, crater_defs = _create_synthetic_crater_scene(512, 512)
        # Use known synthetic craters directly for clean graph demonstration
        synthetic_craters = [
            CraterCandidate(x=float(cx), y=float(cy), radius=float(r), confidence=0.92)
            for cx, cy, r in crater_defs
        ]
        run_graph_experiment(canvas, synthetic_craters, data_label="synthetic_constellation", is_synthetic=True)


if __name__ == "__main__":
    main()
