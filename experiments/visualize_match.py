"""
experiments/visualize_match.py

Demonstration experiment script for Lunar Match Visualization & Explainable Outputs.
Loads lunar image pair, runs matching & registration, generates visualization assets,
and outputs a summary of results.
"""

import os
import sys
import cv2
import numpy as np

from src.matching.hybrid_matcher import HybridMatcher
from src.registration.registration_engine import RegistrationEngine
from src.visualization.match_visualizer import MatchVisualizer


def main():
    print("==================================================")
    print(" Lunar Location Match Visualizer Demonstration ")
    print("==================================================")

    # 1. Load input images
    path_a = "data/raw/image_a.png"
    path_b = "data/raw/image_b.png"

    img_a = cv2.imread(path_a, cv2.IMREAD_GRAYSCALE) if os.path.exists(path_a) else None
    img_b = cv2.imread(path_b, cv2.IMREAD_GRAYSCALE) if os.path.exists(path_b) else None

    if img_a is None or img_b is None or img_a.size == 0 or img_b.size == 0:
        print("[INFO] Sample raw image pair not found in data/raw/. Generating synthetic pair for demonstration...")
        img_a = np.zeros((300, 300), dtype=np.uint8)
        cv2.circle(img_a, (100, 100), 35, 220, -1)
        cv2.circle(img_a, (200, 180), 45, 180, -1)
        cv2.circle(img_a, (80, 220), 25, 255, -1)

        M = np.float32([[1.0, 0.0, 10.0], [0.0, 1.0, 5.0]])
        img_b = cv2.warpAffine(img_a, M, (300, 300))

    print(f"Image A shape: {img_a.shape}")
    print(f"Image B shape: {img_b.shape}\n")

    # 2. Run Hybrid Matcher
    print("Running HybridMatcher...")
    matcher = HybridMatcher()
    match_result = matcher.match(img_a, img_b)

    # 3. Run Registration Engine (if matched)
    reg_result = None
    if match_result.matched and match_result.transform is not None:
        print("Running RegistrationEngine...")
        reg_engine = RegistrationEngine()
        reg_result = reg_engine.register(img_a, img_b, match_result)

    # 4. Generate Visualizations
    print("Generating Explainability Visualizations...")
    visualizer = MatchVisualizer(config={"output_dir": "data/processed/visualizations"})
    viz_result = visualizer.generate(
        image_a=img_a,
        image_b=img_b,
        match_result=match_result,
        registration_result=reg_result,
        job_id="demo_pair",
    )

    # 5. Print Summary
    print("\n--------------------------------------------------")
    print(f"Match: {match_result.matched}")
    print(f"Inliers: {len(match_result.inlier_indices)}")
    print(f"Inlier ratio: {match_result.inlier_ratio * 100:.1f}%")
    rmse_str = f"{reg_result.rmse:.2f} px" if (reg_result and reg_result.rmse is not None) else "N/A"
    print(f"RMSE: {rmse_str}")
    conf_val = reg_result.confidence if reg_result else match_result.inlier_ratio
    print(f"Confidence: {conf_val:.2f}")
    print("\nGenerated Visualizations under data/processed/visualizations/demo_pair/:")
    if viz_result.confidence_map_a:
        print("  - confidence_map_a.png")
    if viz_result.confidence_map_b:
        print("  - confidence_map_b.png")
    if viz_result.correspondence_image:
        print("  - correspondence_image.png")
    if viz_result.registration_overlay:
        print("  - registration_overlay.png")
    if viz_result.checkerboard:
        print("  - checkerboard.png")
    print("--------------------------------------------------\n")


if __name__ == "__main__":
    main()
