"""
experiments/inspect_constellation_matching.py

Demonstration and verification of Phase 8: Local Crater Constellation Matching.
Evaluates synthetic crater constellation matching under known similarity transformations:
- Source constellation (6 craters)
- Target constellation transformed by known scale (1.6x), rotation (55 deg), and translation (220, 140)
- Additional distractor craters in target scene
- Ground-truth correspondence evaluation (Precision, Recall, TP, FP, FN)
- Two-panel side-by-side visual correspondence plot

NOTE: Strictly software verification on synthetic data.
Do NOT imply this is real OHRC/LROC data.
"""

import os
import sys
import math
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.crater_detection.types import CraterCandidate
from src.crater_graph.builder import build_crater_graph
from src.crater_graph.constellation_matcher import match_crater_constellations

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def transform_constellation(
    craters: list[CraterCandidate],
    scale: float = 1.0,
    angle_deg: float = 0.0,
    tx: float = 0.0,
    ty: float = 0.0,
) -> list[CraterCandidate]:
    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    out = []
    for c in craters:
        rx = scale * (cos_a * c.x - sin_a * c.y) + tx
        ry = scale * (sin_a * c.x + cos_a * c.y) + ty
        rr = scale * c.radius
        out.append(CraterCandidate(x=rx, y=ry, radius=rr, confidence=c.confidence))
    return out


def main():
    logger.info("============================================================")
    logger.info("Phase 8 Constellation Matching Inspection")
    logger.info("SYNTHETIC SOFTWARE VERIFICATION")
    logger.info("============================================================")

    # 1. Source constellation (6 craters with distinct geometry)
    craters_a = [
        CraterCandidate(x=100.0, y=100.0, radius=12.0, confidence=0.95),  # 0
        CraterCandidate(x=240.0, y=110.0, radius=22.0, confidence=0.90),  # 1
        CraterCandidate(x=280.0, y=240.0, radius=16.0, confidence=0.92),  # 2
        CraterCandidate(x=190.0, y=310.0, radius=20.0, confidence=0.88),  # 3
        CraterCandidate(x=80.0, y=230.0, radius=14.0, confidence=0.93),   # 4
        CraterCandidate(x=175.0, y=190.0, radius=18.0, confidence=0.91),  # 5 (center)
    ]

    # 2. Known ground truth transform: scale=1.6, rotation=55 deg, tx=220, ty=140
    known_scale = 1.6
    known_angle = 55.0
    known_tx = 220.0
    known_ty = 140.0

    craters_b_true = transform_constellation(
        craters_a, scale=known_scale, angle_deg=known_angle, tx=known_tx, ty=known_ty
    )

    # Add 2 distractor craters to target image that do not belong to the source
    craters_b = list(craters_b_true)
    craters_b.append(CraterCandidate(x=620.0, y=620.0, radius=35.0, confidence=0.85))  # 6 (distractor)
    craters_b.append(CraterCandidate(x=720.0, y=550.0, radius=28.0, confidence=0.80))  # 7 (distractor)

    ground_truth = {i: i for i in range(len(craters_a))}  # Source i -> Target i

    # 3. Build CraterGraphs
    graph_a = build_crater_graph(craters_a, config={"method": "delaunay"})
    graph_b = build_crater_graph(craters_b, config={"method": "delaunay"})

    # 4. Perform Phase 8 Constellation Matching
    logger.info("Executing match_crater_constellations()...")
    result = match_crater_constellations(graph_a, graph_b)

    logger.info("Matching complete: %d matches found.", result.match_count)

    # 5. Evaluate Precision, Recall, and Ground Truth Correctness
    matched_pairs = result.get_match_pairs()
    matched_dict = dict(matched_pairs)

    tp = 0
    fp = 0
    for src_id, tgt_id in matched_pairs:
        if ground_truth.get(src_id) == tgt_id:
            tp += 1
        else:
            fp += 1

    fn = len(ground_truth) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # 6. Display Metrics Table
    print("\n" + "=" * 95)
    print("PHASE 8 CONSTELLATION MATCHING VERIFICATION TABLE (SYNTHETIC GROUND TRUTH)")
    print("=" * 95)
    print(f"Source Craters : {result.source_descriptor_count}")
    print(f"Target Craters : {result.target_descriptor_count} (including 2 distractor craters)")
    print(f"Raw Candidates : {result.candidate_count}")
    print(f"Final Matches  : {result.match_count}")
    print(f"Ground Truth TP: {tp} | FP: {fp} | FN: {fn}")
    print(f"Precision      : {precision * 100:.1f}%")
    print(f"Recall         : {recall * 100:.1f}%")
    print("-" * 95)
    print(f"{'Source':<8} | {'Target':<8} | {'GT Target':<10} | {'Distance':<10} | {'Confidence':<12} | {'Triangles':<10} | {'Status'}")
    print("-" * 95)

    for m in result.matches:
        gt_t = ground_truth.get(m.source_node_id, -1)
        status = "CORRECT (TP)" if gt_t == m.target_node_id else "INCORRECT (FP)"
        print(
            f"C{m.source_node_id:<7} | C{m.target_node_id:<7} | C{gt_t:<9} | "
            f"{m.descriptor_distance:<10.5f} | {m.confidence:<12.4f} | {m.supporting_triangle_count:<10} | {status}"
        )
    print("=" * 95)

    # 7. Generate Side-by-Side Visual Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), facecolor="#0a0e17")
    fig.suptitle(
        "SYNTHETIC CONSTELLATION MATCHING DEMO\n"
        "[Software Verification: Invariant Descriptor Correspondence Across s=1.6, theta=55 deg, t=(220, 140)]",
        color="white", fontsize=13, fontweight="bold", y=0.98
    )

    for ax, g, title, is_target in [(ax1, graph_a, "Constellation A (Source)", False),
                                     (ax2, graph_b, "Constellation B (Transformed Target + Distractors)", True)]:
        ax.set_facecolor("#121720")
        # Edges
        for u, v in g.edges():
            cu, cv = g.get_crater(u), g.get_crater(v)
            ax.plot([cu.x, cv.x], [cu.y, cv.y], color="#4a5568", linestyle="--", linewidth=1.0, zorder=1)

        # Craters
        for nid in g.nodes():
            c = g.get_crater(nid)
            # Color code: distractors in red, matched in cyan/orange
            if is_target and nid in (6, 7):
                edge_col = "#e53e3e"
                label = f"D{nid}"
            elif not is_target and nid in matched_dict:
                edge_col = "#4fd1c5"
                label = f"C{nid}"
            elif is_target and nid in matched_dict.values():
                edge_col = "#f6ad55"
                label = f"C{nid}"
            else:
                edge_col = "#a0aec0"
                label = f"C{nid}"

            circle = patches.Circle((c.x, c.y), c.radius, fill=False, edgecolor=edge_col, linewidth=2.0, zorder=2)
            ax.add_patch(circle)
            ax.scatter(c.x, c.y, color=edge_col, s=30, zorder=3)
            ax.text(c.x + c.radius + 3, c.y + 3, label, color="white", fontsize=9, fontweight="bold", zorder=4)

        ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=8)
        ax.set_aspect("equal", adjustable="datalim")
        ax.tick_params(colors="gray")
        for spine in ax.spines.values():
            spine.set_color("#2d3748")

    # Connect corresponding crater centers between the two figures using ConnectionPatch
    from matplotlib.patches import ConnectionPatch
    for m in result.matches:
        ca = graph_a.get_crater(m.source_node_id)
        cb = graph_b.get_crater(m.target_node_id)
        con = ConnectionPatch(
            xyA=(ca.x, ca.y), coordsA=ax1.transData,
            xyB=(cb.x, cb.y), coordsB=ax2.transData,
            color="#38b2ac", linestyle="solid", linewidth=1.8, alpha=0.75, zorder=5
        )
        fig.add_artist(con)

    # Global synthetic labeling banner
    fig.text(
        0.5, 0.02,
        "SYNTHETIC CONSTELLATION MATCHING DEMO  |  NO REAL OHRC/LROC IMAGERY USED  |  PHASE 8 COMPLETE",
        ha="center", fontsize=10, color="#a0aec0", fontweight="bold"
    )

    out_dir = os.path.join("data", "processed", "constellation_matching")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "synthetic_constellation_matching_demo.png")
    plt.tight_layout(rect=[0, 0.04, 1, 0.94])
    plt.savefig(out_path, dpi=160, facecolor=fig.get_facecolor())
    plt.close()

    logger.info("Visual matching plot saved to: %s", out_path)
    logger.info("Phase 8 Constellation Matching Inspection complete.")


if __name__ == "__main__":
    main()
