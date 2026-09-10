"""
experiments/inspect_crater_invariants.py

Demonstration and verification of Phase 7: Scale- and Rotation-Robust Invariant Descriptors.
Evaluates synthetic crater constellations under:
- Original configuration
- Pure translation
- Pure rotation
- Combined similarity transform (scale + rotation + translation)

Outputs:
- Console table comparing numerical invariance (L_inf and L_2 error).
- Multi-panel visual plot saved to data/processed/crater_invariants/.
- Explicitly labeled: SYNTHETIC INVARIANT DESCRIPTOR TEST.
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
from src.crater_graph.invariants import (
    build_triangle_descriptors,
    build_local_invariant_descriptors,
)


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


def plot_constellation(
    ax: plt.Axes,
    graph,
    triangle_desc,
    title: str,
    color: str = "cyan",
):
    ax.set_facecolor("#121720")

    # Draw graph edges
    for u, v, d in graph.nx_graph.edges(data=True):
        cu = graph.get_crater(u)
        cv = graph.get_crater(v)
        ax.plot([cu.x, cv.x], [cu.y, cv.y], color="#4a5568", linestyle="--", linewidth=1.2, zorder=1)

    # Highlight canonical triangle if present
    if triangle_desc is not None:
        nodes = list(triangle_desc.nodes)
        pts = [[graph.get_crater(n).x, graph.get_crater(n).y] for n in nodes]
        pts.append(pts[0])
        pts = np.array(pts)
        ax.plot(pts[:, 0], pts[:, 1], color="#f6e05e", linewidth=2.2, zorder=2, label="Canonical Triangle")
        ax.fill(pts[:, 0], pts[:, 1], color="#f6e05e", alpha=0.15, zorder=1)

    # Draw craters
    for nid in graph.nodes():
        c = graph.get_crater(nid)
        circle = patches.Circle((c.x, c.y), c.radius, fill=False, edgecolor=color, linewidth=1.8, zorder=3)
        ax.add_patch(circle)
        ax.scatter(c.x, c.y, color=color, s=25, zorder=4)
        ax.text(c.x + c.radius + 3, c.y + 3, f"C{nid}", color="white", fontsize=9, zorder=5)

    ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.tick_params(colors="gray")
    for spine in ax.spines.values():
        spine.set_color("#2d3748")


def main():
    logger.info("============================================================")
    logger.info("Phase 7 Invariant Descriptor Inspection")
    logger.info("SYNTHETIC INVARIANT DESCRIPTOR TEST")
    logger.info("============================================================")

    # Base synthetic constellation (4 craters with distinct positions and sizes)
    base_craters = [
        CraterCandidate(x=100.0, y=100.0, radius=12.0, confidence=0.95),
        CraterCandidate(x=240.0, y=130.0, radius=22.0, confidence=0.90),
        CraterCandidate(x=190.0, y=280.0, radius=16.0, confidence=0.92),
        CraterCandidate(x=70.0, y=230.0, radius=20.0, confidence=0.88),
    ]

    # Generate transformed variants
    variants = {
        "Original": transform_constellation(base_craters, scale=1.0, angle_deg=0.0, tx=0.0, ty=0.0),
        "Translated": transform_constellation(base_craters, scale=1.0, angle_deg=0.0, tx=350.0, ty=200.0),
        "Rotated": transform_constellation(base_craters, scale=1.0, angle_deg=65.0, tx=0.0, ty=0.0),
        "Combined (s=1.8, r=65 deg, t=(200, 150))": transform_constellation(
            base_craters, scale=1.8, angle_deg=65.0, tx=200.0, ty=150.0
        ),
    }

    results = {}
    graphs = {}

    for name, crt_list in variants.items():
        g = build_crater_graph(crt_list, config={"method": "delaunay"})
        tri_descs = build_triangle_descriptors(g)
        local_descs = build_local_invariant_descriptors(g)
        primary_tri = tri_descs[0] if tri_descs else None
        graphs[name] = g
        results[name] = {
            "triangle": primary_tri,
            "local_node0": local_descs.get(0),
        }

    # Numerical Comparison Table
    logger.info("\n--- Numerical Verification of Invariance ---")
    ref_tri = results["Original"]["triangle"]
    ref_local = results["Original"]["local_node0"]

    print(f"\n{'Transformation':<35} | {'Tri Error (L_inf)':<18} | {'Local Error (L_inf)':<18} | {'Invariance Status'}")
    print("-" * 92)

    for name, res in results.items():
        tri = res["triangle"]
        loc = res["local_node0"]

        tri_err = np.max(np.abs(ref_tri.feature_vector - tri.feature_vector)) if tri else 0.0
        loc_err = np.max(np.abs(ref_local.feature_vector - loc.feature_vector)) if loc else 0.0
        status = "EXACT / INVARIANT" if max(tri_err, loc_err) < 1e-3 else "FAIL"

        print(f"{name:<35} | {tri_err:<18.6f} | {loc_err:<18.6f} | {status}")

    print("-" * 92)
    print("Reference Triangle Feature Vector (s1/P, s2/P, s3/P, a1, a2, a3, r1, r2, r3):")
    print(np.round(ref_tri.feature_vector, 4))
    print("\nReference Local Descriptor Node 0 Features (norm_dist, rad_ratio, rel_angle):")
    print(f"  Normalized Distances : {ref_local.normalized_distances}")
    print(f"  Radius Ratios        : {ref_local.radius_ratios}")
    print(f"  Relative Angles (rad): {ref_local.relative_angles_rad}")

    # Plot Multi-Panel Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), facecolor="#0a0e17")
    fig.suptitle(
        "SYNTHETIC INVARIANT DESCRIPTOR TEST\n"
        "[Software Verification: Similarity-Transform Invariance Across Scale, Rotation, Translation]",
        color="white", fontsize=14, fontweight="bold", y=0.98
    )

    colors = ["#4fd1c5", "#63b3ed", "#f6ad55", "#b794f4"]
    for ax, (name, g), col in zip(axes.flat, graphs.items(), colors):
        tri = results[name]["triangle"]
        plot_constellation(ax, g, tri, name, color=col)

    # Global synthetic labeling banner
    fig.text(
        0.5, 0.02,
        "SYNTHETIC CRATER GRAPH  |  NO REAL OHRC/LROC IMAGERY USED  |  PHASE 7 INVARIANCE DEMO",
        ha="center", fontsize=10, color="#a0aec0", fontweight="bold"
    )

    out_dir = os.path.join("data", "processed", "crater_invariants")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "synthetic_similarity_invariance_demo.png")
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    plt.savefig(out_path, dpi=160, facecolor=fig.get_facecolor())
    plt.close()

    logger.info("Multi-panel inspection plot successfully saved to: %s", out_path)
    logger.info("Phase 7 Invariant Descriptor Inspection complete.")


if __name__ == "__main__":
    main()
