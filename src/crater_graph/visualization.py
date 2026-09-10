"""
src/crater_graph/visualization.py

Visualization utilities for crater graphs.
Overlays crater circles, center markers, node IDs, and spatial graph edges
onto lunar or synthetic surface images.
"""

from typing import Optional
import numpy as np
import cv2

from src.crater_graph.graph import CraterGraph


def draw_crater_graph(image: np.ndarray,
                      graph: CraterGraph,
                      data_label: str = "SYNTHETIC CRATER GRAPH",
                      edge_color: tuple = (255, 200, 0),       # Cyan-yellow in BGR
                      crater_color: tuple = (0, 230, 50),       # Bright green
                      center_color: tuple = (0, 0, 255),        # Red
                      show_node_ids: bool = True) -> np.ndarray:
    """
    Render visual overlay of a CraterGraph on top of an image.

    Args:
        image: Base image array (grayscale or BGR).
        graph: CraterGraph instance.
        data_label: Text indicating data source (e.g. 'SYNTHETIC CRATER GRAPH').
        edge_color: BGR color for graph edges.
        crater_color: BGR color for crater circle outlines.
        center_color: BGR color for crater center markers.
        show_node_ids: Whether to draw integer node IDs near centers.

    Returns:
        Annotated BGR numpy array.
    """
    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    h, w = vis.shape[:2]

    # Draw semi-transparent header banner
    header_h = 42
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (w, header_h), (30, 30, 30), thickness=-1)
    cv2.addWeighted(overlay, 0.85, vis, 0.15, 0, vis)

    header_text = (
        f"Graph: {graph.creation_method.upper()} | Nodes: {graph.num_nodes} | "
        f"Edges: {graph.num_edges} | Image: {data_label}"
    )
    cv2.putText(vis, header_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 2)

    # 1. Draw Edges first (underneath node centers)
    for u, v, d in graph.nx_graph.edges(data=True):
        nu = graph.get_node(u)
        nv = graph.get_node(v)
        pt_u = (int(round(nu["x"])), int(round(nu["y"])))
        pt_v = (int(round(nv["x"])), int(round(nv["y"])))
        cv2.line(vis, pt_u, pt_v, edge_color, thickness=2, lineType=cv2.LINE_AA)

    # 2. Draw Crater Circles & Node Centers
    for nid in graph.nodes():
        node = graph.get_node(nid)
        cx, cy, r = int(round(node["x"])), int(round(node["y"])), int(round(node["radius"]))

        # Crater perimeter circle
        cv2.circle(vis, (cx, cy), r, crater_color, thickness=2, lineType=cv2.LINE_AA)

        # Center dot
        cv2.circle(vis, (cx, cy), 3, center_color, thickness=-1)

        # Node ID label
        if show_node_ids:
            lbl = f"N{nid}"
            lbl_x = min(w - 30, cx + 5)
            lbl_y = max(header_h + 15, cy - 5)
            cv2.putText(vis, lbl, (lbl_x, lbl_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    return vis
