"""
src/crater_graph/graph.py

Graph representation for detected crater constellations.
Encapsulates networkx.Graph with typed node attributes (crater candidate, center, radius, confidence)
and edge attributes (Euclidean distance, normalized distance, relative orientation, radius ratio).
"""

from typing import Dict, Any, List, Optional, Tuple, Iterator
import math
import networkx as nx
from src.crater_detection.types import CraterCandidate


class CraterGraph:
    """
    Graph G = (V, E) representing spatial relationships among detected craters.

    Attributes:
        nx_graph: The underlying networkx.Graph instance.
        creation_method: Name of the strategy used (e.g. 'delaunay', 'knn', 'radius').
        image_shape: Optional (height, width) of the source image.
        params: Construction parameters used.
    """

    def __init__(self,
                 nx_graph: Optional[nx.Graph] = None,
                 creation_method: str = "custom",
                 image_shape: Optional[Tuple[int, int]] = None,
                 params: Optional[Dict[str, Any]] = None):
        self.nx_graph: nx.Graph = nx_graph if nx_graph is not None else nx.Graph()
        self.creation_method: str = creation_method
        self.image_shape: Optional[Tuple[int, int]] = image_shape
        self.params: Dict[str, Any] = params or {}

    @property
    def num_nodes(self) -> int:
        """Number of nodes in the graph."""
        return self.nx_graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        """Number of edges in the graph."""
        return self.nx_graph.number_of_edges()

    def __len__(self) -> int:
        return self.num_nodes

    def __iter__(self) -> Iterator[int]:
        return iter(sorted(self.nx_graph.nodes()))

    def nodes(self) -> List[int]:
        """Return sorted list of node IDs."""
        return sorted(self.nx_graph.nodes())

    def edges(self, data: bool = False):
        """Return canonical sorted list of edges."""
        if data:
            return sorted(self.nx_graph.edges(data=True), key=lambda e: (min(e[0], e[1]), max(e[0], e[1])))
        return sorted([(min(u, v), max(u, v)) for u, v in self.nx_graph.edges()], key=lambda e: (e[0], e[1]))

    def get_node(self, node_id: int) -> Dict[str, Any]:
        """Get attributes of a specific node."""
        if node_id not in self.nx_graph:
            raise KeyError(f"Node {node_id} does not exist in CraterGraph.")
        return dict(self.nx_graph.nodes[node_id])

    def get_edge(self, u: int, v: int) -> Dict[str, Any]:
        """Get attributes of an edge between u and v."""
        if not self.nx_graph.has_edge(u, v):
            raise KeyError(f"Edge ({u}, {v}) does not exist in CraterGraph.")
        return dict(self.nx_graph.edges[u, v])

    def get_crater(self, node_id: int) -> CraterCandidate:
        """Get the original CraterCandidate for a node."""
        data = self.get_node(node_id)
        crater = data.get("crater")
        if isinstance(crater, CraterCandidate):
            return crater
        # If reconstructed from dict
        return CraterCandidate.from_dict(data["crater"])

    def craters(self) -> List[CraterCandidate]:
        """Return list of all CraterCandidate objects ordered by node_id."""
        return [self.get_crater(i) for i in self.nodes()]

    def neighbors(self, node_id: int) -> List[int]:
        """Get sorted list of neighbor node IDs for a given node."""
        if node_id not in self.nx_graph:
            raise KeyError(f"Node {node_id} does not exist in CraterGraph.")
        return sorted(self.nx_graph.neighbors(node_id))

    def degree(self, node_id: int) -> int:
        """Degree of a specific node."""
        return self.nx_graph.degree(node_id)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize graph to primitive dictionary (JSON-friendly, no pickle).
        """
        nodes_list = []
        for n in self.nodes():
            nd = dict(self.nx_graph.nodes[n])
            # Serialize crater object safely
            crater_obj = nd.get("crater")
            crater_dict = crater_obj.to_dict() if isinstance(crater_obj, CraterCandidate) else crater_obj
            nodes_list.append({
                "node_id": int(n),
                "x": float(nd["x"]),
                "y": float(nd["y"]),
                "radius": float(nd["radius"]),
                "confidence": float(nd["confidence"]),
                "diameter": float(nd.get("diameter", 2.0 * nd["radius"])),
                "area": float(nd.get("area", math.pi * (nd["radius"] ** 2))),
                "crater": crater_dict,
            })

        edges_list = []
        for u, v, ed in sorted(self.nx_graph.edges(data=True), key=lambda e: (min(e[0], e[1]), max(e[0], e[1]))):
            u_min, v_max = min(u, v), max(u, v)
            edges_list.append({
                "u": int(u_min),
                "v": int(v_max),
                "distance_px": round(float(ed["distance_px"]), 4),
                "normalized_distance": round(float(ed.get("normalized_distance", ed["distance_px"])), 4),
                "relative_angle_rad": round(float(ed.get("relative_angle_rad", 0.0)), 4),
                "radius_ratio": round(float(ed.get("radius_ratio", 1.0)), 4),
                "weight": round(float(ed.get("weight", ed["distance_px"])), 4),
            })

        return {
            "creation_method": self.creation_method,
            "image_shape": list(self.image_shape) if self.image_shape is not None else None,
            "params": self.params,
            "num_nodes": len(nodes_list),
            "num_edges": len(edges_list),
            "nodes": nodes_list,
            "edges": edges_list,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CraterGraph":
        """
        Deserialize graph from primitive dictionary.
        """
        g = nx.Graph()
        for nd in data.get("nodes", []):
            nid = int(nd["node_id"])
            crater_data = nd.get("crater", {})
            crater = CraterCandidate.from_dict(crater_data) if crater_data else CraterCandidate(
                x=float(nd["x"]), y=float(nd["y"]), radius=float(nd["radius"]), confidence=float(nd["confidence"])
            )
            g.add_node(
                nid,
                node_id=nid,
                x=float(nd["x"]),
                y=float(nd["y"]),
                radius=float(nd["radius"]),
                confidence=float(nd["confidence"]),
                diameter=float(nd.get("diameter", 2.0 * float(nd["radius"]))),
                area=float(nd.get("area", math.pi * (float(nd["radius"]) ** 2))),
                crater=crater,
            )

        for ed in data.get("edges", []):
            u, v = int(ed["u"]), int(ed["v"])
            g.add_edge(
                u, v,
                distance_px=float(ed["distance_px"]),
                normalized_distance=float(ed.get("normalized_distance", ed["distance_px"])),
                relative_angle_rad=float(ed.get("relative_angle_rad", 0.0)),
                radius_ratio=float(ed.get("radius_ratio", 1.0)),
                weight=float(ed.get("weight", ed["distance_px"])),
            )

        img_shape = tuple(data["image_shape"]) if data.get("image_shape") is not None else None
        return cls(
            nx_graph=g,
            creation_method=data.get("creation_method", "custom"),
            image_shape=img_shape,
            params=data.get("params", {}),
        )
