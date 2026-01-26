"""Shortest-path tree oracle under custom edge lengths."""

from __future__ import annotations

from typing import Dict

import networkx as nx

from ..core import io
from ..core.tree import Tree


def shortest_path_tree(graph: nx.Graph, lengths: Dict[tuple[str, str], float], root: str) -> Tree:
    """Compute a shortest-path tree from ``root`` using provided edge lengths."""

    def weight(u: str, v: str, data: dict) -> float:
        return float(lengths[io.edge_key(u, v)])

    preds, _ = nx.dijkstra_predecessor_and_distance(graph, root, weight=weight)
    parent: Dict[str, str | None] = {root: None}
    for node, plist in preds.items():
        if node == root:
            continue
        parent[node] = plist[0] if plist else None
    # Include isolated nodes (if any)
    for node in graph.nodes:
        parent.setdefault(node, None if node == root else root)
    return Tree.from_parent_map(root, parent)
