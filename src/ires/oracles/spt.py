"""Shortest-path tree oracle, the baseline candidate generator.

Shortest paths do not solve the weighted tree-score objective in general, so this
carries no approximation guarantee; see ``docs/mathematics.md``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import networkx as nx
import numpy as np

from ..core.io import Edge, edge_key
from ..core.tree import Tree


def shortest_path_tree(graph: nx.Graph, lengths: Mapping[Edge, float], root: str) -> Tree:
    """Shortest-path tree from ``root`` under ``lengths``.

    ``lengths`` must cover every edge; a missing entry raises ``KeyError`` rather than
    falling back to a default, which would silently return a tree for other lengths.
    """

    if root not in graph:
        raise ValueError(f"root {root!r} is not a vertex of the graph")

    def weight(u: str, v: str, _data: Any) -> float:
        return lengths[edge_key(u, v)]

    preds, _ = nx.dijkstra_predecessor_and_distance(graph, root, weight=weight)
    edges: list[Edge] = []
    for node, plist in preds.items():
        if node == root:
            continue
        if not plist:
            raise ValueError(f"node {node!r} unreachable from {root!r}")
        edges.append((plist[0], node))
    return Tree.from_edges(edges, root=root, nodes=graph.nodes)


def random_root_spt(
    graph: nx.Graph, lengths: Mapping[Edge, float], rng: np.random.Generator
) -> Tree:
    """Shortest-path tree from a uniformly random root."""

    return shortest_path_tree(graph, lengths, str(rng.choice(sorted(graph.nodes))))
