"""Shortest-path tree oracle: the baseline to compare other oracles against.

A shortest-path tree from a random root is cheap but carries no approximation
guarantee for minimum average congestion, so the MWU bound it produces is a
heuristic rather than a certificate. See :mod:`ires.oracles.mpx` for a
low-diameter-decomposition alternative.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import networkx as nx
import numpy as np

from ..core.io import Edge, edge_key
from ..core.tree import Tree


def shortest_path_tree(
    graph: nx.Graph, lengths: Mapping[Edge, float], rng: np.random.Generator
) -> Tree:
    """Shortest-path tree from a uniformly random root under ``lengths``.

    ``lengths`` must cover every edge; a missing entry raises ``KeyError`` rather
    than falling back to a default, which would silently return a wrong tree.
    """

    root = str(rng.choice(sorted(graph.nodes)))
    return shortest_path_tree_from(graph, lengths, root)


def shortest_path_tree_from(graph: nx.Graph, lengths: Mapping[Edge, float], root: str) -> Tree:
    """Shortest-path tree from an explicit ``root``."""

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
