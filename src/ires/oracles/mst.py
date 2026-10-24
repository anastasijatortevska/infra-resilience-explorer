"""Minimum spanning tree oracle under the current MWU edge lengths.

Deterministic: all candidate draws in one iteration return the same tree, so
``--candidates`` buys nothing here. Diversity across the mixture comes from the
lengths changing between iterations instead.
"""

from __future__ import annotations

from collections.abc import Mapping

import networkx as nx
import numpy as np

from ..core.io import Edge, edge_key
from ..core.tree import Tree


def minimum_spanning_tree(
    graph: nx.Graph, lengths: Mapping[Edge, float], rng: np.random.Generator
) -> Tree:
    """Minimum-weight spanning tree, weighting each edge by its MWU length."""

    del rng  # deterministic oracle
    weighted = nx.Graph()
    weighted.add_nodes_from(graph.nodes)
    for u, v in graph.edges:
        weighted.add_edge(u, v, length=lengths[edge_key(u, v)])
    mst = nx.minimum_spanning_tree(weighted, weight="length")
    root = min(graph.nodes)
    return Tree.from_edges(mst.edges(), root=root, nodes=graph.nodes)
