"""Cut extraction utilities."""

from __future__ import annotations

from typing import Dict, Iterable, List, Set, Tuple

import networkx as nx

from . import io
from .tree import Tree


def cut_capacity(graph: nx.Graph, subset: Set[str]) -> float:
    """Exact capacity of the cut (subset, complement)."""

    total = 0.0
    for u, v, w in io.iter_edges(graph):
        if (u in subset) != (v in subset):
            total += w
    return total


def extract_tree_cuts(
    graph: nx.Graph, tree: Tree, top_k: int | None = None
) -> List[Dict[str, object]]:
    """Return candidate cuts induced by each tree edge, sorted by capacity."""

    cuts: List[Dict[str, object]] = []
    subtree_cache: Dict[str, List[str]] = {}
    for parent, child in tree.edges():
        if child not in subtree_cache:
            subtree_cache[child] = tree.subtree_nodes(child)
        nodes = subtree_cache[child]
        cap = cut_capacity(graph, set(nodes))
        cuts.append(
            {
                "edge": (parent, child),
                "capacity": cap,
                "nodes": nodes,
            }
        )
    cuts.sort(key=lambda x: x["capacity"])  # type: ignore[index]
    return cuts[:top_k] if top_k is not None else cuts
