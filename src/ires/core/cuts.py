"""Cut extraction utilities."""

from __future__ import annotations

import networkx as nx

from . import io
from .tree import Tree


def cut_capacity(graph: nx.Graph, subset: set[str]) -> float:
    """Exact capacity of the cut (subset, complement)."""

    total = 0.0
    for u, v, w in io.iter_edges(graph):
        if (u in subset) != (v in subset):
            total += w
    return total


def extract_tree_cuts(
    graph: nx.Graph, tree: Tree, top_k: int | None = None
) -> list[dict[str, object]]:
    """Return candidate cuts induced by each tree edge, sorted by capacity."""

    cuts: list[dict[str, object]] = []
    subtree_cache: dict[str, list[str]] = {}
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
