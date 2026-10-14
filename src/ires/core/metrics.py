"""Fundamental-cut capacities and the experimental cut-path score."""

from __future__ import annotations

import math

import networkx as nx

from . import io
from .lca import LCA
from .tree import Tree


def compute_tree_capacities(graph: nx.Graph, tree: Tree, lca: LCA) -> dict[tuple[str, str], float]:
    """Sum original capacities crossing each tree edge's fundamental cut.

    Equivalently, route demand c(u,v) on the tree path for every original edge:
    this is the resulting load on each tree edge, before dividing by a capacity.
    """

    contributions = {node: [] for node in tree.nodes}
    for u, v, w in io.iter_edges(graph):
        ancestor = lca.lca(u, v)
        contributions[u].append(w)
        contributions[v].append(w)
        contributions[ancestor].extend((-w, -w))

    # fsum limits rounding loss from signed endpoint/LCA contributions.
    # Keep child totals in the same sum instead of rounding a local delta first.
    c_t: dict[tuple[str, str], float] = {}
    for node in reversed(tree.bfs_order):
        parent = tree.parent[node]
        if parent is not None:
            try:
                capacity = math.fsum(contributions[node])
            except OverflowError as exc:
                raise ValueError(f"Induced capacity overflow at {(parent, node)}") from exc
            if not math.isfinite(capacity) or capacity <= 0:
                raise ValueError(
                    f"Invalid induced capacity at {(parent, node)}; reduce capacity range"
                )
            contributions[parent].append(capacity)
            c_t[(parent, node)] = capacity
    return c_t


def compute_edge_scores(
    graph: nx.Graph, tree: Tree, lca: LCA, c_t: dict[tuple[str, str], float]
) -> dict[tuple[str, str], float]:
    """Return s_T(u,v) = sum(c_T(f) for f in path_T(u,v)) / c(u,v).

    This is a dimensionless heuristic, not routing congestion on (u,v).
    See docs/mathematics.md for definitions and a routing counterexample.
    """

    lca.set_edge_weights(c_t)
    scores: dict[tuple[str, str], float] = {}
    for u, v, capacity in io.iter_edges(graph):
        key = io.edge_key(u, v)
        io.positive_finite(capacity, f"capacity of {key}")
        dist = lca.dist(u, v)
        scores[key] = dist / capacity
        if not math.isfinite(scores[key]) or scores[key] <= 0:
            raise ValueError(
                f"Nonfinite or nonpositive cut-path score for {key}; reduce capacity range"
            )
    return scores
