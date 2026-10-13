"""Fundamental-cut capacities and the experimental cut-path score."""

from __future__ import annotations

import networkx as nx

from . import io
from .lca import LCA
from .tree import Tree


def compute_tree_capacities(graph: nx.Graph, tree: Tree, lca: LCA) -> dict[tuple[str, str], float]:
    """Sum original capacities crossing each tree edge's fundamental cut.

    Equivalently, route demand c(u,v) on the tree path for every original edge:
    this is the resulting load on each tree edge, before dividing by a capacity.
    """

    add = {node: 0.0 for node in tree.nodes}
    for u, v, w in io.iter_edges(graph):
        ancestor = lca.lca(u, v)
        add[u] += w
        add[v] += w
        add[ancestor] -= 2 * w

    total = dict(add)
    c_t: dict[tuple[str, str], float] = {}
    for node in reversed(tree.bfs_order):
        parent = tree.parent[node]
        if parent is not None:
            total[parent] += total[node]
            c_t[(parent, node)] = total[node]
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
        if capacity <= 0:
            scores[key] = float("inf")
            continue
        dist = lca.dist(u, v)
        scores[key] = dist / capacity
    return scores
