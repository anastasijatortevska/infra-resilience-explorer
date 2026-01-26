"""Capacities and congestion computations on trees."""

from __future__ import annotations

from typing import Dict, Tuple

import networkx as nx

from . import io
from .lca import LCA
from .tree import Tree


def compute_tree_capacities(graph: nx.Graph, tree: Tree, lca: LCA) -> Dict[Tuple[str, str], float]:
    """Compute induced capacities c_T for each tree edge using the LCA trick."""

    add = {node: 0.0 for node in tree.nodes}
    for u, v, w in io.iter_edges(graph):
        ancestor = lca.lca(u, v)
        add[u] += w
        add[v] += w
        add[ancestor] -= 2 * w

    total = dict(add)
    c_t: Dict[Tuple[str, str], float] = {}
    for node in reversed(tree.bfs_order):
        parent = tree.parent[node]
        if parent is not None:
            total[parent] += total[node]
            c_t[(parent, node)] = total[node]
    return c_t


def compute_edge_congestion(
    graph: nx.Graph, tree: Tree, lca: LCA, c_t: Dict[Tuple[str, str], float]
) -> Dict[Tuple[str, str], float]:
    """Compute cong_T(e) for each original edge."""

    lca.set_edge_weights(c_t)
    congestions: Dict[Tuple[str, str], float] = {}
    for u, v, capacity in io.iter_edges(graph):
        key = io.edge_key(u, v)
        if capacity <= 0:
            congestions[key] = float("inf")
            continue
        dist = lca.dist(u, v)
        congestions[key] = dist / capacity
    return congestions
