"""Spanning tree from recursive low-diameter decomposition.

Uses the exponential-start-time clustering of Miller, Peng and Xu, "Parallel graph
decompositions using random shifts" (SPAA 2013): each node draws a shift
``delta_v ~ Exp(1/scale)``, a multi-source Dijkstra started at time
``max(delta) - delta_v`` assigns every node to a cluster, and the settling edges form
a spanning forest of the clusters. Contracting the clusters and repeating until one
node remains yields a spanning tree of the original graph.

Clusters have radius O(scale) and an edge is cut with probability O(length/scale), so
each level keeps distances short while shrinking the graph geometrically, the
standard route to low stretch. The stretch constant of this particular variant is not
proved here; ``scripts/compare_oracles.py`` measures average stretch against the
shortest-path-tree baseline so the claim stays empirical.
"""

from __future__ import annotations

import heapq
from collections.abc import Mapping

import networkx as nx
import numpy as np

from ..core.io import Edge, edge_key
from ..core.tree import Tree

LENGTH_KEY = "length"
ORIGIN_KEY = "origin"


def _mean_length(graph: nx.Graph) -> float:
    """Average edge length, used as the cluster radius for the next level."""

    if graph.number_of_edges() == 0:
        return 1.0
    total = sum(data[LENGTH_KEY] for _, _, data in graph.edges(data=True))
    return total / graph.number_of_edges()


def _decompose(
    graph: nx.Graph, rng: np.random.Generator, scale: float
) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """Cluster ``graph`` by random shifts; return (node -> cluster centre, forest edges)."""

    shift = {node: scale * float(rng.exponential()) for node in graph.nodes}
    latest = max(shift.values())

    heap: list[tuple[float, str, str, str | None]] = [
        (latest - shift[node], node, node, None) for node in graph.nodes
    ]
    heapq.heapify(heap)

    centre: dict[str, str] = {}
    parent: dict[str, str] = {}
    while heap:
        arrival, node, source, via = heapq.heappop(heap)
        if node in centre:
            continue
        centre[node] = source
        if via is not None:
            parent[node] = via
        for neighbour, data in graph[node].items():
            if neighbour not in centre:
                heapq.heappush(heap, (arrival + data[LENGTH_KEY], neighbour, source, node))

    forest = [(via, node) for node, via in parent.items()]
    return centre, forest


def _contract(graph: nx.Graph, centre: Mapping[str, str]) -> nx.Graph:
    """Collapse each cluster to its centre, keeping the shortest crossing edge."""

    contracted = nx.Graph()
    contracted.add_nodes_from(set(centre.values()))
    for u, v, data in graph.edges(data=True):
        cu, cv = centre[u], centre[v]
        if cu == cv:
            continue
        existing = contracted.get_edge_data(cu, cv)
        if existing is not None and existing[LENGTH_KEY] <= data[LENGTH_KEY]:
            continue
        contracted.add_edge(cu, cv, **{LENGTH_KEY: data[LENGTH_KEY], ORIGIN_KEY: data[ORIGIN_KEY]})
    return contracted


def low_diameter_spanning_tree(
    graph: nx.Graph, lengths: Mapping[Edge, float], rng: np.random.Generator
) -> Tree:
    """Build a spanning tree by repeated low-diameter decomposition and contraction."""

    current = nx.Graph()
    current.add_nodes_from(graph.nodes)
    for u, v in graph.edges:
        length = lengths[edge_key(u, v)]
        current.add_edge(u, v, **{LENGTH_KEY: length, ORIGIN_KEY: (u, v)})

    if current.number_of_nodes() == 1:
        return Tree(root=str(next(iter(graph.nodes))), parent={str(next(iter(graph.nodes))): None})

    scale = _mean_length(current)

    chosen: list[Edge] = []
    while current.number_of_nodes() > 1:
        centre, forest = _decompose(current, rng, scale)
        if len(set(centre.values())) == current.number_of_nodes():
            # Every node became its own cluster: shifts were too small to merge
            # anything. Grow the radius and retry; this terminates because a large
            # enough scale collapses the whole graph into one cluster.
            scale *= 2.0
            continue
        chosen.extend(current[u][v][ORIGIN_KEY] for u, v in forest)
        current = _contract(current, centre)
        # Re-read the radius from the contracted metric rather than doubling blindly:
        # contraction changes the length scale, and tracking it measures lower average
        # stretch in scripts/compare_oracles.py.
        scale = 2.0 * _mean_length(current)

    root = str(next(iter(current.nodes)))
    return Tree.from_edges(chosen, root=root, nodes=graph.nodes)
