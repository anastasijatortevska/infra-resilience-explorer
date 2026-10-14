"""Shortest-path tree oracle under custom edge lengths."""

from __future__ import annotations

import math

import networkx as nx

from ..core import io
from ..core.tree import Tree


def shortest_path_tree(graph: nx.Graph, lengths: dict[tuple[str, str], float], root: str) -> Tree:
    """Compute a shortest-path tree from ``root`` using provided edge lengths."""

    io.validate_graph(graph)
    if root not in graph:
        raise ValueError(f"Root {root!r} is not a graph vertex")
    expected = {io.edge_key(u, v) for u, v in graph.edges}
    if set(lengths) != expected:
        raise ValueError("Lengths must cover exactly the graph edges")
    if any(not math.isfinite(value) or value <= 0 for value in lengths.values()):
        raise ValueError("Shortest-path lengths must be finite and positive")

    ordered = nx.Graph()
    ordered.add_nodes_from(sorted(graph))
    ordered.add_edges_from(sorted(expected))

    def weight(u: str, v: str, data: dict) -> float:
        return float(lengths[io.edge_key(u, v)])

    _, paths = nx.single_source_dijkstra(ordered, root, weight=weight)
    if set(paths) != set(graph):
        raise ValueError("Shortest-path tree cannot cover unreachable vertices")
    parent = {node: path[-2] if node != root else None for node, path in paths.items()}
    tree = Tree.from_parent_map(root, parent)
    tree.validate_against(graph)
    return tree
