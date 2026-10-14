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
    return rank_cuts(graph, cuts, top_k)


def rank_cuts(
    graph: nx.Graph, candidates: list[dict[str, object]], top_k: int | None = None
) -> list[dict[str, object]]:
    """Globally rank unique partitions by capacity, then size/labels/edge.

    Choose the smaller side; for equally sized sides choose lexicographically.
    The inducing edge is only provenance, not the cut's identity.
    """

    if top_k is not None:
        io.positive_count(top_k, "top_k")
    records: dict[tuple[str, ...], dict[str, object]] = {}
    for candidate in candidates:
        subset = set(candidate["nodes"])
        if not subset or not subset < set(graph):
            raise ValueError("Candidate cut must be a nonempty proper subset of graph vertices")
        sides = (tuple(sorted(subset)), tuple(sorted(set(graph) - subset)))
        partition = min(sides, key=lambda side: (len(side), side))
        edge = io.edge_key(*candidate["edge"])
        record = {"edge": edge, "capacity": candidate["capacity"], "nodes": list(partition)}
        if partition not in records or edge < records[partition]["edge"]:
            records[partition] = record
    ranked = sorted(
        records.values(),
        key=lambda rec: (rec["capacity"], len(rec["nodes"]), tuple(rec["nodes"]), rec["edge"]),
    )
    return ranked[:top_k] if top_k is not None else ranked
