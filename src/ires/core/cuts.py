"""Cut extraction from the fundamental cuts of mixture trees.

A tree edge ``(parent, child)`` induces the cut ``(subtree(child), rest)``, whose
capacity is exactly the induced capacity ``C_T[(parent, child)]`` from
:mod:`ires.core.metrics`. A graph edge crosses that cut iff one endpoint lies in the
subtree, which is what the LCA accumulation already counts, so capacities are read out
of ``C_T`` rather than recomputed: O(n) per tree instead of the O(n*m) of one edge-list
scan per tree edge. :func:`cut_capacity` keeps the direct definition for tests and
cross-checks.

The same partition is usually induced by many trees of the mixture. Cuts are grouped by
a 64-bit XOR fingerprint of the subtree's node set, canonicalized so a partition and its
complement agree, which keeps grouping O(n) per tree; the reported cuts are then
regrouped on their materialized node sets, so the output is exactly deduplicated
regardless of fingerprint collisions. Node sets are materialized only for cuts that can
still reach the reported prefix.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field, replace

import networkx as nx

from . import io
from .io import Edge
from .tree import Tree

_FINGERPRINT_BYTES = 8


def cut_capacity(graph: nx.Graph, subset: set[str]) -> float:
    """Exact capacity of the cut (subset, complement), by direct definition.

    The reference implementation. Mixture cut extraction reads capacities from the
    induced capacities instead; see the module docstring.
    """

    try:
        total = math.fsum(w for u, v, w in io.iter_edges(graph) if (u in subset) != (v in subset))
    except OverflowError as exc:
        raise ValueError("Cut capacity overflow; reduce capacity range") from exc
    if not math.isfinite(total):
        raise ValueError("Nonfinite cut capacity; reduce capacity range")
    return total


def _node_label(node: str) -> int:
    digest = hashlib.blake2b(node.encode("utf-8"), digest_size=_FINGERPRINT_BYTES).digest()
    return int.from_bytes(digest, "big")


def subtree_fingerprints(tree: Tree) -> dict[str, int]:
    """XOR fingerprint of each node's subtree node set, in one O(n) pass."""

    acc = {node: _node_label(node) for node in tree.parent}
    for node in reversed(tree.bfs_order):
        par = tree.parent[node]
        if par is not None:
            acc[par] ^= acc[node]
    return acc


@dataclass
class Cut:
    """A partition induced by a tree edge, identified by the partition itself.

    The inducing edge is provenance, not identity: the same partition reached from
    different trees is one cut.
    """

    edge: Edge
    capacity: float
    size: int
    key: int
    # The subtree endpoint of the inducing edge. ``edge`` is canonicalized for
    # reporting, which discards the orientation, so the side is recorded separately.
    child: str
    # Provenance, not identity: excluded so ranking is order-independent.
    tree: Tree = field(repr=False, compare=False)
    mixture_mass: float = 0.0
    found_in_trees: int = 0

    def nodes(self) -> list[str]:
        """Materialize the canonical side: the smaller one, lexicographic on ties."""

        inside = sorted(self.tree.subtree_nodes(self.child))
        if 2 * len(inside) < len(self.tree.parent):
            return inside
        outside = sorted(set(self.tree.parent) - set(inside))
        return min((inside, outside), key=lambda side: (len(side), side))


def tree_cuts(tree: Tree, c_t: dict[Edge, float]) -> Iterator[Cut]:
    """Yield one :class:`Cut` per tree edge, reading capacities from ``c_t``."""

    sizes = tree.subtree_sizes()
    prints = subtree_fingerprints(tree)
    total = prints[tree.root]
    for parent, child in tree.edges():
        mine = prints[child]
        yield Cut(
            edge=io.edge_key(parent, child),
            capacity=c_t[(parent, child)],
            size=sizes[child],
            key=min(mine, total ^ mine),
            child=child,
            tree=tree,
        )


def _canonical_size(cut: Cut) -> int:
    return min(cut.size, len(cut.tree.parent) - cut.size)


def _representative_key(cut: Cut) -> tuple[int, Edge, int, str]:
    """Total order over the cuts inducing one partition, so the pick is stable.

    Several trees induce the same partition from different edges and orientations;
    ordering on every field that survives into the output keeps the chosen
    representative independent of the order the candidates arrive in.
    """

    return (_canonical_size(cut), cut.edge, cut.size, cut.child)


def rank_cuts(cuts: Iterable[Cut], top_k: int | None = None) -> list[Cut]:
    """Globally rank distinct partitions by capacity, then size, node labels and edge.

    Duplicates are merged, accumulating the probability mass of the trees that induce
    the partition into ``mixture_mass``, so a cut several high-probability trees agree
    on is distinguishable from one seen once.
    """

    if top_k is not None:
        io.positive_count(top_k, "top_k")

    # Accumulate into copies: ranking the same candidates twice must not double-count.
    merged: dict[int, Cut] = {}
    for cut in cuts:
        seen = merged.get(cut.key)
        if seen is None:
            merged[cut.key] = replace(cut)
        else:
            seen.mixture_mass += cut.mixture_mass
            seen.found_in_trees += cut.found_in_trees
            if _representative_key(cut) < _representative_key(seen):
                seen.edge, seen.size = cut.edge, cut.size
                seen.child, seen.tree = cut.child, cut.tree

    ordered = sorted(merged.values(), key=lambda c: (c.capacity, _canonical_size(c), c.edge))
    if top_k is None:
        candidates = ordered
    else:
        # Keep everything that could still tie into the reported prefix, so the exact
        # node-label tie-break below sees every contender without materializing the rest.
        boundary = ordered[top_k - 1] if top_k <= len(ordered) else ordered[-1]
        cutoff = (boundary.capacity, _canonical_size(boundary))
        candidates = [c for c in ordered if (c.capacity, _canonical_size(c)) <= cutoff]

    # Regroup on materialized node sets: exact identity, independent of fingerprints.
    exact: dict[tuple[str, ...], Cut] = {}
    for cut in candidates:
        partition = tuple(cut.nodes())
        seen = exact.get(partition)
        if seen is None:
            exact[partition] = replace(cut)
        else:
            seen.mixture_mass += cut.mixture_mass
            seen.found_in_trees += cut.found_in_trees
            if _representative_key(cut) < _representative_key(seen):
                seen.edge, seen.size = cut.edge, cut.size
                seen.child, seen.tree = cut.child, cut.tree

    ranked = sorted(
        exact.items(), key=lambda item: (item[1].capacity, len(item[0]), item[0], item[1].edge)
    )
    result = [cut for _, cut in ranked]
    return result[:top_k] if top_k is not None else result


def collect_cuts(weighted_trees: Iterable[tuple[Tree, dict[Edge, float], float]]) -> list[Cut]:
    """Aggregate the fundamental cuts of a mixture, deduplicated and ranked.

    ``weighted_trees`` yields ``(tree, c_t, probability)``.
    """

    gathered: list[Cut] = []
    for tree, c_t, probability in weighted_trees:
        for cut in tree_cuts(tree, c_t):
            cut.mixture_mass = probability
            cut.found_in_trees = 1
            gathered.append(cut)
    return rank_cuts(gathered)


def extract_tree_cuts(graph: nx.Graph, tree: Tree, top_k: int | None = None) -> list[Cut]:
    """Fundamental cuts of a single tree, computing ``C_T`` on the way."""

    from .lca import LCA
    from .metrics import compute_tree_capacities

    lca = LCA(tree)
    c_t = compute_tree_capacities(graph, tree, lca)
    cuts = list(tree_cuts(tree, c_t))
    for cut in cuts:
        cut.mixture_mass = 1.0
        cut.found_in_trees = 1
    return rank_cuts(cuts, top_k)


def as_records(cuts: Sequence[Cut], max_nodes: int = 30) -> list[dict[str, object]]:
    """Serialize ranked cuts, truncating the node list for readability."""

    records: list[dict[str, object]] = []
    for cut in cuts:
        nodes = cut.nodes()
        records.append(
            {
                "edge": list(cut.edge),
                "capacity": cut.capacity,
                "side_size": len(nodes),
                "mixture_mass": cut.mixture_mass,
                "found_in_trees": cut.found_in_trees,
                "nodes": nodes[:max_nodes],
                "truncated": len(nodes) > max_nodes,
            }
        )
    return records
