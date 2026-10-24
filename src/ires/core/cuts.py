"""Cut extraction from tree-induced subtree cuts.

A tree edge ``(parent, child)`` induces the cut ``(subtree(child), rest)``, whose
capacity is exactly ``c_T[(parent, child)]``. A graph edge crosses that cut iff one of
its endpoints lies in the subtree, which is what the LCA accumulation in
:mod:`ires.core.congestion` already counts, so no capacity is recomputed here. Reading
them out of ``c_T`` keeps cut extraction O(n) per tree; rescanning the edge list once per
tree edge would cost O(n*m).

Cuts are deduplicated across the trees of the mixture by a 64-bit XOR fingerprint of the
subtree's node set, canonicalized so a cut and its complement agree. Fingerprints are
derived from node names, so they are stable across runs and processes.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from .io import Edge
from .tree import Tree

_FINGERPRINT_BYTES = 8


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
    """A cut induced by a tree edge, with the side inside the subtree."""

    edge: Edge
    capacity: float
    size: int
    key: int
    tree: Tree = field(repr=False)
    mixture_mass: float = 0.0
    found_in_trees: int = 0

    def nodes(self) -> list[str]:
        """Materialize the subtree side. O(size); only called for reported cuts."""

        return self.tree.subtree_nodes(self.edge[1])


def tree_cuts(tree: Tree, c_t: dict[Edge, float]) -> Iterator[Cut]:
    """Yield one :class:`Cut` per tree edge, reading capacities straight from ``c_t``.

    O(n) overall: no per-edge graph scan and no node sets materialized.
    """

    sizes = tree.subtree_sizes()
    prints = subtree_fingerprints(tree)
    total = prints[tree.root]
    for parent, child in tree.edges():
        mine = prints[child]
        yield Cut(
            edge=(parent, child),
            capacity=c_t[(parent, child)],
            size=sizes[child],
            key=min(mine, total ^ mine),
            tree=tree,
        )


def collect_cuts(weighted_trees: Iterable[tuple[Tree, dict[Edge, float], float]]) -> list[Cut]:
    """Aggregate cuts over a mixture, deduplicated and sorted by ascending capacity.

    ``weighted_trees`` yields ``(tree, c_t, probability)``. The same cut is typically
    induced by many trees of the mixture. Duplicates are merged, accumulating the
    probability mass of the trees that induce the cut into ``mixture_mass``, so a cut
    several high-probability trees agree on is distinguishable from one seen once. Ties
    in capacity are broken by descending mass, then by the smaller side.
    """

    merged: dict[int, Cut] = {}
    for tree, c_t, prob in weighted_trees:
        for cut in tree_cuts(tree, c_t):
            seen = merged.get(cut.key)
            if seen is None:
                cut.mixture_mass = prob
                cut.found_in_trees = 1
                merged[cut.key] = cut
            else:
                seen.mixture_mass += prob
                seen.found_in_trees += 1
                if cut.size < seen.size:
                    # Keep the representative whose subtree side is the smaller one.
                    seen.edge, seen.size, seen.tree = cut.edge, cut.size, cut.tree
    out = list(merged.values())
    out.sort(key=lambda c: (c.capacity, -c.mixture_mass, c.size))
    return out
