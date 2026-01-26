"""Binary lifting LCA with path-sum support."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

from .tree import Tree


class LCA:
    """Binary-lifting LCA solver bound to a Tree."""

    def __init__(self, tree: Tree):
        self.tree = tree
        self.nodes = tree.bfs_order
        self.max_log = math.ceil(math.log2(len(self.nodes))) if self.nodes else 0
        self.up: Dict[int, Dict[str, str | None]] = {0: {}}
        for node, par in tree.parent.items():
            self.up[0][node] = par

        for k in range(1, self.max_log + 1):
            self.up[k] = {}
            for node in self.nodes:
                mid = self.up[k - 1].get(node)
                self.up[k][node] = self.up[k - 1].get(mid) if mid is not None else None

        self.prefix: Dict[str, float] = {}
        self.edge_weights: Dict[Tuple[str, str], float] = {}
        self.set_edge_weights({})

    def set_edge_weights(self, weights: Dict[Tuple[str, str], float]) -> None:
        """Set oriented edge weights (parent, child) and recompute prefix sums."""

        self.edge_weights = dict(weights)
        self.prefix = {self.tree.root: 0.0}
        for node in self.tree.bfs_order:
            if node == self.tree.root:
                continue
            parent = self.tree.parent[node]
            if parent is None:
                raise ValueError(f"Node {node} missing parent")
            w = self.edge_weights.get((parent, node), 0.0)
            self.prefix[node] = self.prefix[parent] + w

    def lca(self, u: str, v: str) -> str:
        """Lowest common ancestor of two nodes."""

        if self.tree.depth[u] < self.tree.depth[v]:
            u, v = v, u
        # Lift u up to depth of v.
        diff = self.tree.depth[u] - self.tree.depth[v]
        for k in range(self.max_log + 1):
            if diff & (1 << k):
                u = self.up[k][u]  # type: ignore[index]
                if u is None:
                    break
        if u == v:
            return u

        for k in reversed(range(self.max_log + 1)):
            if self.up[k][u] != self.up[k][v]:
                u = self.up[k][u]  # type: ignore[index]
                v = self.up[k][v]  # type: ignore[index]
        return self.tree.parent[u]  # type: ignore[return-value]

    def dist(self, u: str, v: str) -> float:
        """Path length using current edge weights."""

        ancestor = self.lca(u, v)
        return self.prefix[u] + self.prefix[v] - 2 * self.prefix[ancestor]
