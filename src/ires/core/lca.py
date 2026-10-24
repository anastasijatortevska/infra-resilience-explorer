"""Binary lifting LCA with path-sum support."""

from __future__ import annotations

import math

from .tree import Tree


class LCA:
    """Binary-lifting LCA solver bound to a Tree."""

    def __init__(self, tree: Tree):
        self.tree = tree
        self.nodes = tree.bfs_order
        self.max_log = math.ceil(math.log2(len(self.nodes))) if self.nodes else 0
        self.up: dict[int, dict[str, str | None]] = {0: {}}
        for node, par in tree.parent.items():
            self.up[0][node] = par

        for k in range(1, self.max_log + 1):
            self.up[k] = {}
            for node in self.nodes:
                mid = self.up[k - 1].get(node)
                self.up[k][node] = self.up[k - 1].get(mid) if mid is not None else None

        self.prefix: dict[str, float] = {}
        self.edge_weights: dict[tuple[str, str], float] = {}
        self.set_edge_weights({})

    def set_edge_weights(self, weights: dict[tuple[str, str], float]) -> None:
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
                lifted = self.up[k][u]
                if lifted is None:
                    raise ValueError(f"depth of {u!r} exceeds its ancestor chain")
                u = lifted
        if u == v:
            return u

        # Descend together while the ancestors differ. At equal depth both lookups are
        # None or both are nodes, so a mismatch guarantees two real ancestors.
        for k in reversed(range(self.max_log + 1)):
            up_u, up_v = self.up[k][u], self.up[k][v]
            if up_u != up_v:
                assert up_u is not None and up_v is not None
                u, v = up_u, up_v
        parent = self.tree.parent[u]
        if parent is None:
            raise ValueError(f"{u!r} and {v!r} have no common ancestor")
        return parent

    def dist(self, u: str, v: str) -> float:
        """Path length using current edge weights."""

        ancestor = self.lca(u, v)
        return self.prefix[u] + self.prefix[v] - 2 * self.prefix[ancestor]
