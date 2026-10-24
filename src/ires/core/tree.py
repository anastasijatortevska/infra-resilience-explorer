"""Rooted tree utilities."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable


class Tree:
    """Represents a rooted tree with parent/child relationships."""

    def __init__(self, root: str, parent: dict[str, str | None]):
        if root not in parent:
            raise ValueError("root must be present in parent map")
        if parent[root] is not None:
            raise ValueError(f"root {root!r} must have no parent")
        self.root = root
        self.parent = dict(parent)

        self.children: dict[str, list[str]] = defaultdict(list)
        for node, par in self.parent.items():
            if par is not None:
                self.children[par].append(node)

        self.depth: dict[str, int] = {}
        self._bfs_order: list[str] = []
        queue: deque[str] = deque([root])
        self.depth[root] = 0
        while queue:
            u = queue.popleft()
            self._bfs_order.append(u)
            for child in self.children.get(u, []):
                self.depth[child] = self.depth[u] + 1
                queue.append(child)

        missing_depth = [n for n in self.parent if n not in self.depth]
        if missing_depth:
            raise ValueError(f"Parent map does not form a connected tree: {missing_depth}")

    @property
    def nodes(self) -> list[str]:
        return list(self.parent.keys())

    @property
    def bfs_order(self) -> list[str]:
        return list(self._bfs_order)

    @property
    def postorder(self) -> list[str]:
        """Return nodes in postorder (children before parent)."""

        order: list[str] = []
        stack: list[tuple[str, int]] = [(self.root, 0)]
        while stack:
            node, state = stack.pop()
            if state == 0:
                stack.append((node, 1))
                for child in self.children.get(node, []):
                    stack.append((child, 0))
            else:
                order.append(node)
        return order

    def edges(self) -> list[tuple[str, str]]:
        """List oriented edges (parent, child)."""

        return [(par, node) for node, par in self.parent.items() if par is not None]

    def subtree_nodes(self, node: str) -> list[str]:
        """Return all nodes in the subtree rooted at ``node``."""

        out: list[str] = []
        stack = [node]
        while stack:
            u = stack.pop()
            out.append(u)
            stack.extend(self.children.get(u, []))
        return out

    def subtree_sizes(self) -> dict[str, int]:
        """Return the subtree size of every node in one O(n) pass."""

        sizes = dict.fromkeys(self.parent, 1)
        for node in reversed(self._bfs_order):
            par = self.parent[node]
            if par is not None:
                sizes[par] += sizes[node]
        return sizes

    @classmethod
    def from_parent_map(cls, root: str, parents: dict[str, str | None]) -> Tree:
        return cls(root=root, parent=parents)

    @classmethod
    def from_edges(
        cls, edges: Iterable[tuple[str, str]], root: str, nodes: Iterable[str] | None = None
    ) -> Tree:
        """Build a rooted tree from an undirected edge set.

        Orients the edges by breadth-first search from ``root``. Raises ``ValueError``
        unless the edges form a spanning tree of ``nodes`` (defaulting to the nodes
        touched by ``edges``): no fabricated edges, no silently dropped component.
        """

        adjacency: dict[str, list[str]] = defaultdict(list)
        edge_list = list(edges)
        for u, v in edge_list:
            adjacency[u].append(v)
            adjacency[v].append(u)

        expected = set(adjacency) if nodes is None else set(nodes)
        if root not in expected:
            raise ValueError(f"root {root!r} is not among the tree's nodes")
        extra = set(adjacency) - expected
        if extra:
            raise ValueError(f"edges touch nodes outside the graph: {sorted(extra)[:5]}")
        if len(edge_list) != len(expected) - 1:
            raise ValueError(
                f"expected {len(expected) - 1} edges for a spanning tree on "
                f"{len(expected)} nodes, got {len(edge_list)}"
            )

        parent: dict[str, str | None] = {root: None}
        queue: deque[str] = deque([root])
        while queue:
            u = queue.popleft()
            for v in adjacency[u]:
                if v not in parent:
                    parent[v] = u
                    queue.append(v)

        unreached = expected - set(parent)
        if unreached:
            raise ValueError(f"edges do not span the graph; unreached: {sorted(unreached)[:5]}")
        return cls(root=root, parent=parent)
