"""Rooted tree utilities."""

from __future__ import annotations

from collections import defaultdict, deque


class Tree:
    """Represents a rooted tree with parent/child relationships."""

    def __init__(self, root: str, parent: dict[str, str | None]):
        if root not in parent:
            raise ValueError("root must be present in parent map")
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

    @classmethod
    def from_parent_map(cls, root: str, parents: dict[str, str | None]) -> Tree:
        return cls(root=root, parent=parents)
