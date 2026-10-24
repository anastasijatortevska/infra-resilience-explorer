"""Typed mixture records and shared mixture evaluation.

Both ``ires fit`` and ``ires report`` score a mixture the same way, so they share
:func:`evaluate_mixture`. Keeping one implementation makes their agreement structural
rather than a coincidence of two parallel loops.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import networkx as nx

from . import congestion, io
from .cuts import Cut, collect_cuts
from .io import Edge
from .lca import LCA
from .tree import Tree


@dataclass(frozen=True)
class TreeRecord:
    """One distinct tree of the mixture, with the number of iterations that chose it."""

    root: str
    edges: tuple[Edge, ...]
    count: int

    def probability(self, total: int) -> float:
        return self.count / total if total else 0.0

    def to_dict(self, total: int) -> dict[str, Any]:
        return {
            "root": self.root,
            "edges": [list(edge) for edge in self.edges],
            "count": self.count,
            "prob": self.probability(total),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TreeRecord:
        try:
            root = str(payload["root"])
            edges = tuple((str(u), str(v)) for u, v in payload["edges"])
            count = int(payload["count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"malformed tree record: {exc}") from exc
        if count <= 0:
            raise ValueError(f"tree record count must be positive, got {count}")
        return cls(root=root, edges=edges, count=count)

    def to_tree(self, graph: nx.Graph) -> Tree:
        """Rebuild the rooted tree, checking it really is a spanning tree of ``graph``."""

        missing = [(u, v) for u, v in self.edges if not graph.has_edge(u, v)]
        if missing:
            raise ValueError(
                f"mixture references {len(missing)} edge(s) absent from the graph, e.g. "
                f"{missing[:3]}. Was the mixture fitted on a different graph?"
            )
        return Tree.from_edges(self.edges, root=self.root, nodes=graph.nodes)

    @classmethod
    def from_tree(cls, tree: Tree, count: int) -> TreeRecord:
        return cls(root=tree.root, edges=tuple(tree.edges()), count=count)


def signature(tree: Tree) -> tuple[Edge, ...]:
    """Root-independent identity of a spanning tree: its sorted undirected edge set."""

    return tuple(sorted(io.edge_key(u, v) for u, v in tree.edges()))


def load_records(payload: Mapping[str, Any]) -> list[TreeRecord]:
    """Parse the ``trees`` array of a mixture.json payload."""

    trees = payload.get("trees")
    if not isinstance(trees, Sequence) or not trees:
        raise ValueError("mixture file has no non-empty 'trees' array")
    return [TreeRecord.from_dict(item) for item in trees]


@dataclass
class MixtureEvaluation:
    """Everything the report needs about a fitted mixture."""

    expected_congestion: dict[Edge, float]
    cuts: list[Cut]
    total_samples: int

    @property
    def alpha_hat(self) -> float:
        """Max expected congestion: a rigorous upper bound on the optimal alpha.

        It is *achieved* by the explicit mixture in ``mixture.json``, so the optimum
        over all tree distributions is no larger.
        """

        return max(self.expected_congestion.values(), default=0.0)


def evaluate_mixture(graph: nx.Graph, records: Iterable[TreeRecord]) -> MixtureEvaluation:
    """Score a mixture: expected per-edge congestion plus deduplicated cuts."""

    records = list(records)
    total = sum(record.count for record in records)
    expected: dict[Edge, float] = dict.fromkeys(io.capacities(graph), 0.0)
    weighted: list[tuple[Tree, dict[Edge, float], float]] = []

    for record in records:
        tree = record.to_tree(graph)
        lca = LCA(tree)
        c_t = congestion.compute_tree_capacities(graph, tree, lca)
        cong = congestion.compute_edge_congestion(graph, tree, lca, c_t)
        weight = record.probability(total)
        for edge, value in cong.items():
            expected[edge] += weight * value
        weighted.append((tree, c_t, weight))

    return MixtureEvaluation(
        expected_congestion=expected,
        cuts=collect_cuts(weighted),
        total_samples=total,
    )
