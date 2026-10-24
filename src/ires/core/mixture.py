"""Strict saved-tree validation and shared mixture recomputation.

``ires fit`` and ``ires report`` score a mixture through :func:`evaluate_mixture`, so
their agreement is structural rather than two parallel loops that happen to match.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import networkx as nx

from . import cuts as cut_utils
from . import io, metrics
from .cuts import Cut
from .io import Edge
from .lca import LCA
from .tree import Tree

METRIC = "cut_path_score"
SCHEMA_VERSION = 1


def tree_signature(tree: Tree) -> tuple[Edge, ...]:
    """Root-independent identity of a spanning tree: its sorted undirected edge set."""

    return tuple(sorted(io.edge_key(u, v) for u, v in tree.edges()))


@dataclass(frozen=True)
class TreeRecord:
    """One distinct tree of the mixture, with the iterations that selected it."""

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
    def from_tree(cls, tree: Tree, count: int) -> TreeRecord:
        return cls(root=tree.root, edges=tuple(tree.edges()), count=count)


def record_to_tree(record: Mapping[str, Any], graph: nx.Graph) -> Tree:
    """Read oriented parent-child edges without supplying absent parents."""

    if not isinstance(record, Mapping):
        raise ValueError("Each mixture tree must be an object")
    root = record.get("root")
    edges = record.get("edges")
    if not isinstance(root, str) or root not in graph:
        raise ValueError("Mixture tree root must be a graph vertex")
    if not isinstance(edges, list) or len(edges) != len(graph) - 1:
        raise ValueError("Mixture tree must have exactly n-1 oriented edges")
    parent: dict[str, str | None] = {root: None}
    for edge in edges:
        if not isinstance(edge, list | tuple) or len(edge) != 2:
            raise ValueError("Tree edges must be parent-child pairs")
        u, v = edge
        if not isinstance(u, str) or not isinstance(v, str) or not graph.has_edge(u, v):
            raise ValueError(f"Mixture tree edge {edge!r} is not a graph edge")
        if v in parent:
            raise ValueError(f"Mixture tree repeats a child or assigns a parent to root: {v}")
        parent[v] = u
    tree = Tree(root, parent)
    tree.validate_against(graph)
    return tree


def validate_records(graph: nx.Graph, records: Any, total: int) -> list[Tree]:
    """Check counts, probabilities and distinctness before anything is recomputed."""

    io.positive_count(total, "mixture sample count")
    if not isinstance(records, Sequence) or isinstance(records, str) or not records:
        raise ValueError("Mixture trees must be a nonempty list")
    trees: list[Tree] = []
    signatures: set[tuple[Edge, ...]] = set()
    probabilities: list[float] = []
    counts: list[int] = []
    for record in records:
        tree = record_to_tree(record, graph)
        count = io.positive_count(record.get("count"), "tree count")
        probability = record.get("prob")
        if (
            isinstance(probability, bool)
            or not isinstance(probability, int | float)
            or not math.isfinite(probability)
            or not 0 < probability <= 1
        ):
            raise ValueError("Tree probability must be finite and in (0, 1]")
        if not math.isclose(probability, count / total, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("Tree probability must match count / total samples")
        signature = tree_signature(tree)
        if signature in signatures:
            raise ValueError("Duplicate unrooted trees must be aggregated by count")
        signatures.add(signature)
        trees.append(tree)
        counts.append(count)
        probabilities.append(float(probability))
    if sum(counts) != total:
        raise ValueError("Mixture counts must sum to the recorded iteration count")
    if not math.isclose(math.fsum(probabilities), 1, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Mixture probabilities must sum to one")
    return trees


def validate_payload(graph: nx.Graph, payload: Any) -> None:
    """Reject a mixture file that does not belong to this graph and metric."""

    if not isinstance(payload, Mapping) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported mixture schema; regenerate using ires fit")
    if payload.get("metric") != METRIC:
        raise ValueError(f"Mixture metric must be {METRIC}")
    if payload.get("graph_fingerprint") != io.graph_fingerprint(graph):
        raise ValueError("Mixture graph fingerprint does not match node labels and capacities")
    io.positive_count(payload.get("iters"), "iters")
    io.positive_count(payload.get("candidates"), "candidates")
    io.positive_finite(payload.get("alpha"), "alpha")
    io.positive_finite(payload.get("eta"), "eta")
    if type(payload.get("seed")) is not int or payload["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    validate_records(graph, payload.get("trees"), payload["iters"])


@dataclass
class MixtureEvaluation:
    """Expected per-edge scores and ranked cuts for a fitted mixture."""

    expected_scores: dict[Edge, float]
    cuts: list[Cut]
    total_samples: int

    @property
    def max_expected_score(self) -> float:
        """max_e of the expected cut-path score: the value the mixture achieves."""

        return max(self.expected_scores.values(), default=0.0)


def evaluate_mixture(
    graph: nx.Graph, records: Sequence[Mapping[str, Any]], total: int
) -> MixtureEvaluation:
    """Recompute expected scores and cuts from counts, identically for fit and report."""

    trees = validate_records(graph, records, total)
    contributions: dict[Edge, list[float]] = {edge: [] for edge in io.capacities(graph)}
    weighted: list[tuple[Tree, dict[Edge, float], float]] = []

    for record, tree in zip(records, trees, strict=True):
        lca = LCA(tree)
        capacities = metrics.compute_tree_capacities(graph, tree, lca)
        scores = metrics.compute_edge_scores(graph, tree, lca, capacities)
        probability = record["count"] / total
        for edge, score in scores.items():
            contributions[edge].append(probability * score)
        weighted.append((tree, capacities, probability))

    expected = {edge: math.fsum(values) for edge, values in contributions.items()}
    if any(not math.isfinite(value) or value <= 0 for value in expected.values()):
        raise ValueError("Nonfinite or nonpositive expected cut-path score")
    return MixtureEvaluation(
        expected_scores=expected,
        cuts=cut_utils.collect_cuts(weighted),
        total_samples=total,
    )


def summarize(
    graph: nx.Graph, records: Sequence[Mapping[str, Any]], total: int
) -> tuple[dict[Edge, float], list[Cut]]:
    """Tuple form of :func:`evaluate_mixture`, for callers that want just the pair."""

    evaluation = evaluate_mixture(graph, records, total)
    return evaluation.expected_scores, evaluation.cuts


def records_to_dicts(records: Iterable[TreeRecord], total: int) -> list[dict[str, Any]]:
    return [record.to_dict(total) for record in records]
