"""Strict saved-tree validation and shared mixture recomputation."""

from __future__ import annotations

import math

import networkx as nx

from . import cuts, io, metrics
from .lca import LCA
from .tree import Tree

METRIC = "cut_path_score"
SCHEMA_VERSION = 1


def tree_signature(tree: Tree) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(io.edge_key(u, v) for u, v in tree.edges()))


def record_to_tree(record: dict, graph: nx.Graph) -> Tree:
    """Read oriented parent-child edges without supplying absent parents."""

    if not isinstance(record, dict):
        raise ValueError("Each mixture tree must be an object")
    root = record.get("root")
    edges = record.get("edges")
    if not isinstance(root, str) or root not in graph:
        raise ValueError("Mixture tree root must be a graph vertex")
    if not isinstance(edges, list) or len(edges) != len(graph) - 1:
        raise ValueError("Mixture tree must have exactly n-1 oriented edges")
    parent: dict[str, str | None] = {root: None}
    for edge in edges:
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
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


def validate_records(graph: nx.Graph, records: list, total: int) -> list[Tree]:
    io.positive_count(total, "mixture sample count")
    if not isinstance(records, list) or not records:
        raise ValueError("Mixture trees must be a nonempty list")
    trees = []
    signatures = set()
    probabilities = []
    counts = []
    for record in records:
        tree = record_to_tree(record, graph)
        count = record.get("count")
        io.positive_count(count, "tree count")
        probability = record.get("prob")
        if (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
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
        probabilities.append(probability)
    if sum(counts) != total:
        raise ValueError("Mixture counts must sum to the recorded iteration count")
    if not math.isclose(math.fsum(probabilities), 1, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Mixture probabilities must sum to one")
    return trees


def validate_payload(graph: nx.Graph, payload: dict) -> None:
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported mixture schema; regenerate using ires fit")
    if payload.get("metric") != METRIC:
        raise ValueError("Mixture metric must be cut_path_score")
    if payload.get("graph_fingerprint") != io.graph_fingerprint(graph):
        raise ValueError("Mixture graph fingerprint does not match node labels and capacities")
    io.positive_count(payload.get("iters"), "iters")
    io.positive_count(payload.get("candidates"), "candidates")
    validate_records(graph, payload.get("trees"), payload["iters"])


def summarize(
    graph: nx.Graph, records: list[dict], total: int
) -> tuple[dict[tuple[str, str], float], list[dict[str, object]]]:
    """Recompute expected scores from counts, identically for fit and report."""

    trees = validate_records(graph, records, total)
    contributions = {io.edge_key(u, v): [] for u, v, _ in io.iter_edges(graph)}
    candidates = []
    for record, tree in zip(records, trees, strict=True):
        lca = LCA(tree)
        capacities = metrics.compute_tree_capacities(graph, tree, lca)
        scores = metrics.compute_edge_scores(graph, tree, lca, capacities)
        for edge, score in scores.items():
            contributions[edge].append(record["count"] / total * score)
        candidates.extend(cuts.extract_tree_cuts(graph, tree))
    expected = {edge: math.fsum(values) for edge, values in contributions.items()}
    return expected, cuts.rank_cuts(graph, candidates)
