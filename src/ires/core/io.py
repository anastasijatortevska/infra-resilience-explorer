"""Input/output helpers for Infra Resilience Explorer."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator
from pathlib import Path

import networkx as nx

CAPACITY_KEY = "capacity"

Edge = tuple[str, str]


def load_graph(path: str | Path) -> nx.Graph:
    """Load an undirected weighted graph from an edge list file.

    Each non-empty line must have three tokens: ``u v w`` where ``u`` and ``v`` are
    node identifiers (kept as strings) and ``w`` is a finite positive capacity.
    Duplicate edges between the same unordered pair are merged by summing weights.
    """

    graph = nx.Graph()
    for line_number, raw in enumerate(Path(path).read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 3:
            raise ValueError(f"Line {line_number}: expected three tokens u v capacity")
        u, v, w_txt = parts
        try:
            weight = float(w_txt)
        except ValueError as exc:
            raise ValueError(f"Line {line_number}: invalid capacity {w_txt!r}") from exc
        if u == v:
            raise ValueError(f"Line {line_number}: self-loops are not supported")
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"Line {line_number}: capacity must be finite and positive")
        if graph.has_edge(u, v):
            graph[u][v][CAPACITY_KEY] += weight
            if not math.isfinite(graph[u][v][CAPACITY_KEY]):
                raise ValueError(f"Line {line_number}: merged capacity overflow for {(u, v)}")
        else:
            graph.add_edge(u, v, **{CAPACITY_KEY: weight})
    validate_graph(graph)
    # Canonical insertion order also makes tied shortest paths reproducible.
    ordered = nx.Graph()
    ordered.add_nodes_from(sorted(graph.nodes))
    for u, v, capacity in iter_edges(graph):
        ordered.add_edge(u, v, **{CAPACITY_KEY: capacity})
    return ordered


def validate_graph(graph: nx.Graph) -> None:
    """Reject unsupported graphs rather than synthesizing missing connectivity."""

    if graph.is_directed() or graph.is_multigraph():
        raise ValueError("Expected a simple undirected graph; merge parallel edges first")
    if graph.number_of_nodes() < 2:
        raise ValueError("Graph must contain at least two vertices")
    if any(not isinstance(node, str) for node in graph):
        raise ValueError("Node identifiers must be strings")
    if nx.number_of_selfloops(graph):
        raise ValueError("Self-loops are not supported")
    for u, v, data in graph.edges(data=True):
        try:
            value = data[CAPACITY_KEY]
            capacity = float(value)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Edge {(u, v)} requires a finite positive capacity") from exc
        if isinstance(value, bool) or not math.isfinite(capacity) or capacity <= 0:
            raise ValueError(f"Edge {(u, v)} requires a finite positive capacity")
    if not nx.is_connected(graph):
        raise ValueError("Graph must be connected")


def graph_fingerprint(graph: nx.Graph) -> str:
    """Identify node labels and capacities independent of file/edge ordering."""

    payload = {"nodes": sorted(graph), "edges": list(iter_edges(graph))}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def positive_count(value: object, name: str) -> int:
    """Require a positive ``int``, returning it so callers can narrow the type."""

    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def positive_finite(value: object, name: str) -> float:
    """Require a finite positive real, returning it so callers can narrow the type."""

    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def edge_key(u: str, v: str) -> Edge:
    """Return a canonical unordered edge key."""

    return (u, v) if u <= v else (v, u)


def iter_edges(graph: nx.Graph) -> Iterator[tuple[str, str, float]]:
    """Yield edges with capacity in canonical sorted order."""

    for u, v in sorted(edge_key(u, v) for u, v in graph.edges):
        yield u, v, float(graph[u][v][CAPACITY_KEY])


def capacities(graph: nx.Graph) -> dict[Edge, float]:
    """Return a canonical edge-key to capacity map."""

    return {edge_key(u, v): w for u, v, w in iter_edges(graph)}
