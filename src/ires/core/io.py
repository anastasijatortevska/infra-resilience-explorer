"""Input/output helpers for Infra Resilience Explorer."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Iterable

import networkx as nx

CAPACITY_KEY = "capacity"


def load_graph(path: str | Path) -> nx.Graph:
    """Load an undirected weighted graph from an edge list file.

    Each non-empty line must have three tokens: ``u v w`` where ``u`` and ``v`` are
    node identifiers (kept as strings) and ``w`` is a float capacity. Duplicate edges
    between the same unordered pair are merged by summing their weights.
    """

    graph = nx.Graph()
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 3:
            raise ValueError(f"Malformed line in edge list: {line!r}")
        u, v, w_txt = parts
        weight = float(w_txt)
        if graph.has_edge(u, v):
            graph[u][v][CAPACITY_KEY] += weight
        else:
            graph.add_edge(u, v, **{CAPACITY_KEY: weight})
    return graph


def edge_key(u: str, v: str) -> tuple[str, str]:
    """Return a canonical unordered edge key."""

    return (u, v) if u <= v else (v, u)


def iter_edges(graph: nx.Graph) -> Iterable[tuple[str, str, float]]:
    """Yield edges with capacity in a consistent form."""

    for u, v, data in graph.edges(data=True):
        yield u, v, float(data.get(CAPACITY_KEY, 0.0))
