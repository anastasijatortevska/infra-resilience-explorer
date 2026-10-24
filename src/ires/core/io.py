"""Input/output helpers for Infra Resilience Explorer."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import networkx as nx

CAPACITY_KEY = "capacity"

Edge = tuple[str, str]


def load_graph(path: str | Path) -> nx.Graph:
    """Load an undirected weighted graph from an edge list file.

    Each non-empty line must have three tokens: ``u v w`` where ``u`` and ``v`` are
    node identifiers (kept as strings) and ``w`` is a positive float capacity.
    Duplicate edges between the same unordered pair are merged by summing their
    weights.

    Raises ``ValueError`` for malformed lines, self-loops, non-positive capacities,
    and graphs that are empty or disconnected. The congestion model assumes a
    single connected component with strictly positive capacities: a disconnected
    input has no spanning tree, and a zero capacity makes congestion infinite.
    """

    graph = nx.Graph()
    for lineno, raw in enumerate(Path(path).read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 3:
            raise ValueError(f"{path}:{lineno}: expected 'u v w', got {line!r}")
        u, v, w_txt = parts
        if u == v:
            raise ValueError(f"{path}:{lineno}: self-loop on node {u!r}")
        try:
            weight = float(w_txt)
        except ValueError:
            raise ValueError(f"{path}:{lineno}: capacity {w_txt!r} is not a number") from None
        if weight <= 0.0 or weight != weight:
            raise ValueError(f"{path}:{lineno}: capacity must be positive, got {weight}")
        if graph.has_edge(u, v):
            graph[u][v][CAPACITY_KEY] += weight
        else:
            graph.add_edge(u, v, **{CAPACITY_KEY: weight})

    if graph.number_of_nodes() == 0:
        raise ValueError(f"{path}: no edges found")
    if not nx.is_connected(graph):
        components = sorted((len(c) for c in nx.connected_components(graph)), reverse=True)
        raise ValueError(
            f"{path}: graph is disconnected ({len(components)} components, "
            f"sizes {components[:5]}{'...' if len(components) > 5 else ''}). "
            "Spanning-tree routing requires a connected graph; "
            "split the input and analyse each component separately."
        )
    return graph


def edge_key(u: str, v: str) -> Edge:
    """Return a canonical unordered edge key."""

    return (u, v) if u <= v else (v, u)


def iter_edges(graph: nx.Graph) -> Iterator[tuple[str, str, float]]:
    """Yield edges with capacity in a consistent form."""

    for u, v, data in graph.edges(data=True):
        yield u, v, float(data[CAPACITY_KEY])


def capacities(graph: nx.Graph) -> dict[Edge, float]:
    """Return a canonical edge-key to capacity map."""

    return {edge_key(u, v): w for u, v, w in iter_edges(graph)}
