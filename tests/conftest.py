from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from ires.core import io

DATA = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture
def tiny_graph() -> nx.Graph:
    return io.load_graph(DATA / "tiny.edgelist")


def write_edgelist(path: Path, graph: nx.Graph) -> Path:
    path.write_text("\n".join(f"{u} {v} {w:.10g}" for u, v, w in io.iter_edges(graph)) + "\n")
    return path


def random_capacitated_graph(
    nodes: int, degree: int = 4, seed: int = 0, spread: float = 1.0
) -> nx.Graph:
    """Connected random graph with lognormal capacities (``spread`` = sigma)."""

    graph = nx.gnm_random_graph(nodes, nodes * degree // 2, seed=seed)
    graph = graph.subgraph(max(nx.connected_components(graph), key=len)).copy()
    graph = nx.relabel_nodes(graph, lambda i: f"n{i:04d}")
    rng = np.random.default_rng(seed)
    for u, v in graph.edges:
        graph[u][v][io.CAPACITY_KEY] = float(np.exp(rng.normal(0.0, spread)))
    return graph


def cycle_graph(nodes: int) -> nx.Graph:
    graph = nx.cycle_graph(nodes)
    graph = nx.relabel_nodes(graph, lambda i: f"v{i:03d}")
    for u, v in graph.edges:
        graph[u][v][io.CAPACITY_KEY] = 1.0
    return graph


def brute_force_cut_capacity(graph: nx.Graph, subset: set[str]) -> float:
    """Reference implementation: scan every edge. O(m) per cut, used only in tests."""

    return sum(w for u, v, w in io.iter_edges(graph) if (u in subset) != (v in subset))
