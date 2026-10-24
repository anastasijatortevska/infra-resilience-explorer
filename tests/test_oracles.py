from __future__ import annotations

import statistics

import networkx as nx
import numpy as np
import pytest
from conftest import random_capacitated_graph

from ires.core import io
from ires.core.lca import LCA
from ires.core.tree import Tree
from ires.oracles import DEFAULT_ORACLE, ORACLES, get_oracle


def unit_grid(side: int) -> nx.Graph:
    graph = nx.relabel_nodes(nx.grid_2d_graph(side, side), lambda rc: f"r{rc[0]}c{rc[1]}")
    for u, v in graph.edges:
        graph[u][v][io.CAPACITY_KEY] = 1.0
    return graph


def average_stretch(graph: nx.Graph, tree: Tree, lengths: dict[io.Edge, float]) -> float:
    """Mean ratio of tree path length to direct edge length under ``lengths``."""

    lca = LCA(tree)
    lca.set_edge_weights({(p, c): lengths[io.edge_key(p, c)] for p, c in tree.edges()})
    return statistics.fmean(
        lca.dist(u, v) / lengths[io.edge_key(u, v)] for u, v, _ in io.iter_edges(graph)
    )


@pytest.mark.parametrize("oracle", sorted(ORACLES))
def test_every_oracle_returns_a_spanning_tree(oracle: str) -> None:
    graph = random_capacitated_graph(120, seed=8, spread=3.0)
    lengths = {edge: 1.0 / capacity for edge, capacity in io.capacities(graph).items()}
    rng = np.random.default_rng(0)
    build = ORACLES[oracle]
    for _ in range(10):
        tree = build(graph, lengths, rng)
        assert len(tree.edges()) == graph.number_of_nodes() - 1
        assert set(tree.nodes) == set(graph.nodes)
        assert all(graph.has_edge(u, v) for u, v in tree.edges())


@pytest.mark.parametrize("oracle", sorted(ORACLES))
def test_oracles_require_complete_lengths(oracle: str) -> None:
    graph = random_capacitated_graph(20, seed=0)
    lengths = dict.fromkeys(io.capacities(graph), 1.0)
    lengths.pop(next(iter(lengths)))
    with pytest.raises(KeyError):
        ORACLES[oracle](graph, lengths, np.random.default_rng(0))


def test_registry_lookup() -> None:
    assert get_oracle(DEFAULT_ORACLE) is ORACLES[DEFAULT_ORACLE]
    with pytest.raises(ValueError, match="unknown oracle"):
        get_oracle("nope")


def test_decomposition_oracle_has_lower_stretch_than_the_baseline() -> None:
    """The point of the decomposition oracle: lower average stretch than an SPT.

    Measured on a unit-capacity grid under the resistance metric 1/capacity, which is
    the metric low-stretch spanning tree constructions are stated for.
    """

    graph = unit_grid(15)
    lengths = {edge: 1.0 / capacity for edge, capacity in io.capacities(graph).items()}

    def mean_stretch(name: str) -> float:
        rng = np.random.default_rng(0)
        return statistics.fmean(
            average_stretch(graph, ORACLES[name](graph, lengths, rng), lengths) for _ in range(5)
        )

    ldd, spt = mean_stretch("ldd"), mean_stretch("spt")
    assert ldd < spt, f"ldd stretch {ldd:.2f} should beat spt stretch {spt:.2f}"


def test_stretch_is_finite_on_heavy_tailed_capacities() -> None:
    graph = random_capacitated_graph(80, seed=3, spread=4.0)
    lengths = {edge: 1.0 / capacity for edge, capacity in io.capacities(graph).items()}
    rng = np.random.default_rng(0)
    for name in sorted(ORACLES):
        value = average_stretch(graph, ORACLES[name](graph, lengths, rng), lengths)
        assert value > 0 and value == value
