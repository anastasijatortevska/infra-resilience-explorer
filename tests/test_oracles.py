from __future__ import annotations

import os
import statistics
import subprocess
import sys
import textwrap
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
from conftest import random_capacitated_graph, resistance_lengths, unit_grid

from ires.core import io
from ires.core.lca import LCA
from ires.core.tree import Tree
from ires.oracles import DEFAULT_ORACLE, ORACLES, get_oracle


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
    lengths = resistance_lengths(graph)
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
    lengths = resistance_lengths(graph)

    def mean_stretch(name: str) -> float:
        rng = np.random.default_rng(0)
        return statistics.fmean(
            average_stretch(graph, ORACLES[name](graph, lengths, rng), lengths) for _ in range(5)
        )

    ldd, spt = mean_stretch("ldd"), mean_stretch("spt")
    assert ldd < spt, f"ldd stretch {ldd:.2f} should beat spt stretch {spt:.2f}"


def test_stretch_is_finite_on_heavy_tailed_capacities() -> None:
    graph = random_capacitated_graph(80, seed=3, spread=4.0)
    lengths = resistance_lengths(graph)
    rng = np.random.default_rng(0)
    for name in sorted(ORACLES):
        value = average_stretch(graph, ORACLES[name](graph, lengths, rng), lengths)
        assert value > 0 and value == value


@pytest.mark.parametrize("oracle", sorted(ORACLES))
def test_oracles_are_reproducible_across_processes(oracle: str) -> None:
    """Same seed, same result, whatever PYTHONHASHSEED is.

    Iterating a set of node labels varies with the per-process string hash seed, so
    a run that does it is reproducible within one process and not between them. Only
    a subprocess with a different seed catches that.
    """

    program = textwrap.dedent(
        f"""
        from conftest import unit_grid
        from ires.core import io
        from ires.oracles import ORACLES
        import numpy as np

        graph = unit_grid(8)
        lengths = {{e: 1.0 for e in io.capacities(graph)}}
        rng = np.random.default_rng(0)
        trees = [ORACLES[{oracle!r}](graph, lengths, rng) for _ in range(3)]
        print(sorted(io.edge_key(u, v) for tree in trees for u, v in tree.edges()))
        """
    )
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parent)}
    outputs = {
        subprocess.run(  # noqa: S603
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            check=True,
            env={**env, "PYTHONHASHSEED": seed},
        ).stdout
        for seed in ("0", "1", "2")
    }
    assert len(outputs) == 1, f"{oracle} depends on PYTHONHASHSEED"
