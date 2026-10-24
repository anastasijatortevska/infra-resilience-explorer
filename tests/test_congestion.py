from __future__ import annotations

import networkx as nx
import pytest
from conftest import brute_force_cut_capacity, cycle_graph, random_capacitated_graph

from ires.core import congestion, io
from ires.core.lca import LCA
from ires.oracles.spt import shortest_path_tree_from


def unit_lengths(graph: nx.Graph) -> dict[io.Edge, float]:
    return dict.fromkeys(io.capacities(graph), 1.0)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_tree_capacity_equals_cut_capacity(seed: int) -> None:
    """c_T on a tree edge is exactly the capacity of the cut it induces."""

    graph = random_capacitated_graph(60, seed=seed)
    root = min(graph.nodes)
    tree = shortest_path_tree_from(graph, unit_lengths(graph), root)
    lca = LCA(tree)
    c_t = congestion.compute_tree_capacities(graph, tree, lca)

    assert len(c_t) == graph.number_of_nodes() - 1
    for (_, child), value in c_t.items():
        reference = brute_force_cut_capacity(graph, set(tree.subtree_nodes(child)))
        assert value == pytest.approx(reference)


@pytest.mark.parametrize("seed", [0, 1])
def test_congestion_is_at_least_one_on_tree_edges(seed: int) -> None:
    """An edge crosses its own cut, so c_T(e) >= cap(e) on every tree edge."""

    graph = random_capacitated_graph(60, seed=seed)
    tree = shortest_path_tree_from(graph, unit_lengths(graph), min(graph.nodes))
    lca = LCA(tree)
    c_t = congestion.compute_tree_capacities(graph, tree, lca)
    cong = congestion.compute_edge_congestion(graph, tree, lca, c_t)

    for parent, child in tree.edges():
        assert cong[io.edge_key(parent, child)] >= 1.0 - 1e-12


@pytest.mark.parametrize("n", [4, 7, 12])
def test_cycle_has_exact_known_congestion(n: int) -> None:
    """Unit-capacity cycle: c_T = 2 on every tree edge, max congestion 2(n-1).

    Each tree edge's cut is crossed by itself and by the single omitted edge, which in
    turn routes along all n-1 tree edges.
    """

    graph = cycle_graph(n)
    tree = shortest_path_tree_from(graph, unit_lengths(graph), "v000")
    lca = LCA(tree)
    c_t = congestion.compute_tree_capacities(graph, tree, lca)
    cong = congestion.compute_edge_congestion(graph, tree, lca, c_t)

    assert all(value == pytest.approx(2.0) for value in c_t.values())
    tree_keys = {io.edge_key(p, c) for p, c in tree.edges()}
    (omitted,) = set(cong) - tree_keys
    assert cong[omitted] == pytest.approx(2.0 * (n - 1))
    assert max(cong.values()) == pytest.approx(2.0 * (n - 1))


def test_congestion_is_root_independent() -> None:
    """c_T and congestion depend on the undirected tree only, not on the root."""

    graph = random_capacitated_graph(40, seed=5)
    lengths = unit_lengths(graph)
    reference: dict[io.Edge, float] | None = None
    tree = shortest_path_tree_from(graph, lengths, min(graph.nodes))
    for root in list(graph.nodes)[:5]:
        rerooted = type(tree).from_edges(tree.edges(), root=root, nodes=graph.nodes)
        lca = LCA(rerooted)
        c_t = congestion.compute_tree_capacities(graph, rerooted, lca)
        cong = congestion.compute_edge_congestion(graph, rerooted, lca, c_t)
        if reference is None:
            reference = cong
        else:
            for edge, value in cong.items():
                assert value == pytest.approx(reference[edge])
