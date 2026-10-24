from __future__ import annotations

import pytest
from conftest import brute_force_cut_capacity, random_capacitated_graph

from ires.core import congestion, io
from ires.core.cuts import collect_cuts, subtree_fingerprints, tree_cuts
from ires.core.lca import LCA
from ires.core.mwu import MwuParams, run_mwu
from ires.oracles.spt import shortest_path_tree_from


def test_fingerprint_identifies_a_cut_and_its_complement() -> None:
    graph = random_capacitated_graph(40, seed=2)
    lengths = dict.fromkeys(io.capacities(graph), 1.0)
    nodes = sorted(graph.nodes)
    prints_a = subtree_fingerprints(shortest_path_tree_from(graph, lengths, nodes[0]))
    prints_b = subtree_fingerprints(shortest_path_tree_from(graph, lengths, nodes[1]))
    # The fingerprint of the whole node set is root-independent.
    assert prints_a[nodes[0]] == prints_b[nodes[1]]


def test_tree_cuts_capacities_match_brute_force() -> None:
    graph = random_capacitated_graph(50, seed=3)
    tree = shortest_path_tree_from(graph, dict.fromkeys(io.capacities(graph), 1.0), "n0000")
    lca = LCA(tree)
    c_t = congestion.compute_tree_capacities(graph, tree, lca)

    cuts = list(tree_cuts(tree, c_t))
    assert len(cuts) == graph.number_of_nodes() - 1
    for cut in cuts:
        nodes = cut.nodes()
        assert len(nodes) == cut.size
        assert cut.capacity == pytest.approx(brute_force_cut_capacity(graph, set(nodes)))


def test_collect_cuts_dedupes_and_sorts_globally() -> None:
    """The report's bottleneck list must be the globally cheapest distinct cuts.

    Concatenating per-tree lists and slicing returns the first tree's cheapest cuts,
    and the same cut found by k trees appears k times.
    """

    graph = random_capacitated_graph(60, seed=4)
    result = run_mwu(graph, MwuParams(iters=20, candidates=3, seed=1))

    weighted = []
    raw: list[float] = []
    total = sum(record.count for record in result.records)
    seen_keys: set[int] = set()
    for record in result.records:
        tree = record.to_tree(graph)
        lca = LCA(tree)
        c_t = congestion.compute_tree_capacities(graph, tree, lca)
        weighted.append((tree, c_t, record.probability(total)))
        for cut in tree_cuts(tree, c_t):
            raw.append(cut.capacity)
            seen_keys.add(cut.key)

    collected = collect_cuts(weighted)
    assert len(result.records) > 1, "need several trees for this to be meaningful"
    assert len(collected) == len(seen_keys) < len(raw), "duplicates were not merged"

    capacities = [cut.capacity for cut in collected]
    assert capacities == sorted(capacities)

    # Every distinct cut capacity is present, cheapest first.
    distinct_sorted = sorted({cut.key: cut.capacity for cut in collected}.values())
    assert capacities == distinct_sorted
    assert all(cut.found_in_trees >= 1 for cut in collected)
    assert sum(cut.found_in_trees for cut in collected) == len(raw)


def test_mixture_mass_is_bounded_by_one() -> None:
    graph = random_capacitated_graph(40, seed=6)
    result = run_mwu(graph, MwuParams(iters=15, candidates=2, seed=2))
    total = sum(record.count for record in result.records)
    weighted = []
    for record in result.records:
        tree = record.to_tree(graph)
        lca = LCA(tree)
        c_t = congestion.compute_tree_capacities(graph, tree, lca)
        weighted.append((tree, c_t, record.probability(total)))
    for cut in collect_cuts(weighted):
        assert 0.0 < cut.mixture_mass <= 1.0 + 1e-12
