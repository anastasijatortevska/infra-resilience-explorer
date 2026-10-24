from __future__ import annotations

import pytest
from conftest import random_capacitated_graph, unit_lengths

from ires.core import cuts, io, metrics, mixture
from ires.core.lca import LCA
from ires.core.mwu import MwuParams, run_mwu
from ires.oracles.spt import shortest_path_tree


def test_fingerprint_is_root_independent() -> None:
    graph = random_capacitated_graph(40, seed=2)
    lengths = unit_lengths(graph)
    nodes = sorted(graph.nodes)
    first = cuts.subtree_fingerprints(shortest_path_tree(graph, lengths, nodes[0]))
    second = cuts.subtree_fingerprints(shortest_path_tree(graph, lengths, nodes[1]))
    assert first[nodes[0]] == second[nodes[1]], "whole-graph fingerprint must not depend on root"


def test_tree_cut_capacities_and_sides_match_brute_force() -> None:
    graph = random_capacitated_graph(50, seed=3)
    tree = shortest_path_tree(graph, unit_lengths(graph), "n0000")
    c_t = metrics.compute_tree_capacities(graph, tree, LCA(tree))

    extracted = list(cuts.tree_cuts(tree, c_t))
    assert len(extracted) == graph.number_of_nodes() - 1
    for cut in extracted:
        nodes = set(cut.nodes())
        assert cut.capacity == pytest.approx(cuts.cut_capacity(graph, nodes))
        assert 2 * len(nodes) <= graph.number_of_nodes()


def _weighted(graph, records, total):
    weighted = []
    for record in records:
        tree = mixture.record_to_tree(record, graph)
        c_t = metrics.compute_tree_capacities(graph, tree, LCA(tree))
        weighted.append((tree, c_t, record["count"] / total))
    return weighted


def test_collect_cuts_dedupes_and_ranks_globally() -> None:
    graph = random_capacitated_graph(60, seed=4)
    result = run_mwu(graph, MwuParams(iters=20, candidates=3, seed=1))
    records = mixture.records_to_dicts(result.records, 20)
    assert len(records) > 1, "need several trees for this to be meaningful"

    weighted = _weighted(graph, records, 20)
    raw = [cut for tree, c_t, _ in weighted for cut in cuts.tree_cuts(tree, c_t)]
    collected = cuts.collect_cuts(weighted)

    assert len(collected) < len(raw), "duplicates were not merged"
    capacities = [cut.capacity for cut in collected]
    assert capacities == sorted(capacities)
    partitions = [tuple(cut.nodes()) for cut in collected]
    assert len(partitions) == len(set(partitions))
    assert sum(cut.found_in_trees for cut in collected) == len(raw)


def test_mixture_mass_is_a_probability() -> None:
    graph = random_capacitated_graph(40, seed=6)
    result = run_mwu(graph, MwuParams(iters=15, candidates=2, seed=2))
    records = mixture.records_to_dicts(result.records, 15)
    for cut in cuts.collect_cuts(_weighted(graph, records, 15)):
        assert 0.0 < cut.mixture_mass <= 1.0 + 1e-12


def test_ranking_does_not_mutate_its_input() -> None:
    """Ranking the same candidates twice must not double-count their mass."""

    graph = random_capacitated_graph(30, seed=8)
    tree = shortest_path_tree(graph, unit_lengths(graph), "n0000")
    c_t = metrics.compute_tree_capacities(graph, tree, LCA(tree))
    candidates = list(cuts.tree_cuts(tree, c_t))
    for cut in candidates:
        cut.mixture_mass, cut.found_in_trees = 1.0, 1
    assert cuts.rank_cuts(candidates) == cuts.rank_cuts(candidates)


def test_reported_cuts_are_the_globally_cheapest() -> None:
    graph = random_capacitated_graph(60, seed=10)
    result = run_mwu(graph, MwuParams(iters=20, candidates=3, seed=0))
    records = mixture.records_to_dicts(result.records, 20)
    collected = cuts.collect_cuts(_weighted(graph, records, 20))
    top = cuts.rank_cuts(collected, top_k=5)
    assert [cut.capacity for cut in top] == sorted(c.capacity for c in collected)[:5]


def test_top_k_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        cuts.rank_cuts([], top_k=0)


def test_serialized_records_truncate_long_node_lists() -> None:
    graph = random_capacitated_graph(80, seed=11)
    tree = shortest_path_tree(graph, unit_lengths(graph), "n0000")
    c_t = metrics.compute_tree_capacities(graph, tree, LCA(tree))
    extracted = list(cuts.tree_cuts(tree, c_t))
    records = cuts.as_records(extracted, max_nodes=5)
    for cut, record in zip(extracted, records, strict=True):
        nodes = cut.nodes()
        assert record["nodes"] == nodes[:5]
        assert record["side_size"] == len(nodes)
        assert record["truncated"] == (len(nodes) > 5)
        assert tuple(cut.edge) == io.edge_key(*cut.edge)
