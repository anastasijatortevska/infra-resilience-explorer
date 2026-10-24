import itertools
import math

import networkx as nx
import pytest

from ires.core import cuts, mixture
from scripts.evaluate import run_evaluation, synthetic_graphs


def test_exact_min_cut_reference_and_reported_candidate_capacities():
    graphs = synthetic_graphs()
    for name, graph in graphs.items():
        nodes = sorted(graph)
        # Enumerate each partition once by including the first vertex.
        capacities = []
        for size in range(len(nodes) - 1):
            for side in itertools.combinations(nodes[1:], size):
                capacities.append(cuts.cut_capacity(graph, {nodes[0], *side}))
        smallest = {"path": 1, "cycle": 2, "clusters": 0.5}[name]
        assert min(capacities) == smallest
        assert nx.stoer_wagner(graph, weight="capacity")[0] == smallest
    result = run_evaluation(iters=3, candidates=2, seeds=[0])
    assert len(result["runs"]) == 9
    for row in result["runs"]:
        graph = graphs[row["family"]]
        records = row["trees"]
        total = sum(rec["count"] for rec in records)
        expected, candidate_cuts = mixture.summarize(graph, records, total)
        assert row["max_expected_cut_path_score"] == max(expected.values())
        assert row["best_candidate_cut_capacity"] == candidate_cuts[0].capacity
        assert row["candidate_cut_gap"] >= 0
        assert math.fsum(rec["prob"] for rec in records) == pytest.approx(1)
        if row["family"] == "path":
            assert row["unique_trees"] == row["max_expected_cut_path_score"] == 1
    budgets = {row["method"]: row["budget"] for row in result["runs"] if row["family"] == "cycle"}
    assert (
        budgets["iterative"]["shortest_path_trees"]
        == budgets["uniform_spt"]["shortest_path_trees"]
        == 6
    )
    assert budgets["fixed_spt"]["shortest_path_trees"] == 1
    assert budgets["iterative"]["selected_tree_samples"] == 3
    assert budgets["uniform_spt"]["selected_tree_samples"] == 6


def test_evaluation_repeats_all_substantive_results():
    first = run_evaluation(iters=3, candidates=2, seeds=[0, 1])
    second = run_evaluation(iters=3, candidates=2, seeds=[0, 1])
    for result in [first, second]:
        for row in result["runs"]:
            row.pop("runtime_seconds")
    assert first == second
