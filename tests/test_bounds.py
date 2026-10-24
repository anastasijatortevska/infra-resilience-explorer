from __future__ import annotations

import math

import pytest
from conftest import cycle_graph, random_capacitated_graph

from ires.core import io, mixture
from ires.core.mwu import MwuParams, certified_lower_bound, run_mwu, suggested_eta
from ires.oracles import ORACLES


def evaluate(graph, result, iters):
    records = mixture.records_to_dicts(result.records, iters)
    return mixture.evaluate_mixture(graph, records, iters)


@pytest.mark.parametrize("oracle", sorted(ORACLES))
def test_bounds_do_not_cross(oracle: str) -> None:
    """certified_lower <= F* <= achieved, so the two ends must not cross."""

    graph = random_capacitated_graph(60, seed=1)
    result = run_mwu(graph, MwuParams(iters=25, candidates=3, seed=0, oracle=oracle))
    evaluation = evaluate(graph, result, 25)
    assert result.certified_lower > 0.0
    assert result.certified_lower <= evaluation.max_expected_score + 1e-9


def test_certified_bound_is_the_mst_weight_and_is_oracle_free() -> None:
    """On a cycle every spanning tree uses n-1 of the n edges."""

    graph = cycle_graph(8)
    uniform = dict.fromkeys(io.capacities(graph), 1.0 / graph.number_of_edges())
    expected = (graph.number_of_nodes() - 1) / graph.number_of_edges()
    assert certified_lower_bound(graph, uniform) == pytest.approx(expected)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_certified_bound_cannot_exceed_one(seed: int) -> None:
    """It keeps only tree edges, so it is bounded by sum_e q_e = 1 by construction."""

    graph = random_capacitated_graph(50, seed=seed)
    result = run_mwu(graph, MwuParams(iters=20, candidates=2, seed=seed))
    assert 0.0 < result.certified_lower <= 1.0


def test_dual_estimate_does_not_exceed_what_the_mixture_achieves() -> None:
    graph = random_capacitated_graph(50, seed=4)
    result = run_mwu(graph, MwuParams(iters=30, candidates=3, seed=0))
    evaluation = evaluate(graph, result, 30)
    assert 0.0 < result.dual_estimate <= evaluation.max_expected_score + 1e-9


def test_history_is_complete_and_the_certificate_only_improves() -> None:
    graph = random_capacitated_graph(50, seed=3)
    result = run_mwu(graph, MwuParams(iters=20, candidates=2, seed=0))
    assert [entry.iteration for entry in result.history] == list(range(20))
    lowers = [entry.certified_lower for entry in result.history]
    assert lowers == sorted(lowers)
    assert all(math.isfinite(entry.objective) for entry in result.history)


def test_running_max_matches_the_final_mixture() -> None:
    """The last running average is the value the emitted mixture achieves."""

    graph = random_capacitated_graph(40, seed=7)
    result = run_mwu(graph, MwuParams(iters=24, candidates=2, seed=1))
    evaluation = evaluate(graph, result, 24)
    assert result.history[-1].running_max_expected == pytest.approx(
        evaluation.max_expected_score, rel=1e-9
    )


def test_suggested_eta_tracks_the_iteration_budget() -> None:
    """Offered for --eta, not imposed: the default stays the historical 0.6."""

    graph = random_capacitated_graph(60, seed=2)
    m = graph.number_of_edges()
    for iters in (20, 200):
        assert suggested_eta(graph, iters) == pytest.approx(
            min(1.0, math.sqrt(math.log(m) / iters))
        )
    assert suggested_eta(graph, 50) > suggested_eta(graph, 500)
    assert MwuParams(iters=50).eta == 0.6


def test_mixture_counts_sum_to_iterations() -> None:
    graph = random_capacitated_graph(40, seed=4)
    result = run_mwu(graph, MwuParams(iters=18, candidates=2, seed=0))
    assert sum(record.count for record in result.records) == 18


def test_same_seed_is_reproducible() -> None:
    graph = random_capacitated_graph(40, seed=5)
    params = MwuParams(iters=12, candidates=3, seed=11)
    assert run_mwu(graph, params) == run_mwu(graph, params)


def test_rejects_unknown_oracle() -> None:
    graph = random_capacitated_graph(20, seed=0)
    with pytest.raises(ValueError, match="unknown oracle"):
        run_mwu(graph, MwuParams(iters=1, oracle="nope"))
