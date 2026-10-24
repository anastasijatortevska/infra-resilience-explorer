from __future__ import annotations

import math

import pytest
from conftest import cycle_graph, random_capacitated_graph

from ires.core import io
from ires.core.mixture import evaluate_mixture
from ires.core.mwu import MwuParams, certified_lower_bound, run_mwu
from ires.oracles import ORACLES


@pytest.mark.parametrize("oracle", sorted(ORACLES))
def test_bounds_sandwich_the_optimum(oracle: str) -> None:
    """certified_lower <= alpha* <= alpha_hat, so the two must not cross."""

    graph = random_capacitated_graph(60, seed=1)
    result = run_mwu(graph, MwuParams(iters=25, candidates=3, seed=0, oracle=oracle))
    evaluation = evaluate_mixture(graph, result.records)
    assert result.certified_lower > 0.0
    assert result.certified_lower <= evaluation.alpha_hat + 1e-9


def test_certified_lower_bound_is_oracle_free_and_exact() -> None:
    """The bound is the minimum spanning tree weight under the edge distribution."""

    graph = cycle_graph(8)
    uniform = dict.fromkeys(io.capacities(graph), 1.0 / graph.number_of_edges())
    # On a cycle every spanning tree uses n-1 of the n edges.
    expected = (graph.number_of_nodes() - 1) / graph.number_of_edges()
    assert certified_lower_bound(graph, uniform) == pytest.approx(expected)


def test_certified_lower_bound_never_exceeds_achieved_alpha_on_cycle() -> None:
    graph = cycle_graph(10)
    result = run_mwu(graph, MwuParams(iters=30, candidates=2, seed=0))
    evaluation = evaluate_mixture(graph, result.records)
    assert result.certified_lower <= evaluation.alpha_hat


def test_wide_capacity_spread_does_not_overflow() -> None:
    """Log-space weights survive a spread that overflows math.exp on raw weights.

    The in-place update ``w *= exp(eta * (cong/alpha - 1))`` raises OverflowError once
    cong/alpha passes about 1180, which a heavy-tailed capacity distribution reaches.
    """

    graph = random_capacitated_graph(80, seed=7, spread=6.0)
    result = run_mwu(graph, MwuParams(iters=30, candidates=2, seed=0, alpha=1e-3))
    probabilities = result.final_probs
    assert math.isclose(sum(probabilities.values()), 1.0, rel_tol=1e-9)
    assert all(math.isfinite(value) and value >= 0.0 for value in probabilities.values())
    assert all(math.isfinite(entry.max_congestion) for entry in result.history)


def test_eta_tracks_the_iteration_budget() -> None:
    """Regret-optimal step size shrinks as sqrt(log m / T); a fixed eta ignores T."""

    graph = random_capacitated_graph(60, seed=2)
    m = graph.number_of_edges()
    for iters in (20, 200):
        eta = MwuParams(iters=iters).resolved_eta(graph)
        assert eta == pytest.approx(min(1.0, math.sqrt(math.log(m) / iters)))
    assert MwuParams(iters=50).resolved_eta(graph) > MwuParams(iters=500).resolved_eta(graph)
    assert MwuParams(iters=50, eta=0.25).resolved_eta(graph) == 0.25


def test_history_is_complete_and_monotone_in_the_certificate() -> None:
    graph = random_capacitated_graph(50, seed=3)
    result = run_mwu(graph, MwuParams(iters=20, candidates=2, seed=0))
    assert [entry.iteration for entry in result.history] == list(range(20))
    lowers = [entry.certified_lower for entry in result.history]
    assert lowers == sorted(lowers), "the best certificate seen can only improve"


def test_mixture_counts_sum_to_iterations() -> None:
    graph = random_capacitated_graph(40, seed=4)
    result = run_mwu(graph, MwuParams(iters=18, candidates=2, seed=0))
    assert sum(record.count for record in result.records) == 18


def test_same_seed_is_reproducible() -> None:
    graph = random_capacitated_graph(40, seed=5)
    params = MwuParams(iters=12, candidates=3, seed=11)
    first, second = run_mwu(graph, params), run_mwu(graph, params)
    assert [r.edges for r in first.records] == [r.edges for r in second.records]
    assert first.certified_lower == second.certified_lower


@pytest.mark.parametrize(
    "params",
    [
        MwuParams(iters=0),
        MwuParams(candidates=0),
        MwuParams(iters=1, eta=0.0),
        MwuParams(iters=1, alpha=-1.0),
    ],
)
def test_rejects_degenerate_parameters(params: MwuParams) -> None:
    graph = random_capacitated_graph(20, seed=0)
    with pytest.raises(ValueError, match="must be positive"):
        run_mwu(graph, params)


def test_rejects_unknown_oracle() -> None:
    graph = random_capacitated_graph(20, seed=0)
    with pytest.raises(ValueError, match="unknown oracle"):
        run_mwu(graph, MwuParams(iters=1, oracle="nope"))
