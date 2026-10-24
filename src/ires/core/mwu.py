"""Multiplicative-weights loop over tree oracles.

Finds a distribution over spanning trees minimizing ``alpha = max_e E[cong_T(e)]``.
Taking the LP dual gives, for every probability vector ``p`` over edges,

    alpha* >= min_T sum_e p_e cong_T(e)

so the weights maintained here are a search for a good ``p``. A run reports three
numbers.

``alpha_hat``
    Max expected congestion of the emitted mixture. Rigorous upper bound on ``alpha*``,
    since an explicit mixture achieves it.

``certified_lower``
    Rigorous lower bound. Every tree edge has ``cong_T(e) = c_T(e)/cap(e) >= 1``, because
    an edge crosses its own cut, so for any ``p``

        sum_e p_e cong_T(e) >= sum_{e in T} p_e >= minimum spanning tree weight under p,

    which is exactly computable and needs no oracle. Keeping only the tree edges discards
    the non-tree edges that carry the load, so the value can never exceed ``sum_e p_e = 1``.
    It is a sanity floor rather than a tight certificate; a tight bound needs to bound
    ``cong_T(e)`` below on non-tree edges, which is the minimum-weighted-stretch problem.

``dual_objective``
    The dual value at the final ``p``, evaluated against the trees the oracle produced.
    ``alpha_hat / dual_objective`` is the useful estimate of remaining slack. It bounds
    ``alpha*`` below only for an exact minimizing oracle, so it is reported flagged
    uncertified; an oracle with a proved approximation factor ``c`` certifies it after
    dividing by ``c``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np

from ..oracles import DEFAULT_ORACLE, get_oracle
from . import congestion, io
from .io import Edge
from .lca import LCA
from .mixture import TreeRecord, signature
from .tree import Tree


@dataclass(frozen=True)
class MwuParams:
    """Parameters of one MWU run."""

    iters: int = 80
    candidates: int = 8
    seed: int = 0
    oracle: str = DEFAULT_ORACLE
    eta: float | None = None
    alpha: float | None = None

    def resolved_alpha(self, graph: nx.Graph) -> float:
        """Width parameter of the update; defaults to the target congestion scale."""

        if self.alpha is not None:
            if self.alpha <= 0:
                raise ValueError("alpha must be positive")
            return self.alpha
        return 10.0 * math.log2(graph.number_of_nodes() + 1)

    def resolved_eta(self, graph: nx.Graph) -> float:
        """Step size; defaults to the regret-optimal ``sqrt(log m / T)``.

        Standard multiplicative-weights regret over ``m`` edges and ``T`` rounds is
        minimized at ``eta ~ sqrt(log m / T)``, so the step size has to shrink as the
        iteration budget grows. Hard-coding one value decouples them and leaves the
        run either under-mixed or oscillating.
        """

        if self.eta is not None:
            if self.eta <= 0:
                raise ValueError("eta must be positive")
            return self.eta
        m = max(graph.number_of_edges(), 2)
        return min(1.0, math.sqrt(math.log(m) / max(self.iters, 1)))


@dataclass(frozen=True)
class IterationLog:
    """Per-iteration convergence trace."""

    iteration: int
    objective: float
    max_congestion: float
    alpha_running: float
    certified_lower: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "objective": self.objective,
            "max_congestion": self.max_congestion,
            "alpha_running": self.alpha_running,
            "certified_lower": self.certified_lower,
        }


@dataclass
class MwuResult:
    """Output of a run: the mixture, the convergence trace and the bounds."""

    records: list[TreeRecord]
    history: list[IterationLog]
    certified_lower: float
    dual_objective: float
    eta: float
    alpha: float
    final_probs: dict[Edge, float] = field(repr=False, default_factory=dict)


def _normalize(log_weights: Mapping[Edge, float]) -> dict[Edge, float]:
    """Softmax of log-weights.

    The weights are kept in log space and shifted by their maximum before
    exponentiating. Updating ``w *= exp(eta * (cong/alpha - 1))`` directly overflows
    ``math.exp`` once ``cong/alpha`` passes ~1180, which a wide capacity spread reaches.
    """

    largest = max(log_weights.values())
    scaled = {edge: math.exp(value - largest) for edge, value in log_weights.items()}
    total = sum(scaled.values())
    return {edge: value / total for edge, value in scaled.items()}


def certified_lower_bound(graph: nx.Graph, probs: Mapping[Edge, float]) -> float:
    """Minimum spanning tree weight under ``probs``: a rigorous lower bound on alpha*.

    Exact, and needs no oracle. See the module docstring for the derivation.
    """

    weighted = nx.Graph()
    weighted.add_nodes_from(graph.nodes)
    for u, v in graph.edges:
        weighted.add_edge(u, v, p=probs[io.edge_key(u, v)])
    mst = nx.minimum_spanning_tree(weighted, weight="p")
    return float(sum(data["p"] for _, _, data in mst.edges(data=True)))


def _score(
    graph: nx.Graph, tree: Tree, probs: Mapping[Edge, float]
) -> tuple[float, dict[Edge, float]]:
    """Return (average congestion under ``probs``, per-edge congestion) for one tree."""

    lca = LCA(tree)
    c_t = congestion.compute_tree_capacities(graph, tree, lca)
    cong = congestion.compute_edge_congestion(graph, tree, lca, c_t)
    objective = sum(probs[edge] * value for edge, value in cong.items())
    return objective, cong


def run_mwu(
    graph: nx.Graph,
    params: MwuParams,
    progress: Any = None,
) -> MwuResult:
    """Run the multiplicative-weights loop and return the mixture plus its bounds."""

    if params.iters <= 0:
        raise ValueError("iters must be positive")
    if params.candidates <= 0:
        raise ValueError("candidates must be positive")

    oracle = get_oracle(params.oracle)
    alpha = params.resolved_alpha(graph)
    eta = params.resolved_eta(graph)

    rng = np.random.default_rng(params.seed)
    capacities = io.capacities(graph)
    log_weights: dict[Edge, float] = dict.fromkeys(capacities, 0.0)
    running_total: dict[Edge, float] = dict.fromkeys(capacities, 0.0)

    counts: dict[tuple[Edge, ...], tuple[Tree, int]] = {}
    history: list[IterationLog] = []
    certified = 0.0
    probs: dict[Edge, float] = _normalize(log_weights)

    iterations: Any = range(params.iters)
    if progress is not None:
        iterations = progress(iterations)

    for step in iterations:
        probs = _normalize(log_weights)
        lengths = {edge: probs[edge] / capacities[edge] for edge in capacities}
        certified = max(certified, certified_lower_bound(graph, probs))

        best: tuple[float, Tree, dict[Edge, float]] | None = None
        for _ in range(params.candidates):
            tree = oracle(graph, lengths, rng)
            objective, cong = _score(graph, tree, probs)
            if best is None or objective < best[0]:
                best = (objective, tree, cong)
        assert best is not None
        objective, tree, cong = best

        for edge, value in cong.items():
            log_weights[edge] += eta * (value / alpha - 1.0)
            running_total[edge] += value

        key = signature(tree)
        previous = counts.get(key)
        counts[key] = (tree, 1) if previous is None else (previous[0], previous[1] + 1)

        done = step + 1
        history.append(
            IterationLog(
                iteration=step,
                objective=objective,
                max_congestion=max(cong.values()),
                alpha_running=max(total / done for total in running_total.values()),
                certified_lower=certified,
            )
        )

    records = [TreeRecord.from_tree(tree, count) for tree, count in counts.values()]
    records.sort(key=lambda record: record.count, reverse=True)

    # Score the dual at the weights after the final update, not at the distribution the
    # last iteration happened to start from.
    probs = _normalize(log_weights)
    certified = max(certified, certified_lower_bound(graph, probs))
    dual = min(_score(graph, record.to_tree(graph), probs)[0] for record in records)

    return MwuResult(
        records=records,
        history=history,
        certified_lower=certified,
        dual_objective=dual,
        eta=eta,
        alpha=alpha,
        final_probs=probs,
    )
