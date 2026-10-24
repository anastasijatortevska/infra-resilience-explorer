"""Multiplicative-weights loop over tree oracles.

Experimentally minimizes ``F(lambda) = max_e E_lambda[s_T(e)]``, the maximum expected
cut-path score over a distribution of spanning trees. See ``docs/mathematics.md`` for
the definition of ``s_T`` and for what this procedure does and does not guarantee: the
candidate oracles are heuristics, and ``alpha`` is an update scale, not a proved bound.

Three numbers bracket the result. Only the first two are certificates, and the
derivations are in ``docs/mathematics.md``.

``achieved``
    ``max_e`` expected score of the emitted mixture. Rigorous upper bound on the optimum
    ``F*``, since an explicit mixture attains it.

``certified_lower``
    Rigorous lower bound. Every tree edge has ``s_T(f) = C_T(f)/c(f) >= 1``, because an
    edge crosses its own fundamental cut, so for any edge distribution ``q``

        sum_e q_e s_T(e) >= sum_{f in T} q_f >= minimum spanning tree weight under q,

    which is exactly computable and needs no oracle. Keeping only tree edges discards
    the non-tree edges that carry the large scores, so this can never exceed
    ``sum_e q_e = 1``. It is a sanity floor, not a tight certificate.

``dual_estimate``
    The LP dual value ``min_T sum_e q_e s_T(e)`` at the final ``q``, evaluated against
    the trees the oracle produced. ``achieved / dual_estimate`` is the useful measure of
    remaining slack, but the quantity bounds ``F*`` below only for an exact minimizing
    oracle, so it is reported flagged uncertified.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np

from ..oracles import DEFAULT_ORACLE, get_oracle
from . import io, metrics
from .io import Edge
from .lca import LCA
from .mixture import TreeRecord, tree_signature
from .tree import Tree

DEFAULT_ETA = 0.6


@dataclass(frozen=True)
class MwuParams:
    """Parameters of one run. Only the ratio ``eta / alpha`` affects the update."""

    iters: int = 80
    candidates: int = 8
    seed: int = 0
    oracle: str = DEFAULT_ORACLE
    eta: float = DEFAULT_ETA
    alpha: float | None = None

    def resolved_alpha(self, graph: nx.Graph) -> float:
        """Update scale. The historical default has no proved bound; it is tuning."""

        if self.alpha is not None:
            return io.positive_finite(self.alpha, "alpha")
        return 10.0 * math.log2(graph.number_of_nodes() + 1)

    def validate(self, graph: nx.Graph) -> tuple[float, float]:
        io.positive_count(self.iters, "iters")
        io.positive_count(self.candidates, "candidates")
        eta = io.positive_finite(self.eta, "eta")
        alpha = self.resolved_alpha(graph)
        io.positive_finite(eta / alpha, "eta / alpha")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        return eta, alpha


@dataclass(frozen=True)
class IterationLog:
    """Per-iteration convergence trace."""

    iteration: int
    objective: float
    max_score: float
    running_max_expected: float
    certified_lower: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "objective": self.objective,
            "max_score": self.max_score,
            "running_max_expected": self.running_max_expected,
            "certified_lower": self.certified_lower,
        }


@dataclass
class MwuResult:
    """The mixture, its convergence trace and the bounds from the run."""

    records: list[TreeRecord]
    history: list[IterationLog]
    certified_lower: float
    dual_estimate: float
    eta: float
    alpha: float
    final_probs: dict[Edge, float] = field(repr=False, default_factory=dict)


def edge_distribution(
    log_weights: Mapping[Edge, float], capacities: Mapping[Edge, float]
) -> tuple[dict[Edge, float], dict[Edge, float]]:
    """Normalize weights and rescale lengths by a common positive factor.

    Subtracting the largest log value prevents a common scale from causing overflow.
    Rescaling lengths preserves all shortest paths. Unrepresentable relative
    weights/lengths are rejected rather than silently becoming zero.
    """

    if any(not math.isfinite(value) for value in log_weights.values()):
        raise ValueError("Nonfinite log weights; increase alpha or reduce eta/iterations")
    peak = max(log_weights.values())
    relative = {edge: value - peak for edge, value in log_weights.items()}
    weights = {edge: math.exp(value) for edge, value in relative.items()}
    total = math.fsum(weights.values())
    probs = {edge: weight / total for edge, weight in weights.items()}
    if any(value <= 0 for value in probs.values()):
        raise ValueError("Edge weights underflow; increase alpha or reduce eta/iterations")
    log_lengths = {
        edge: relative[edge] - math.log(capacity) for edge, capacity in capacities.items()
    }
    length_peak = max(log_lengths.values())
    lengths = {edge: math.exp(value - length_peak) for edge, value in log_lengths.items()}
    if any(not math.isfinite(value) or value <= 0 for value in lengths.values()):
        raise ValueError("Shortest-path lengths underflow; reduce the capacity/weight range")
    return probs, lengths


def update_log_weights(
    log_weights: Mapping[Edge, float], scores: Mapping[Edge, float], step: float
) -> dict[Edge, float]:
    """Positive MWU update, removing the common largest score before scaling."""

    peak_score = max(scores.values())
    updated = {
        edge: value + step * (scores[edge] - peak_score) for edge, value in log_weights.items()
    }
    if any(not math.isfinite(value) for value in updated.values()):
        raise ValueError("Nonfinite weight update; increase alpha or reduce eta")
    peak = max(updated.values())
    return {edge: value - peak for edge, value in updated.items()}


def certified_lower_bound(graph: nx.Graph, probs: Mapping[Edge, float]) -> float:
    """Minimum spanning tree weight under ``probs``: a rigorous lower bound on F*.

    Exact, and needs no oracle. See the module docstring for the derivation, including
    why the value cannot exceed 1.
    """

    weighted = nx.Graph()
    weighted.add_nodes_from(graph.nodes)
    for u, v in graph.edges:
        weighted.add_edge(u, v, q=probs[io.edge_key(u, v)])
    mst = nx.minimum_spanning_tree(weighted, weight="q")
    return math.fsum(data["q"] for _, _, data in mst.edges(data=True))


def score_tree(
    graph: nx.Graph, tree: Tree, probs: Mapping[Edge, float]
) -> tuple[float, dict[Edge, float]]:
    """Return (weighted score under ``probs``, per-edge scores) for one tree."""

    lca = LCA(tree)
    c_t = metrics.compute_tree_capacities(graph, tree, lca)
    scores = metrics.compute_edge_scores(graph, tree, lca, c_t)
    objective = math.fsum(probs[edge] * value for edge, value in scores.items())
    if not math.isfinite(objective):
        raise ValueError("Nonfinite weighted cut-path score")
    return objective, scores


def suggested_eta(graph: nx.Graph, iters: int) -> float:
    """``sqrt(log m / iters)``, the step size regret analysis would pick.

    Offered for ``--eta``, not imposed as the default: the regret bound it comes from
    does not transfer to this objective with a heuristic candidate oracle, and the
    committed evaluation baseline uses the historical 0.6.
    """

    return min(1.0, math.sqrt(math.log(max(graph.number_of_edges(), 2)) / max(iters, 1)))


def run_mwu(
    graph: nx.Graph,
    params: MwuParams,
    progress: Callable[[Iterable[int]], Iterable[int]] | None = None,
) -> MwuResult:
    """Run the loop and return the mixture with its bounds and convergence trace."""

    io.validate_graph(graph)
    eta, alpha = params.validate(graph)
    step = eta / alpha
    oracle = get_oracle(params.oracle)

    rng = np.random.default_rng(params.seed)
    capacities = io.capacities(graph)
    log_weights: dict[Edge, float] = dict.fromkeys(capacities, 0.0)
    running: dict[Edge, list[float]] = {edge: [] for edge in capacities}

    counts: dict[tuple[Edge, ...], tuple[Tree, int]] = {}
    history: list[IterationLog] = []
    certified = 0.0
    probs: dict[Edge, float] = {}

    iterations: Iterable[int] = range(params.iters)
    if progress is not None:
        iterations = progress(iterations)

    for iteration in iterations:
        probs, lengths = edge_distribution(log_weights, capacities)
        certified = max(certified, certified_lower_bound(graph, probs))

        best: tuple[float, Tree, dict[Edge, float]] | None = None
        for _ in range(params.candidates):
            tree = oracle(graph, lengths, rng)
            objective, scores = score_tree(graph, tree, probs)
            if best is None or objective < best[0]:
                best = (objective, tree, scores)
        assert best is not None
        objective, tree, scores = best

        if iteration + 1 < params.iters:
            log_weights = update_log_weights(log_weights, scores, step)

        signature = tree_signature(tree)
        previous = counts.get(signature)
        counts[signature] = (tree, 1) if previous is None else (previous[0], previous[1] + 1)

        done = iteration + 1
        for edge, value in scores.items():
            running[edge].append(value)
        history.append(
            IterationLog(
                iteration=iteration,
                objective=objective,
                max_score=max(scores.values()),
                running_max_expected=max(math.fsum(v) / done for v in running.values()),
                certified_lower=certified,
            )
        )

    records = [TreeRecord.from_tree(tree, count) for tree, count in counts.values()]
    records.sort(key=lambda record: (-record.count, tree_signature_of(record)))

    # Score the dual at the weights after the final update, not at the distribution the
    # last iteration started from.
    probs, _ = edge_distribution(log_weights, capacities)
    certified = max(certified, certified_lower_bound(graph, probs))
    dual = min(
        score_tree(graph, Tree.from_edges(record.edges, record.root, graph.nodes), probs)[0]
        for record in records
    )

    return MwuResult(
        records=records,
        history=history,
        certified_lower=certified,
        dual_estimate=dual,
        eta=eta,
        alpha=alpha,
        final_probs=probs,
    )


def tree_signature_of(record: TreeRecord) -> tuple[Edge, ...]:
    return tuple(sorted(io.edge_key(u, v) for u, v in record.edges))
