"""Report assembly helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import networkx as nx

from . import io
from .cuts import Cut, as_records, rank_cuts
from .io import Edge
from .mixture import METRIC, MixtureEvaluation

MAX_REPORTED_NODES = 30


@dataclass(frozen=True)
class Bounds:
    """Where the optimum ``F* = min_lambda max_e E[s_T(e)]`` is known to lie.

    Derivations are in ``docs/mathematics.md``. ``achieved`` and ``certified_lower``
    are certificates; ``dual_estimate`` is not.
    """

    achieved: float
    certified_lower: float
    dual_estimate: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "achieved": self.achieved,
            "achieved_note": (
                "rigorous upper bound on the optimum: the emitted mixture attains it"
            ),
            "certified_lower": self.certified_lower,
            "certified_lower_note": (
                "rigorous lower bound, but at most 1 by construction, since it keeps only "
                "tree edges where the score is smallest; a sanity floor, not a tight "
                "certificate"
            ),
        }
        if self.dual_estimate is not None:
            payload["dual_estimate"] = self.dual_estimate
            payload["dual_certified"] = False
            payload["estimated_gap"] = (
                self.achieved / self.dual_estimate if self.dual_estimate > 0 else None
            )
            payload["dual_note"] = (
                "LP dual value at the learned edge distribution, evaluated against the "
                "trees the oracle produced. It bounds the optimum below only for an exact "
                "minimizing oracle; divide by the oracle's approximation factor to "
                "certify it. estimated_gap = achieved / dual_estimate."
            )
        return payload


def scored_edges(
    graph: nx.Graph, expected_scores: dict[Edge, float], top_k: int = 10
) -> list[dict[str, Any]]:
    """Return top-k edges by expected cut-path score (a heuristic)."""

    io.positive_count(top_k, "top_k")
    records: list[dict[str, Any]] = []
    for u, v, capacity in io.iter_edges(graph):
        key = io.edge_key(u, v)
        records.append(
            {
                "edge": list(key),
                "capacity": capacity,
                "expected_cut_path_score": expected_scores[key],
            }
        )
    records.sort(key=lambda record: (-record["expected_cut_path_score"], record["edge"]))
    return records[:top_k]


def assemble_report(
    graph: nx.Graph,
    mixture: Sequence[dict[str, Any]],
    evaluation: MixtureEvaluation,
    params: dict[str, Any],
    bounds: Bounds | None = None,
    history: Sequence[dict[str, Any]] = (),
    top_k_edges: int = 10,
    top_k_cuts: int = 10,
) -> dict[str, Any]:
    """Build a summary report ready to serialize."""

    ranked: list[Cut] = rank_cuts(evaluation.cuts, top_k_cuts)
    report: dict[str, Any] = {
        "graph": {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
        },
        "parameters": params,
        "metric": METRIC,
        "max_expected_cut_path_score": evaluation.max_expected_score,
        "mixture": {
            "unique_trees": len(mixture),
            "total_samples": evaluation.total_samples,
            "trees": list(mixture),
        },
        "scored_edges": scored_edges(graph, evaluation.expected_scores, top_k_edges),
        "candidate_cuts": as_records(ranked, MAX_REPORTED_NODES),
        "distinct_candidate_cuts": len(evaluation.cuts),
    }
    if bounds is not None:
        report["bounds"] = bounds.to_dict()
    if history:
        report["history"] = list(history)
    return report
