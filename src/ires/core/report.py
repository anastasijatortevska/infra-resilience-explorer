"""Report assembly."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import networkx as nx

from . import io
from .cuts import Cut
from .mixture import MixtureEvaluation, TreeRecord

MAX_REPORTED_NODES = 30


@dataclass(frozen=True)
class Bounds:
    """Interval the optimal alpha is known to lie in."""

    alpha_hat: float
    certified_lower: float
    dual_objective: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "alpha_hat": self.alpha_hat,
            "alpha_hat_note": (
                "rigorous upper bound on the optimum: the emitted mixture achieves it"
            ),
            "certified_lower": self.certified_lower,
            "certified_lower_note": (
                "rigorous lower bound, but bounded above by 1 by construction (it keeps "
                "only tree edges, where congestion is lowest); a sanity floor rather "
                "than a tight certificate"
            ),
        }
        if self.dual_objective is not None:
            payload["dual_objective"] = self.dual_objective
            payload["dual_certified"] = False
            payload["estimated_gap"] = (
                self.alpha_hat / self.dual_objective if self.dual_objective > 0 else None
            )
            payload["dual_note"] = (
                "LP dual value at the learned edge distribution, evaluated against the "
                "trees the oracle produced. Certified only for an exact minimizing "
                "oracle; divide by the oracle's approximation factor to certify it. "
                "estimated_gap = alpha_hat / dual_objective."
            )
        return payload


def critical_edges(
    graph: nx.Graph, expected_cong: dict[io.Edge, float], top_k: int = 10
) -> list[dict[str, Any]]:
    """Return top-k edges by expected congestion."""

    records: list[dict[str, Any]] = []
    for u, v, capacity in io.iter_edges(graph):
        records.append(
            {
                "edge": [u, v],
                "capacity": capacity,
                "expected_congestion": expected_cong.get(io.edge_key(u, v), 0.0),
            }
        )
    records.sort(key=lambda record: -record["expected_congestion"])
    return records[:top_k]


def cut_entries(cuts: Sequence[Cut], top_k: int = 10) -> list[dict[str, Any]]:
    """Serialize the cheapest ``top_k`` cuts.

    ``cuts`` is expected already deduplicated and globally sorted by
    :func:`ires.core.cuts.collect_cuts`; node sets are materialized here, for the
    reported cuts only.
    """

    entries: list[dict[str, Any]] = []
    for cut in cuts[:top_k]:
        nodes = cut.nodes()
        entries.append(
            {
                "edge": list(cut.edge),
                "capacity": cut.capacity,
                "size": cut.size,
                "mixture_mass": cut.mixture_mass,
                "found_in_trees": cut.found_in_trees,
                "nodes": nodes[:MAX_REPORTED_NODES],
                "truncated": len(nodes) > MAX_REPORTED_NODES,
            }
        )
    return entries


def assemble_report(
    graph: nx.Graph,
    records: Sequence[TreeRecord],
    evaluation: MixtureEvaluation,
    params: dict[str, Any],
    bounds: Bounds,
    history: Sequence[dict[str, Any]] = (),
    top_k_edges: int = 10,
    top_k_cuts: int = 10,
) -> dict[str, Any]:
    """Build a summary report ready to serialize."""

    report: dict[str, Any] = {
        "graph": {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
        },
        "parameters": params,
        "bounds": bounds.to_dict(),
        "mixture": {
            "unique_trees": len(records),
            "total_samples": evaluation.total_samples,
            "trees": [record.to_dict(evaluation.total_samples) for record in records],
        },
        "critical_edges": critical_edges(graph, evaluation.expected_congestion, top_k_edges),
        "bottleneck_cuts": cut_entries(evaluation.cuts, top_k_cuts),
        "distinct_cuts": len(evaluation.cuts),
    }
    if history:
        report["history"] = list(history)
    return report
