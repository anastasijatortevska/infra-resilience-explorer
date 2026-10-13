"""Report assembly helpers."""

from __future__ import annotations

import networkx as nx

from . import io


def scored_edges(
    graph: nx.Graph, expected_scores: dict[tuple[str, str], float], top_k: int = 10
) -> list[dict[str, object]]:
    """Return top-k edges by expected cut-path score (a heuristic)."""

    records: list[dict[str, object]] = []
    for u, v, capacity in io.iter_edges(graph):
        key = io.edge_key(u, v)
        records.append(
            {
                "edge": (u, v),
                "capacity": capacity,
                "expected_cut_path_score": expected_scores[key],
            }
        )
    records.sort(key=lambda r: r["expected_cut_path_score"], reverse=True)  # type: ignore[index]
    return records[:top_k]


def assemble_report(
    graph: nx.Graph,
    mixture: list[dict[str, object]],
    expected_scores: dict[tuple[str, str], float],
    cuts: list[dict[str, object]],
    params: dict[str, object],
    top_k_edges: int = 10,
    top_k_cuts: int = 10,
) -> dict[str, object]:
    """Build a summary report ready to serialize."""

    cut_entries: list[dict[str, object]] = []
    for item in cuts[:top_k_cuts]:
        nodes = item["nodes"]  # type: ignore[index]
        cut_entries.append(
            {
                "edge": item["edge"],
                "capacity": item["capacity"],
                "nodes": nodes[:30],
                "truncated": len(nodes) > 30,
            }
        )

    return {
        "graph": {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
        },
        "parameters": params,
        "metric": "cut_path_score",
        "max_expected_cut_path_score": max(expected_scores.values()),
        "mixture": {
            "unique_trees": len(mixture),
            "total_samples": sum(int(t["count"]) for t in mixture),
            "trees": mixture,
        },
        "scored_edges": scored_edges(graph, expected_scores, top_k_edges),
        "candidate_cuts": cut_entries,
    }
