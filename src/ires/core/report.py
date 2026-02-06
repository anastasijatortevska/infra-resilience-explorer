"""Report assembly helpers."""

from __future__ import annotations

import networkx as nx

from . import io


def critical_edges(
    graph: nx.Graph, expected_cong: dict[tuple[str, str], float], top_k: int = 10
) -> list[dict[str, object]]:
    """Return top-k edges by expected congestion."""

    records: list[dict[str, object]] = []
    for u, v, capacity in io.iter_edges(graph):
        key = io.edge_key(u, v)
        records.append(
            {
                "edge": (u, v),
                "capacity": capacity,
                "expected_congestion": expected_cong.get(key, 0.0),
            }
        )
    records.sort(key=lambda r: r["expected_congestion"], reverse=True)  # type: ignore[index]
    return records[:top_k]


def assemble_report(
    graph: nx.Graph,
    mixture: list[dict[str, object]],
    expected_cong: dict[tuple[str, str], float],
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
        "mixture": {
            "unique_trees": len(mixture),
            "total_samples": sum(int(t["count"]) for t in mixture),
            "trees": mixture,
        },
        "critical_edges": critical_edges(graph, expected_cong, top_k_edges),
        "bottleneck_cuts": cut_entries,
    }
