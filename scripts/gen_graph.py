#!/usr/bin/env python3
"""Generate edge lists bigger than the toy fixture, for scaling and oracle comparison.

python scripts/gen_graph.py grid --rows 40 --cols 40 --out data/grid40.edgelist
python scripts/gen_graph.py random --nodes 2000 --degree 4 --out data/random2k.edgelist
"""

from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
import numpy as np


def build(kind: str, args: argparse.Namespace) -> nx.Graph:
    if kind == "grid":
        graph = nx.grid_2d_graph(args.rows, args.cols)
        return nx.relabel_nodes(graph, lambda rc: f"r{rc[0]}c{rc[1]}")
    if kind == "random":
        graph = nx.gnm_random_graph(args.nodes, args.nodes * args.degree // 2, seed=args.seed)
    elif kind == "scalefree":
        graph = nx.barabasi_albert_graph(args.nodes, max(args.degree // 2, 1), seed=args.seed)
    else:  # pragma: no cover - argparse restricts the choices
        raise ValueError(f"unknown kind {kind!r}")
    largest = max(nx.connected_components(graph), key=len)
    graph = graph.subgraph(largest).copy()
    return nx.relabel_nodes(graph, lambda i: f"n{i}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["grid", "random", "scalefree"])
    parser.add_argument("--rows", type=int, default=20)
    parser.add_argument("--cols", type=int, default=20)
    parser.add_argument("--nodes", type=int, default=1000)
    parser.add_argument("--degree", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--capacity",
        choices=["unit", "lognormal"],
        default="unit",
        help="unit capacities, or a heavy-tailed spread that stresses the weight update",
    )
    parser.add_argument("--sigma", type=float, default=1.0, help="lognormal capacity spread")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    graph = build(args.kind, args)
    rng = np.random.default_rng(args.seed)
    lines = []
    for u, v in sorted(tuple(sorted((str(a), str(b)))) for a, b in graph.edges):
        weight = 1.0 if args.capacity == "unit" else float(np.exp(rng.normal(0.0, args.sigma)))
        lines.append(f"{u} {v} {weight:.6g}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")


if __name__ == "__main__":
    main()
