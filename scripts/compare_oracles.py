#!/usr/bin/env python3
"""Compare tree oracles on one graph: alpha bounds, average stretch and runtime.

    python scripts/compare_oracles.py data/tiny.edgelist --iters 40

Average stretch is measured under the fixed resistance metric ``1/capacity``: for each
graph edge ``(u, v)``, the ratio of its tree path length to its own length, averaged
over edges. It is the quantity low-stretch spanning tree constructions aim to minimize.
A fixed metric is used rather than the final MWU lengths so the number compares oracles
rather than the states their runs happened to end in.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import networkx as nx

from ires.core import io
from ires.core.lca import LCA
from ires.core.mixture import TreeRecord, evaluate_mixture, records_to_dicts
from ires.core.mwu import MwuParams, run_mwu
from ires.core.tree import Tree
from ires.oracles import ORACLES


def average_stretch(graph: nx.Graph, record: TreeRecord, lengths: dict[io.Edge, float]) -> float:
    """Mean ratio of tree path length to direct edge length, under ``lengths``."""

    tree = Tree.from_edges(record.edges, record.root, graph.nodes)
    lca = LCA(tree)
    lca.set_edge_weights({(p, c): lengths[io.edge_key(p, c)] for p, c in tree.edges()})
    ratios = [lca.dist(u, v) / lengths[io.edge_key(u, v)] for u, v, _ in io.iter_edges(graph)]
    return sum(ratios) / len(ratios)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph", type=Path)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    graph = io.load_graph(args.graph)
    lengths = {edge: 1.0 / capacity for edge, capacity in io.capacities(graph).items()}
    print(f"{args.graph}: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    header = (
        f"{'oracle':8s} {'floor':>9s} {'achieved':>10s} {'gap':>7s} "
        f"{'trees':>6s} {'stretch':>8s} {'sec':>6s}"
    )
    print(header)
    print("-" * len(header))

    for name in sorted(ORACLES):
        params = MwuParams(
            iters=args.iters, candidates=args.candidates, seed=args.seed, oracle=name
        )
        started = time.perf_counter()
        result = run_mwu(graph, params)
        elapsed = time.perf_counter() - started
        records = records_to_dicts(result.records, args.iters)
        evaluation = evaluate_mixture(graph, records, args.iters)

        stretch = min(average_stretch(graph, record, lengths) for record in result.records)
        achieved = evaluation.max_expected_score
        gap = achieved / result.dual_estimate if result.dual_estimate else 0.0
        print(
            f"{name:8s} {result.certified_lower:9.4f} {achieved:10.4f} "
            f"{gap:6.1f}x {len(result.records):6d} {stretch:8.2f} {elapsed:6.2f}"
        )


if __name__ == "__main__":
    main()
