"""Compare the heuristic with two SPT baselines on small synthetic graphs."""

from __future__ import annotations

import argparse
import math
import platform
import time
from importlib.metadata import version
from pathlib import Path

import networkx as nx
import numpy as np

from ires.cli import _edge_distribution, _tree_to_record, run_mwu, write_json
from ires.core import io, mixture
from ires.oracles.spt import shortest_path_tree


def synthetic_graphs() -> dict[str, nx.Graph]:
    """Eight vertices per family; two clusters joined by two weak links."""

    path = nx.path_graph([str(i) for i in range(8)])
    cycle = nx.cycle_graph([str(i) for i in range(8)])
    nx.set_edge_attributes(path, 1.0, "capacity")
    nx.set_edge_attributes(cycle, 1.0, "capacity")
    clusters = nx.Graph()
    for prefix in ["a", "b"]:
        cluster = nx.complete_graph([f"{prefix}{i}" for i in range(4)])
        nx.set_edge_attributes(cluster, 3.0, "capacity")
        clusters.update(cluster)
    clusters.add_edge("a0", "b0", capacity=0.25)
    clusters.add_edge("a1", "b1", capacity=0.25)
    return {"path": path, "cycle": cycle, "clusters": clusters}


def static_mixture(graph: nx.Graph, samples: int, seed: int, fixed: bool = False):
    """Equal sample weights, initial lengths proportional to inverse capacity.

    The fixed baseline roots at the first sorted label. The sampled baseline
    draws roots uniformly with replacement; equal samples need not give equal
    probabilities to distinct unrooted trees.
    """

    capacities = {io.edge_key(u, v): c for u, v, c in io.iter_edges(graph)}
    _, lengths = _edge_distribution({e: 0.0 for e in capacities}, capacities)
    nodes = sorted(graph)
    rng = np.random.default_rng(seed)
    counts = {}
    for _ in range(samples):
        root = nodes[0] if fixed else rng.choice(nodes)
        tree = shortest_path_tree(graph, lengths, root)
        signature = mixture.tree_signature(tree)
        if signature not in counts:
            counts[signature] = [tree, 0]
        counts[signature][1] += 1
    records = [_tree_to_record(tree, count, samples) for tree, count in counts.values()]
    records.sort(
        key=lambda rec: (-rec["count"], tuple(sorted(io.edge_key(*e) for e in rec["edges"])))
    )
    expected, _ = mixture.summarize(graph, records, samples)
    return records, expected


def run_evaluation(iters: int = 80, candidates: int = 8, seeds=(0, 1, 2), alpha=None) -> dict:
    io.positive_count(iters, "iters")
    io.positive_count(candidates, "candidates")
    seeds = list(seeds)
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("seeds must be nonnegative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be distinct")
    if alpha is not None:
        io.positive_finite(alpha, "alpha")

    payload = {
        "schema_version": 1,
        "metric": mixture.METRIC,
        "configuration": {
            "iters": iters,
            "candidates": candidates,
            "seeds": seeds,
            "alpha_override": alpha,
            "eta": 0.6,
            "default_alpha": "10 * log2(n + 1)",
            "static_lengths": "proportional to 1 / capacity",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "packages": {
                name: version(name)
                for name in ["networkx", "numpy", "typer", "pydantic", "rich", "pytest", "ruff"]
            },
        },
        "graphs": {},
        "runs": [],
    }
    for family, graph in synthetic_graphs().items():
        io.validate_graph(graph)
        exact, _ = nx.stoer_wagner(graph, weight=io.CAPACITY_KEY)
        scale = alpha if alpha is not None else 10 * math.log2(len(graph) + 1)
        payload["graphs"][family] = {
            "nodes": sorted(graph),
            "edges": list(io.iter_edges(graph)),
            "fingerprint": io.graph_fingerprint(graph),
            "exact_global_min_cut_capacity": exact,
        }
        for seed in seeds:
            for method in ["fixed_spt", "uniform_spt", "iterative"]:
                budget = iters * candidates if method != "fixed_spt" else 1
                started = time.perf_counter()
                if method == "iterative":
                    records, _ = run_mwu(
                        graph, iters, candidates, eta=0.6, alpha=scale, seed=seed, progress=False
                    )
                    total = iters
                else:
                    records, _ = static_mixture(graph, budget, seed, fixed=method == "fixed_spt")
                    total = budget
                expected, candidate_cuts = mixture.summarize(graph, records, total)
                elapsed = time.perf_counter() - started
                best = candidate_cuts[0]["capacity"]
                payload["runs"].append(
                    {
                        "family": family,
                        "seed": seed,
                        "method": method,
                        "alpha": scale,
                        "max_expected_cut_path_score": max(expected.values()),
                        "unique_trees": len(records),
                        "runtime_seconds": elapsed,
                        "best_candidate_cut_capacity": best,
                        "exact_global_min_cut_capacity": exact,
                        "candidate_cut_gap": best - exact,
                        "budget": {
                            "shortest_path_trees": budget,
                            "selection_score_evaluations": budget if method == "iterative" else 0,
                            "selected_tree_samples": total,
                            "summary_tree_score_evaluations": 2 * len(records),
                        },
                        "trees": records,
                    }
                )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--out", type=Path, default=Path("evaluation/results.json"))
    args = parser.parse_args()
    try:
        payload = run_evaluation(args.iters, args.candidates, args.seeds, args.alpha)
        write_json(args.out, payload)
    except (ValueError, OSError, OverflowError) as exc:
        parser.error(str(exc))
    print("family    seed method        max score  trees  seconds  candidate / exact cut")
    for row in payload["runs"]:
        print(
            f"{row['family']:9} {row['seed']:4} {row['method']:13} "
            f"{row['max_expected_cut_path_score']:9.4f} {row['unique_trees']:6} "
            f"{row['runtime_seconds']:8.4f}  "
            f"{row['best_candidate_cut_capacity']:.4g} / {row['exact_global_min_cut_capacity']:.4g}"
        )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
