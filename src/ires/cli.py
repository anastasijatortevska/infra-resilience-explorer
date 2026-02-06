"""Typer CLI for Infra Resilience Explorer."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Annotated

import networkx as nx
import numpy as np
import typer
from rich.console import Console
from rich.progress import track

from .core import congestion, cuts, io, report
from .core.lca import LCA
from .core.tree import Tree
from .oracles.spt import shortest_path_tree

console = Console()
app = typer.Typer(help="Infra Resilience Explorer")


def _tree_signature(tree: Tree) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(io.edge_key(u, v) for u, v in tree.edges()))


def _tree_to_record(tree: Tree, count: int, total: int) -> dict[str, object]:
    return {
        "root": tree.root,
        "edges": [(u, v) for u, v in tree.edges()],
        "count": count,
        "prob": count / total if total else 0.0,
    }


def _record_to_tree(record: dict[str, object], nodes: list[str]) -> Tree:
    root = record["root"]  # type: ignore[index]
    edges = record["edges"]  # type: ignore[index]
    parent: dict[str, str | None] = {root: None}
    for u, v in edges:  # type: ignore[misc]
        parent[v] = u
    for node in nodes:
        parent.setdefault(node, None if node == root else root)
    return Tree.from_parent_map(root, parent)


def run_mwu(
    graph: nx.Graph,
    iters: int,
    r: int,
    eta: float,
    alpha: float,
    seed: int,
) -> tuple[list[dict[str, object]], dict[tuple[str, str], float]]:
    """Run the multiplicative-weights update loop and return (mixture, expected_cong)."""

    rng = np.random.default_rng(seed)
    nodes = list(graph.nodes)
    capacities = {io.edge_key(u, v): w for u, v, w in io.iter_edges(graph)}
    weights = {edge: 1.0 for edge in capacities}
    expected_cong = {edge: 0.0 for edge in capacities}

    mixture_counts: dict[tuple[tuple[str, str], ...], dict[str, object]] = {}

    for _ in track(range(iters), description="Running MWU"):
        total_w = sum(weights.values())
        probs = {e: weights[e] / total_w for e in weights}
        lengths = {e: probs[e] / capacities[e] for e in capacities}

        candidates = []
        for _ in range(r):
            root = rng.choice(nodes)
            tree = shortest_path_tree(graph, lengths, root)
            lca = LCA(tree)
            c_t = congestion.compute_tree_capacities(graph, tree, lca)
            cong = congestion.compute_edge_congestion(graph, tree, lca, c_t)
            obj = sum(probs[e] * cong[e] for e in cong)
            candidates.append((obj, tree, cong))

        obj, tree, cong = min(candidates, key=lambda x: x[0])

        for edge, val in cong.items():
            weights[edge] *= math.exp(eta * (val / alpha - 1))
            expected_cong[edge] += val / iters

        sig = _tree_signature(tree)
        if sig not in mixture_counts:
            mixture_counts[sig] = {"tree": tree, "count": 0}
        mixture_counts[sig]["count"] = mixture_counts[sig]["count"] + 1  # type: ignore[index]

    mixture = [
        _tree_to_record(info["tree"], info["count"], iters)  # type: ignore[index]
        for info in mixture_counts.values()
    ]
    mixture.sort(key=lambda rec: rec["count"], reverse=True)  # type: ignore[index]
    return mixture, expected_cong


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


@app.command()
def fit(
    graph: Annotated[
        Path,
        typer.Option(..., exists=True, help="Path to undirected weighted edge list"),
    ],
    iters: Annotated[int, typer.Option(help="Number of MWU iterations")] = 80,
    candidates: Annotated[int, typer.Option(help="Candidate trees per iteration")] = 8,
    seed: Annotated[int, typer.Option(help="Random seed")] = 0,
    out: Annotated[Path, typer.Option(help="Output directory")] = Path("outputs"),
) -> None:
    """Fit a tree mixture and emit mixture.json and report.json."""

    g = io.load_graph(graph)
    alpha = 10 * math.log2(g.number_of_nodes() + 1)
    console.print(
        f"[bold]Graph[/]: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges. "
        f"alpha={alpha:.2f}"
    )

    mixture, expected_cong = run_mwu(g, iters=iters, r=candidates, eta=0.6, alpha=alpha, seed=seed)

    mixture_payload = {
        "graph": graph.as_posix(),
        "iters": iters,
        "candidates": candidates,
        "seed": seed,
        "alpha": alpha,
        "trees": mixture,
    }
    write_json(out / "mixture.json", mixture_payload)

    all_cuts: list[dict[str, object]] = []
    for rec in mixture:
        tree = _record_to_tree(rec, list(g.nodes))
        all_cuts.extend(cuts.extract_tree_cuts(g, tree))

    rpt = report.assemble_report(
        g,
        mixture,
        expected_cong,
        cuts=all_cuts,
        params={"iters": iters, "candidates": candidates, "seed": seed, "alpha": alpha},
    )
    write_json(out / "report.json", rpt)
    console.print(f"[green]Wrote[/] {out/'mixture.json'} and {out/'report.json'}")


@app.command("report")
def report_cmd(
    graph: Annotated[Path, typer.Option(..., exists=True, help="Path to graph edge list")],
    mixture: Annotated[Path, typer.Option(..., exists=True, help="Path to mixture.json")],
    out: Annotated[Path, typer.Option(help="Output directory")] = Path("outputs"),
) -> None:
    """Recompute report.json from a stored mixture."""

    g = io.load_graph(graph)
    payload = json.loads(mixture.read_text())
    mixture_records = payload["trees"]
    total_samples = sum(int(t["count"]) for t in mixture_records)

    expected_cong = {io.edge_key(u, v): 0.0 for u, v, _ in io.iter_edges(g)}
    all_cuts: list[dict[str, object]] = []
    for rec in mixture_records:
        tree = _record_to_tree(rec, list(g.nodes))
        lca = LCA(tree)
        c_t = congestion.compute_tree_capacities(g, tree, lca)
        cong = congestion.compute_edge_congestion(g, tree, lca, c_t)
        weight = rec["count"] / total_samples  # type: ignore[index]
        for edge, val in cong.items():
            expected_cong[edge] += weight * val
        all_cuts.extend(cuts.extract_tree_cuts(g, tree))

    rpt = report.assemble_report(
        g,
        mixture_records,
        expected_cong,
        cuts=all_cuts,
        params={
            "iters": payload.get("iters"),
            "candidates": payload.get("candidates"),
            "seed": payload.get("seed"),
            "alpha": payload.get("alpha"),
        },
    )
    write_json(out / "report.json", rpt)
    console.print(f"[green]Wrote[/] {out/'report.json'}")


if __name__ == "__main__":
    app()
