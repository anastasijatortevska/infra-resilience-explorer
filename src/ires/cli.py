"""Typer CLI for Infra Resilience Explorer."""

from __future__ import annotations

import json
import math
from functools import wraps
from pathlib import Path
from typing import Annotated

import networkx as nx
import numpy as np
import typer
from rich.console import Console
from rich.progress import track

from .core import io, metrics, report
from .core import mixture as mix_utils
from .core.lca import LCA
from .core.tree import Tree
from .oracles.spt import shortest_path_tree

console = Console()
app = typer.Typer(help="Infra Resilience Explorer")


def _input_errors(command):
    """Turn invalid user input into a short CLI error and nonzero exit."""

    @wraps(command)
    def checked(*args, **kwargs):
        try:
            return command(*args, **kwargs)
        except (ValueError, OSError, OverflowError) as exc:
            console.print(f"Error: {exc}", markup=False)
            raise typer.Exit(code=2) from exc

    return checked


def _tree_to_record(tree: Tree, count: int, total: int) -> dict[str, object]:
    return {
        "root": tree.root,
        "edges": [(u, v) for u, v in tree.edges()],
        "count": count,
        "prob": count / total if total else 0.0,
    }


def _edge_distribution(
    log_weights: dict[tuple[str, str], float], capacities: dict[tuple[str, str], float]
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float]]:
    """Normalize weights and rescale lengths by a common positive factor.

    Subtracting the largest log value prevents a common scale from causing
    overflow. Rescaling lengths preserves all shortest paths. Unrepresentable
    relative weights/lengths are rejected rather than silently becoming zero.
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


def _update_log_weights(
    log_weights: dict[tuple[str, str], float], scores: dict[tuple[str, str], float], step: float
) -> dict[tuple[str, str], float]:
    """Positive MWU update, removing the common largest score before scaling."""

    peak_score = max(scores.values())
    updated = {
        edge: value + step * (scores[edge] - peak_score) for edge, value in log_weights.items()
    }
    if any(not math.isfinite(value) for value in updated.values()):
        raise ValueError("Nonfinite weight update; increase alpha or reduce eta")
    peak = max(updated.values())
    return {edge: value - peak for edge, value in updated.items()}


def run_mwu(
    graph: nx.Graph,
    iters: int,
    r: int,
    eta: float,
    alpha: float,
    seed: int,
    progress: bool = True,
) -> tuple[list[dict[str, object]], dict[tuple[str, str], float]]:
    """Experimentally minimize the maximum expected cut-path score.

    Candidates minimize the current weighted score; the positive update raises
    the next iteration's attention to high-scoring edges. The SPT candidate
    generator is a heuristic, and alpha is only an update scale.
    """

    io.validate_graph(graph)
    io.positive_count(iters, "iters")
    io.positive_count(r, "candidates")
    io.positive_finite(eta, "eta")
    io.positive_finite(alpha, "alpha")
    step = eta / alpha
    io.positive_finite(step, "eta / alpha")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    nodes = sorted(graph.nodes)
    capacities = {io.edge_key(u, v): w for u, v, w in io.iter_edges(graph)}
    log_weights = {edge: 0.0 for edge in capacities}

    mixture_counts: dict[tuple[tuple[str, str], ...], dict[str, object]] = {}

    for iteration in track(range(iters), description="Running MWU", disable=not progress):
        probs, lengths = _edge_distribution(log_weights, capacities)

        candidates = []
        for _ in range(r):
            root = rng.choice(nodes)
            tree = shortest_path_tree(graph, lengths, root)
            lca = LCA(tree)
            c_t = metrics.compute_tree_capacities(graph, tree, lca)
            scores = metrics.compute_edge_scores(graph, tree, lca, c_t)
            obj = math.fsum(probs[e] * scores[e] for e in scores)
            if not math.isfinite(obj):
                raise ValueError(f"Iteration {iteration}: nonfinite weighted cut-path score")
            candidates.append((obj, tree, scores))

        obj, tree, scores = min(candidates, key=lambda x: x[0])

        if iteration + 1 < iters:
            log_weights = _update_log_weights(log_weights, scores, step)

        sig = mix_utils.tree_signature(tree)
        if sig not in mixture_counts:
            mixture_counts[sig] = {"tree": tree, "count": 0}
        mixture_counts[sig]["count"] = mixture_counts[sig]["count"] + 1  # type: ignore[index]

    mixture = [
        _tree_to_record(info["tree"], info["count"], iters)  # type: ignore[index]
        for info in mixture_counts.values()
    ]
    mixture.sort(
        key=lambda rec: (-rec["count"], tuple(sorted(io.edge_key(*e) for e in rec["edges"])))
    )
    expected_scores, _ = mix_utils.summarize(graph, mixture, iters)
    return mixture, expected_scores


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


@app.command()
@_input_errors
def fit(
    graph: Annotated[
        Path,
        typer.Option(..., exists=True, help="Path to undirected weighted edge list"),
    ],
    iters: Annotated[int, typer.Option(min=1, help="Number of MWU iterations")] = 80,
    candidates: Annotated[int, typer.Option(min=1, help="Candidate trees per iteration")] = 8,
    seed: Annotated[int, typer.Option(min=0, help="Random seed")] = 0,
    alpha: Annotated[
        float | None, typer.Option(help="Positive update scale (a tuning parameter)")
    ] = None,
    out: Annotated[Path, typer.Option(help="Output directory")] = Path("outputs"),
) -> None:
    """Fit a tree mixture and emit mixture.json and report.json."""

    g = io.load_graph(graph)
    # Historical tuning default; no bound or approximation guarantee is implied.
    alpha = alpha if alpha is not None else 10 * math.log2(g.number_of_nodes() + 1)
    eta = 0.6
    console.print(
        f"[bold]Graph[/]: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges. "
        f"alpha={alpha:.2f}"
    )

    mixture, expected_scores = run_mwu(
        g, iters=iters, r=candidates, eta=eta, alpha=alpha, seed=seed
    )

    mixture_payload = {
        "schema_version": mix_utils.SCHEMA_VERSION,
        "metric": mix_utils.METRIC,
        "graph_fingerprint": io.graph_fingerprint(g),
        "graph": graph.as_posix(),
        "iters": iters,
        "candidates": candidates,
        "seed": seed,
        "alpha": alpha,
        "eta": eta,
        "trees": mixture,
    }
    write_json(out / "mixture.json", mixture_payload)

    expected_scores, all_cuts = mix_utils.summarize(g, mixture, iters)

    rpt = report.assemble_report(
        g,
        mixture,
        expected_scores,
        cuts=all_cuts,
        params={"iters": iters, "candidates": candidates, "seed": seed, "alpha": alpha, "eta": eta},
    )
    write_json(out / "report.json", rpt)
    console.print(f"[green]Wrote[/] {out / 'mixture.json'} and {out / 'report.json'}")


@app.command("report")
@_input_errors
def report_cmd(
    graph: Annotated[Path, typer.Option(..., exists=True, help="Path to graph edge list")],
    mixture: Annotated[Path, typer.Option(..., exists=True, help="Path to mixture.json")],
    out: Annotated[Path, typer.Option(help="Output directory")] = Path("outputs"),
) -> None:
    """Recompute report.json from a stored mixture."""

    g = io.load_graph(graph)
    payload = json.loads(mixture.read_text())
    mix_utils.validate_payload(g, payload)
    mixture_records = payload["trees"]
    expected_scores, all_cuts = mix_utils.summarize(g, mixture_records, payload["iters"])

    rpt = report.assemble_report(
        g,
        mixture_records,
        expected_scores,
        cuts=all_cuts,
        params={
            "iters": payload.get("iters"),
            "candidates": payload.get("candidates"),
            "seed": payload.get("seed"),
            "alpha": payload.get("alpha"),
            "eta": payload.get("eta", 0.6),
        },
    )
    write_json(out / "report.json", rpt)
    console.print(f"[green]Wrote[/] {out / 'report.json'}")


if __name__ == "__main__":
    app()
