"""Typer CLI for Infra Resilience Explorer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import networkx as nx
import typer
from rich.console import Console
from rich.progress import track
from rich.table import Table

from .core import io
from .core.mixture import MixtureEvaluation, TreeRecord, evaluate_mixture, load_records
from .core.mwu import MwuParams, MwuResult, run_mwu
from .core.report import Bounds, assemble_report
from .oracles import DEFAULT_ORACLE, ORACLES

console = Console()
app = typer.Typer(help="Infra Resilience Explorer", no_args_is_help=True)

DEFAULT_OUT = Path("reports")

GraphOption = Annotated[
    Path, typer.Option(..., exists=True, dir_okay=False, help="Path to undirected edge list")
]
OutOption = Annotated[Path, typer.Option(help="Output directory")]


def _load_graph(path: Path) -> nx.Graph:
    """Load a graph, reporting validation failures as a usage error."""

    try:
        return io.load_graph(path)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--graph") from exc


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _print_summary(graph: nx.Graph, evaluation: MixtureEvaluation, bounds: Bounds) -> None:
    table = Table(title="Summary", show_header=False, box=None)
    table.add_row("graph", f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    table.add_row("alpha_hat", f"{bounds.alpha_hat:.4g}  (upper bound, achieved)")
    if bounds.dual_objective:
        table.add_row("dual", f"{bounds.dual_objective:.4g}  (lower estimate, uncertified)")
        table.add_row("estimated gap", f"{bounds.alpha_hat / bounds.dual_objective:.2f}x")
    table.add_row("certified floor", f"{bounds.certified_lower:.4g}  (rigorous, <= 1)")
    table.add_row("distinct cuts", str(len(evaluation.cuts)))
    console.print(table)


def _params_payload(graph_path: Path, params: MwuParams, result: MwuResult) -> dict[str, Any]:
    return {
        "graph": graph_path.as_posix(),
        "iters": params.iters,
        "candidates": params.candidates,
        "seed": params.seed,
        "oracle": params.oracle,
        "eta": result.eta,
        "alpha": result.alpha,
    }


def _mixture_payload(
    graph_path: Path, params: MwuParams, result: MwuResult, total: int
) -> dict[str, Any]:
    return {
        **_params_payload(graph_path, params, result),
        "certified_lower": result.certified_lower,
        "trees": [record.to_dict(total) for record in result.records],
    }


@app.command()
def fit(
    graph: GraphOption,
    iters: Annotated[int, typer.Option(help="Number of MWU iterations")] = 80,
    candidates: Annotated[int, typer.Option(help="Candidate trees per iteration")] = 8,
    seed: Annotated[int, typer.Option(help="Random seed")] = 0,
    oracle: Annotated[
        str, typer.Option(help=f"Tree oracle ({', '.join(sorted(ORACLES))})")
    ] = DEFAULT_ORACLE,
    eta: Annotated[
        float | None, typer.Option(help="MWU step size (default: sqrt(log m / iters))")
    ] = None,
    alpha: Annotated[
        float | None, typer.Option(help="Update width (default: 10*log2(n+1))")
    ] = None,
    out: OutOption = DEFAULT_OUT,
) -> None:
    """Fit a tree mixture and emit mixture.json and report.json."""

    g = _load_graph(graph)
    params = MwuParams(
        iters=iters, candidates=candidates, seed=seed, oracle=oracle, eta=eta, alpha=alpha
    )
    try:
        result = run_mwu(
            g, params, progress=lambda it: track(it, description="Running MWU", console=console)
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    evaluation = evaluate_mixture(g, result.records)
    bounds = Bounds(
        alpha_hat=evaluation.alpha_hat,
        certified_lower=result.certified_lower,
        dual_objective=result.dual_objective,
    )
    params_payload = _params_payload(graph, params, result)

    write_json(
        out / "mixture.json",
        _mixture_payload(graph, params, result, evaluation.total_samples),
    )
    write_json(
        out / "report.json",
        assemble_report(
            g,
            result.records,
            evaluation,
            params=params_payload,
            bounds=bounds,
            history=[entry.to_dict() for entry in result.history],
        ),
    )
    _print_summary(g, evaluation, bounds)
    console.print(f"[green]Wrote[/] {out / 'mixture.json'} and {out / 'report.json'}")


@app.command("report")
def report_cmd(
    graph: GraphOption,
    mixture: Annotated[
        Path, typer.Option(..., exists=True, dir_okay=False, help="Path to mixture.json")
    ],
    out: OutOption = DEFAULT_OUT,
) -> None:
    """Recompute report.json from a stored mixture."""

    g = _load_graph(graph)
    payload = json.loads(mixture.read_text())
    try:
        records: list[TreeRecord] = load_records(payload)
        evaluation = evaluate_mixture(g, records)
    except ValueError as exc:
        raise typer.BadParameter(f"{mixture}: {exc}") from exc

    bounds = Bounds(alpha_hat=evaluation.alpha_hat, certified_lower=0.0)
    stored = payload.get("certified_lower")
    if isinstance(stored, int | float):
        bounds = Bounds(alpha_hat=evaluation.alpha_hat, certified_lower=float(stored))

    write_json(
        out / "report.json",
        assemble_report(
            g,
            records,
            evaluation,
            params={key: payload.get(key) for key in ("iters", "candidates", "seed", "oracle")},
            bounds=bounds,
        ),
    )
    _print_summary(g, evaluation, bounds)
    console.print(f"[green]Wrote[/] {out / 'report.json'}")


@app.command()
def oracles() -> None:
    """List the available tree oracles."""

    for name, fn in sorted(ORACLES.items()):
        summary = (fn.__doc__ or "").strip().splitlines()[0]
        marker = " (default)" if name == DEFAULT_ORACLE else ""
        console.print(f"[bold]{name}[/]{marker}: {summary}")


if __name__ == "__main__":
    app()
