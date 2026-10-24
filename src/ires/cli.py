"""Typer CLI for Infra Resilience Explorer."""

from __future__ import annotations

import json
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Annotated, Any, TypeVar

import networkx as nx
import typer
from rich.console import Console
from rich.progress import track
from rich.table import Table

from .core import io, report
from .core import mixture as mix_utils
from .core.mixture import MixtureEvaluation, evaluate_mixture
from .core.mwu import (
    DEFAULT_ETA,
    MwuParams,
    MwuResult,
    edge_distribution,
    run_mwu,
    suggested_eta,
    update_log_weights,
)
from .core.report import Bounds
from .oracles import DEFAULT_ORACLE, ORACLES

__all__ = [
    "app",
    "edge_distribution",
    "fit",
    "report_cmd",
    "run_mwu",
    "update_log_weights",
    "write_json",
]

console = Console()
app = typer.Typer(help="Infra Resilience Explorer", no_args_is_help=True)

DEFAULT_OUT = Path("reports")

F = TypeVar("F", bound=Callable[..., Any])


def _input_errors(command: F) -> F:
    """Turn invalid user input into a short CLI error and nonzero exit."""

    @wraps(command)
    def checked(*args: Any, **kwargs: Any) -> Any:
        try:
            return command(*args, **kwargs)
        except (ValueError, OSError, OverflowError) as exc:
            console.print(f"Error: {exc}", markup=False)
            raise typer.Exit(code=2) from exc

    return checked  # type: ignore[return-value]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def _print_summary(graph: nx.Graph, evaluation: MixtureEvaluation, bounds: Bounds | None) -> None:
    table = Table(title="Summary", show_header=False, box=None)
    table.add_row("graph", f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    table.add_row("max expected score", f"{evaluation.max_expected_score:.4g}  (upper bound)")
    if bounds is not None and bounds.dual_estimate:
        table.add_row("dual", f"{bounds.dual_estimate:.4g}  (lower estimate, uncertified)")
        table.add_row("estimated gap", f"{bounds.achieved / bounds.dual_estimate:.2f}x")
    if bounds is not None:
        table.add_row("certified floor", f"{bounds.certified_lower:.4g}  (rigorous, <= 1)")
    table.add_row("distinct cuts", str(len(evaluation.cuts)))
    console.print(table)


def _parameters(params: MwuParams, result: MwuResult) -> dict[str, Any]:
    return {
        "iters": params.iters,
        "candidates": params.candidates,
        "seed": params.seed,
        "oracle": params.oracle,
        "eta": result.eta,
        "alpha": result.alpha,
    }


@app.command()
@_input_errors
def fit(
    graph: Annotated[
        Path,
        typer.Option(..., exists=True, dir_okay=False, help="Path to undirected edge list"),
    ],
    iters: Annotated[int, typer.Option(min=1, help="Number of MWU iterations")] = 80,
    candidates: Annotated[int, typer.Option(min=1, help="Candidate trees per iteration")] = 8,
    seed: Annotated[int, typer.Option(min=0, help="Random seed")] = 0,
    oracle: Annotated[
        str, typer.Option(help=f"Candidate generator ({', '.join(sorted(ORACLES))})")
    ] = DEFAULT_ORACLE,
    eta: Annotated[
        float | None,
        typer.Option(help=f"Step size, tuning only (default {DEFAULT_ETA}; only eta/alpha acts)"),
    ] = None,
    alpha: Annotated[
        float | None, typer.Option(help="Positive update scale (a tuning parameter)")
    ] = None,
    out: Annotated[Path, typer.Option(help="Output directory")] = DEFAULT_OUT,
) -> None:
    """Fit a tree mixture and emit mixture.json and report.json."""

    g = io.load_graph(graph)
    params = MwuParams(
        iters=iters,
        candidates=candidates,
        seed=seed,
        oracle=oracle,
        eta=DEFAULT_ETA if eta is None else eta,
        alpha=alpha,
    )
    console.print(
        f"[bold]Graph[/]: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges. "
        f"oracle={oracle} suggested_eta={suggested_eta(g, iters):.3f}"
    )

    result = run_mwu(
        g, params, progress=lambda it: track(it, description="Running MWU", console=console)
    )
    trees = mix_utils.records_to_dicts(result.records, params.iters)
    evaluation = evaluate_mixture(g, trees, params.iters)
    bounds = Bounds(
        achieved=evaluation.max_expected_score,
        certified_lower=result.certified_lower,
        dual_estimate=result.dual_estimate,
    )
    parameters = _parameters(params, result)

    write_json(
        out / "mixture.json",
        {
            "schema_version": mix_utils.SCHEMA_VERSION,
            "metric": mix_utils.METRIC,
            "graph_fingerprint": io.graph_fingerprint(g),
            "graph": graph.as_posix(),
            **parameters,
            "certified_lower": result.certified_lower,
            "dual_estimate": result.dual_estimate,
            "history": [entry.to_dict() for entry in result.history],
            "trees": trees,
        },
    )
    write_json(
        out / "report.json",
        report.assemble_report(
            g,
            trees,
            evaluation,
            params=parameters,
            bounds=bounds,
        ),
    )
    _print_summary(g, evaluation, bounds)
    console.print(f"[green]Wrote[/] {out / 'mixture.json'} and {out / 'report.json'}")


@app.command("report")
@_input_errors
def report_cmd(
    graph: Annotated[
        Path, typer.Option(..., exists=True, dir_okay=False, help="Path to graph edge list")
    ],
    mixture: Annotated[
        Path, typer.Option(..., exists=True, dir_okay=False, help="Path to mixture.json")
    ],
    out: Annotated[Path, typer.Option(help="Output directory")] = DEFAULT_OUT,
) -> None:
    """Recompute report.json from a stored mixture."""

    g = io.load_graph(graph)
    payload = json.loads(mixture.read_text())
    mix_utils.validate_payload(g, payload)
    evaluation = evaluate_mixture(g, payload["trees"], payload["iters"])

    def stored(name: str) -> float | None:
        value = payload.get(name)
        return float(value) if isinstance(value, int | float) else None

    bounds = Bounds(
        achieved=evaluation.max_expected_score,
        certified_lower=stored("certified_lower") or 0.0,
        dual_estimate=stored("dual_estimate"),
    )

    write_json(
        out / "report.json",
        report.assemble_report(
            g,
            payload["trees"],
            evaluation,
            params={
                key: payload.get(key)
                for key in ("iters", "candidates", "seed", "oracle", "eta", "alpha")
            },
            bounds=bounds,
        ),
    )
    _print_summary(g, evaluation, bounds)
    console.print(f"[green]Wrote[/] {out / 'report.json'}")


@app.command()
def oracles() -> None:
    """List the available candidate generators."""

    for name, fn in sorted(ORACLES.items()):
        summary = (fn.__doc__ or "").strip().splitlines()[0]
        marker = " (default)" if name == DEFAULT_ORACLE else ""
        console.print(f"[bold]{name}[/]{marker}: {summary}")


if __name__ == "__main__":
    app()
