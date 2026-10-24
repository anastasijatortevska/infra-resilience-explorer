from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from conftest import random_capacitated_graph, write_edgelist
from typer.testing import CliRunner

from ires.cli import app

runner = CliRunner()
TINY = "data/tiny.edgelist"


def run(*args: str) -> Any:
    result = runner.invoke(app, list(args))
    if result.exit_code != 0:
        raise AssertionError(f"ires {' '.join(args)} failed:\n{result.output}\n{result.exception}")
    return result


def test_fit_writes_mixture_and_report(tmp_path: Path) -> None:
    run("fit", "--graph", TINY, "--iters", "6", "--candidates", "2", "--out", str(tmp_path))
    mixture = json.loads((tmp_path / "mixture.json").read_text())
    report = json.loads((tmp_path / "report.json").read_text())

    assert sum(tree["count"] for tree in mixture["trees"]) == 6
    assert mixture["certified_lower"] > 0
    assert report["bounds"]["certified_lower"] <= report["bounds"]["alpha_hat"]
    assert len(report["history"]) == 6
    assert report["critical_edges"]
    assert report["bottleneck_cuts"]


def test_report_reproduces_fit(tmp_path: Path) -> None:
    """`fit` and `report` must score the same mixture identically."""

    graph = write_edgelist(tmp_path / "g.edgelist", random_capacitated_graph(50, seed=9))
    run("fit", "--graph", str(graph), "--iters", "15", "--candidates", "3", "--out", str(tmp_path))
    first = json.loads((tmp_path / "report.json").read_text())

    run(
        "report",
        "--graph",
        str(graph),
        "--mixture",
        str(tmp_path / "mixture.json"),
        "--out",
        str(tmp_path / "again"),
    )
    second = json.loads((tmp_path / "again" / "report.json").read_text())

    assert first["critical_edges"] == second["critical_edges"]
    assert first["bottleneck_cuts"] == second["bottleneck_cuts"]
    assert first["bounds"]["alpha_hat"] == pytest.approx(second["bounds"]["alpha_hat"])
    assert first["distinct_cuts"] == second["distinct_cuts"]


def test_bottleneck_cuts_are_the_globally_cheapest_distinct_cuts(tmp_path: Path) -> None:
    graph = write_edgelist(tmp_path / "g.edgelist", random_capacitated_graph(60, seed=10))
    run("fit", "--graph", str(graph), "--iters", "20", "--candidates", "3", "--out", str(tmp_path))
    report = json.loads((tmp_path / "report.json").read_text())

    capacities = [cut["capacity"] for cut in report["bottleneck_cuts"]]
    assert capacities == sorted(capacities)
    sizes = [
        tuple(sorted(cut["nodes"])) for cut in report["bottleneck_cuts"] if not cut["truncated"]
    ]
    assert len(sizes) == len(set(sizes)), "the same cut was reported twice"
    assert report["distinct_cuts"] >= len(capacities)


def test_same_seed_gives_identical_report(tmp_path: Path) -> None:
    for name in ("a", "b"):
        run("fit", "--graph", TINY, "--iters", "10", "--seed", "3", "--out", str(tmp_path / name))
    assert (tmp_path / "a" / "report.json").read_text() == (
        tmp_path / "b" / "report.json"
    ).read_text()


@pytest.mark.parametrize("oracle", ["spt", "mst", "ldd"])
def test_every_oracle_runs_end_to_end(tmp_path: Path, oracle: str) -> None:
    run("fit", "--graph", TINY, "--iters", "8", "--oracle", oracle, "--out", str(tmp_path / oracle))
    report = json.loads((tmp_path / oracle / "report.json").read_text())
    assert report["parameters"]["oracle"] == oracle


def test_rejects_mixture_from_a_different_graph(tmp_path: Path) -> None:
    other = write_edgelist(tmp_path / "other.edgelist", random_capacitated_graph(30, seed=11))
    run("fit", "--graph", str(other), "--iters", "4", "--out", str(tmp_path))
    result = runner.invoke(
        app,
        [
            "report",
            "--graph",
            TINY,
            "--mixture",
            str(tmp_path / "mixture.json"),
            "--out",
            str(tmp_path),
        ],
    )
    assert result.exit_code != 0
    assert "different graph" in result.output


def test_rejects_disconnected_graph(tmp_path: Path) -> None:
    path = tmp_path / "split.edgelist"
    path.write_text("a b 1\nc d 1\n")
    result = runner.invoke(app, ["fit", "--graph", str(path), "--out", str(tmp_path)])
    assert result.exit_code != 0
    assert "disconnected" in result.output


def test_rejects_bad_parameters(tmp_path: Path) -> None:
    result = runner.invoke(app, ["fit", "--graph", TINY, "--iters", "0", "--out", str(tmp_path)])
    assert result.exit_code != 0
    assert "must be positive" in result.output


def test_default_output_directory_is_reports(tmp_path: Path, monkeypatch: Any) -> None:
    """The default target is reports/, kept separate from anything else on disk."""

    graph = Path(TINY).resolve()
    monkeypatch.chdir(tmp_path)
    run("fit", "--graph", str(graph), "--iters", "4")
    assert (tmp_path / "reports" / "report.json").exists()
    assert not (tmp_path / "outputs").exists()


def test_oracles_command_lists_the_registry() -> None:
    output = run("oracles").output
    for name in ("spt", "mst", "ldd"):
        assert name in output
