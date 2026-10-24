from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from conftest import TINY, random_capacitated_graph, write_edgelist
from typer.testing import CliRunner

from ires.cli import app

runner = CliRunner()


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
    assert len(mixture["history"]) == 6
    assert mixture["certified_lower"] > 0
    assert report["bounds"]["certified_lower"] <= report["bounds"]["achieved"]
    assert report["bounds"]["dual_certified"] is False
    assert report["scored_edges"] and report["candidate_cuts"]
    assert report["max_expected_cut_path_score"] == report["bounds"]["achieved"]


def test_bottleneck_cuts_are_distinct_and_cheapest_first(tmp_path: Path) -> None:
    graph = write_edgelist(tmp_path / "g.edgelist", random_capacitated_graph(60, seed=10))
    run("fit", "--graph", str(graph), "--iters", "20", "--candidates", "3", "--out", str(tmp_path))
    report = json.loads((tmp_path / "report.json").read_text())

    capacities = [cut["capacity"] for cut in report["candidate_cuts"]]
    assert capacities == sorted(capacities)
    sides = [tuple(cut["nodes"]) for cut in report["candidate_cuts"] if not cut["truncated"]]
    assert len(sides) == len(set(sides)), "the same cut was reported twice"
    assert report["distinct_candidate_cuts"] >= len(capacities)


@pytest.mark.parametrize("oracle", ["spt", "mst", "ldd"])
def test_every_oracle_runs_end_to_end(tmp_path: Path, oracle: str) -> None:
    run("fit", "--graph", TINY, "--iters", "8", "--oracle", oracle, "--out", str(tmp_path / oracle))
    report = json.loads((tmp_path / oracle / "report.json").read_text())
    assert report["parameters"]["oracle"] == oracle


def test_rejects_unknown_oracle(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["fit", "--graph", TINY, "--oracle", "nope", "--out", str(tmp_path)]
    )
    assert result.exit_code == 2
    assert "unknown oracle" in result.output


def test_eta_override_is_recorded(tmp_path: Path) -> None:
    run("fit", "--graph", TINY, "--iters", "5", "--eta", "0.25", "--out", str(tmp_path))
    assert json.loads((tmp_path / "mixture.json").read_text())["eta"] == 0.25


def test_default_output_directory_is_reports(tmp_path: Path, monkeypatch: Any) -> None:
    """The default target is reports/, kept separate from anything else on disk."""

    graph = Path(TINY).resolve()
    monkeypatch.chdir(tmp_path)
    run("fit", "--graph", str(graph), "--iters", "4")
    assert (tmp_path / "reports" / "report.json").exists()
    assert not (tmp_path / "outputs").exists()


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
    assert result.exit_code == 2
    assert "fingerprint" in result.output


def test_oracles_command_lists_the_registry() -> None:
    output = run("oracles").output
    for name in ("spt", "mst", "ldd"):
        assert name in output
