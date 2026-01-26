import json
from pathlib import Path

import networkx as nx
import pytest
from typer.testing import CliRunner

from ires.cli import app
from ires.core import congestion, io
from ires.core.lca import LCA
from ires.oracles.spt import shortest_path_tree


@pytest.fixture
def tiny_graph() -> nx.Graph:
    return io.load_graph(Path("data/tiny.edgelist"))


def test_tree_capacities_non_negative(tiny_graph: nx.Graph) -> None:
    lengths = {io.edge_key(u, v): 1.0 for u, v, _ in io.iter_edges(tiny_graph)}
    tree = shortest_path_tree(tiny_graph, lengths, root=next(iter(tiny_graph.nodes)))
    lca = LCA(tree)
    c_t = congestion.compute_tree_capacities(tiny_graph, tree, lca)
    assert all(value >= 0 for value in c_t.values())


def test_congestion_non_negative(tiny_graph: nx.Graph) -> None:
    lengths = {io.edge_key(u, v): 1.0 for u, v, _ in io.iter_edges(tiny_graph)}
    tree = shortest_path_tree(tiny_graph, lengths, root=next(iter(tiny_graph.nodes)))
    lca = LCA(tree)
    c_t = congestion.compute_tree_capacities(tiny_graph, tree, lca)
    cong = congestion.compute_edge_congestion(tiny_graph, tree, lca, c_t)
    assert all(value >= 0 for value in cong.values())


def test_cli_fit_outputs(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "fit",
            "--graph",
            "data/tiny.edgelist",
            "--iters",
            "4",
            "--candidates",
            "2",
            "--seed",
            "0",
            "--out",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0
    mix_path = tmp_path / "mixture.json"
    rpt_path = tmp_path / "report.json"
    assert mix_path.exists()
    assert rpt_path.exists()
    mixture = json.loads(mix_path.read_text())
    assert "trees" in mixture and len(mixture["trees"]) > 0
    report = json.loads(rpt_path.read_text())
    assert "critical_edges" in report
