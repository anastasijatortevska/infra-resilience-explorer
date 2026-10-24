import json
import math
from typing import Any

import networkx as nx
import numpy as np
import pytest
from conftest import TINY
from typer.testing import CliRunner

from ires.cli import app
from ires.core import cuts, io, mixture
from ires.core.mwu import MwuParams, edge_distribution, run_mwu, update_log_weights


def run(graph, **overrides):
    """Run the loop and return the (records, expected scores) pair the tests compare."""

    settings: dict[str, Any] = dict(iters=25, candidates=4, eta=0.6, alpha=30, seed=7)
    settings.update(overrides)
    result = run_mwu(graph, MwuParams(**settings))
    iters = int(settings["iters"])
    records = mixture.records_to_dicts(result.records, iters)
    expected, _ = mixture.summarize(graph, records, iters)
    return records, expected


def test_normalized_update_agrees_with_original_exponent_and_attention_sign():
    edges = [("a", "b"), ("b", "c")]
    old = dict(zip(edges, [2.0, 5.0], strict=True))
    scores = dict(zip(edges, [1.5, 4.0], strict=True))
    eta, alpha = 0.6, 3.0
    updated = update_log_weights({e: math.log(w) for e, w in old.items()}, scores, eta / alpha)
    observed, _ = edge_distribution(updated, {e: 1 for e in edges})
    raw = {e: w * math.exp(eta * (scores[e] / alpha - 1)) for e, w in old.items()}
    assert observed == pytest.approx({e: w / sum(raw.values()) for e, w in raw.items()})
    assert observed[edges[1]] / observed[edges[0]] > old[edges[1]] / old[edges[0]]


def test_thousands_of_identical_updates_do_not_underflow():
    graph = nx.path_graph(["a", "b", "c"])
    nx.set_edge_attributes(graph, 1.0, "capacity")
    records, expected = run(graph, iters=2000, candidates=1)
    assert len(records) == 1
    assert records[0]["count"] == 2000
    assert set(expected.values()) == {1.0}


def test_common_large_update_and_extreme_common_capacity_scale_are_safe():
    graph = nx.Graph()
    graph.add_edge("a", "b", capacity=1e308)
    _, expected = run(graph, alpha=1e-308)
    assert expected == {("a", "b"): 1.0}
    probs, lengths = edge_distribution({("a", "b"): 0}, {("a", "b"): 1e-308})
    assert probs == lengths == {("a", "b"): 1.0}


@pytest.mark.parametrize(
    "name,value",
    [
        ("iters", 0),
        ("candidates", 0),
        ("eta", 0),
        ("eta", -1),
        ("eta", math.nan),
        ("eta", math.inf),
        ("alpha", 0),
        ("alpha", -1),
        ("alpha", math.nan),
        ("alpha", math.inf),
        ("alpha", 1e-320),
        ("seed", -1),
        ("seed", True),
    ],
)
def test_invalid_update_parameters_are_rejected(name, value):
    graph = io.load_graph(TINY)
    with pytest.raises(ValueError):
        run(graph, **{name: value})


def test_unrepresentable_relative_weights_fail_clearly():
    with pytest.raises(ValueError, match="weights underflow"):
        edge_distribution({("a", "b"): 0, ("b", "c"): -1000}, {("a", "b"): 1, ("b", "c"): 1})
    with pytest.raises(ValueError, match="lengths underflow"):
        edge_distribution({("a", "b"): 0, ("b", "c"): 0}, {("a", "b"): 1e308, ("b", "c"): 1e-308})


def test_nonfinite_metric_fails_instead_of_producing_json_infinity():
    graph = nx.Graph()
    graph.add_weighted_edges_from(
        [("a", "b", 1e308), ("b", "c", 1e308), ("a", "c", 1e308)], weight="capacity"
    )
    with pytest.raises(ValueError, match="overflow"):
        run(graph)


def test_seed_and_edge_insertion_order_are_reproducible():
    graph = io.load_graph(TINY)
    reversed_graph = nx.Graph()
    for u, v, capacity in reversed(list(io.iter_edges(graph))):
        reversed_graph.add_edge(v, u, capacity=capacity)
    assert run(graph) == run(graph) == run(reversed_graph)
    assert io.graph_fingerprint(graph) == io.graph_fingerprint(reversed_graph)


def test_moderate_varied_capacity_stress_matches_direct_recomputation():
    rng = np.random.default_rng(123)
    graph = nx.watts_strogatz_graph(24, 4, 0.2, seed=4)
    graph = nx.relabel_nodes(graph, {n: str(n) for n in graph})
    for u, v in sorted(graph.edges):
        graph[u][v]["capacity"] = float(2 ** rng.uniform(-1, 4))
    records, expected = run(graph, iters=100, candidates=5, alpha=200)
    assert math.fsum(rec["prob"] for rec in records) == pytest.approx(1)
    assert sum(rec["count"] for rec in records) == 100
    direct = {e: 0.0 for e in expected}
    for rec in records:
        tree = mixture.record_to_tree(rec, graph)
        undirected = nx.Graph(tree.edges())
        fundamental = {}
        for u, v in tree.edges():
            split = undirected.copy()
            split.remove_edge(u, v)
            side = nx.node_connected_component(split, u)
            fundamental[io.edge_key(u, v)] = cuts.cut_capacity(graph, side)
        for u, v, capacity in io.iter_edges(graph):
            path = nx.shortest_path(undirected, u, v)
            score = (
                math.fsum(
                    fundamental[io.edge_key(a, b)] for a, b in zip(path, path[1:], strict=False)
                )
                / capacity
            )
            direct[io.edge_key(u, v)] += rec["prob"] * score
    assert expected == pytest.approx(direct, rel=1e-12)
    assert all(math.isfinite(value) and value > 0 for value in expected.values())


def test_cli_records_scale_and_reproduces_json(tmp_path):
    runner = CliRunner()
    for name in ["one", "two"]:
        result = runner.invoke(
            app,
            [
                "fit",
                "--graph",
                TINY,
                "--iters",
                "20",
                "--candidates",
                "3",
                "--seed",
                "2",
                "--alpha",
                "15",
                "--out",
                str(tmp_path / name),
            ],
        )
        assert result.exit_code == 0, result.output
    for filename in ["mixture.json", "report.json"]:
        assert (tmp_path / "one" / filename).read_bytes() == (
            tmp_path / "two" / filename
        ).read_bytes()
    payload = json.loads((tmp_path / "one" / "mixture.json").read_text())
    assert payload["alpha"] == 15 and payload["eta"] == 0.6
