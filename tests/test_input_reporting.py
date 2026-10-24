import copy
import json

import networkx as nx
import pytest
from conftest import TINY
from typer.testing import CliRunner

from ires.cli import app
from ires.core import cuts, io, mixture
from ires.core.mwu import MwuParams, run_mwu
from ires.core.tree import Tree
from ires.oracles.spt import shortest_path_tree


@pytest.mark.parametrize(
    "text,message",
    [
        ("", "at least two"),
        ("# no vertices\n", "at least two"),
        ("a a 1\n", "self-loop"),
        ("a b 1\nc d 2\n", "connected"),
        ("a b 0\n", "finite and positive"),
        ("a b -1\n", "finite and positive"),
        ("a b nan\n", "finite and positive"),
        ("a b inf\n", "finite and positive"),
        ("a b -inf\n", "finite and positive"),
        ("a b nope\n", "invalid capacity"),
        ("a b\n", "three tokens"),
        ("a b 1e308\nb a 1e308\n", "merged capacity overflow"),
        ("a b 2\nb a -1\n", "finite and positive"),
    ],
)
def test_reject_invalid_edgelists(tmp_path, text, message):
    path = tmp_path / "graph.edgelist"
    path.write_text(text)
    with pytest.raises(ValueError, match=message):
        io.load_graph(path)


@pytest.mark.parametrize("nodes", [[], ["a"]])
def test_reject_graphs_with_fewer_than_two_vertices(nodes):
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    with pytest.raises(ValueError, match="at least two"):
        io.validate_graph(graph)


def test_parallel_edges_merge_by_capacity_sum(tmp_path):
    path = tmp_path / "parallel.edgelist"
    path.write_text("a b 2\nb a 3.5\nb c 1\n")
    graph = io.load_graph(path)
    assert list(io.iter_edges(graph)) == [("a", "b", 5.5), ("b", "c", 1)]


def test_unreachable_vertex_does_not_get_a_fake_parent():
    graph = nx.Graph()
    graph.add_edge("a", "b", capacity=1)
    graph.add_node("z")
    with pytest.raises(ValueError, match="connected"):
        shortest_path_tree(graph, {("a", "b"): 1}, "a")


@pytest.mark.parametrize("iters,candidates", [(0, 1), (-1, 1), (1, 0), (1, -1)])
def test_reject_nonpositive_experiment_counts(iters, candidates):
    graph = io.load_graph(TINY)
    with pytest.raises(ValueError, match="positive integer"):
        run_mwu(graph, MwuParams(iters=iters, candidates=candidates, alpha=10))


def cluster_graph():
    graph = nx.complete_graph(list("abcd"))
    for u, v in graph.edges:
        graph[u][v]["capacity"] = 10 if {u, v} in ({"a", "b"}, {"c", "d"}) else 1
    return graph


def test_globally_rank_a_better_cut_from_a_later_tree():
    """A cheaper cut found by a later tree must outrank the first tree's best.

    Concatenating per-tree lists and slicing reports the first tree's cheapest cuts.
    """

    graph = cluster_graph()
    first = Tree("a", {"a": None, "c": "a", "d": "a", "b": "c"})
    later = Tree("a", {"a": None, "b": "a", "c": "a", "d": "c"})
    first_cuts = cuts.extract_tree_cuts(graph, first)
    assert first_cuts[0].capacity == 12
    candidates = first_cuts + cuts.extract_tree_cuts(graph, later)
    ranked = cuts.rank_cuts(candidates, top_k=1)
    assert ranked[0].capacity == 4
    assert ranked[0].nodes() == ["a", "b"]


def test_cut_complements_deduplicate_and_ties_are_deterministic():
    """A partition and its complement are one cut, however it was induced."""

    graph = cluster_graph()
    trees = [
        Tree("a", {"a": None, "b": "a", "c": "b", "d": "c"}),
        Tree("a", {"a": None, "c": "a", "d": "c", "b": "a"}),
        Tree("d", {"d": None, "c": "d", "b": "c", "a": "b"}),
    ]
    candidates = [cut for tree in trees for cut in cuts.extract_tree_cuts(graph, tree)]
    ranked = cuts.rank_cuts(candidates)
    partitions = [cut.nodes() for cut in ranked]
    assert len(partitions) == len(set(map(tuple, partitions))), "complements not merged"
    assert [cut.capacity for cut in ranked] == sorted(cut.capacity for cut in ranked)
    assert ranked == cuts.rank_cuts(list(reversed(candidates)))


@pytest.fixture
def saved_mixture():
    graph = nx.Graph()
    graph.add_weighted_edges_from([("a", "b", 1), ("b", "c", 2), ("c", "d", 3)], weight="capacity")
    payload = {
        "schema_version": mixture.SCHEMA_VERSION,
        "metric": mixture.METRIC,
        "graph_fingerprint": io.graph_fingerprint(graph),
        "iters": 2,
        "candidates": 1,
        "seed": 0,
        "alpha": 10,
        "eta": 0.6,
        "trees": [
            {"root": "a", "edges": [["a", "b"], ["b", "c"], ["c", "d"]], "count": 2, "prob": 1.0}
        ],
    }
    return graph, payload


@pytest.mark.parametrize(
    "field,value",
    [
        ("count", 0),
        ("count", -1),
        ("count", 1.5),
        ("count", True),
        ("count", 3),
        ("prob", 0),
        ("prob", -1),
        ("prob", float("nan")),
        ("prob", float("inf")),
        ("prob", 0.5),
        ("prob", True),
        ("root", "z"),
        ("edges", [["a", "b"], ["b", "c"]]),
        ("edges", [["a", "b"], ["b", "c"], ["a", "d"]]),
        ("edges", [["a", "b"], ["b", "c"], ["b", "c"]]),
        ("edges", [["a", "b"], ["c", "d"], ["d", "c"]]),
        ("edges", [["b", "a"], ["b", "c"], ["c", "d"]]),
        ("edges", [["a", "b"], ["b", "z"], ["z", "d"]]),
    ],
)
def test_invalid_saved_trees_are_rejected(saved_mixture, field, value):
    graph, original = saved_mixture
    payload = copy.deepcopy(original)
    payload["trees"][0][field] = value
    with pytest.raises(ValueError):
        mixture.validate_payload(graph, payload)


def test_reject_duplicate_trees_and_total_count_mismatch(saved_mixture):
    graph, payload = saved_mixture
    payload["trees"][0].update(count=1, prob=0.5)
    payload["trees"].append(copy.deepcopy(payload["trees"][0]))
    with pytest.raises(ValueError, match="Duplicate"):
        mixture.validate_payload(graph, payload)
    payload["trees"].pop()
    payload["trees"][0]["prob"] = 1
    with pytest.raises(ValueError):
        mixture.validate_payload(graph, payload)


def test_saved_graph_capacity_must_match(saved_mixture):
    graph, payload = saved_mixture
    graph["a"]["b"]["capacity"] = 2
    with pytest.raises(ValueError, match="fingerprint"):
        mixture.validate_payload(graph, payload)


def test_comments_and_blank_lines_are_ignored(tmp_path):
    path = tmp_path / "commented.edgelist"
    path.write_text("# header\n\na b 1\n\n# trailing\nb c 2\n")
    assert list(io.iter_edges(io.load_graph(path))) == [("a", "b", 1.0), ("b", "c", 2.0)]


def test_report_rebuild_reproduces_entire_report(tmp_path):
    runner = CliRunner()
    fit_out = tmp_path / "fit"
    rebuilt = tmp_path / "rebuilt"
    args = ["--graph", TINY]
    fitted = runner.invoke(
        app,
        ["fit", *args, "--iters", "12", "--candidates", "3", "--seed", "1", "--out", str(fit_out)],
    )
    assert fitted.exit_code == 0, fitted.output
    restored = runner.invoke(
        app, ["report", *args, "--mixture", str(fit_out / "mixture.json"), "--out", str(rebuilt)]
    )
    assert restored.exit_code == 0, restored.output
    assert json.loads((fit_out / "report.json").read_text()) == json.loads(
        (rebuilt / "report.json").read_text()
    )


def test_cli_prints_a_clear_input_error(tmp_path):
    path = tmp_path / "invalid.edgelist"
    path.write_text("a b nan\n")
    result = CliRunner().invoke(app, ["fit", "--graph", str(path)])
    assert result.exit_code == 2
    assert "capacity must be finite and positive" in result.output
