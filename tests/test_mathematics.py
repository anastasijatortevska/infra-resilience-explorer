import itertools

import networkx as nx
import pytest

from ires.core import cuts, io, metrics
from ires.core.lca import LCA
from ires.core.tree import Tree
from ires.oracles.spt import shortest_path_tree


@pytest.mark.parametrize(
    "edges,parent,capacities,scores",
    [
        (
            [("a", "b", 2), ("b", "c", 3)],
            {"a": None, "b": "a", "c": "b"},
            {("a", "b"): 2, ("b", "c"): 3},
            {("a", "b"): 1, ("b", "c"): 1},
        ),
        (
            [("a", "b", 2), ("b", "c", 3), ("a", "c", 5)],
            {"a": None, "b": "a", "c": "b"},
            {("a", "b"): 7, ("b", "c"): 8},
            {("a", "b"): 3.5, ("b", "c"): 8 / 3, ("a", "c"): 3},
        ),
        (
            [("a", "b", 2), ("b", "c", 3), ("a", "c", 5), ("c", "d", 4)],
            {"a": None, "b": "a", "c": "b", "d": "c"},
            {("a", "b"): 7, ("b", "c"): 8, ("c", "d"): 4},
            {("a", "b"): 3.5, ("b", "c"): 8 / 3, ("a", "c"): 3, ("c", "d"): 1},
        ),
    ],
)
def test_hand_computed_metrics(edges, parent, capacities, scores):
    graph = nx.Graph()
    graph.add_weighted_edges_from(edges, weight="capacity")
    tree = Tree("a", parent)
    lca = LCA(tree)
    actual = metrics.compute_tree_capacities(graph, tree, lca)
    assert actual == pytest.approx(capacities)
    for p, child in tree.edges():
        subset = set(tree.subtree_nodes(child))
        assert actual[p, child] == pytest.approx(cuts.cut_capacity(graph, subset))
    assert metrics.compute_edge_scores(graph, tree, lca, actual) == pytest.approx(scores)


def test_lca_and_path_sums_against_simple_references():
    # Branching and depth differences exercise both lifting stages.
    tree = Tree(
        "a", {"a": None, "b": "a", "c": "a", "d": "b", "e": "b", "f": "d", "g": "c", "h": "g"}
    )
    weights = {edge: (i + 1) / 2 for i, edge in enumerate(tree.edges())}
    lca = LCA(tree)
    lca.set_edge_weights(weights)
    reference = nx.Graph()
    reference.add_edges_from(tree.edges())
    for u, v in itertools.product(tree.nodes, repeat=2):
        ancestors = set()
        node = u
        while node is not None:
            ancestors.add(node)
            node = tree.parent[node]
        node = v
        while node not in ancestors:
            node = tree.parent[node]
        assert lca.lca(u, v) == node
        path = nx.shortest_path(reference, u, v)
        expected = sum(
            weights.get((a, b), weights.get((b, a), 0)) for a, b in itertools.pairwise(path)
        )
        assert lca.dist(u, v) == pytest.approx(expected)


def test_induced_capacities_on_all_small_complete_graph_trees():
    graph = nx.complete_graph(list("abcd"))
    for i, (u, v) in enumerate(graph.edges):
        graph[u][v]["capacity"] = i + 0.25
    for edges in itertools.combinations(graph.edges, 3):
        candidate = nx.Graph()
        candidate.add_nodes_from(graph)
        candidate.add_edges_from(edges)
        if not nx.is_tree(candidate):
            continue
        parent = {"a": None, **{v: u for u, v in nx.bfs_edges(candidate, "a")}}
        tree = Tree("a", parent)
        actual = metrics.compute_tree_capacities(graph, tree, LCA(tree))
        for p, child in tree.edges():
            assert actual[p, child] == pytest.approx(
                cuts.cut_capacity(graph, set(tree.subtree_nodes(child)))
            )


def test_shortest_path_tree_changes_with_lengths():
    graph = nx.Graph()
    graph.add_weighted_edges_from([("a", "b", 2), ("b", "c", 3), ("a", "c", 5)], weight="capacity")
    lengths = {io.edge_key(u, v): 1.0 for u, v in graph.edges}
    lengths["a", "c"] = 100
    tree = shortest_path_tree(graph, lengths, "a")
    assert {io.edge_key(*e) for e in tree.edges()} == {("a", "b"), ("b", "c")}
    lengths["a", "c"] = 0.5
    changed = shortest_path_tree(graph, lengths, "a")
    assert {io.edge_key(*e) for e in changed.edges()} == {("a", "b"), ("a", "c")}
