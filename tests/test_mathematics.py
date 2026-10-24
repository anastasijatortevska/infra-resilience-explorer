import itertools

import networkx as nx
import pytest
from conftest import cycle_graph, random_capacitated_graph, unit_lengths

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
        walk: str | None = u
        while walk is not None:
            ancestors.add(walk)
            walk = tree.parent[walk]
        node = v
        while node not in ancestors:
            parent = tree.parent[node]
            assert parent is not None
            node = parent
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


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_induced_capacity_equals_cut_capacity_on_random_graphs(seed):
    """C_T on a tree edge is exactly the capacity of the cut it induces.

    The identity that lets cut extraction read capacities out of C_T rather than
    rescanning the edge list once per tree edge.
    """

    graph = random_capacitated_graph(60, seed=seed)
    tree = shortest_path_tree(graph, unit_lengths(graph), min(graph.nodes))
    lca = LCA(tree)
    c_t = metrics.compute_tree_capacities(graph, tree, lca)

    assert len(c_t) == graph.number_of_nodes() - 1
    for (_, child), value in c_t.items():
        reference = cuts.cut_capacity(graph, set(tree.subtree_nodes(child)))
        assert value == pytest.approx(reference)


@pytest.mark.parametrize("seed", [0, 1])
def test_score_is_at_least_one_on_tree_edges(seed):
    """An edge crosses its own fundamental cut, so s_T(f) >= 1 for every tree edge.

    The inequality the certified lower bound rests on.
    """

    graph = random_capacitated_graph(60, seed=seed)
    tree = shortest_path_tree(graph, unit_lengths(graph), min(graph.nodes))
    lca = LCA(tree)
    c_t = metrics.compute_tree_capacities(graph, tree, lca)
    scores = metrics.compute_edge_scores(graph, tree, lca, c_t)
    for parent, child in tree.edges():
        assert scores[io.edge_key(parent, child)] >= 1.0 - 1e-12


@pytest.mark.parametrize("n", [4, 7, 12])
def test_cycle_has_exact_known_scores(n):
    """Unit-capacity cycle: C_T = 2 on every tree edge, max score 2(n-1).

    Each tree edge's cut is crossed by itself and by the single omitted edge, which
    in turn routes along all n-1 tree edges.
    """

    graph = cycle_graph(n)
    tree = shortest_path_tree(graph, unit_lengths(graph), "v000")
    lca = LCA(tree)
    c_t = metrics.compute_tree_capacities(graph, tree, lca)
    scores = metrics.compute_edge_scores(graph, tree, lca, c_t)

    assert all(value == pytest.approx(2.0) for value in c_t.values())
    (omitted,) = set(scores) - {io.edge_key(p, c) for p, c in tree.edges()}
    assert scores[omitted] == pytest.approx(2.0 * (n - 1))
    assert max(scores.values()) == pytest.approx(2.0 * (n - 1))


def test_scores_are_root_independent():
    """C_T and the scores depend on the undirected tree only, not on the root."""

    graph = random_capacitated_graph(40, seed=5)
    tree = shortest_path_tree(graph, unit_lengths(graph), min(graph.nodes))
    reference = None
    for root in sorted(graph.nodes)[:5]:
        rerooted = Tree.from_edges(tree.edges(), root=root, nodes=graph.nodes)
        lca = LCA(rerooted)
        c_t = metrics.compute_tree_capacities(graph, rerooted, lca)
        scores = metrics.compute_edge_scores(graph, rerooted, lca, c_t)
        if reference is None:
            reference = scores
        else:
            assert scores == pytest.approx(reference)
