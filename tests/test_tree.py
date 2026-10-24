from __future__ import annotations

import pytest

from ires.core.tree import Tree


def test_from_edges_orients_by_bfs() -> None:
    tree = Tree.from_edges([("a", "b"), ("b", "c"), ("c", "d")], root="b")
    assert tree.depth == {"b": 0, "a": 1, "c": 1, "d": 2}
    assert tree.subtree_sizes() == {"b": 4, "a": 1, "c": 2, "d": 1}
    assert sorted(tree.subtree_nodes("c")) == ["c", "d"]


def test_subtree_sizes_matches_subtree_nodes() -> None:
    edges = [("r", "a"), ("r", "b"), ("a", "c"), ("a", "d"), ("d", "e")]
    tree = Tree.from_edges(edges, root="r")
    sizes = tree.subtree_sizes()
    for node in tree.nodes:
        assert sizes[node] == len(tree.subtree_nodes(node))


def test_postorder_places_children_first() -> None:
    tree = Tree.from_edges([("r", "a"), ("a", "b")], root="r")
    order = tree.postorder
    assert order.index("b") < order.index("a") < order.index("r")


@pytest.mark.parametrize(
    ("edges", "root", "nodes", "message"),
    [
        ([("a", "b"), ("c", "d")], "a", None, "spanning tree"),
        ([("a", "b"), ("b", "c"), ("a", "c")], "a", None, "spanning tree"),
        ([("a", "b")], "z", None, "not among"),
        ([("a", "b")], "a", ["a", "b", "c"], "spanning tree"),
        ([("a", "b"), ("b", "z")], "a", ["a", "b"], "outside the graph"),
    ],
)
def test_from_edges_rejects_non_trees(
    edges: list[tuple[str, str]], root: str, nodes: list[str] | None, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        Tree.from_edges(edges, root=root, nodes=nodes)


def test_root_must_have_no_parent() -> None:
    with pytest.raises(ValueError, match="no parent"):
        Tree(root="a", parent={"a": "b", "b": None})
