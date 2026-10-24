from __future__ import annotations

from pathlib import Path

import pytest

from ires.core import io


def test_loads_tiny_fixture(tiny_graph: object) -> None:
    assert io.load_graph(Path("data/tiny.edgelist")).number_of_edges() == 8


def test_parallel_edges_sum(tmp_path: Path) -> None:
    path = tmp_path / "g.edgelist"
    path.write_text("a b 1\nb a 2.5\nb c 1\n")
    graph = io.load_graph(path)
    assert graph["a"]["b"][io.CAPACITY_KEY] == pytest.approx(3.5)


def test_comments_and_blank_lines_ignored(tmp_path: Path) -> None:
    path = tmp_path / "g.edgelist"
    path.write_text("# header\n\na b 1\n\n# trailing\nb c 2\n")
    assert io.load_graph(path).number_of_edges() == 2


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("a b 1\nc d 1\n", "disconnected"),
        ("a b 0\n", "must be positive"),
        ("a b -3\n", "must be positive"),
        ("a a 1\n", "self-loop"),
        ("a b\n", "expected 'u v w'"),
        ("a b 1 2\n", "expected 'u v w'"),
        ("a b x\n", "not a number"),
        ("", "no edges"),
        ("# only a comment\n", "no edges"),
    ],
)
def test_rejects_bad_input(tmp_path: Path, body: str, message: str) -> None:
    path = tmp_path / "bad.edgelist"
    path.write_text(body)
    with pytest.raises(ValueError, match=message):
        io.load_graph(path)


def test_edge_key_is_canonical() -> None:
    assert io.edge_key("b", "a") == io.edge_key("a", "b") == ("a", "b")
