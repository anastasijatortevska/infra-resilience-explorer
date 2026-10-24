"""Tree oracles for the MWU loop.

An oracle takes the graph, the current per-edge lengths and a random generator, and
returns a spanning tree. The MWU loop asks for several candidates per iteration and
keeps the one with the lowest average congestion under the current length function.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

import networkx as nx
import numpy as np

from ..core.io import Edge
from ..core.tree import Tree
from .mpx import low_diameter_spanning_tree
from .mst import minimum_spanning_tree
from .spt import shortest_path_tree, shortest_path_tree_from


class Oracle(Protocol):
    def __call__(
        self, graph: nx.Graph, lengths: Mapping[Edge, float], rng: np.random.Generator
    ) -> Tree: ...


ORACLES: dict[str, Oracle] = {
    "spt": shortest_path_tree,
    "mst": minimum_spanning_tree,
    "ldd": low_diameter_spanning_tree,
}

DEFAULT_ORACLE = "spt"


def get_oracle(name: str) -> Oracle:
    try:
        return ORACLES[name]
    except KeyError:
        raise ValueError(
            f"unknown oracle {name!r}; available: {', '.join(sorted(ORACLES))}"
        ) from None


__all__ = [
    "DEFAULT_ORACLE",
    "ORACLES",
    "Oracle",
    "get_oracle",
    "low_diameter_spanning_tree",
    "minimum_spanning_tree",
    "shortest_path_tree",
    "shortest_path_tree_from",
]
