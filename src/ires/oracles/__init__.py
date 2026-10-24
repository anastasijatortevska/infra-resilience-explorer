"""Tree oracles for the MWU loop.

An oracle takes the graph, the current per-edge lengths and a random generator, and
returns a spanning tree. The loop asks for several candidates per iteration and keeps
the one with the lowest weighted cut-path score. None of them solves that objective;
they are candidate generators, as ``docs/mathematics.md`` sets out.
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
from .spt import random_root_spt, shortest_path_tree


class Oracle(Protocol):
    def __call__(
        self, graph: nx.Graph, lengths: Mapping[Edge, float], rng: np.random.Generator
    ) -> Tree: ...


ORACLES: dict[str, Oracle] = {
    "spt": random_root_spt,
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
    "random_root_spt",
    "shortest_path_tree",
]
