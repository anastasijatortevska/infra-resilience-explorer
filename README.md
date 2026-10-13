Infra Resilience Explorer
=========================

This small Python experiment samples spanning-tree mixtures on undirected,
capacitated graphs. It reports a heuristic **cut-path score** for original edges
and the capacities of candidate fundamental cuts. These are not traffic
congestion or validated infrastructure criticality measures.

The exact definitions, a hand-checkable example, and the relationship to the
iterative update are in [the mathematical contract](docs/mathematics.md).
There is no approximation guarantee; the historical alpha formula is a tuning
default, not a proven bound.

Quick start
-----------

1) Install dependencies (Python 3.11+):
```
pip install -e .[dev]
```

2) Fit on the provided tiny example:
```
ires fit --graph data/tiny.edgelist --iters 20 --candidates 4 --seed 0 --out outputs/
```
This writes `outputs/mixture.json` and `outputs/report.json`.

3) Rebuild a report from a stored mixture:
```
ires report --graph data/tiny.edgelist --mixture outputs/mixture.json --out outputs/
```

Edge list format
----------------
Each line contains `u v w` (nodes are strings, `w` is a float capacity). Parallel edges are merged by summing capacities.

Outputs
-------
- `mixture.json`: the sampled trees and their probabilities.
- `report.json`: maximum expected cut-path score, high-scoring edges, and low-capacity candidate cuts (node lists are truncated to 30 for readability).
