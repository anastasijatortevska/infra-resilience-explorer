Infra Resilience Explorer
=========================

This tool learns a mixture of spanning trees that highlight weak points in an undirected, capacitated network. It reports:
- **Critical edges**: original edges with high expected congestion when traffic is routed along the learned tree mixture.
- **Bottleneck cuts**: node partitions induced by tree edges whose removal disconnects a sparse part of the graph.

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
- `report.json`: summary with critical edges and top bottleneck cuts (node lists are truncated to 30 for readability).
