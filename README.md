Infra Resilience Explorer
=========================

Learns a distribution over spanning trees that exposes weak points in an undirected,
capacitated network, and reports how far that distribution is from optimal.

The problem
-----------

Route all traffic along spanning trees drawn from a distribution `λ`. A tree `T`
induces, on each of its edges `f`, the capacity `c_T(f)` of the cut that removing `f`
would create, and loads each original edge `e = (u, v)` with

    cong_T(e) = (sum of c_T over the T-path from u to v) / cap(e)

The goal is the distribution minimizing the worst expected load,

    alpha* = min over λ of max over e of E_λ[cong_T(e)]

This is the congestion-minimization objective of Räcke's oblivious routing framework,
specialized to spanning trees. `ires` attacks it with multiplicative weights: maintain
a probability vector `p` over edges, ask an oracle for a tree that is cheap under `p`,
reweight the edges that the tree overloads, repeat.

What comes out
--------------

- **Bounds** on `alpha*`, so a run says how good its answer is:
  - `alpha_hat`: *rigorous upper bound*. The emitted mixture achieves it, so the
    optimum is no larger.
  - `dual_objective`: the LP dual value at the learned `p`. The useful estimate of
    remaining slack (`estimated_gap = alpha_hat / dual_objective`), but a lower bound
    only for an exact minimizing oracle; divide by the oracle's approximation factor
    to certify it.
  - `certified_lower`: rigorous lower bound, from `cong_T(e) >= 1` on tree edges and
    an exact minimum spanning tree computation. Weak, though: it cannot exceed 1 by
    construction, so it is a sanity floor rather than a tight certificate.
- **Critical edges**: original edges with the highest expected congestion.
- **Bottleneck cuts**: the cheapest distinct cuts induced by tree edges of the
  mixture, deduplicated across trees and reported with the probability mass of the
  trees that agree on each one.
- **History**: per-iteration objective, max congestion and running `alpha`, for
  checking whether the run actually converged.

Quick start
-----------

```
pip install -e ".[dev]"

# Fit on the toy fixture
ires fit --graph data/tiny.edgelist --iters 40 --candidates 4 --seed 0 --out reports/

# Or on a 400-node grid
ires fit --graph data/grid20.edgelist --iters 60 --out reports/

# Rebuild the report from a stored mixture
ires report --graph data/tiny.edgelist --mixture reports/mixture.json --out reports/
```

Tuning
------

| Flag | Default | Notes |
| --- | --- | --- |
| `--iters` | 80 | MWU rounds. Check `history` in the report to see if it plateaued. |
| `--candidates` | 8 | Trees drawn per round; the cheapest under the current `p` wins. |
| `--eta` | `sqrt(log m / iters)` | Step size. Multiplicative-weights regret over `m` edges and `T` rounds is minimized near this value, so it must shrink as `--iters` grows; a fixed step either under-mixes or oscillates. |
| `--alpha` | `10*log2(n+1)` | Width of the update, i.e. the congestion scale the run targets. |
| `--oracle` | `spt` | See below. |

Oracles
-------

`ires oracles` lists them. All three return a spanning tree given the current lengths:

- `spt`: shortest-path tree from a random root. Cheap, no approximation guarantee.
- `mst`: minimum spanning tree under the current lengths. Deterministic, so
  `--candidates` buys nothing.
- `ldd`: recursive low-diameter decomposition using the exponential-shift clustering
  of Miller, Peng and Xu. Lower average stretch than `spt` (measurably: 5.0 vs 6.1 on a
  15x15 unit grid under the resistance metric), which is the property low-stretch
  spanning tree constructions target.

Lower stretch does not automatically mean lower congestion. On the grids tested, `spt`
still reaches a smaller `alpha_hat` than `ldd` despite the worse stretch;
`scripts/compare_oracles.py` prints the comparison.

```
python scripts/gen_graph.py grid --rows 25 --cols 25 --out /tmp/grid.edgelist
python scripts/compare_oracles.py /tmp/grid.edgelist --iters 30
```

Scaling
-------

A tree edge's induced capacity `c_T` *is* the capacity of the cut it induces, so cut
capacities are read out of `c_T` instead of rescanning the edge list once per tree edge.
Extracting all `n-1` cuts of a tree is then O(n) rather than O(n*m). Measured on one tree
(`scripts/compare_oracles.py` graphs, this machine):

| graph | nodes | edges | cuts from `c_T` | per-edge rescan |
| --- | --- | --- | --- | --- |
| 40x40 grid | 1600 | 3120 | 2.7 ms | 2.0 s |
| random | 4989 | 15000 | 11.8 ms | 38.1 s |

Node sets are materialized only for the cuts that actually get reported. A 20-iteration
fit on the 4989-node graph takes about 13 s end to end.

Edge list format
----------------

Each line is `u v w`: node identifiers (kept as strings) and a positive float capacity.
Blank lines and `#` comments are skipped, and parallel edges are merged by summing
capacities. Self-loops, non-positive capacities, and disconnected or empty graphs are
rejected: spanning-tree routing has no meaning on a disconnected graph, and a zero
capacity makes congestion infinite.

Development
-----------

```
ruff check . && ruff format --check . && mypy && pytest -q
```

References
----------

- H. Räcke. *Optimal hierarchical decompositions for congestion minimization in
  networks.* STOC 2008.
- N. Alon, R. Karp, D. Peleg, D. West. *A graph-theoretic game and its application to
  the k-server problem.* SIAM J. Computing, 1995. (low-stretch spanning trees)
- G. Miller, R. Peng, S. C. Xu. *Parallel graph decompositions using random shifts.*
  SPAA 2013. (the `ldd` oracle's decomposition)
- S. Plotkin, D. Shmoys, É. Tardos. *Fast approximation algorithms for fractional
  packing and covering problems.* Math. of OR, 1995. (the multiplicative-weights form
  used here)
