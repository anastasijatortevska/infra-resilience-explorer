# Infra Resilience Explorer

A small Python experiment on undirected graphs with positive capacities. It
samples shortest-path spanning trees, uses multiplicative edge weights to
choose a mixture, and reports a **cut-path score** and low-capacity candidate
cuts. The name is historical: this is a synthetic graph-analysis experiment,
with no real infrastructure validation or theoretical approximation guarantee.

## Quick start

Python 3.11+ is required. These commands were verified in a new virtual
environment with Python 3.14.8; see [evaluation notes](docs/evaluation.md).

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
ires fit --graph data/tiny.edgelist --iters 20 --candidates 4 --seed 0 --out reports/fit
ires report --graph data/tiny.edgelist --mixture reports/fit/mixture.json --out reports/rebuilt
python -m pytest -q
ruff check . && ruff format --check . && mypy
```

Fit writes mixture.json and report.json; `--out` defaults to `reports/`. Report
validates the saved mixture and rebuilds the same complete report, including
floating-point values. The optional `--alpha` sets a positive update scale and
`--eta` the step size, defaulting to 0.6; only their ratio affects the update.
`--oracle` selects the candidate generator (`ires oracles` lists them). Outputs
record iterations, candidate count, seed, oracle, alpha and eta. The run-specific
values, the per-iteration history and the bounds below live in mixture.json, so
report.json is reproducible from it alone.

## Input and output example

Each line is `u v capacity`, with string node labels. Blank lines and lines
starting with # are ignored. For example:

```text
a b 2
b c 3
```

This path has one possible unrooted spanning tree. Its induced capacities are
2 and 3, and both edge scores are 1. A fit reports these substantive fields:

```json
{
  "metric": "cut_path_score",
  "max_expected_cut_path_score": 1.0,
  "scored_edges": [
    {"edge": ["a", "b"], "capacity": 2.0, "expected_cut_path_score": 1.0},
    {"edge": ["b", "c"], "capacity": 3.0, "expected_cut_path_score": 1.0}
  ],
  "candidate_cuts": [
    {"edge": ["a", "b"], "capacity": 2.0, "nodes": ["a"], "side_size": 1, "truncated": false},
    {"edge": ["b", "c"], "capacity": 3.0, "nodes": ["c"], "side_size": 1, "truncated": false}
  ],
  "bounds": {"achieved": 1.0, "certified_lower": 1.0, "dual_estimate": 1.0}
}
```

This excerpt omits parameters and mixture records. Cut partitions use the
smaller side; equal-size sides use sorted labels. Node lists are truncated to
30 with full side size retained.

Inputs must be connected with at least two vertices, no self-loops, and finite
positive capacities. Parallel input edges are merged by capacity sum and the
merged value is checked. Nonpositive iteration/candidate counts are rejected.
Saved mixtures must cover every vertex with real tree edges and consistent
positive counts/probabilities. A fingerprint binds the mixture to node labels
and capacities. Old files without schema/metric metadata must be regenerated.

## What the algorithm measures

Remove tree edge f to obtain a partition. Its induced capacity C_T(f) is the
sum of original capacities crossing that cut. For an original edge e=(u,v):

```text
s_T(e) = sum(C_T(f) for f on the u-v tree path) / capacity(e)
expected_score(e) = sum(probability(T) * s_T(e) for T in the mixture)
F = max(expected_score(e) for e in the original graph)
```

Lower F is the target. This **is not routing congestion**: for a unit triangle
whose tree is a-b-c, the unused edge a-c has score 4, although it carries no
routed demand. High-scoring edges have no established infrastructure-criticality
interpretation.

Each iteration normalizes weights q, sets lengths proportional to
q(e)/capacity(e), samples roots, and generates shortest-path trees. It selects
the candidate minimizing sum(q(e)*s_T(e)). The positive update raises attention
on high-scoring edges. One selected tree per iteration forms an equally
weighted sample mixture; repeated unrooted trees are aggregated by count.

Weights use normalized logs and lengths receive a common rescaling. Invalid or
unrepresentable intermediate values fail explicitly. Alpha's default
10*log2(n+1) is a historical tuning choice with no proven bound. The candidate
oracle does not generally minimize the weighted score over all spanning trees.
See [the mathematical contract](docs/mathematics.md) for the derivation,
reference examples and floating-point limitations.

Cuts are a separate output: all selected trees' fundamental cuts are globally
sorted by absolute capacity, complements are deduplicated, and ties use side
size, labels, then inducing edge. Side size and volume do not normalize this
score. A cut is identified by its partition rather than by the edge that induced
it, and `mixture_mass` records the probability of the trees that agree on it.
Capacities come from C_T directly, since a tree edge's induced capacity is by
definition the capacity of the cut it induces, which makes extraction O(n) per
tree rather than O(nm); the identity is asserted against a direct edge scan in
the tests.

## Bounds on the objective

A run brackets F*, the smallest achievable maximum expected score:

| Field | Meaning | Certificate? |
| --- | --- | --- |
| `achieved` | max expected score of the emitted mixture | yes, an upper bound: the mixture attains it |
| `dual_estimate` | LP dual value at the learned weights | **no**: valid only for an exact minimizing oracle |
| `certified_lower` | minimum spanning tree weight under those weights | yes, a lower bound, but at most 1 by construction |

`achieved / dual_estimate` is the informative measure of remaining slack.
`certified_lower` is honest but weak: it keeps only tree edges, where the score
is smallest, so it is a floor rather than a tight certificate. The derivations,
and why a tight lower bound is not available here, are in
[the mathematical contract](docs/mathematics.md).

## Candidate generators

`--oracle` selects one; none of them minimizes the per-iteration objective.

- `spt` (default): shortest-path tree from a uniformly random root.
- `mst`: minimum spanning tree under the current lengths. Deterministic, so
  extra candidates within one iteration are identical.
- `ldd`: recursive low-diameter decomposition by random shifts (Miller, Peng and
  Xu, SPAA 2013).

`python scripts/compare_oracles.py <graph>` prints each one's bounds, average
stretch and runtime. On `data/grid20.edgelist` (400 nodes, 30 iterations) `ldd`
has the lowest average stretch, 4.72 against 6.07 for `spt` under the resistance
metric 1/c(e), and still the *larger* maximum expected score, 887.9 against
508.0. Low stretch is not a proxy for this objective; the comparison is measured
rather than assumed.

## Reproducible evaluation

```sh
python scripts/evaluate.py --iters 80 --candidates 8 --seeds 0 1 2 --out evaluation/results.json
```

The deterministic eight-vertex families are a unit path, a unit cycle, and two
four-vertex cliques with internal capacity 3 joined by two capacity-0.25 links.
Baselines are a fixed SPT rooted at the first sorted label and an equal-weight
mixture of 640 SPT samples with uniformly sampled roots. Static lengths are
proportional to inverse capacity. Iteration also makes 640 SPT calls but selects
80 trees and evaluates all 640 candidates' scores. SPT budgets match the sampled
baseline; total computation differs.

Mean maximum expected scores over seeds 0, 1 and 2 (lower is better):

| Graph | Fixed SPT | Sampled SPT mixture | Iterative mixture |
| --- | ---: | ---: | ---: |
| Path | 1.0000 | 1.0000 | 1.0000 |
| Cycle | 14.0000 | 4.8313 | 3.7500 |
| Two clusters | 37.0000 | 37.0000 | 75.5125 |

All methods recover a capacity-1 cut on the path and a capacity-2 cut on the
cycle. On the cluster graph, both baselines' best candidate cut is 9, while
iteration finds 0.5, the exact minimum. Iteration has **worse F** there:
recovering a smaller cut is distinct from optimizing the cut-path score.
There is no general improvement guarantee. The path correctly yields one
unique tree for every method; diversity is not required.

The saved run uses Python 3.14.8, NetworkX 3.7 and NumPy 2.5.4 on Linux x86_64.
Full configuration, graph edges, mixtures, per-seed runtimes and budgets are in
[results.json](evaluation/results.json). Dependency pins and detailed
runtime/unique-tree results are in [evaluation notes](docs/evaluation.md).
Identical software, input, parameters and seed reproduce substantive results;
wall-clock timings vary. These small measurements are not a performance study.

## Scope and limitations

- Synthetic small graphs only; no claims about actual infrastructure resilience.
- A heuristic score and restricted SPT candidates, with no approximation,
  convergence, monotonic improvement or minimum-cut recovery guarantee.
- Candidate cuts can miss the global minimum. The exact comparison checks
  absolute minimum cut, not normalized sparsity or general resilience.
- Double-precision arithmetic: log weighting removes common-scale failures,
  but extreme relative weights/capacities can still be rejected or lose accuracy.
- Deterministic ordering within a recorded environment; dependency versions can
  change shortest-path ties.
- `certified_lower` cannot exceed 1, so for graphs where the score is large it
  confirms very little. Only `achieved` is a tight certificate.

Tests cover hand calculations, exhaustive four-node spanning trees, LCA/path
references, input failures, saved-tree validation, global cut ranking, bounds
consistency, oracle validity, numerical stability and reproduction.
[CI](.github/workflows/tests.yml) runs pytest, Ruff and mypy on Python 3.11
and 3.14. `scripts/gen_graph.py` generates larger grid, random and scale-free
instances for the oracle comparison.
