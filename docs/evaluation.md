# Evaluation evidence

Run from the repository root with Python 3.14.8:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r evaluation/requirements.txt -e '.[dev]'
python scripts/evaluate.py --iters 80 --candidates 8 --seeds 0 1 2 --out evaluation/results.json
```

The pinned environment was installed in a second clean virtual environment and
the command rerun. All 27 result rows, stored trees, configuration and graph
definitions matched the first run exactly after excluding runtime. The saved
artifact contains the clean-environment rerun. Do not compare timing values for
exact equality. The pins describe the measured Python 3.14 environment; normal
installation resolves compatible versions for other supported Python versions.

Environment: Linux 7.2.9-arch1-1, x86_64, glibc 2.44; Python 3.14.8; NetworkX
3.7; NumPy 2.5.4; Typer 0.27.3; Rich 15.0.0. All package versions are recorded
in results.json and the dependency file. Alpha is 10*log2(9), approximately
31.6993, for every family; eta is 0.6.

## Graphs and budgets

Each family has eight vertices. Path and cycle capacities are 1. The cluster
graph contains two four-vertex cliques with internal capacities 3, joined by
two links with capacity 0.25. The complete edge lists and fingerprints are
saved so the measured inputs are inspectable.

| Method | SPT calls | Candidate score evaluations | Selected samples |
| --- | ---: | ---: | ---: |
| Fixed SPT | 1 | 0 | 1 |
| Uniform SPT samples | 640 | 0 | 640 |
| Iterative | 640 | 640 | 80 |

The fixed root is the first sorted node label. The sampled baseline draws roots
uniformly with replacement at static lengths proportional to inverse capacity.
Each sample has equal probability; distinct unrooted trees need not be uniform.
The iterative method uses the same number of SPT calls, with changing lengths
and score-based selection. Both random methods use the recorded seed and draw
640 roots, but only iteration evaluates scores for candidate selection.

Runtime is measured with perf_counter around each method's tree generation,
selection and shared report recomputation. Each method includes two summary
passes, each evaluating one score vector per unique tree; their exact counts
are recorded separately. Graph construction and the exact reference minimum
cut are outside the timer. No warmup, repeated timing trials or machine-load
control is performed. These single-process, tiny-graph times illustrate the
computation performed; they do not establish a performance benchmark.

## Observations

Scores and runtimes are means over seeds 0, 1 and 2; unique-tree columns show
the range over those seeds. Lower maximum expected cut-path score is better.

| Graph | Method | Max expected score | Unique trees | Mean seconds | Best candidate / exact cut |
| --- | --- | ---: | ---: | ---: | ---: |
| Path | Fixed | 1.0000 | 1 | 0.002496 | 1 / 1 |
| Path | Sampled | 1.0000 | 1 | 0.065438 | 1 / 1 |
| Path | Iterative | 1.0000 | 1 | 0.109947 | 1 / 1 |
| Cycle | Fixed | 14.0000 | 1 | 0.000800 | 2 / 2 |
| Cycle | Sampled | 4.8313 | 7 | 0.068167 | 2 / 2 |
| Cycle | Iterative | 3.7500 | 7–8 | 0.118648 | 2 / 2 |
| Clusters | Fixed | 37.0000 | 1 | 0.001114 | 9 / 0.5 |
| Clusters | Sampled | 37.0000 | 8 | 0.082945 | 9 / 0.5 |
| Clusters | Iterative | 75.5125 | 4 | 0.152725 | 0.5 / 0.5 |

The path sanity check gives score 1 and one unique tree in every run. Iteration
improves the observed cycle score over both baselines. On the cluster graph it
worsens the score in every seed, yet exposes the capacity-0.5 cluster cut that
both baselines miss. These are distinct outputs with distinct objectives.
The restricted shortest-path oracle has no guarantee of finding the best
weighted-score tree, and these outcomes do not justify a general superiority
claim. A run with one tree is not by itself a bug.

The exact reference uses NetworkX's Stoer–Wagner algorithm. A test checks its
answer against exhaustive partition enumeration for all three fixtures. This
checks absolute global minimum cut only. It does not validate normalized
sparsity, traffic congestion, infrastructure criticality or resilience.

The full evaluation is small and uses three fixed seeds, with no parameter
search or unseen real-world data. Repeated seeds are reproducibility checks,
not statistically independent graph samples. Runtimes vary, and floating-point
ties may change with dependency versions.
