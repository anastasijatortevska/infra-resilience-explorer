# Mathematical contract

The input is a connected, simple, undirected graph G=(V,E), with at least two
vertices and finite positive capacities c(e). Input parallel edges are merged
by summing capacities. A tree T is a spanning tree using edges of G.

For a tree edge f, removing f partitions V into S_f and its complement. Its
**induced capacity** is C_T(f) = sum of c(e) over original edges crossing that
partition. It is also the load on f if each original edge e=(u,v) supplies demand
c(e), routed on the unique u-v path in T. The implementation adds c(e) at both
endpoints and subtracts 2c(e) at their lowest common ancestor, then accumulates
subtrees from leaves to root. An original edge contributes once to a subtree
sum exactly when one endpoint is inside and the other is outside.

For an original edge e=(u,v), the retained **cut-path score** is

    s_T(e) = (sum of C_T(f) over f on path_T(u,v)) / c(e).

It is dimensionless and unchanged by multiplying all capacities by the same
positive constant. It is not standard routing congestion: under the demands
above, load/capacity on original edges would be C_T(f)/c(f) for tree edges and
zero for unused edges. For a unit-capacity triangle with tree a-b-c, C_T(ab)=2
and C_T(bc)=2. The score of ac is 4, although ac carries no routed demand.
The project therefore makes no traffic or infrastructure criticality claim.

A nonuniform example: c(ab)=2, c(bc)=3, c(ac)=5, with the same tree. The induced
capacities are 7 and 8; edge scores are 7/2, 8/3, and (7+8)/5=3. Appending bridge
c-d with capacity 4 gives C_T(cd)=4 and score(cd)=1. On any input that is itself
a tree, every original edge has score 1, regardless of its positive capacity.

For a mixture with probabilities p_i, the expected score is
bar_s(e)=sum_i p_i s_Ti(e). The experimental target is to minimize
F(p)=max_e bar_s(e). This is distinct from maximizing or minimizing cut capacity.
Candidate cuts are ranked by absolute crossing capacity, smallest first;
neither side size nor volume normalizes this ranking. A tree edge's induced
capacity C_T(f) is by definition the capacity of the cut that removing f induces,
so ranking reads capacities out of C_T rather than rescanning the edge list per
tree edge: O(n) per tree instead of O(nm). The identity is asserted against a
direct scan in the tests. Cuts are identified by the partition, not the inducing
edge, so a partition and its complement are one cut however many trees induce it;
`mixture_mass` accumulates the probability of the trees that induce it.

At iteration t, normalized edge weights q_t define lengths ell_t(e)=q_t(e)/c(e).
Random roots generate shortest-path-tree candidates under those lengths. Among
the sampled candidates, the algorithm minimizes J_t(T)=sum_e q_t(e)s_T(e).
Lengths influence which candidates are available; shortest paths do not solve
the weighted tree-score objective in general. Expanding that objective gives
J_t(T)=sum_f C_T(f) sum_{e: f in path_T(e)} q_t(e)/c(e), which is not simply the
sum of shortest-path distances.

The multiplicative update is q_{t+1}(e) proportional to
q_t(e) exp(eta*s_T(e)/alpha). Its positive sign shifts attention toward edges
with high scores, while candidate selection minimizes the weighted score.
The original exponent included a common -eta term; it cancels on normalization.
The mixture averages one selected tree per iteration, aggregating identical
unrooted trees by count. It need not improve monotonically or contain multiple
trees: all shortest-path candidates may coincide, or one may keep winning.

Alpha is a positive update scale. The historical default 10*log2(|V|+1) has no
proved bound here. Eta/alpha controls relative weight changes; these parameters
are not independent in the normalized update. Only their ratio enters the update,
so `--eta` and `--alpha` are one degree of freedom, not two. `suggested_eta`
reports sqrt(log|E| / iters), the step a multiplicative-weights regret analysis
would pick; it is offered, not imposed, because that analysis assumes an oracle
that minimizes the per-round objective, which none of the candidate generators
does. There is no approximation, convergence, or minimum-cut recovery guarantee
for these candidate oracles. This is an experimental heuristic, not an
implementation of a theorem-backed tree embedding or resilience algorithm.

Candidate generators are interchangeable (`--oracle`). `spt` draws a
shortest-path tree from a uniformly random root; `mst` takes a minimum spanning
tree under the current lengths and is deterministic, so extra candidates in one
iteration are identical; `ldd` contracts a random-shift low-diameter
decomposition recursively (Miller, Peng and Xu, SPAA 2013). On the grids in
`scripts/compare_oracles.py`, `ldd` has lower average stretch than `spt` under
the fixed resistance metric 1/c(e) and still reaches a *larger* maximum expected
score, so low stretch is not a proxy for this objective. None of the three
minimizes J_t.

## What the run can and cannot certify

Write F(p) = max_e bar_s(e) for a mixture p, and F* for its minimum over all
distributions of spanning trees. Three numbers appear under `bounds`.

`achieved` is max_e bar_s(e) for the emitted mixture. Because that mixture is
explicit and attains the value, F* <= achieved. This is a certificate.

`certified_lower` is a certificate in the other direction. For any probability
vector q over edges and any distribution p,

    max_e bar_s(e) >= sum_e q(e) bar_s(e) = E_p[sum_e q(e) s_T(e)]
                   >= min_T sum_e q(e) s_T(e),

so every q lower-bounds F*. Bounding the inner minimum without an oracle uses
s_T(f) = C_T(f)/c(f) >= 1 for f in T, which holds because f crosses its own
fundamental cut:

    sum_e q(e) s_T(e) >= sum_{f in T} q(f) >= min over spanning trees of sum_{f in T} q(f),

and that last quantity is the minimum spanning tree weight under q, computed
exactly. The run reports the largest value seen over its iterations. The bound is
weak by construction: discarding the non-tree edges, which carry the large
scores, leaves at most sum_e q(e) = 1. It is a floor that confirms the pipeline is
not reporting something impossible, not a tight certificate. A tight bound would
need to bound s_T(e) below on non-tree edges, which is a minimum-weighted-stretch
problem and is not solved here.

`dual_estimate` is min over the trees the oracle produced of sum_e q(e) s_T(e) at
the final q. It is the practically informative number, since
achieved / dual_estimate measures the remaining slack, but it is **not** a
certificate: the derivation above needs the exact minimum over all spanning
trees, and a heuristic oracle returns a value that can exceed it. An oracle with a
proved approximation factor c would certify dual_estimate / c. It is reported with
`dual_certified: false`.

Numerically, weights are stored as logs. Each update adds
(eta/alpha)*(s_T(e)-max_f s_T(f)), then subtracts the largest resulting log
weight. Both subtractions remove only a common scale, preserving the normalized
update and avoiding the old common exponential decay. A softmax gives q_t.
Lengths are computed in logs and divided by their largest value: multiplying
all lengths by the same positive constant preserves shortest paths.

Capacity sums use `math.fsum`; tree distances subtract ancestor prefixes before
adding the two sides. These reduce rounding and avoid unnecessary intermediate
overflow, but are still double-precision calculations. Nonfinite scores/sums,
nonpositive computed capacities, and unrepresentable relative weights/lengths
are rejected. Extreme capacity ranges can still suffer cancellation; this is
not an arbitrary-precision solver. Reproducibility means the same input,
parameters, seed, and software environment; it is not a promise of identical
ties across dependency versions or machines.
