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
neither side size nor volume normalizes this ranking.

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
are not independent in the normalized update. There is no approximation,
convergence, or minimum-cut recovery guarantee for this candidate oracle.
This is an experimental heuristic, not an implementation of a theorem-backed
tree embedding or resilience algorithm.
