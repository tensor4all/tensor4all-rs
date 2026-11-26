# Streaming TreeTCI leaf-side enumeration (#1441)

The third global-search stage now merges sorted coordinate blocks with a
min-heap of sparse mixed-radix cursors. Each cursor borrows its retained
anchor and owns only the side's nonunit coordinates. The existing
32-configuration side limit bounds those coordinates to five. Equal heap
heads advance together, eliminating a whole-point seen set and preserving
the eager `BTreeSet`'s lexicographic order, including equal-residual ties.

Retaining all accepted points would reintroduce the same memory growth after
generation. The stage instead retains its strongest caller-limited pivots,
with stable ties, while evaluating every residual for `max_error`. Existing
batch sizes, thresholds, fallback gating and random-stream consumption are
unchanged. The generator is a private search component; there is no new
public API, dependency or numerical policy.

A first-generating-block filter, as used by TreeACI's already-streamed Guard,
would also avoid the seen set but would change TreeTCI's established point
order. Preserving that order makes the sorted union preferable here. The
TreeACI path was inspected for the related defect; it already streams and
truncates accepted candidates at each flush, so no cross-crate change is
needed. Neither implementation reaches through the other's private state.

The [acceptance evidence](../experiments/issue-1441/README.md) compares all 70
baseline/candidate cells: both box sizes and both stop rules on 11 seeds,
the reopened #1352 table on 11 seeds, and the two comb cases. Evaluated point
costs, trajectories and independent errors are identical. Small exhaustive
references and the 200-arm structural test cover ordering, overlaps and
storage; four injected faults fail the private regressions.

The storage guarantee concerns enumeration-generated points and accepted
pivots. Graph/side discovery metadata, input anchors, the target memo and
the approximation readout cache remain under their existing contracts.
No runtime speedup or peak-RSS reduction is claimed. The root README's
features and crate map remain accurate; the durable algorithm and storage
contract are recorded in the [termination design](../design/treetci-termination.md).
