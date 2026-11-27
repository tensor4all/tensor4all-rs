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
No runtime speedup or full-interpolation peak-RSS reduction is claimed.
The [isolated enumeration-storage RSS measurement](../experiments/issue-1441/README.md#peak-rss-on-the-motivating-point-enumeration-workload)
reports median 240.00 to 10.18 MiB, with matching points/checksums and all six
processes CLEAN. Its incidental generation-only timings are slower for the
streaming arm; this is disclosed without an end-to-end runtime conclusion.
`report_rss.py` now regenerates the RSS summary from the accepted raw records. The root README's
features and crate map remain accurate; the durable algorithm and storage
contract are recorded in the [termination design](../design/treetci-termination.md).


## Cross-provider review follow-up

The branch is synchronized with main `9ddffd17`. No production enumeration
code changed in this follow-up. The evidence now distinguishes the 70-cell
trajectory comparison from direct unit/search coverage of stage-3 eviction.
The PR's model disclosure also includes `openai/gpt-6-luna` for the Copilot CLI
comment contribution. The historical RSS controller and all raw data stay
pinned; summary regeneration does not rerun or reinterpret the measurement.
