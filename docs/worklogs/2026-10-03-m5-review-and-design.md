# M5 evidence review and engine-seam design

## Decisions

The split selector remains undecided. A forced root split with uncapped,
from-scratch child interpolation cannot establish a selector for recursive
capped pQTCI. Follow-up investigation therefore separates the placement of a
site-carrying junction from the fit between a function and its tree topology.
The local investigation is maintained on `investigate/m5-split-oracle`;
its data are not promoted to an implemented capability on the patching branch.

Oracle parameter counts describe collapsed active-child networks with fixed
local dimensions set to one. The driver reattaches full-dimension fixed-site
indices using one-hot factors before storing a patch. Oracle counts therefore
do not establish stored driver tensor size, memory savings, or its stopping
decisions.

The [patch-size plan](../design/tree-pqtci-patch-size-bounds.md) preserves the
user's decisions: count one generalized bit per active site, and distinguish
an accepted patch that misses its allowance from a certified result. Spatial
size and nonbinary-site notices remain deferred. Unresolved capped-outcome
semantics and option naming remain proposals rather than adopted contracts.

The [edge-pivot proposal](../design/tree-interpolation-edge-pivots.md) keeps
selected side coordinates at the interpolation owner. Projecting the union
of joined seed points cannot recover the selected sets of a particular edge.
Optional collection documents its storage cost and avoids adding that cost
to ordinary interpolation. No production API or selector is implemented by
this design work.

## Verification conclusions and constraints

The original oracle raw data are preserved on the investigation branch.
A Codex reviewer subagent reproduced the corrected tables and headline
numeric claims; those findings remain limited to the original single-root
experiment. The follow-up protocol fixes topology, initial-pivot variation,
cost and accuracy comparisons before measurement. The fixed follow-up
completed all four root oracles and their required controls in 118.90
minutes; its evidence is retained on that branch.

A later Claude review (2026-10-04) re-verified the manifest, the source and
binary hashes, and the byte-for-byte regeneration of the derived files, and
found the analyzer's printed set order to vary between runs (the
postprocessing script now pins the hash seed). It also corrected the
interpretation: `hub_x03` (degree 4) and the scale backbone (degree 4-5)
exceed the degree-three limit of practical trees. The x03 counterexample to
E1's band and the scale suitability result are therefore not selector
evidence. The practically relevant tree case is `hub_y00`, which has the
shape of the downstream comb topology: E0 and E1 both choose x00 while the
cheapest base-tolerance site is the junction y00, and matched-cost selection
remains inconclusive because x00 extrapolates. The junction saving comes
from the active local dimension that the driver's one-hot embedding
restores, so it depends on the patch representation (M4). The ridge
function is not a downstream workload. These findings concern common
sampled points and collapsed active networks, not the recursive capped
driver.

Premeasurement review identified topology-dependent random-test streams:
site-free dimension-one vertices consumed RNG draws despite carrying no
physical bit. Follow-up sampling must generate physical bit coordinates
first and map the same points into each topology. The related fixed-depth
sampler generates physical bits directly and does not have this defect.

The [fixed-depth report](../../benchmarks/results/2026-10-03-m5-fixed-depth-exploration.md#known-defects-of-this-data)
now records the missing thread settings and the relation to issue #670.
Equal thread settings would not prove equal oversubscription effects for a
large unpartitioned network and smaller patches; timing ratios need pinned
thread-count measurements before a performance decision.

Design-document link and consistency checks cover the new proposal and the
patch-size plan. They establish neither an implemented API nor a numerical
correctness or performance result for a new selector.

Independent plan review corrected certification wording: an exhaustive
measurement can retain a rigorous error bound while missing the tolerance;
such a patch is not certified. No-effect option equivalence is restricted
to valid configurations, and the proposed dense check uses the existing
M3 rounding certificate. These are design corrections, not changes to
implemented test tolerances.
