# Tree zip-up rank cancellation

## Decision and literature boundary

The second M6 slice implements cancellation only for deterministic SVD on
the existing general-tree zip-up schedule. A static completion policy shares
the numerical body with ordinary contraction; ordinary callers keep the
network return type and allocate no rank report. A qualifying edge frees the
intermediates and returns only its endpoints and retained dimension.

The [design's literature and proof section](../design/treetn-contraction-outcome.md#literature-basis-and-tree-specific-argument)
separates references already in the provenance policy from our own derivation.
No upstream source code was consulted or translated for this hook.

| Already implemented or added here | Basis and limit |
| --- | --- |
| Ordinary/selected adaptive analytical baselines | Independently constructed exact-value oracles, establishing software behavior; no new algorithm or literature theorem |
| Post-hoc outcome API, report ordering, independent threshold | Our software contract over unchanged numerical entries; equality is consistent with the rank-cap criterion in Grosso et al. §4.3, but the API itself is not specified by that paper |
| Local zip-up rank-triggered stop | Explicit observation in Grosso et al. §4.3; the retained-link proof for our general-tree schedule is independent and documented in the design |
| Keeping the chain path post-hoc | The chain implementation has a final compression pass, consistent with the zip-up literature already cited in the design; earlier ranks need not be final |
| Fit, Naive and randomized routes | Existing execution is reused unchanged. This slice does not establish new accuracy guarantees or claim a complete audit of those algorithms |

Each retained tree link is immutable after its factorization, and its
presence prevents its ancestors from being pruned. The scalar-pruning path
creates no link. No rank-changing pass follows the tree sweep. These are the
implementation premises of cancellation equivalence, and changes to them
require a new proof. Stored rank is not minimal Schmidt rank: a regression
constructs an all-ones product tensor with final stored ranks `[1, 2, 2]`.

Option validation and the caller-context entry's whole-input ownership checks
precede cancellation. Errors from executed numerical operations retain their
sources; errors in skipped later work are deliberately not evaluated. Caller
RNG streams and seeded RSVD/SRC still run in full. The adaptive driver has not
adopted cancellation yet; layout refinement, Hadamard, sibling merging and
their error ledgers remain separate work.

## Verification conclusions

Per-thread test-only instrumentation counts factorizations in the public
entry's actual tree edge loop. On a three-edge star, ordinary and threshold-free
calls execute three factorizations; a threshold reached at the first edge
executes one, on both compatibility and explicit-context routes. A nonuniform
rank fixture executes two before stopping at the second edge. Deeper tree
tests also compare the hook with completed reports and the known dense oracle.
These are executed-work counts, not wall-time speedup measurements.

Coverage includes real/complex data, chains, multiple branches that saturate,
scalar subtrees and edgeless outputs, threshold equality and independence
from truncation caps, source-chain preservation, pre-stop versus skipped
failures, context rejection, existing AD oracles and unchanged RNG consumption.
Single-precision support/refusal remains inherited. No existing numerical
tolerance or coverage threshold is changed.

The performance audit found no new tensor-element loop, payload copy,
implicit dense materialization, backend construction or per-edge report
allocation. The hook reads the returned factorization rank and allocates only
the saturated edge's endpoint names. Existing contractions and factorization
leaves remain at their owners.

Default CPU tests and documentation checks cover this slice. CUDA,
non-default providers, end-to-end adaptive speedup and parallel scaling remain
unverified; the hook makes no such claim. Hosted CI and coverage require an
approved push/PR.

Local verification passed the full treetn suite (997 tests, two existing
ignored tests unchanged), all 207 rustdoc examples, three downstream adaptive
baselines and all 32 mdBook chapters. The focused outcome suite has 21 tests,
including seven new cancellation regressions. Clippy with warnings and public
error/panic documentation lints, rustdoc with warnings denied, formatting,
local Markdown paths, crate-boundary and test-registration checks, and the
deterministic repository-rules review passed. The extra second-operand context
check was rerun in the focused suite after the full suite; production code was
unchanged. Tests use the ordinary profile; executed-work counts are structural
evidence and do not make an optimized wall-time claim.
