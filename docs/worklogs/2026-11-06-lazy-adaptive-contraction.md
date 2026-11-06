# Lazy Sequential adaptive contraction

## Decision and basis

The fourth M6 slice connects TreeTN rank outcomes to ordinary and selected
adaptive contraction. A shared metadata planner validates templates, orders
operands through the existing logical-key/content-digest owner, filters
compatibility/selection and predicts output projectors before tensor work.
Non-adaptive contraction reuses this planner and retains bounded refinement.
The planner is a visitor: ordinary contraction consumes pairs as they are
visited, preserving its interleaving and avoiding an extra full pair list.
The adaptive consumer collects group operands before starting tensor work.

Groups execute in first-appearance order of that canonical pair plan. Within
each group, the existing contribution order is preserved. This replaces the
old sort on completed groups' first numerical contribution, which cannot be
obtained for a cancelled probe. The final truncation still sorts completed
patches using their actual metadata/content keys at the existing budget owner.
No input HashMap iteration determines the contribution fold.

With a Sequential cap and an available candidate, contributions are attempted
one at a time. A saturated outcome stops the group immediately; a completed
bondless result still has subdomain rank one, preserving the cap-one trigger.
Every child is recomputed from projected original operands of **all** pairs,
including those skipped at the parent. Completed groups retain the exact
sum engine, one post-sum truncation and final norm/volume budget allocation.
ExactParameterGain, absent caps and exhausted candidates evaluate every pair.

Explicit patch orders derive the candidate from output-index/projector metadata.
Empty orders complete their first contribution and use its actual node/site
order before later attempts can stop. This deliberately refines the original
design's stronger before-any-contraction proposal: predicting the numerical
method's local axis order would risk changing the default split layout.

[Grosso et al., §4.3](https://arxiv.org/html/2602.22372v1#S4.SS3) motivates local
rank stopping and contributor-only refinement. The lazy loop is independently
implemented against this repository's existing behavior; no upstream source
code was consulted or translated. Its argument is a short-circuit of the
**existing** predicate: once one contribution reaches the cap, further terms
cannot make that existential predicate false. Since this branch discards
contributions and projects original operands, its retained prefix is never used
as a sum. This is not a theorem about the final sum's rank: cancellation can
lower it. The TreeTN hook's stored-edge proof and partial adapter have their
own [design record](../design/treetn-contraction-outcome.md).

## Verification and limits

Public-entry counters distinguish planned, attempted and thresholded pairs.
A two-pair saturated group attempts one parent pair, then recomputes both
pairs on each child. The matrix covers chains and a degree-three tree,
ordinary/selected entries, reversed insertion order, Zipup, Fit, seeded SRC
and RSVD. It also covers a completed low-rank first pair followed by saturation,
implicit order, cap-one edgeless results, no cap, no candidate and exact gain.
The pre-outcome real/complex analytical baselines remain unchanged, including
layout, threshold equality, selection ancestry and Binary/Nary sums.

The subdomain adapter shares validation and result wrapping with its ordinary
entry, preserving typed source errors, projector filtering and inherited budget.
Conflicting regions return no contribution; invalid options are checked first.
Template validation still precedes an empty selection. Numerical errors in
work skipped after a stop are not evaluated. Per-attempt seeded method options
are unchanged; there is no new cross-group caller-RNG stream.

The structural performance review found no new numerical kernel, tensor-element
loop, payload readback, device transfer or dense materialization. Pair planning
still enumerates `N_A * N_B` metadata candidates and retains grouped operands;
project-first retries retain the existing operand clone/projection caches and
`64 * eps` compression. Saving numerical attempts is established by counts;
wall-time speedup, parallel scaling and non-default providers are unmeasured.
No numerical tolerance, coverage threshold, overlap/fusion policy, sibling merge
or public signature changes. Documentation now describes the split fields
actually used by adaptive contraction; the old claim that they were unused was
stale. The strict doc build also exposed broken `CpuEngine` links in lanes
documentation; the corresponding lanes/MPI prose now uses code-form names.

Hosted CI and coverage require an approved push/PR. Partitioned Hadamard,
adaptive overlap handling and the multi-level sibling-merge error ledger remain
separate M6 work.

Local checks passed the full changed-crate suite (390 tests, 13 existing ignored
tests unchanged), all 124 rustdoc examples and all 32 mdBook chapters. After
refining the planner into a visitor, all 132 unit tests and 25 contraction-related
integration tests passed again; the unrelated earlier integration evidence
remains applicable. Eight new regressions exercise the lazy paths. Clippy with
warnings and public error/panic-doc lints, strict rustdoc, formatting, public-error
docs, crate boundaries, test registration, local links and deterministic rules
review passed. The root README remains accurate and the API inventory accounts
for all 17 public crates. All runtime checks use ordinary profiles; no release
timing claim or test-tolerance change is made.
