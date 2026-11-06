# Partial-contraction rank outcomes

## Decision

The third M6 slice adds `partial_contract_with_outcome`, with the existing
partial entry's operands, specification, center and options, plus the independent
positive rank threshold. Diagonal pairs also express the existing Hadamard
operation. No additional Hadamard convenience wrapper is needed for this slice.

Ordinary and outcome calls share preparation and restoration at their owner.
Preparation includes option/pair validation, local structured-copy expansion,
index renaming, movement of contracted site indices and dimension-one spectator
connections. Ordinary calls keep the same numerical dispatch, output ordering
and errors; they allocate no report. Outcome calls reuse the contraction outcome
owner and its final-report helper, without a parallel rank-inspection algorithm.

The [adapter argument](../design/treetn-contraction-outcome.md#partial-contraction-adapter)
extends our independent tree-loop proof, with references already documented in
that design. The patching paper motivates the stop criterion; it does not prove
the partial adapter's semantics. Local copy tensors are consumed inside existing
vertices, and successful restoration only renames external indices of equal
dimensions. Neither step introduces a graph edge to exclude from the report.
All retained output links, including dimension-one structural links, count.

Two paths intentionally classify only after the ordinary operation completes.
Output ordering validates the final survivor set and rebuilds node enumeration;
even though it does not resize links, its errors and final report order must
remain visible. Mismatched-topology operations can recover from a structural
failure through an explicitly bounded dense/reference fallback. Stopping that
structural attempt could misclassify the fallback's final result, so this slice
preserves the whole existing pipeline. It adds no new dense fallback or implicit
materialization.

The partial API currently has no explicit-context or caller-RNG entry. This
slice preserves compatibility-context acceptance/refusal and the existing seeded
RSVD/SRC options. A future context/RNG extension must carry execution ownership
through preparation and restoration as well as the central contraction.
It cannot be implemented by forwarding just that middle call.

## Verification and limits

Known GHZ Hadamard values test real and complex chains/branched trees, including
same-ID indices with distinct prime levels. Known uniform references exercise
contract-only, diagonal-only, mixed and outer specifications, including dummy
spectator preparation. A moved contract pair checks index-alignment behavior.
The same matrix covers deterministic Zipup/Fit, Naive, seeded SRC and RSVD
Zipup/Fit. Single-precision support or refusal is inherited from the ordinary
entry. A one-node Hadamard derivative has the exact oracle `[5, 7]`.

Public-entry counters distinguish three tree edge factorizations from one on
early saturation. Ordered-output calls still execute all three, and a
mismatched star still executes all four. These counts exclude preparation and
make no wall-time claim. Error comparisons cover invalid options, pairs, output
ordering, bounded fallback limits and context refusal, retaining source chains.
A forced scalar fallback verifies its successful value and edgeless rule.

Local verification passed all 45 partial-contraction tests (11 new regressions),
the full treetn suite (1008 passed, two existing ignored tests unchanged),
all 208 rustdoc examples, three downstream adaptive baselines and all 32 mdBook
chapters. Clippy with warnings and error/panic-documentation lints, rustdoc with
warnings denied, formatting, public-error documentation, crate boundaries,
test-target registration, local Markdown paths and the deterministic rules
review passed. The regenerated API inventory accounts for all 17 public crates
and includes the new entry. The root README remains accurate. Ordinary test
profiles suffice for this adapter and the structural work-count claims;
Nextest was unavailable, so changed-crate validation used Cargo's test runner.

The performance review checks the changed paths structurally: no new numerical
kernel, tensor-element loop, device transfer, payload readback or unbounded dense
path is introduced. Existing local copy/swap operations remain before the stop
and their preparation cost is still paid. Existing numerical tolerances and
coverage thresholds are unchanged.

Adaptive-driver adoption and partitioned Hadamard remain future work. No
end-to-end adaptive speedup, CUDA or non-default-provider qualification is
claimed. Hosted CI/coverage require an approved push/PR.
