# Exact dimension-one patch pairs

Issue #1409 agrees on the link-based shortcut alone; compact prefactor patches
remain the separate design in #1410. A pair whose operand has only dimension-one
links needs no new bond space, regardless of its local values. The partition
adapter reuses the public local exact operator-apply path with already-bound
indices, preserving the other operand's bond dimensions without a factorization.
Existing exact group sums and their single truncation remain unchanged.

Related owner contracts matter to this shortcut. Local exact apply must not
claim an orthogonal gauge it did not establish; it now leaves canonical metadata
empty. Its binary contraction primitive also handles outer products when both
local site spaces are nonempty and disjoint. The adapter calls the owning typed
`ContractionOptions::validate` method before selecting a shortcut, so invalid
initialization, RSVD and SRC
settings cannot become silently accepted. Standalone subdomain contraction and
the adaptive adapter retain their existing compressed-pair policy, including
scalar-subtree pruning and stopping behavior. Keeping the exact dispatch in
ordinary partition algebra avoids combining preserved and pruned topologies
inside an adaptive group.

Regression coverage includes real and complex local signs, dense and singular
maps, three-leg copy tensors, either operand position, bondless and branched
trees, full index identities with different prime levels, exact contributions
followed by one group truncation, unchanged nonunit pairs, and option failures
before empty selections. The gauge regression checks dense values, norm and
subsequent truncation. No runtime improvement is claimed without measurements.

Local validation passes: 213 partition library tests (two pre-existing ignored
tests), 54 operator-apply tests, 63 focused partition/subdomain/adaptive
integration tests, 148 partition doctests and 10 relevant TreeTN doctests.
Both changed crates pass strict Clippy and rustdoc generation; formatting,
implicit-engine, public-error-doc and deterministic repository-rules checks
pass. Full workspace coverage and feature/hardware matrices remain hosted-CI
validation; no performance measurement was taken.
