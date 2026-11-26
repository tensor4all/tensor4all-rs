# Exact dimension-one patch pairs

Issue #1409 agrees on the link-based shortcut alone; compact prefactor patches
remain the separate design in #1410. A pair whose operand has only dimension-one
links needs no new bond space, regardless of its local values. Ordinary partition
algebra calls TreeTN's typed `contract_local_exact` seam. It performs local
contractions and link fusion without factorization, preserving the other
operand's dimensions on retained bonds. Existing exact group sums and their
single truncation remain unchanged.

Mixed exact and compressed pairs must have compatible output topologies. Review
of PR #1469 exposed a branched mixed call failing with `TopologyMismatch`:
ordinary Zipup pruned a site-free subtree while exact operator apply retained it.
The owner now shares one metadata topology plan between Zipup emission and exact
cleanup, selected by `ContractionTopology`. Rooted-tree pruning retains scalar
bridges and the center; chain planning preserves the existing final-block
convention, which can omit a scalar endpoint but retain an adjacent bridge.
Exact cleanup absorbs omitted terminal tensors by local pair contraction only.
Chain-vs-rooted routing also retains the legacy comparison of full site spaces,
including directed bra/ket sets that fully contract. Fit and SRC retain the
input topology. Nonunit pairs keep their selected
algorithm, numerical schedule and truncation policy. Standalone subdomain and
adaptive contraction keep their existing compressed-pair and stopping policies.

The dedicated exact seam owns the former naive-apply kernel, so both APIs share
local contraction and fusion without an empty-map operator adapter. Local exact
contraction leaves canonical metadata empty because it establishes no orthogonal
gauge, and its binary primitive handles disjoint nonempty local site spaces.
It validates topology, center, full site-index assignment and product bond sizes
before numerical work. This factors existing generic `TensorLike` operations;
it creates no backend/context and adds no implicit-engine ratchet sites. Engine
migration remains governed by #1376 rather than introducing a second execution
route in this fix.

The adapter calls typed `ContractionOptions::validate` before shortcut dispatch.
Common invalid partition options use `InvalidOptions`; initialization/RSVD/SRC
validation preserves `TreeTNOperationError::InvalidConfiguration` under the
partition's `TreeTN` variant. Both error routes and the distinction between
standalone and ordinary partition contraction are documented publicly.

Regression coverage includes real/complex local signs, dense/singular maps,
three-leg copy tensors, either operand, bondless and branched trees, full index
identities at different prime levels, retained bond products, exact group sums,
nonunit direct-reference parity and valid/invalid empty selections. Mixed calls
cover Zipup/Fit/SRC, both public partition entry points, distinct output groups
and multiple contributions per group. Owner tests fix the expected topology for
chain endpoints, scalar bridges, fully scalar results and branched subtrees,
under both topology policies. The gauge test checks dense values, norm and
subsequent truncation. No numerical tolerance was changed and no runtime
improvement is claimed.

Local validation passes: 118 TreeTN contraction tests (one pre-existing ignored
heavy test), 54 operator-apply tests, 215 partition library tests (two
pre-existing ignored tests), and 63 focused partition/subdomain/adaptive
integration tests. Both changed crates pass strict Clippy, rustdoc generation and formatting.
All 148 partition doctests and 11 relevant TreeTN doctests pass. Public error
docs, deterministic repository rules, the unchanged implicit-engine ratchet and
the regenerated API inventory pass; README and bundled usage-skill claims remain
accurate. Full workspace coverage and feature/hardware matrices remain hosted-CI
validation; no performance measurement was taken.

A separate nonunit SRC case with disjoint factors on site-free surviving leaves
remains unsupported by the existing batched contraction helper
(`src_probe.rs::contract_retaining`). That numerical path is unchanged here;
its outer-product handling warrants a separate audit. The mixed method-parity
fixture supplies shared local inputs to isolate the topology contract.
