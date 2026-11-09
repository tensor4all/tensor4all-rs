# Partial apply: skipping the untouched zip-up nodes

## Decisions

- A partial `apply_linear_operator` extends its operator with identities and then
  zip-ups the whole state. The nodes the operator does not cover carry an
  identity factor, so their block's rank is the state tensor's own rank and
  factorizing them only rebuilds the bond the state already has. The general tree
  schedule of the zip-up now emits such a node as its own state tensor, with the
  identity's input site indices renamed to its output ones, and passes no
  remainder to its parent.
- The pass-through stands in for a factorization, so it is allowed only where
  that factorization would have been exact (`zipup_skip_is_exact`):
  - no truncation control at all (`svd_policy` and `max_bond_dim` both unset) —
    every factorization keeps its full rank, certified or not;
  - a relative `svd_policy` with no bond cap, for a node the network certifies as
    isometric toward the parent (`ortho_towards` on the parent edge points at the
    parent, i.e. the state is in canonical form with the centre on that side). An
    isometric node's block is full rank with all singular values one, and its
    remainder is unitary, so the parent's cutoff sees the same spectrum it would
    without the node.
  - a `max_bond_dim` never qualifies. The cap truncates the block itself rather
    than a relative tail: measured on a canonical comb, a cap of two moves the
    truncated result by `7e-4` (and by `7e-3` when a node whose bond exceeds the
    cap is skipped), which the pass-through cannot express.
- An uncertified node keeps the ordinary contraction and factorization. Skipping
  one is wrong: a leaf with `diag(1e-8, 1e8)` under a hub with `diag(1, 1e-8)`
  loses an amplitude of 1 under `rtol = 1e-6`, and that reproduction is the
  regression test
  `apply_zipup_does_not_skip_a_node_that_is_not_isometric_toward_the_center`.
- A node that receives a remainder from an operator-bearing child is factorized
  as usual, and so is a site-free node (the pruned-scalar-subtree branch keeps
  precedence).
- Chain-shaped states keep the chain schedule. It hands one remainder per edge to
  its successor, so it cannot emit a node without one, and it additionally
  conditions both operands with an exact QR and truncates at the centre; changing
  it is a separate change.
- The untouched node's bonds stay on the edges they came from, and the parent's
  copy of the identity bond is summed out when the parent is processed
  (`sum_out_identity_legs`, which refuses a leg that is not dimension one). Noise
  is avoided by summing a dimension-one leg out rather than leaving it as a
  stray index: before that, three existing tests failed on dense index counts.
- The identity data is carried, not rebuilt: the emitted tensor is the state
  tensor with the site index renamed (`replaceind`), not a delta contraction.

## Verification conclusions and constraints

- The untouched region is unchanged bit for bit: the tests compare the result
  nodes against the input state with `f64::to_bits`, including their site index
  identity and order, and check that the untouched bond dimensions are the
  input's.
- A table test compares every skip decision against the same apply with the skip
  switched off, over `(no control, three relative policies, three bond caps)`
  combinations, and asserts both the skip count and the result.
- Without truncation controls the pass-through is also more accurate than the
  factorize-and-contract round trip it replaces: on a fixture with a 1e16
  dynamic range it lands `2.2e-16` from the exact local apply where the
  factorized path lands `1e-8`.
- The whole result is *not* bit-identical to the pre-change path, and cannot be:
  the old path rebuilt every untouched node from an SVD of an isometry, so its
  rounding is part of the old result. The measured deviation is at the last-bit
  level (`1.9e-17` on a 128-amplitude fixture, largest amplitude `1.4e-2`), and
  both paths agree with the exact local apply to `~5e-18`.
- `max_bond_dim` and `svd_policy` are consequently not applied inside an
  untouched region, where the pre-change path could only compress a
  rank-deficient block; truncate the result if that region should be compressed.
- The saving is shape-dependent and measured, not assumed: on a 61-site comb a
  whole-branch operator gains nothing measurable (the operator's own branch
  dominates), while a three-site operator on the same state is about 9.6x faster (0.077 s to 0.008 s per apply)
  because the untouched nodes are then the whole job. Numbers, harness and raw
  output: [partial apply untouched nodes](../experiments/2026-11-08-partial-apply-untouched-nodes.md).
- Not covered: the chain schedule (see above) and the pre-existing
  dimension-one legs that a partial apply leaves at the centre.
