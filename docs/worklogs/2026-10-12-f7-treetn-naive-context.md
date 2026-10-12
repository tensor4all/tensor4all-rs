# treetn Naive contraction in a caller-owned context (issue #859 F7)

## Summary

The last F7 method: `contract_naive_to_treetn` (the Naive dense reference) now
has a caller-owned route, so `tensor4all_treetn::contract_in` and
`tensor4all-itensorlike::contract_in` serve all four methods.

## Design

1. The three Naive stages take the existing `LeafOps<T>` strategy:
   `contract_to_tensor_with_leaves` (the tree-to-dense contraction),
   `contract_naive_with_leaves` (the two dense products and their pairwise
   product) and `contract_naive_to_treetn_with_leaves` (plus the
   decomposition's `factorize_tensor_to_treetn_with_leaves`). The public
   entries keep their signatures and pass `GlobalLeaves`.
2. Option validation moved into the leaves variant so the public wrapper and
   the context route validate identically (including the dense-reference limit
   and the `qr_rtol` refusal).
3. `contract_in`'s Naive branch calls
   `contract_naive_to_treetn_with_leaves(..., &SessionLeaves(context))`;
   itensorlike's Naive branch calls that dispatch, so all four methods are
   wired.

## Verification

- itensorlike `tests/contract_context.rs` (10 tests): SRC, Zip-up, Fit and
  Naive parity (norm-based residual <= 1e-10), the dense-reference refusal, the
  SRC/Zip-up/Fit identity regressions.
- Whole treetn suite (29 test binaries, 564 lib tests) passes unchanged
  through `GlobalLeaves`.
- Overhead (`benches/f7_contract_src_overhead.rs`, release, criterion, one
  thread): SRC plain `0.912 ms` vs context `1.103 ms`; Zip-up plain
  `0.103 ms` vs context `0.539 ms`; Fit plain `4.024 ms` vs context
  `11.495 ms`; **Naive plain `0.168 ms` vs context `0.616 ms`**. The context
  routes pay the per-leaf adoption round-trip (recorded with the earlier
  slices).

## Status

Every F7 consumer method in `tensor4all-itensorlike` and every contraction
method in `tensor4all-treetn` now has a caller-owned context route. Remaining
follow-ups from this pass: skip the adoption round-trip when operands already
share an AD context, and an in-context Gram factorization path.
