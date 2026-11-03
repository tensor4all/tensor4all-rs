# N-ary exact direct sum (#1107, part 1)

First part of #1107 (selectable sum engines). This part only makes the exact
sum of `N` networks cheap; engine selection, the density-matrix-seeded fit,
cross interpolation of a sum and the patched split rule are later parts.

## Decisions

- One N-ary kernel owns the direct sum. `direct_sum_many` (core free function
  and the required `TensorContractionLike::direct_sum_many`) stacks all
  operands in one pass; the binary `direct_sum` is its two-operand form (a
  provided trait method), so no second implementation exists.
  `BlockTensor` and the itensorlike `TensorTrain` keep returning an
  unsupported-operation error.
- The kernel permutes each operand once to the result axis order (common
  indices, then its grouped indices) through the backend permutation and copies
  contiguous runs of `common_len * dims[0]` elements into its diagonal block.
  The previous binary kernel decoded a multi-index per result element; that
  per-element work was the main cost of a pairwise add (#1107 finding 6).
  All four floating dtypes are served; the old kernel served `f64` and
  `Complex64` only.
- `TreeTN::add_many(&[&TreeTN])` is the N-ary counterpart of `TreeTN::add`, and
  `add` now delegates to it. The name follows the `direct_sum` /
  `direct_sum_many` pair. The block order is the term order and the leg layout
  follows `terms[0]`, so `add_many(&[a, b, c])` has bit-identical node tensors
  to `a.add(b).add(c)`, differing only in bond index IDs.
- Every exact add chain in `tensor4all-partitionedtreetn` that is followed by
  one truncation (or none) now uses one N-ary sum: `to_treetn`, the duplicate
  projector groups of `contract`/`contract_selected`, the two group probes in
  `patching.rs`, and the equal-key merge of `add_with_patching`. These chains
  were exact before truncating, so the summed network, and hence the truncated
  result, is unchanged. `SubDomainTreeTN::add_many` stays crate-private until a
  public need appears.
- The pairwise merges of `reconstruct` and `schedule_merge_refine` are not
  touched: they decide per merge whether to keep it, which is a sum-engine
  question for the next part.

## Verification conclusions and constraints

- Acceptance criterion 1 of #1107 holds with the issue's reproducer (R = 12,
  N = 64, release, one core, Ryzen 9 6900HX, median of 3): `TreeTN::add_many`
  takes 0.035 / 0.024 / 0.023 s (smooth / random / mixed) against 2.67 / 2.52 /
  2.50 s for the sequential `add` chain at `origin/main` d8fcf4f5, a 75-111x
  reduction. The baseline chain is itself 4x faster on this branch because
  `add` uses the new kernel. Dense errors equal the chain's to the printed
  digits.
- `add_many` is about 2x slower than the issue's hand-built `sum_nary`, which
  writes the cores directly. The difference is the per-operand permutation and
  validation; it was not optimized further.
- Each new test was turned red by an injected change (16 mutations, all red);
  the list is in the PR description.
- A zero-dimensional bond cannot occur: `IdxTensor` rejects zero dimensions at
  construction, which the run-length copy relies on (recorded as an
  `INVARIANT`).
