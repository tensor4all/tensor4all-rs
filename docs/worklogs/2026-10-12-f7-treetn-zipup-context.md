# treetn zip-up in a caller-owned context (issue #859 F7)

## Summary

The F7 inventory recorded treetn's zip-up as the blocker for itensorlike's
Zip-up pass-through. This slice adds `TreeTN::contract_zipup_with_in` and
completes the Zip-up half of `tensor4all-itensorlike::contract_in`.

## Design

1. **A leaf-execution strategy, not a runtime downcast.** The zip-up schedule
   operates on the generic node tensor `T`, and the context-scoped core
   entries (`contract_pair_on_session`, `factorize_in`) are `IdxTensor`
   operations. `contract_zipup_impl`/`contract_zipup_chain` therefore take a
   `&L: ZipupLeafOps<T>` strategy: `GlobalLeaves` keeps the existing
   process-global calls for every `T`, and `SessionLeaves` implements the
   trait for `IdxTensor` only. The public entry is an
   `impl<V> TreeTN<IdxTensor, V>` block, so the context route is selected
   statically - no downcast, no `unsafe`, no silent global fallback for other
   tensor types.
2. **Multi-tensor leaves fold through pairs.** The global route keeps the
   multi-tensor planner (`T::contract(&[..])`); the session route folds the
   same list left-to-right through `contract_pair_on_session`, which is
   numerically equivalent for the small `R + A + B` blocks zip-up forms.
3. **`LU`/`CI` forms are refused** (typed error): core's `factorize_in` has no
   context-scoped LU/CI path. Refusing beats silently factorizing globally.
4. **The `factorize_auto` Gram shortcut is excluded** (recorded): its
   eigendecomposition has no context-scoped twin, so the session route uses the
   direct truncated-SVD factorization with the same options. Results agree to
   tight tolerance, not bitwise.
5. **itensorlike's Zip-up half is completed**: `contract_in` routes Zip-up
   through the new twin; Fit/Naive stay refused with a typed error naming the
   method (their sweep/dense-reference internals still need treetn twins).

## Verification

- `crates/tensor4all-treetn/tests/f7_zipup_context.rs`: parity with and
  without an SVD truncation policy on the shared-site general path, on the
  chain path (distinct per-node site spaces), and on a star (branch) path;
  typed refusal for `LU`/`CI`. All pass; the pre-existing treetn contraction
  tests (135 lib + integration) pass unchanged through `GlobalLeaves`.
- `crates/tensor4all-itensorlike/tests/contract_context.rs`: Zip-up parity
  through `contract_in`, Fit/Naive refusal.
- Overhead: `benches/f7_contract_src_overhead.rs` (release, criterion, one
  thread): SRC plain `0.922 ms` vs context `1.092 ms`; **Zip-up plain
  `0.103 ms` vs context `0.356 ms`**. The Zip-up context route is ~3.4x slower
  on this fixture: each leaf takes the context's session (lock + eager
  session acquisition) instead of the process-global path, and the Gram
  shortcut is unavailable in-context. Reported as measured; a follow-up could
  add an in-context Gram path (`hermitian_eigendecomposition` session twin).

## Follow-ups

- Fit and Naive contraction twins (`contract_fit` sweeps, naive dense
  reference) to complete itensorlike's pass-through.
- An in-context Gram factorization twin to remove the Zip-up overhead.
