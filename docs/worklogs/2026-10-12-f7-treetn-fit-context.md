# treetn fit contraction in a caller-owned context (issue #859 F7)

## Summary

The F7 inventory recorded Fit as the second remaining method behind
itensorlike's pass-through. This slice adds `contract_fit_in` (plus a
`tensor4all_treetn::contract_in` dispatch) and completes the Fit half of
`tensor4all-itensorlike::contract_in`. Naive stays refused with a typed error.

## Design

1. **The leaf strategy generalizes.** `ZipupLeafOps` became `LeafOps<T>`
   (object-safe) so the fit schedule reuses the same strategy: `GlobalLeaves`
   keeps the process-global calls, `SessionLeaves` runs
   `contract_pair_on_session` / `factorize_in` in the caller-owned context.
   The fit updater carries its strategy as `Box<dyn LeafOps<T>>` (a private
   field, so `FitUpdater`'s public shape is unchanged) and the environment
   cache, leaf-environment and two-site-update helpers take `&dyn LeafOps<T>`.
2. **`contract_in` is the public dispatch** for `IdxTensor` trees: Zip-up ->
   `contract_zipup_with_in`, Fit -> `contract_fit_in`, SRC ->
   `contract_src_with_rng_in`, Naive -> typed refusal. itensorlike calls it
   for the Fit method.
3. **Context adoption inside the session strategy.** The plain helpers create
   some intermediates on the process-global route (for example the singleton
   boundaries of `factorize_allowing_empty_side`, which the fit initializer
   reaches), and the session's own results carry the session's eager AD
   context. Mixing those with context-resident inputs fails the AD-context
   check even when every tensor validates against the same execution context.
   `SessionLeaves` therefore **adopts every operand** (rebuilds it through
   `IdxTensor::from_dense_in`) before contracting or factorizing, so all
   operands share one AD context. This is the price of correctness here and is
   recorded in the measured overhead below; skipping adoption when the AD
   contexts already agree needs an API to compare them (follow-up).

## Verification

- `crates/tensor4all-treetn/tests/f7_zipup_context.rs` (zip-up paths) and the
  itensorlike `contract_context.rs` suite (8 tests: SRC parity/provenance/
  identity, Zip-up parity, Fit parity, Naive refusal) pass.
- The whole treetn suite (29 test binaries, 564 lib tests) passes unchanged
  through `GlobalLeaves`.
- Overhead (`benches/f7_contract_src_overhead.rs`, release, criterion, one
  thread): SRC plain `0.890 ms` vs context `1.107 ms`; Zip-up plain
  `0.103 ms` vs context `0.532 ms`; **Fit plain `3.964 ms` vs context
  `11.456 ms`**. The Zip-up/Fit context routes pay the per-leaf adoption
  round-trip (plus, for Zip-up, the missing Gram shortcut). Reported as
  measured.

## Known divergence (recorded)

The caller-owned Fit route rejects a shape in which one operand's site index is
the *prime* of the other's (same id, different prime level): its factorization
path refuses the unpaired primed site with a typed error, while the plain route
happens to factorize it. The routes therefore differ on that pathological
shape; the plain route's result is pinned as distinct from the same-prime-level
result (`fit_context_route_handles_same_id_primed_indices_like_the_plain_route`).
A conflation bug would instead make the two results equal, so the regression
still catches identity errors.

## Follow-ups

- Naive's context-scoped dense-reference path.
- Skip the adoption round-trip when operands already share an AD context.
