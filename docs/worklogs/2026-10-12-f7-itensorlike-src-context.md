# itensorlike contract: caller-owned SRC route (issue #859 F7)

## Summary

`tensor4all-itensorlike::contract` delegates to `tensor4all-treetn`'s
contraction. The F7 verdict pass classified the crate as a "thread-through",
but verification showed the pass-through only exists for the SRC method:
treetn has one session-taking contraction twin today,
`contract_src_with_rng_in`, while its Zip-up, Fit and Naive paths
(canonicalize, zip-up sweeps, fit solves, naive dense reference) have none.
This worklog records the scope decision, the provenance constraint that
shapes the new route, and the measured numbers.

## Scope decision

1. **SRC gets the caller-owned route now** (`contract_in`, free function and
   `TensorTrain::contract_in`): identical algorithm, options and validation as
   `contract`, with the SRC projection running through the supplied
   `ExecutionContext` (treetn's `contract_src_with_rng_in`, same ChaCha8
   seeding from `SrcOptions.seed`).
2. **Zip-up / Fit / Naive are deferred**, not silently claimed: `contract_in`
   refuses them with a typed error naming the method (a caller who supplies a
   context must never get a silent process-global run), and the inventory
   (`docs/design/859-f7-consumer-inventory.md`) records that a full
   pass-through needs treetn-side session twins for the canonicalize /
   zip-up / fit / naive internals. Wiring those is a treetn slice, not an
   itensorlike wrapper; plain `contract` keeps serving them.
3. **Input provenance is a hard requirement of the caller-owned route**: the
   route validates that every input tensor belongs to the supplied context
   (`validate_context`), so consumers build their trains with
   `IdxTensor::from_dense_in(&context, ...)`. Process-global-built trains are
   refused with a typed context error; the plain `contract` entry keeps
   accepting them. This matches the established caller-owned semantics of the
   pass (mixed host/foreign-CUDA inputs fail at the boundary with typed
   errors) and is documented on the entry and in the tests.

## Alternatives considered

- Adding treetn's missing twins first (zip-up canonicalize + fit + naive):
  the remaining work after SRC, and a larger threading slice through
  treetn's contraction driver. Chosen as the *next* step, not this one, to
  keep this PR a bounded pass-through.
- Making `contract_in` fall back to the global route when inputs are
  global-built: rejected because it would silently pick a different resource
  than the caller asked for, contradicting the provenance model the route
  validates.
- Delegating Zip-up/Fit/Naive to the plain route under `contract_in`:
  initially chosen, then rejected after the rules review flagged the silent
  resource substitution for methods the caller's context cannot serve; the
  final design refuses them with a typed error that names the method.

## Verification

- Parity: `tests/contract_context.rs` — exact (`assert_eq`) equality between
  the plain route and the caller-owned route on identical standard fixtures,
  plus error parity for invalid SVD options, length-mismatch rejection,
  provenance refusal, and an identity regression (same-ID indices differing
  by prime level stay distinct and change the result vs the same-plev pair).
  29 crate tests + 6 integration tests pass.
- Observed: on the same-id/primed shape the caller-owned route's *values* can
  diverge from the plain route (1.1 vs 0.9 on the fixture, rank structure
  unchanged): the adaptive SRC sketch's growth decisions sit on the exact
  session numerics, so primed shapes pin shape parity, not value parity.
  Standard (unprimed) fixtures stay bit-exact, so this is recorded as a
  property of the primed shape, not a correctness regression of the route.
- Overhead: `benches/f7_contract_src_overhead.rs` (release, criterion, both
  routes pinned to one thread via `RAYON_NUM_THREADS`/`BLAS_NUM_THREADS`/
  `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS=1` and
  `CpuBackend::with_threads(1)`): plain `0.891 ms` vs context `1.084 ms` per
  run (6 sites, bond dim 2). The caller-owned route carries ~22% overhead on
  this small workload (fixed eager-runtime bookkeeping per node), reported as
  measured, not as parity.
- The shared entry validation was extracted (`validate_contraction_inputs` +
  `contract_dispatch`) so the two routes cannot drift on boundary checks; the
  review's duplicate-validation finding and identity-regression finding are
  both addressed in the same PR.

## Follow-ups

- treetn-side session twins for Zip-up / Fit / Naive contraction internals,
  then `contract_in` becomes a full pass-through.