# partitionedtreetn: the engine carries the caller-owned context (issue #859 F7)

## Summary

`tensor4all-partitionedtreetn::patched_interpolate` is a patch driver over a
pluggable tree interpolation engine (`TreeInterpolator` from
`tensor4all-treetn`). Verification showed the crate holds no numerics of its
own: every numeric step (per-patch optimizer, per-patch materialization)
belongs to the engine the caller passes in. The F7 work therefore lives in the
engine layer, not in the driver.

## Scope decision

1. **The driver and its trait stay untouched.** The `TreeInterpolator` trait
   is the numeric provider boundary; adding a context parameter to it would
   force every engine (including test engines) through a surface they do not
   need. The trait decision raised in the verdict pass is resolved the other
   way: the engine *carries* the resource identity.
2. **A context-carrying engine, `TreeTciInterpolatorIn`, in
   `tensor4all-treetci`:** same options, validation, and stopping rule as
   `TreeTciInterpolator`, but `interpolate` routes both the optimizer
   (`optimize_with_proposer_in`, #937) and the final materialization
   (`to_named_treetn_chunked` with the context, #939-wired) through a
   caller-owned `CpuExecutionContext` instead of the process-global one. The
   single-node shortcut has no numerics and is shared unchanged.
3. **PartitionedtreeTn consumers opt in by passing the in-engine** to
   `patched_interpolate`; nothing else changes. `to_named_treetn_chunked` was
   made `pub(crate)` (same-crate visibility only; no API surface growth).

## Alternatives considered

- Adding `interpolate_in` to the `TreeInterpolator` trait: rejected, it
  leaks resource plumbing into every engine and every driver call site.
- A `patched_interpolate_in(..., context)` entry: rejected, the driver has
  nothing to thread - the engine is where the numerics live.
- Making the plain engine context-aware behind a flag: rejected, mixed
  resource identity in one type is what the provenance validation exists to
  catch.

## Verification

- Parity: `tests/adaptive_interpolation_engines.rs` - `patched_interpolate`
  with the in-engine gives the same patch splits, dense values equal to
  1e-12 relative over the whole domain, and the same accuracy contract as
  with the plain engine on a seeded run.
- Overhead: `benches/f7_engine_context_overhead.rs` (release, criterion,
  both routes pinned to one thread via `RAYON_NUM_THREADS`/`BLAS_NUM_THREADS`/
  `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS=1` and
  `CpuBackend::with_threads(1)`): plain `0.697 ms` vs context `0.705 ms` per
  `interpolate` run (2-node tree, 4x4 sites) - parity within noise.

## Follow-ups

- Any other `TreeInterpolator` consumer can take the same in-engine; the
  treetci API-level entries already had context twins (#937/#939).
