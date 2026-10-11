# TreeACI correctness and resource-contract repairs (#854)

## Decisions

- Repair the eight confirmed findings in [#854](https://github.com/tensor4all/tensor4all-rs/issues/854) at their owning boundaries. Existing investigation-only or partially repaired issues remain separate; this does not claim to resolve #784/#794 or all historical performance work.
- Keep rank truncation and per-edge stability unchanged. Search independently of injection headroom and report a rank limit when significant residuals remain at saturated cuts; do not grant temporary rank beyond the caller's cap or turn off validation.
- Add typed finite-value rejection for guard/exact output, and a shared Core scalar operation for division by a real normalizer. Extend existing generic host CI dispatch to f32/Complex32 rather than widening TreeACI payloads or restricting its advertised scalar surface.
- Allocate aggregate message budgets conservatively across directed caches. Plan candidate metadata before allocation and use a Core-owned LUCI logical working estimate for the later phase. The estimate excludes allocator and provider-private workspace, as documented; no RSS or speedup claim is made.
- Retain the audit's full dense controls and desired-contract probes as upstream regressions. Local-memory probes disable the guard to isolate local admission; otherwise the newly enabled saturated guard could reject the same budget for an unrelated reason.
- A saturated, exact constant-function control exposed an additional Core rrLU error-reporting defect: exiting at a rank ceiling left the previous accepted pivot as the final error. Report and validate the remaining Schur complement at that boundary. Apply the same residual semantics to the backend dense pivot kernel, using its next complete-pivoting diagonal. Test both orientations, ranks 0/1/2, all four dtypes, backend/rrLU parity, and non-finite elimination overflow; preserve pivot selection and tolerances.
- Distinguish guard detection from actual injection when deciding whether final cleanup is needed. A saturated guard failure requires an honest status, without a redundant local pass that cannot incorporate the detected point.

## Verification conclusions and constraints

- Validated locally on `fix/treeaci-854`, based on audited main `633fd264`. No existing numerical tolerance was relaxed, and no RSI code is included. These are local repairs, not claims that the upstream issues have been merged or closed.
- TreeACI: 237 passing default checks (185 unit tests, 31 integration tests, 21 doctests), plus the R=9 low-temperature branch regression in release and three diagnostics-feature checks. The 16 public audit tests include all 432 dense geometry/order/seed/operator/dtype controls. Ten existing profiling tests remain ignored.
- Core: all 1,252 checks passed, including 342 doctests and dtype-preserving f32/Complex32 CI in both canonical directions with zero/nonzero inputs. Two existing opt-in tests remain ignored. The added rank-ceiling controls compare the reported residual with one dense reconstruction, without changing tolerances.
- Shared-path regression checks: 15 TreeTN canonicalization tests and 81 TreeTCI unit/continuation/termination tests passed. The large G0 regression uses release because its numerical workload is impractically costly in debug; ordinary checks use debug builds.
- The guard tests exercise non-finite starts, coordinate walks, reconstruction/subtraction overflow, and threshold overflow. Cache controls cover one/two inputs and budgets of 0/96/512/4096 bytes, including nonzero retention with adequate headroom. Both local-memory regressions reject before the callback and succeed with a generous budget.
- Changed-crate Clippy (all targets, warnings and missing error/panic documentation denied), formatting, public error-doc checks, API inventory generation, and the deterministic repository-rules preview passed. The preview used `--dry-run`; external LLM review and hosted CI have not run. `cargo-nextest` is unavailable locally, so crate checks used `cargo test`.
- Provider-injection library compilation passed with `--no-default-features --features tenferro-provider-inject`. Enabling that feature alongside the default faer provider is an invalid native/BLAS combination and was not used as evidence of a source defect.
- Backend workspace peaks, injected-provider numerical runtime, and historical DMRG/performance follow-ups remain outside the evidence. Saturated sparse-delta initialization still cannot inject a replacement without expansion headroom: it now returns `RankLimited` with its inaccurate partial result, rather than `Converged`. This does not assert that the target's minimum mathematical rank exceeds the cap.

## Confirmed finding coverage

| Finding | Repair boundary | Regression evidence |
| --- | --- | --- |
| F1: saturated guard bypass | Sweep detection and termination; cleanup depends on actual injection | Sparse delta reports `RankLimited` at cap 1; uncapped recovery and exact capped constant report `Converged` |
| F2: non-finite guard target/residual | Typed finite-value checks at callback and residual boundaries | Public NaN/Inf targets and private start/walk/residual/threshold cases reject |
| F3: non-finite exact shortcut | Single-node callback boundary | NaN/Inf and finite-input product overflow reject |
| F4: complex normalization extremes | Core real-scalar division, used by local matrix normalization | Complex64 huge/tiny and absolute/relative controls; Complex32 relative-scale controls |
| F5: aggregate cache overshoot | Divide evaluator quotas among directed caches | Sum of per-evaluator payload peaks remains within the configured total |
| F6: single-precision CI dispatch | Generic host Core CI and full-rank CI entry points | All four dtypes pass the dense oracle; f32/Complex32 factors preserve dtype |
| F7: sweep-ceiling allocation panic | Bounded initial history reservation | `usize::MAX` ceiling converges on the early-stopping fixture |
| F8: local working-budget omissions | Candidate preflight and Core-owned LUCI phase estimate | Metadata and live-factor counterexamples reject before evaluation; generous controls reconstruct exactly |
| Additional: stale rank-ceiling residual | Core rrLU and backend dense pivot-kernel error reporting | All four dtypes/orientations, zero/nonzero rank ceilings, backend parity, and final residual overflow |
