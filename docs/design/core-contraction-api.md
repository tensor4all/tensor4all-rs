# tensor4all-core contraction API direction

Source of the still-live API decision from the deleted
`plan/tensor4all-core-api-cleanup.md` (tensor4all-rs#566 Task 11 debris
removal).

## Direction (current and pending)

- Prefer one public N-ary contraction entry point over separate binary and
  N-ary APIs. Binary contraction is expressed as N-ary contraction with two
  operands.
- The API takes borrowed operands (`contract(&[&a, &b, &c])`). Ordinary and
  prepared calls share the same execution dispatch: singleton clone, binary
  pairwise fast path without retained indices, compact structured/AD N-ary
  execution, or dense native execution. The owned entry points with no
  production callers outside their own implementation in the surveyed
  workspace were removed in #1381; owning operands does not select a
  different kernel.
- The public API accepts structural labels directly instead of building a
  string equation and parsing it again downstream.
- `contract_pair` remains exported as a temporary convenience; the direction
  is to converge on N-ary `contract` plus explicit
  `outer_product`, then remove `contract_pair`.

## Implemented

- `contract`, `contract_with_options`, and `PreparedContraction`:
  `crates/tensor4all-core/src/defaults/contract.rs`
  (re-exported from `crates/tensor4all-core/src/lib.rs`).
- Planning validates shared-label dimensions once, including retained labels
  whose index identity is equal but whose dimensions differ. Prepared plans
  validate metadata before sharing the ordinary execution dispatch. Labels
  are built lazily for ordinary calls, preserving the binary fast path without
  N-ary planning. Structured and AD operands use the compact
  N-ary contraction owner; there is no sequential-pair fallback.
- Semantics tests: `crates/tensor4all-core/src/defaults/contract/tests/mod.rs`.
- `outer_product` for disconnected inputs:
  `crates/tensor4all-core/src/defaults/contract.rs`.

## Follow-up

- Remove the `contract_pair` convenience export once consumers migrate to
  N-ary `contract` (PR 3 API-vocabulary work in tensor4all-rs#566).

## Migration for 0.4.0 (unreleased)

For an owned `Vec<IdxTensor>`, collect references without cloning payloads.
The vector remains owned by the caller; drop it after contraction if it is
no longer needed. Repeated compatible operands can use `PreparedContraction`.

- Before: `contract_owned(tensors)?`.
  After: `let refs: Vec<_> = tensors.iter().collect(); contract(&refs)?`.
- Before: `contract_owned_with_options(tensors, options)?`.
  After: `let refs: Vec<_> = tensors.iter().collect(); contract_with_options(&refs, options)?`.
- Before: `tensor4all_core::legacy::einsum_native_tensors_owned(operands, output_ids)?`.
  After, for an existing backend-level native-tensor caller:
  `tensor4all_tensorbackend::einsum_native_tensors_owned_in(&context, operands, output_ids)?`.

The third removal is a doc-hidden migration bridge, not removal of the
tensorbackend implementation. Both tensorbackend-owned entries remain,
including its feature-gated global convenience entry. An explicit caller
supplies its existing `CpuExecutionContext` to `_in`; an `IdxTensor`
algorithm stays on core's high-level `contract` / `contract_with_options`.
The corresponding `defaults` and `defaults::contract` exports of the first
two functions disappear with the root exports.

## Public-removal survey (#1381)

The [source evidence](../experiments/issue-1381-public-removal-survey.json)
records complete-identifier searches over every tracked text file at these
fixed revisions, including manifests, tests, examples, bindings and docs:

| Repository | Revision | Result |
|---|---|---|
| tensor4all-rs, before removal | `379868e01f2d` | Owned contraction references are confined to core implementation/wrappers, seven unit-test calls, re-exports and documentation. The removed core implementation calls the legacy owned-einsum bridge. |
| tensor4all-rs, after implementation | `153f4f4d1d5e` | No owned contraction entry or core legacy owned-einsum bridge remains. Tensorbackend's distinct owned-einsum implementation, matrix caller, tests and exports remain. |
| tenferro-rs | `5358cbb2a244` | No matching identifiers; no `tensor4all-core` Cargo dependency. |
| tenferro-sci-rs | `dc5026c2afdf` | No matching identifiers; no `tensor4all-core` Cargo dependency. |
| Tensor4all.jl | `22ea6c8b2388` | No matching identifiers; bindings use the C API boundary. |

No `tenferro-mpi-capi` repository was found by the repository-name search.
Within the complete, nontruncated tenferro tree above, the name occurs only
in `docs/design/mpi-native-abi.md`; no corresponding crate or source
implementation exists at that revision. This checks the available design,
not an unavailable deployed implementation.

`cargo-public-api` is not installed. The repository's source-based API dump
was run on both workspace snapshots: core's declarations for the three
removed names go from one each to zero; tensorbackend's global and `_in`
declarations remain one each. This inventory is not a feature-aware rustdoc
or semver compatibility analysis. The `legacy` wildcard export and the
`global-defaults` / refusing branches were inspected separately to identify
the precise compiled public paths. There is no crates.io usage census;
private consumers, other revisions and uninspected repositories are outside
the survey. The evidence supports removal of unused workspace paths, not a
universal claim that no downstream caller exists.

## Validation timing change (#1381)

Shared-label dimension validation moved into `build_contraction_plan`.
Previously the ordinary dense route rejected inconsistent dimensions when
executing a contraction, but `PreparedContraction::new` could successfully
construct a plan with equal-identity retained labels of different extents.
It now returns an error at construction. This is an intentional earlier
rejection of invalid metadata alongside the dispatch refactor, not a new
restriction on well-formed contractions. Structured/AD planning uses the
same check before dispatching to its kernel.

The explicit-session `contract_with_options_on_session` route also uses this
planner. Invalid shared-label dimensions now fail before operand adaptation or
session materialization, with the planner's shape-mismatch message instead of a
downstream session error. The public error remains `IdxTensorError::Operation`.

`retained_label_dimension_mismatch_is_rejected_during_planning` exercises
ordinary, prepared and explicit-session entry points with identity 701 at
dimensions 2 and 3 and requires the same shape-mismatch error. It covers the
shared planner error, including the newly documented session route.
No numerical tolerance changed.

## Prepared-call profiling (#1381)

With contract profiling enabled, prepared dense N-ary and retained contractions
now record through the same collector as ordinary dense calls. A mixed sweep
therefore reports calls and time that were previously absent for those prepared
entries. Singleton clones, the binary shortcut without retained indices and
structured/AD dispatch keep their existing profiling behavior. The numerical
result is unchanged; `prepared_dense_nary_records_profile_output_dimensions`
checks a real three-operand result and one entry with output dimensions `[2,4]`.
