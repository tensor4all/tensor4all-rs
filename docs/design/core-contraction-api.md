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
  execution, or dense native execution. The unused owned entry points were
  removed in #1381; owning operands does not select a different kernel.
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
  validate metadata before sharing the ordinary execution dispatch. Labels are built lazily for ordinary calls, preserving the binary
  fast path without N-ary planning. Structured and AD operands use the compact
  N-ary contraction owner; there is no sequential-pair fallback.
- Semantics tests: `crates/tensor4all-core/src/defaults/contract/tests/mod.rs`.
- `outer_product` for disconnected inputs:
  `crates/tensor4all-core/src/defaults/contract.rs`.

## Follow-up

- Remove the `contract_pair` convenience export once consumers migrate to
  N-ary `contract` (PR 3 API-vocabulary work in tensor4all-rs#566).
