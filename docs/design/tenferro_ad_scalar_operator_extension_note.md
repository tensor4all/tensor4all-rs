# Tenferro AD Scalar Operator Extension Note

## Objective

Provide a non-ad-hoc AD scalar operator foundation required for tensor4all backend cutover:

- Binary AD scalar rules in `chainrules-scalarops`
- Typed operator support on `tenferro-dyadtensor::AdScalar<T>`
- Dynamic operator support on `tenferro-dyadtensor::DynAdValue`
- Mixed operations (`f64`, `Complex64`) with explicit promotion policy

This note defines the implementation contract before coding.

Authoritative AD rule notes for matrix/tensor operators (for example
`solve`, `solve_triangular`, decomposition rules) belong in
`tenferro-rs/docs/AD/*`. This document is intentionally limited to scalar
operator rules.

## Scope

In scope:

- `chainrules-scalarops`: `add/sub/mul/div` primal + `frule` + `rrule`
- `AdScalar<T>`: `Add/Sub/Mul/Div` for AD-aware propagation
- `DynAdValue`: checked binary ops + operator traits
- Mixed scalar operations around `DynAdValue`:
  - `DynAdValue <op> f64`, `f64 <op> DynAdValue`
  - `DynAdValue <op> Complex64`, `Complex64 <op> DynAdValue`
- Tests and rustdoc examples for all new public APIs

Out of scope (separate task):

- Full Hessian/HVP plumbing
- New tape engine design
- Tensor-level AD compose/decompose APIs (`DynAdTensor::complex`, `real`, `imag`) beyond operator-enabling minimum

## Design Principles

- KISS: one rule source, one propagation path.
- DRY: no duplicated derivative algebra between dyadtensor and chainrules layer.
- Layering:
  - Scalar calculus lives in `chainrules-scalarops`
  - AD container mode/tape propagation lives in `tenferro-dyadtensor`
  - tensor4all consumes high-level APIs only

## API Contract

### 1) `chainrules-scalarops`

Add functions:

- `add`, `add_frule`, `add_rrule`
- `sub`, `sub_frule`, `sub_rrule`
- `mul`, `mul_frule`, `mul_rrule`
- `div`, `div_frule`, `div_rrule`

Rule shape:

- `*_frule`: returns `(primal, tangent)`
- `*_rrule`: returns argument cotangents (`(dx, dy)` for binary ops)

Complex convention:

- Keep current package convention already used by `powf/powi/sqrt`: derivative factors are conjugated where appropriate (PyTorch-compatible complex behavior target).

### 2) `AdScalar<T>` operator traits

Implement:

- `impl<T: ScalarAd> Add for AdScalar<T>`
- `impl<T: ScalarAd> Sub for AdScalar<T>`
- `impl<T: ScalarAd> Mul for AdScalar<T>`
- `impl<T: ScalarAd> Div for AdScalar<T>`

Propagation rules:

- Mode precedence: `Reverse > Forward > Primal`
- Tangent propagation:
  - If any operand carries tangent, use scalar `frule`
  - Missing tangent is treated as zero tangent
- Reverse metadata:
  - If reverse mode is present, output remains reverse
  - If both sides are reverse, tapes must match

Tape mismatch handling:

- Enforced check; mismatch is an error in checked paths
- Operator trait path must not silently merge incompatible tapes

### 3) `DynAdValue` operators

Provide checked APIs:

- `try_add`, `try_sub`, `try_mul`, `try_div` returning `Result<DynAdValue>`

Provide operator traits for ergonomic use:

- `Add/Sub/Mul/Div` for `DynAdValue` (`Output = DynAdValue`)
- Mixed scalar operator impls with `f64` and `Complex64` on both sides

Operator trait behavior:

- Internally delegates to checked APIs
- If unsupported pair appears in an operator expression, fail loudly (panic with explicit message)
- Callers that require non-panicking behavior should use `try_*`

Promotion policy (v1):

- `F64` with `C64` promotes to `C64`
- `F32` with `C32` promotes to `C32`
- Same dtype stays same dtype
- Cross precision (`F32` with `F64`, `C32` with `C64`, etc.) is unsupported in v1 checked APIs

## Acceptance Tests

### `chainrules-scalarops`

- Real formulas: `mul/div` frule and rrule for `f64`
- Complex formulas: `mul/div` frule and rrule for `Complex64`
- Sanity: `add/sub` formulas for both real and complex

### `tenferro-dyadtensor` AdScalar

- Forward-mode tangent propagation for `mul` and `div`
- Reverse metadata preservation for binary ops
- Tape mismatch check path is covered

### `tenferro-dyadtensor` DynAdValue

- `f64 * DynAdValue(C64)` works and promotes to `C64`
- `Complex64 / DynAdValue(F64)` works and promotes to `C64`
- Unsupported dtype pair via checked API returns error

## Execution Order

1. Implement and test binary rules in `chainrules-scalarops`
2. Implement and test `AdScalar<T>` operators
3. Implement and test `DynAdValue` checked/operator APIs and mixed scalar impls
4. Run formatting + targeted tests in release mode

## Risk and Mitigation

Risk:

- Inconsistent complex derivative convention across old/new rules

Mitigation:

- Reuse the same conjugation style as existing `powf/powi/sqrt` rules
- Add explicit complex-valued tests comparing analytic formulas

Risk:

- Runtime panic from operator overload in dynamic paths

Mitigation:

- Expose checked `try_*` API and use it internally where recoverability matters
- Keep panic messages explicit and actionable
