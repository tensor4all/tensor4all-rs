# SVD Truncation Policy Design

## Goal

Replace the current `rtol/cutoff/max_rank` truncation interface for SVD-based
operations with an explicit policy object that can represent:

- singular-value vs squared-singular-value thresholds
- per-value vs discarded-tail-sum rules
- relative vs absolute scaling

At the same time, remove the current API confusion that lets `truncate`
superficially expose `LU/CI` even though the actual truncation sweep is SVD
based.

## Context

The current design mixes together several concerns:

- `TruncationParams` is shared across SVD, QR, and higher-level APIs even
  though the semantics differ.
- `cutoff` is only stored as a compatibility alias and is immediately collapsed
  into `rtol = sqrt(cutoff)`.
- `truncate` APIs expose `LU/CI` even though `TreeTN::truncate` ultimately uses
  SVD in the truncation sweep.
- `FactorizeOptions` has a single `rtol` field even though SVD, QR, and LU use
  different truncation/rank-selection criteria.

This prevents expressing the four intended SVD truncation semantics:

1. value + per-value
2. value + discarded-tail-sum
3. squared-value + per-value
4. squared-value + discarded-tail-sum

## Non-Goals

- Do not preserve backwards compatibility for `rtol`, `cutoff`, `maxdim`,
  `TruncateOptions::lu()`, or `TruncateOptions::ci()`. This repository is in
  early development and deprecated compatibility layers should be removed
  immediately.
- Do not force LU/CI to share the same truncation policy type as SVD. Their
  rank selection is not based on singular values.
- Do not redesign the C API in the same change. Rust APIs should be cleaned up
  first, then FFI can follow in a separate pass.

## Primary Type

```rust
pub struct SvdTruncationPolicy {
    pub threshold: f64,
    pub scale: ThresholdScale,
    pub measure: SingularValueMeasure,
    pub rule: TruncationRule,
}

pub enum ThresholdScale {
    Relative,
    Absolute,
}

pub enum SingularValueMeasure {
    Value,
    SquaredValue,
}

pub enum TruncationRule {
    PerValue,
    DiscardedTailSum,
}
```

### Defaults

`SvdTruncationPolicy::new(threshold)` uses:

- `ThresholdScale::Relative`
- `SingularValueMeasure::Value`
- `TruncationRule::PerValue`

This makes the default policy correspond to the current intuitive
`sigma_i / sigma_max <= threshold` style API.

### Builder Surface

```rust
impl SvdTruncationPolicy {
    pub fn new(threshold: f64) -> Self;
    pub fn with_relative(self) -> Self;
    pub fn with_absolute(self) -> Self;
    pub fn with_values(self) -> Self;
    pub fn with_squared_values(self) -> Self;
    pub fn with_per_value(self) -> Self;
    pub fn with_discarded_tail_sum(self) -> Self;
}
```

Examples:

```rust
let default_policy = SvdTruncationPolicy::new(1e-12);

let squared_tail = SvdTruncationPolicy::new(1e-10)
    .with_squared_values()
    .with_discarded_tail_sum();

let absolute_per_value = SvdTruncationPolicy::new(1e-8)
    .with_absolute()
    .with_values()
    .with_per_value();
```

## Semantics

Let `measure_i` denote either `sigma_i` or `sigma_i^2`, depending on
`measure`.

### Relative Per-Value

`ThresholdScale::Relative + TruncationRule::PerValue`

Keep the largest prefix such that:

`measure_i / max(measure) > threshold`

This uses the largest kept singular-value-derived quantity as the natural
reference scale. This is the direct generalization of current `rtol` semantics.

### Relative Discarded Tail Sum

`ThresholdScale::Relative + TruncationRule::DiscardedTailSum`

Discard from the smallest end while:

`(discarded_sum + next_measure) / sum(all_measures) <= threshold`

This is an early-exit suffix accumulation rule. It matches the current ITensor
style definition for relative discarded weight when the measure is squared
singular values.

Reference:

- docs: <https://itensor.github.io/ITensors.jl/stable/ITensorType.html>
- implementation: <https://github.com/ITensor/ITensors.jl/blob/3a62e79627afec50afd63b5ca942c060da1c0c0e/NDTensors/src/truncate.jl>

### Absolute Per-Value

Keep values satisfying:

`measure_i > threshold`

### Absolute Discarded Tail Sum

Discard from the smallest end while:

`discarded_sum + next_measure <= threshold`

### `max_rank` Interaction

`max_rank` remains independent of the policy and is applied as a hard cap on
the retained rank.

The retained rank is:

`min(rank_allowed_by_policy, max_rank.unwrap_or(usize::MAX))`

At least rank 1 is kept for non-empty spectra.

## API Restructuring

### `tensor4all-core`

#### `truncation.rs`

Split the current shared truncation container into algorithm-specific concepts.

- Remove `TruncationParams`
- Remove `HasTruncationParams`
- Keep `DecompositionAlg`
- Add `SvdTruncationPolicy`

This file becomes the home of algorithm-selection enums plus SVD policy types,
not a forced common container for all decomposition families.

#### `SvdOptions`

Current:

```rust
pub struct SvdOptions {
    pub truncation: TruncationParams,
}
```

New:

```rust
pub struct SvdOptions {
    pub max_rank: Option<usize>,
    pub policy: Option<SvdTruncationPolicy>,
}
```

`SvdOptions::with_rtol` and `SvdOptions::with_cutoff` are removed.

New builder:

```rust
SvdOptions::new()
    .with_max_rank(64)
    .with_policy(SvdTruncationPolicy::new(1e-12));
```

Global default SVD truncation should become a default `SvdTruncationPolicy`
instead of a bare `f64 rtol`.

#### `QrOptions`

QR retains its own semantics. It should not reuse `SvdTruncationPolicy`.

Recommended shape:

```rust
pub struct QrOptions {
    pub rtol: Option<f64>,
}
```

This keeps QR rank selection independent and avoids smuggling SVD policy fields
into QR.

#### `FactorizeOptions`

`FactorizeOptions` remains as a thin multi-algorithm facade:

```rust
pub struct FactorizeOptions {
    pub alg: FactorizeAlg,
    pub canonical: Canonical,
    pub max_rank: Option<usize>,
    pub svd_policy: Option<SvdTruncationPolicy>,
    pub qr_rtol: Option<f64>,
}
```

Validation rules:

- `SVD/RSVD`: allow `max_rank` and `svd_policy`; reject `qr_rtol`
- `QR`: allow `max_rank` and `qr_rtol`; reject `svd_policy`
- `LU/CI`: allow `max_rank`; reject `svd_policy` and `qr_rtol`

This keeps the public surface pragmatic while making invalid combinations
explicitly illegal.

### `tensor4all-treetn`

#### `CanonicalizationOptions`

Keep `CanonicalForm::{Unitary, LU, CI}`. Canonicalization is where LU/CI belong.

#### `TruncationOptions`

Current `TruncationOptions` mixes canonical form and truncation settings.

New shape:

```rust
pub struct TruncationOptions {
    pub max_rank: Option<usize>,
    pub svd_policy: Option<SvdTruncationPolicy>,
}
```

Remove:

- `form`
- `with_form`
- `with_rtol`
- `with_cutoff`

`TreeTN::truncate` is then honestly SVD-based.

#### `SplitOptions`

If split continues to be SVD-based when truncation is requested, it should also
hold `svd_policy` explicitly instead of sharing the old `TruncationParams`.

#### `ApplyOptions`, `ContractionOptions`, `FitContractionOptions`, `LinsolveOptions`

Any high-level option that currently exposes `rtol` for an SVD-based truncation
path should be updated to expose an `SvdTruncationPolicy` instead.

For fit-style APIs that allow non-SVD factorization algorithms, the rule is:

- SVD/RSVD: `svd_policy` is legal
- QR: use `qr_rtol`
- LU/CI: no SVD policy accepted

### `tensor4all-itensorlike`

#### `TruncateOptions`

Restrict to SVD-based truncation only.

Remove:

- `TruncateOptions::lu()`
- `TruncateOptions::ci()`
- `with_rtol`
- `with_cutoff`
- `with_maxdim`

New shape:

```rust
pub struct TruncateOptions {
    svd_policy: Option<SvdTruncationPolicy>,
    max_rank: Option<usize>,
    site_range: Option<Range<usize>>,
}
```

#### `ContractOptions` and `LinsolveOptions`

Replace `rtol/cutoff` builders with:

- `with_svd_policy(...)`
- `with_max_rank(...)`

and keep sweep-count / convergence settings as they are.

## Validation Rules

All policy-carrying APIs should reject:

- non-finite thresholds
- negative thresholds
- `max_rank == 0`

Algorithm-dependent validation should reject impossible field combinations
before any numerical work starts.

## Testing Strategy

### Core Policy Tests

Add focused unit tests for:

- default policy values
- builder overrides
- retained-rank computation for all four measure/rule combinations
- relative vs absolute semantics
- `max_rank` cap interaction

### API Cleanup Tests

Add tests that:

- verify `truncate` no longer exposes LU/CI
- verify invalid `FactorizeOptions` combinations return errors
- verify high-level APIs propagate `SvdTruncationPolicy` correctly

### Regression Coverage

Retain behavioral coverage for existing SVD truncation workflows by rewriting
tests to use explicit policies instead of `rtol/cutoff`.

## Risks

### Builder Churn

Many tests and examples currently use `with_rtol` and `with_cutoff`. This is
expected and acceptable because backwards compatibility is explicitly out of
scope.

### Hidden QR Coupling

`TruncationParams` currently leaks SVD terminology into QR. Careless refactors
can accidentally break QR rank selection. QR must keep its own option path and
tests.

### High-Level Fit Semantics

Some fit APIs intentionally treat omitted tolerance and explicit `0.0`
similarly. That behavior needs to be re-expressed carefully once `rtol`
disappears, likely by interpreting `None` as "no policy" and using explicit
bond caps to keep current behavior where intended.

## Follow-Up Work

- Update the C API after the Rust API stabilizes.
- Update Julia/Python bindings after the Rust and C surfaces are settled.
