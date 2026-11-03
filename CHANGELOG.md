# Changelog

## Unreleased — next minor release (0.3.0)

### Breaking Rust API changes

Fixed-rank randomized factorization (#1108) is a breaking pre-1.0 Rust API
change designated for the next minor release. The following changes break
exhaustive matches, complete struct literals, or named public types:

- `tensor4all-core`: `FactorizeAlg::RSVD` and
  `FactorizeError::UnsupportedTruncationPolicy` are new enum variants.
  `FactorizeOptions` adds `randomized` and `rsvd_seed`. Update exhaustive
  matches and use the options constructors or `..Default::default()` in
  struct literals.
- `tensor4all-treetn`: `ContractionOptions` and `FitOptions` add `randomized`
  and `rsvd_seed`. `FitUpdater` gains a lifetime parameter for the borrowed
  execution strategy and RNG. Update named types and complete struct literals.
- `tensor4all-simplett`: remove the unimplemented `FactorizeMethod::RSVD`
  variant and its unused `rsvd_p`/`rsvd_q` options. New randomized consumers
  use the core and TreeTN APIs.
- `tensor4all-capi`: remove the unused Rust `From<FactorizeAlg>` conversion
  to the C enum. The C ABI and C enum values are unchanged.

### Added

- Dense fixed-rank RSVD, Gaussian range finding, caller-owned RNG entries,
  and RSVD selection in TreeTN Fit and Zipup. Supported scalar types are
  `f64` and `Complex64`; `f32` and `Complex32` are unsupported.

This records release impact; package versions remain unchanged until release
preparation.
