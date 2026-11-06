//! Platform default CPU backend for tensor4all.
//!
//! This crate has no items. Depending on it selects tenferro's CPU backend by
//! target: Apple Accelerate (`blas-accelerate`) on macOS and faer (`native`)
//! everywhere else. The tensor4all crates enable it through their default
//! `tenferro-cpu-platform` feature; an explicit backend feature
//! (`tenferro-cpu-faer`, `tenferro-system-blas`, `tenferro-provider-inject`)
//! replaces it under `--no-default-features`.
