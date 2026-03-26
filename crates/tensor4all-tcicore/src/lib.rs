#![warn(missing_docs)]
//! TCI Core infrastructure
//!
//! Shared foundation for tensor cross interpolation algorithms:
//! - Matrix CI: [`MatrixLUCI`], [`MatrixACA`], [`RrLU`]
//! - [`CachedFunction`]: Thread-safe cached function evaluation with wide key support
//! - [`IndexSet`]: Bidirectional index set for pivot management
