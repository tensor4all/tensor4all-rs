//! Trait implementations and operations for TreeTN.
//!
//! This module provides:
//! - `Default` implementation
//! - `Clone` implementation
//! - `Debug` implementation
//! - TODO: Re-enable after TensorLike adds scalar multiplication support:
//!   - `Mul<f64>` and `Mul<Complex64>` for scalar multiplication
//!   - `log_norm` for computing the logarithm of the Frobenius norm

use std::hash::Hash;

use tensor4all_core::TensorLike;

use super::TreeTN;

// ============================================================================
// Default implementation
// ============================================================================

impl<T, V> Default for TreeTN<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Clone implementation
// ============================================================================

impl<T, V> Clone for TreeTN<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    fn clone(&self) -> Self {
        Self {
            graph: self.graph.clone(),
            canonical_center: self.canonical_center.clone(),
            canonical_form: self.canonical_form,
            site_index_network: self.site_index_network.clone(),
            ortho_towards: self.ortho_towards.clone(),
        }
    }
}

// ============================================================================
// Debug implementation
// ============================================================================

impl<T, V> std::fmt::Debug for TreeTN<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TreeTN")
            .field("node_count", &self.node_count())
            .field("edge_count", &self.edge_count())
            .field("canonical_center", &self.canonical_center)
            .finish_non_exhaustive()
    }
}

// ============================================================================
// Scalar multiplication for TreeTN (TODO: Re-enable after TensorLike refactoring)
// ============================================================================
//
// The scalar multiplication implementations are temporarily disabled because they
// access internal tensor fields (`storage`) that are not exposed by the TensorLike trait.
//
// To re-enable, either:
// 1. Add a `scale(&self, factor: f64) -> Self` method to TensorLike
// 2. Add a `scale_complex(&self, factor: Complex64) -> Self` method to TensorLike
// 3. Use a different approach that doesn't require internal field access

// ============================================================================
// Norm Computation (TODO: Re-enable after TensorLike refactoring)
// ============================================================================
//
// The log_norm implementation is temporarily disabled because it requires:
// 1. The `canonicalize_mut` method (from the disabled canonicalize module)
// 2. The `norm_squared` method on tensors (not exposed by TensorLike)
//
// To re-enable:
// 1. Re-enable the canonicalize module
// 2. Add `norm_squared(&self) -> f64` method to TensorLike
