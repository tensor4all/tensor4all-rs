//! Tests for linsolve module
//!
//! These tests verify the basic functionality of the linear equation solver
//! for Tree Tensor Networks.

use std::sync::Arc;

use tensor4all_core::index::{DefaultIndex as Index, DynId, NoSymmSpace};
use tensor4all_core::storage::DenseStorageF64;
use tensor4all_core::{Storage, TensorDynLen};
use tensor4all_treetn::{
    EnvironmentCache, LinsolveOptions, ProjectedOperator, ProjectedState, TreeTN,
};

// ============================================================================
// Test Helpers
// ============================================================================

/// Create a simple 3-site MPS chain for testing.
/// Returns (mps, site indices, bond indices)
fn create_simple_mps_chain() -> (
    TreeTN<DynId, NoSymmSpace, &'static str>,
    Vec<Index<DynId>>,
    Vec<Index<DynId>>,
) {
    let mut mps = TreeTN::<DynId, NoSymmSpace, &'static str>::new();

    // Physical indices (dimension 2 for each site)
    let s0 = Index::new_dyn(2);
    let s1 = Index::new_dyn(2);
    let s2 = Index::new_dyn(2);

    // Bond indices (dimension 4)
    let b01 = Index::new_dyn(4);
    let b12 = Index::new_dyn(4);

    // Create tensors with random data
    // Site 0: [s0, b01] shape (2, 4)
    let t0 = TensorDynLen::new(
        vec![s0.clone(), b01.clone()],
        vec![2, 4],
        Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![1.0; 8]))),
    );

    // Site 1: [b01, s1, b12] shape (4, 2, 4)
    let t1 = TensorDynLen::new(
        vec![b01.clone(), s1.clone(), b12.clone()],
        vec![4, 2, 4],
        Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![1.0; 32]))),
    );

    // Site 2: [b12, s2] shape (4, 2)
    let t2 = TensorDynLen::new(
        vec![b12.clone(), s2.clone()],
        vec![4, 2],
        Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![1.0; 8]))),
    );

    // Add nodes with string names (add_tensor returns Result)
    let n0 = mps.add_tensor("site0", t0).unwrap();
    let n1 = mps.add_tensor("site1", t1).unwrap();
    let n2 = mps.add_tensor("site2", t2).unwrap();

    // Connect nodes (this also updates site spaces by removing bond indices)
    mps.connect(n0, &b01, n1, &b01).unwrap();
    mps.connect(n1, &b12, n2, &b12).unwrap();

    // Site spaces are now automatically managed:
    // - site0: {s0} (b01 removed after connect)
    // - site1: {s1} (b01, b12 removed after connect)
    // - site2: {s2} (b12 removed after connect)

    (mps, vec![s0, s1, s2], vec![b01, b12])
}

// ============================================================================
// EnvironmentCache Tests
// ============================================================================

#[test]
fn test_environment_cache_basic() {
    let cache: EnvironmentCache<DynId, NoSymmSpace, &str> = EnvironmentCache::new();
    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
}

#[test]
fn test_environment_cache_insert_get() {
    let mut cache: EnvironmentCache<DynId, NoSymmSpace, &str> = EnvironmentCache::new();

    // Create a simple tensor to cache
    let idx = Index::new_dyn(2);
    let tensor = TensorDynLen::new(
        vec![idx],
        vec![2],
        Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![1.0, 2.0]))),
    );

    cache.insert("a", "b", tensor.clone());

    assert!(!cache.is_empty());
    assert_eq!(cache.len(), 1);
    assert!(cache.contains(&"a", &"b"));
    assert!(!cache.contains(&"b", &"a"));

    let retrieved = cache.get(&"a", &"b");
    assert!(retrieved.is_some());
}

#[test]
fn test_environment_cache_clear() {
    let mut cache: EnvironmentCache<DynId, NoSymmSpace, &str> = EnvironmentCache::new();

    let idx = Index::new_dyn(2);
    let tensor = TensorDynLen::new(
        vec![idx],
        vec![2],
        Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![1.0, 2.0]))),
    );

    cache.insert("a", "b", tensor);
    assert_eq!(cache.len(), 1);

    cache.clear();
    assert!(cache.is_empty());
}

// ============================================================================
// LinsolveOptions Tests
// ============================================================================

#[test]
fn test_linsolve_options_default() {
    let opts = LinsolveOptions::default();

    assert_eq!(opts.nsweeps, 10);
    assert_eq!(opts.krylov_tol, 1e-10);
    assert_eq!(opts.krylov_maxiter, 100);
    assert_eq!(opts.krylov_dim, 30);
    assert_eq!(opts.a0, 0.0);
    assert_eq!(opts.a1, 1.0);
    assert!(opts.convergence_tol.is_none());
}

#[test]
fn test_linsolve_options_builder() {
    let opts = LinsolveOptions::default()
        .with_nsweeps(5)
        .with_krylov_tol(1e-8)
        .with_krylov_maxiter(50)
        .with_krylov_dim(20)
        .with_coefficients(1.0, -1.0)
        .with_convergence_tol(1e-6);

    assert_eq!(opts.nsweeps, 5);
    assert_eq!(opts.krylov_tol, 1e-8);
    assert_eq!(opts.krylov_maxiter, 50);
    assert_eq!(opts.krylov_dim, 20);
    assert_eq!(opts.a0, 1.0);
    assert_eq!(opts.a1, -1.0);
    assert_eq!(opts.convergence_tol, Some(1e-6));
}

// ============================================================================
// ProjectedState Tests
// ============================================================================

#[test]
fn test_projected_state_creation() {
    let (mps, _, _) = create_simple_mps_chain();
    let projected_state = ProjectedState::new(mps);

    // Verify initial state
    assert!(projected_state.envs.is_empty());
}

// ============================================================================
// ProjectedOperator Tests
// ============================================================================

#[test]
fn test_projected_operator_creation() {
    let (mps, _, _) = create_simple_mps_chain();
    let projected_op = ProjectedOperator::new(mps);

    // Verify initial state
    assert!(projected_op.envs.is_empty());
}

#[test]
fn test_projected_operator_local_dimension() {
    let (mps, _sites, _) = create_simple_mps_chain();

    // Create an "operator" with the same structure but double physical indices
    // For simplicity, use the MPS itself (local_dimension checks site_space)
    let projected_op = ProjectedOperator::new(mps);

    // local_dimension should compute product of site dimensions in region
    let dim = projected_op.local_dimension(&["site0"]);
    // site0 has physical index of dim 2
    assert_eq!(dim, 2);

    let dim = projected_op.local_dimension(&["site1"]);
    assert_eq!(dim, 2);

    let dim = projected_op.local_dimension(&["site0", "site1"]);
    // 2 * 2 = 4
    assert_eq!(dim, 4);
}
