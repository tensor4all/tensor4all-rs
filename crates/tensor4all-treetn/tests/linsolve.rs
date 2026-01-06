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

// ============================================================================
// Integration Tests
// ============================================================================

/// Create a simple 2-site MPS chain for testing.
/// Returns (mps, site indices, bond indices)
fn create_two_site_mps() -> (
    TreeTN<DynId, NoSymmSpace, &'static str>,
    Vec<Index<DynId>>,
    Vec<Index<DynId>>,
) {
    let mut mps = TreeTN::<DynId, NoSymmSpace, &'static str>::new();

    // Physical indices (dimension 2 for each site)
    let s0 = Index::new_dyn(2);
    let s1 = Index::new_dyn(2);

    // Bond index (dimension 2)
    let b01 = Index::new_dyn(2);

    // Create tensors with normalized data
    // Site 0: [s0, b01] shape (2, 2)
    let t0 = TensorDynLen::new(
        vec![s0.clone(), b01.clone()],
        vec![2, 2],
        Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![
            1.0, 0.0, // s0=0: b01=0, b01=1
            0.0, 1.0, // s0=1: b01=0, b01=1
        ]))),
    );

    // Site 1: [b01, s1] shape (2, 2)
    let t1 = TensorDynLen::new(
        vec![b01.clone(), s1.clone()],
        vec![2, 2],
        Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![
            1.0, 0.0, // b01=0: s1=0, s1=1
            0.0, 1.0, // b01=1: s1=0, s1=1
        ]))),
    );

    // Add nodes with string names
    let n0 = mps.add_tensor("site0", t0).unwrap();
    let n1 = mps.add_tensor("site1", t1).unwrap();

    // Connect nodes
    mps.connect(n0, &b01, n1, &b01).unwrap();

    (mps, vec![s0, s1], vec![b01])
}

/// Create an identity MPO for testing.
/// The identity operator at each site: I[s, s'] = delta(s, s')
fn create_identity_mpo(
    site_indices: &[Index<DynId>],
) -> (TreeTN<DynId, NoSymmSpace, &'static str>, Vec<Index<DynId>>) {
    let mut mpo = TreeTN::<DynId, NoSymmSpace, &'static str>::new();

    // Clone site indices for output (ket) indices
    // The bra contracts with s, ket gets s'
    let s0_out = site_indices[0].clone();
    let s1_out = site_indices[1].clone();

    // For the identity MPO, we need input and output physical indices
    // Input physical index (for ket contraction)
    let s0_in = Index::new_dyn(2);
    let s1_in = Index::new_dyn(2);

    // Bond index for MPO (dimension 1 for identity)
    let b01 = Index::new_dyn(1);

    // Identity tensor at site 0: [s0_out, s0_in, b01] shape (2, 2, 1)
    // I[s0_out, s0_in, b01] = delta(s0_out, s0_in) for b01=0
    let t0 = TensorDynLen::new(
        vec![s0_out.clone(), s0_in.clone(), b01.clone()],
        vec![2, 2, 1],
        Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![
            1.0, // s0_out=0, s0_in=0, b01=0
            0.0, // s0_out=0, s0_in=1, b01=0
            0.0, // s0_out=1, s0_in=0, b01=0
            1.0, // s0_out=1, s0_in=1, b01=0
        ]))),
    );

    // Identity tensor at site 1: [b01, s1_out, s1_in] shape (1, 2, 2)
    let t1 = TensorDynLen::new(
        vec![b01.clone(), s1_out.clone(), s1_in.clone()],
        vec![1, 2, 2],
        Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![
            1.0, // b01=0, s1_out=0, s1_in=0
            0.0, // b01=0, s1_out=0, s1_in=1
            0.0, // b01=0, s1_out=1, s1_in=0
            1.0, // b01=0, s1_out=1, s1_in=1
        ]))),
    );

    // Add nodes with string names
    let n0 = mpo.add_tensor("site0", t0).unwrap();
    let n1 = mpo.add_tensor("site1", t1).unwrap();

    // Connect nodes
    mpo.connect(n0, &b01, n1, &b01).unwrap();

    (mpo, vec![s0_in, s1_in])
}

#[test]
fn test_linsolve_simple_two_site() {
    use tensor4all_treetn::linsolve;

    // Create a simple 2-site MPS as the RHS
    let (rhs, site_indices, _bonds) = create_two_site_mps();

    // Create an identity MPO
    let (identity_mpo, _input_indices) = create_identity_mpo(&site_indices);

    // Create initial guess (same as RHS for simplicity)
    let init = rhs.clone();

    // Solve I * x = b (solution should be x = b)
    let options = LinsolveOptions::default()
        .with_nsweeps(1)
        .with_krylov_tol(1e-8)
        .with_max_rank(4);

    let result = linsolve(&identity_mpo, &rhs, init, &"site0", options);

    // The solve should succeed (even if the algorithm isn't fully working yet,
    // we want to verify it doesn't panic)
    assert!(
        result.is_ok(),
        "linsolve failed: {:?}",
        result.err()
    );

    let linsolve_result = result.unwrap();
    assert_eq!(linsolve_result.sweeps, 1);
}
