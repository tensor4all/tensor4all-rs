//! Contraction operations for SubDomainTT and PartitionedTT
//!
//! This module provides helper functions for contracting tensor trains within
//! partitioned structures.
//!
//! The main contraction functionality is implemented in `SubDomainTT::contract`
//! and `PartitionedTT::contract`. This module provides additional utilities.

use crate::error::Result;
use crate::projector::Projector;
use crate::subdomain_tt::SubDomainTT;
use tensor4all_itensorlike::ContractOptions;

/// Contract two SubDomainTTs.
///
/// The contraction is only non-vanishing if the projectors are compatible.
/// Returns `None` if the projectors are incompatible.
pub fn contract(
    m1: &SubDomainTT,
    m2: &SubDomainTT,
    options: &ContractOptions,
) -> Result<Option<SubDomainTT>> {
    m1.contract(m2, options)
}

/// Project two SubDomainTTs to a projector before contracting them.
pub fn proj_contract(
    m1: &SubDomainTT,
    m2: &SubDomainTT,
    proj: &Projector,
    options: &ContractOptions,
) -> Result<Option<SubDomainTT>> {
    // Project both inputs
    let m1_proj = match m1.project(proj) {
        Some(m) => m,
        None => return Ok(None),
    };
    let m2_proj = match m2.project(proj) {
        Some(m) => m,
        None => return Ok(None),
    };

    // Contract the projected tensor trains
    m1_proj.contract(&m2_proj, options)
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;
    use tensor4all_core::index::Index;
    use tensor4all_core::{DynIndex, StorageScalar, TensorDynLen};
    use tensor4all_itensorlike::TensorTrain;

    /// Trait for scalar types used in tests.
    /// Provides generic creation and comparison capabilities.
    trait TestScalar:
        StorageScalar + From<f64> + std::ops::Sub<Output = Self> + std::fmt::Debug
    {
        /// Get the absolute value/norm for comparison
        fn abs_diff(a: Self, b: Self) -> f64;

        /// Create zero value
        fn zero_val() -> Self;

        /// Extract data slice from tensor
        fn extract_slice(tensor: &TensorDynLen) -> &[Self];
    }

    impl TestScalar for f64 {
        fn abs_diff(a: Self, b: Self) -> f64 {
            (a - b).abs()
        }

        fn zero_val() -> Self {
            0.0
        }

        fn extract_slice(tensor: &TensorDynLen) -> &[Self] {
            tensor.as_slice_f64().unwrap()
        }
    }

    impl TestScalar for Complex64 {
        fn abs_diff(a: Self, b: Self) -> f64 {
            (a - b).norm()
        }

        fn zero_val() -> Self {
            Complex64::new(0.0, 0.0)
        }

        fn extract_slice(tensor: &TensorDynLen) -> &[Self] {
            tensor.as_slice_c64().unwrap()
        }
    }

    fn make_index(size: usize) -> DynIndex {
        Index::new_dyn(size)
    }

    /// Create a tensor with generic scalar type
    fn make_tensor_generic<T: TestScalar>(indices: Vec<DynIndex>) -> TensorDynLen {
        let dims: Vec<usize> = indices.iter().map(|i| i.dim).collect();
        let size: usize = dims.iter().product();
        let data: Vec<T> = (0..size).map(|i| T::from((i + 1) as f64)).collect();
        let storage = T::dense_storage_with_shape(data, &dims);
        TensorDynLen::new(indices, storage)
    }

    /// Create indices for testing TT contraction
    /// TT1: s0 -- l01 -- s1
    /// TT2: s1 -- l12 -- s2
    /// s1 is the shared index that will be contracted
    fn make_contraction_indices() -> (DynIndex, DynIndex, DynIndex, DynIndex, DynIndex) {
        let s0 = make_index(2); // external index of TT1
        let l01 = make_index(3); // link index of TT1
        let s1 = make_index(2); // shared index (contracted)
        let l12 = make_index(3); // link index of TT2
        let s2 = make_index(2); // external index of TT2
        (s0, l01, s1, l12, s2)
    }

    /// Create a 2-site tensor train with generic scalar type: site0 -- link -- site1
    fn make_tt_generic<T: TestScalar>(
        site0: &DynIndex,
        link: &DynIndex,
        site1: &DynIndex,
    ) -> TensorTrain {
        let t0 = make_tensor_generic::<T>(vec![site0.clone(), link.clone()]);
        let t1 = make_tensor_generic::<T>(vec![link.clone(), site1.clone()]);
        TensorTrain::new(vec![t0, t1]).unwrap()
    }

    #[test]
    fn test_projector_compatibility_check() {
        let (s0, l01, s1, l12, s2) = make_contraction_indices();

        let tt1 = make_tt_generic::<f64>(&s0, &l01, &s1);
        let m1 = SubDomainTT::new(tt1, Projector::from_pairs([(s0.clone(), 0)]));

        let tt2 = make_tt_generic::<f64>(&s1, &l12, &s2);
        let m2 = SubDomainTT::new(tt2, Projector::from_pairs([(s2.clone(), 0)]));

        // Projectors are compatible (different indices)
        assert!(m1.projector().has_overlap(m2.projector()));
    }

    #[test]
    fn test_projector_incompatibility_check() {
        let (s0, l01, s1, l12, s2) = make_contraction_indices();

        let tt1 = make_tt_generic::<f64>(&s0, &l01, &s1);
        let m1 = SubDomainTT::new(tt1, Projector::from_pairs([(s1.clone(), 0)]));

        let tt2 = make_tt_generic::<f64>(&s1, &l12, &s2);
        // Incompatible: same shared index s1, different values
        let m2 = SubDomainTT::new(tt2, Projector::from_pairs([(s1.clone(), 1)]));

        // Projectors conflict (s1=0 vs s1=1)
        assert!(!m1.projector().has_overlap(m2.projector()));
    }

    #[test]
    fn test_contract_compatible_projectors() {
        let (s0, l01, s1, l12, s2) = make_contraction_indices();

        // TT1 with projector on s0
        let tt1 = make_tt_generic::<f64>(&s0, &l01, &s1);
        let m1 = SubDomainTT::new(tt1, Projector::from_pairs([(s0.clone(), 0)]));

        // TT2 with projector on s2
        let tt2 = make_tt_generic::<f64>(&s1, &l12, &s2);
        let m2 = SubDomainTT::new(tt2, Projector::from_pairs([(s2.clone(), 1)]));

        // Contract: s1 is contracted away, result has s0 and s2
        let options = ContractOptions::default();
        let result = contract(&m1, &m2, &options).unwrap();

        assert!(result.is_some());
        let contracted = result.unwrap();

        // Contracted result should have projectors for s0 and s2
        assert!(contracted.projector().is_projected_at(&s0));
        assert!(contracted.projector().is_projected_at(&s2));
        // s1 was contracted away, should not be projected
        assert!(!contracted.projector().is_projected_at(&s1));
    }

    #[test]
    fn test_contract_incompatible_projectors() {
        let (s0, l01, s1, l12, s2) = make_contraction_indices();

        // TT1 with projector s1=0
        let tt1 = make_tt_generic::<f64>(&s0, &l01, &s1);
        let m1 = SubDomainTT::new(tt1, Projector::from_pairs([(s1.clone(), 0)]));

        // TT2 with conflicting projector s1=1
        let tt2 = make_tt_generic::<f64>(&s1, &l12, &s2);
        let m2 = SubDomainTT::new(tt2, Projector::from_pairs([(s1.clone(), 1)]));

        // Contract should return None due to incompatible projectors
        let options = ContractOptions::default();
        let result = contract(&m1, &m2, &options).unwrap();

        assert!(result.is_none());
    }

    #[test]
    fn test_contract_no_projectors() {
        let (s0, l01, s1, l12, s2) = make_contraction_indices();

        // TT1 without projector
        let tt1 = make_tt_generic::<f64>(&s0, &l01, &s1);
        let m1 = SubDomainTT::from_tt(tt1);

        // TT2 without projector
        let tt2 = make_tt_generic::<f64>(&s1, &l12, &s2);
        let m2 = SubDomainTT::from_tt(tt2);

        // Contract should succeed
        let options = ContractOptions::default();
        let result = contract(&m1, &m2, &options).unwrap();

        assert!(result.is_some());
        let contracted = result.unwrap();

        // Result should have empty projector
        assert!(contracted.projector().is_empty());

        // Result should have site indices s0 and s2 (s1 contracted away)
        let result_indices = contracted.all_indices();
        assert!(result_indices.contains(&s0));
        assert!(result_indices.contains(&s2));
        assert!(!result_indices.contains(&s1));
    }

    #[test]
    fn test_proj_contract_projector_filtering() {
        let (s0, l01, s1, _l12, _s2) = make_contraction_indices();

        let tt1 = make_tt_generic::<f64>(&s0, &l01, &s1);
        let m1 = SubDomainTT::from_tt(tt1);

        let proj = Projector::from_pairs([(s0.clone(), 0)]);

        // Project should work since m1 has empty projector (compatible with anything)
        let projected = m1.project(&proj);
        assert!(projected.is_some());
        assert!(projected.unwrap().projector().is_projected_at(&s0));
    }

    #[test]
    fn test_proj_contract() {
        let (s0, l01, s1, l12, s2) = make_contraction_indices();

        // TT1 without projector
        let tt1 = make_tt_generic::<f64>(&s0, &l01, &s1);
        let m1 = SubDomainTT::from_tt(tt1);

        // TT2 without projector
        let tt2 = make_tt_generic::<f64>(&s1, &l12, &s2);
        let m2 = SubDomainTT::from_tt(tt2);

        // Project both to s0=0, then contract
        let proj = Projector::from_pairs([(s0.clone(), 0)]);
        let options = ContractOptions::default();
        let result = proj_contract(&m1, &m2, &proj, &options).unwrap();

        assert!(result.is_some());
        let contracted = result.unwrap();

        // Result should have s0 projected
        assert!(contracted.projector().is_projected_at(&s0));
    }

    /// Generic numerical correctness test for contraction
    fn test_contract_numerical_correctness_generic<T: TestScalar>() {
        let (s0, l01, s1, l12, s2) = make_contraction_indices();

        // Create TTs
        let tt1 = make_tt_generic::<T>(&s0, &l01, &s1);
        let tt2 = make_tt_generic::<T>(&s1, &l12, &s2);

        // Convert to full tensors
        let t1_full = tt1.to_dense().unwrap();
        let t2_full = tt2.to_dense().unwrap();

        // Contract SubDomainTTs
        let m1 = SubDomainTT::from_tt(tt1);
        let m2 = SubDomainTT::from_tt(tt2);
        // Use Naive method for exact results (no approximation)
        let options = ContractOptions::naive();
        let result = contract(&m1, &m2, &options).unwrap().unwrap();
        let contracted_tt = result.data();

        // Convert contracted TT to full tensor
        let contracted_full = contracted_tt.to_dense().unwrap();

        // Manually compute expected contraction: R = T1 * T2 (contracting s1)
        // T1 has indices [s0, s1], T2 has indices [s1, s2]
        // Result R has indices [s0, s2]
        let expected = t1_full.contract(&t2_full);

        // Compare: both should have the same data
        let contracted_data = T::extract_slice(&contracted_full);
        let expected_data = T::extract_slice(&expected);

        assert_eq!(
            contracted_data.len(),
            expected_data.len(),
            "Contracted and expected tensors should have the same size"
        );

        for (i, (&actual, &exp)) in contracted_data.iter().zip(expected_data.iter()).enumerate() {
            assert!(
                T::abs_diff(actual, exp) < 1e-10,
                "Mismatch at index {}: actual={:?}, expected={:?}",
                i,
                actual,
                exp
            );
        }
    }

    #[test]
    fn test_contract_numerical_correctness_f64() {
        test_contract_numerical_correctness_generic::<f64>();
    }

    #[test]
    fn test_contract_numerical_correctness_c64() {
        test_contract_numerical_correctness_generic::<Complex64>();
    }

    /// Generic numerical correctness test for contraction with projectors
    fn test_contract_with_projectors_numerical_correctness_generic<T: TestScalar>() {
        let (s0, l01, s1, l12, s2) = make_contraction_indices();

        // Create TTs
        let tt1 = make_tt_generic::<T>(&s0, &l01, &s1);
        let tt2 = make_tt_generic::<T>(&s1, &l12, &s2);

        // Convert to full tensors before projection
        let t1_full = tt1.to_dense().unwrap();
        let t2_full = tt2.to_dense().unwrap();

        // Create SubDomainTTs with projectors
        // m1 is projected at s0=0
        // m2 is projected at s2=1
        let proj1 = Projector::from_pairs([(s0.clone(), 0)]);
        let proj2 = Projector::from_pairs([(s2.clone(), 1)]);

        let m1 = SubDomainTT::new(tt1, proj1.clone());
        let m2 = SubDomainTT::new(tt2, proj2.clone());

        // Contract
        // Use Naive method for exact results (no approximation)
        let options = ContractOptions::naive();
        let result = contract(&m1, &m2, &options).unwrap().unwrap();
        let contracted_tt = result.data();

        // Verify projector is correct
        assert!(result.projector().is_projected_at(&s0));
        assert!(result.projector().is_projected_at(&s2));
        assert!(!result.projector().is_projected_at(&s1)); // s1 was contracted

        // Convert contracted TT to full tensor
        let contracted_full = contracted_tt.to_dense().unwrap();

        // Compute expected result manually:
        // 1. Project t1_full to s0=0 (zero out s0=1)
        // 2. Project t2_full to s2=1 (zero out s2=0)
        // 3. Contract the projected tensors

        // Project t1_full: zero out s0=1 slice
        // t1_full has indices [s0, s1], shape [2, 2]
        let t1_data = T::extract_slice(&t1_full);
        let mut t1_proj_data = vec![T::zero_val(); t1_data.len()];
        // Keep only s0=0 row: indices 0, 1 (s0*s1.dim + s1 = 0*2 + 0, 0*2 + 1)
        t1_proj_data[0] = t1_data[0]; // [0, 0]
        t1_proj_data[1] = t1_data[1]; // [0, 1]
                                      // t1_proj_data[2], [3] remain 0 (s0=1 row)

        let t1_proj = TensorDynLen::from_dense_data(vec![s0.clone(), s1.clone()], t1_proj_data);

        // Project t2_full: zero out s2=0 slice
        // t2_full has indices [s1, s2], shape [2, 2]
        let t2_data = T::extract_slice(&t2_full);
        let mut t2_proj_data = vec![T::zero_val(); t2_data.len()];
        // Keep only s2=1 column: indices 1, 3 (s1*s2.dim + s2 = 0*2 + 1, 1*2 + 1)
        t2_proj_data[1] = t2_data[1]; // [0, 1]
        t2_proj_data[3] = t2_data[3]; // [1, 1]

        let t2_proj = TensorDynLen::from_dense_data(vec![s1.clone(), s2.clone()], t2_proj_data);

        // Contract projected tensors
        let expected = t1_proj.contract(&t2_proj);

        // Compare
        let contracted_data = T::extract_slice(&contracted_full);
        let expected_data = T::extract_slice(&expected);

        assert_eq!(
            contracted_data.len(),
            expected_data.len(),
            "Contracted and expected tensors should have the same size"
        );

        for (i, (&actual, &exp)) in contracted_data.iter().zip(expected_data.iter()).enumerate() {
            assert!(
                T::abs_diff(actual, exp) < 1e-10,
                "Mismatch at index {}: actual={:?}, expected={:?}",
                i,
                actual,
                exp
            );
        }
    }

    #[test]
    fn test_contract_with_projectors_numerical_correctness_f64() {
        test_contract_with_projectors_numerical_correctness_generic::<f64>();
    }

    #[test]
    fn test_contract_with_projectors_numerical_correctness_c64() {
        test_contract_with_projectors_numerical_correctness_generic::<Complex64>();
    }

    /// Generic test for contraction when the contracted index has a projector
    fn test_contract_with_projector_on_contracted_index_generic<T: TestScalar>() {
        // Test when the contracted index s1 has a projector
        let (s0, l01, s1, l12, s2) = make_contraction_indices();

        let tt1 = make_tt_generic::<T>(&s0, &l01, &s1);
        let tt2 = make_tt_generic::<T>(&s1, &l12, &s2);

        let t1_full = tt1.to_dense().unwrap();
        let t2_full = tt2.to_dense().unwrap();

        // Both have projector on the contracted index s1
        // m1: s1=0, m2: s1=0 (compatible)
        let proj1 = Projector::from_pairs([(s1.clone(), 0)]);
        let proj2 = Projector::from_pairs([(s1.clone(), 0)]);

        let m1 = SubDomainTT::new(tt1, proj1);
        let m2 = SubDomainTT::new(tt2, proj2);

        // Use Naive method for exact results (no approximation)
        let options = ContractOptions::naive();
        let result = contract(&m1, &m2, &options).unwrap().unwrap();
        let contracted_tt = result.data();

        // s1 is contracted away, so result projector should be empty
        assert!(result.projector().is_empty());

        let contracted_full = contracted_tt.to_dense().unwrap();

        // Compute expected: project both to s1=0, then contract
        let t1_data = T::extract_slice(&t1_full);
        let mut t1_proj_data = vec![T::zero_val(); t1_data.len()];
        // Keep only s1=0 column: indices 0, 2
        t1_proj_data[0] = t1_data[0]; // [s0=0, s1=0]
        t1_proj_data[2] = t1_data[2]; // [s0=1, s1=0]

        let t1_proj = TensorDynLen::from_dense_data(vec![s0.clone(), s1.clone()], t1_proj_data);

        let t2_data = T::extract_slice(&t2_full);
        let mut t2_proj_data = vec![T::zero_val(); t2_data.len()];
        // Keep only s1=0 row: indices 0, 1
        t2_proj_data[0] = t2_data[0]; // [s1=0, s2=0]
        t2_proj_data[1] = t2_data[1]; // [s1=0, s2=1]

        let t2_proj = TensorDynLen::from_dense_data(vec![s1.clone(), s2.clone()], t2_proj_data);

        let expected = t1_proj.contract(&t2_proj);

        let contracted_data = T::extract_slice(&contracted_full);
        let expected_data = T::extract_slice(&expected);

        assert_eq!(contracted_data.len(), expected_data.len());
        for (i, (&actual, &exp)) in contracted_data.iter().zip(expected_data.iter()).enumerate() {
            assert!(
                T::abs_diff(actual, exp) < 1e-10,
                "Mismatch at index {}: actual={:?}, expected={:?}",
                i,
                actual,
                exp
            );
        }
    }

    #[test]
    fn test_contract_with_projector_on_contracted_index_f64() {
        test_contract_with_projector_on_contracted_index_generic::<f64>();
    }

    #[test]
    fn test_contract_with_projector_on_contracted_index_c64() {
        test_contract_with_projector_on_contracted_index_generic::<Complex64>();
    }

    /// Generic test when only one side has a projector
    fn test_contract_one_side_has_projector_generic<T: TestScalar>() {
        // Test when only one side has a projector
        let (s0, l01, s1, l12, s2) = make_contraction_indices();

        let tt1 = make_tt_generic::<T>(&s0, &l01, &s1);
        let tt2 = make_tt_generic::<T>(&s1, &l12, &s2);

        let t1_full = tt1.to_dense().unwrap();
        let t2_full = tt2.to_dense().unwrap();

        // m1 has projector on s0, m2 has no projector
        let proj1 = Projector::from_pairs([(s0.clone(), 1)]);
        let m1 = SubDomainTT::new(tt1, proj1);
        let m2 = SubDomainTT::from_tt(tt2);

        // Use Naive method for exact results (no approximation)
        let options = ContractOptions::naive();
        let result = contract(&m1, &m2, &options).unwrap().unwrap();

        // Result should have projector on s0 only
        assert!(result.projector().is_projected_at(&s0));
        assert!(!result.projector().is_projected_at(&s2));

        let contracted_full = result.data().to_dense().unwrap();

        // Compute expected: project t1 to s0=1, t2 unchanged
        let t1_data = T::extract_slice(&t1_full);
        let mut t1_proj_data = vec![T::zero_val(); t1_data.len()];
        // Keep only s0=1 row: indices 2, 3
        t1_proj_data[2] = t1_data[2]; // [s0=1, s1=0]
        t1_proj_data[3] = t1_data[3]; // [s0=1, s1=1]

        let t1_proj = TensorDynLen::from_dense_data(vec![s0.clone(), s1.clone()], t1_proj_data);

        let expected = t1_proj.contract(&t2_full);

        let contracted_data = T::extract_slice(&contracted_full);
        let expected_data = T::extract_slice(&expected);

        assert_eq!(contracted_data.len(), expected_data.len());
        for (i, (&actual, &exp)) in contracted_data.iter().zip(expected_data.iter()).enumerate() {
            assert!(
                T::abs_diff(actual, exp) < 1e-10,
                "Mismatch at index {}: actual={:?}, expected={:?}",
                i,
                actual,
                exp
            );
        }
    }

    #[test]
    fn test_contract_one_side_has_projector_f64() {
        test_contract_one_side_has_projector_generic::<f64>();
    }

    #[test]
    fn test_contract_one_side_has_projector_c64() {
        test_contract_one_side_has_projector_generic::<Complex64>();
    }

    /// Generic test for proj_contract numerical correctness
    fn test_proj_contract_numerical_correctness_generic<T: TestScalar>() {
        let (s0, l01, s1, l12, s2) = make_contraction_indices();

        let tt1 = make_tt_generic::<T>(&s0, &l01, &s1);
        let tt2 = make_tt_generic::<T>(&s1, &l12, &s2);

        let t1_full = tt1.to_dense().unwrap();
        let t2_full = tt2.to_dense().unwrap();

        // Create SubDomainTTs without projectors
        let m1 = SubDomainTT::from_tt(tt1);
        let m2 = SubDomainTT::from_tt(tt2);

        // proj_contract with projector that projects s0=0 and s2=1
        let proj = Projector::from_pairs([(s0.clone(), 0), (s2.clone(), 1)]);
        // Use Naive method for exact results (no approximation)
        let options = ContractOptions::naive();
        let result = proj_contract(&m1, &m2, &proj, &options).unwrap().unwrap();

        // Verify projectors
        assert!(result.projector().is_projected_at(&s0));
        assert!(result.projector().is_projected_at(&s2));

        let contracted_full = result.data().to_dense().unwrap();

        // Compute expected: project both inputs to the given projector, then contract
        let t1_data = T::extract_slice(&t1_full);
        let mut t1_proj_data = vec![T::zero_val(); t1_data.len()];
        t1_proj_data[0] = t1_data[0]; // [s0=0, s1=0]
        t1_proj_data[1] = t1_data[1]; // [s0=0, s1=1]

        let t1_proj = TensorDynLen::from_dense_data(vec![s0.clone(), s1.clone()], t1_proj_data);

        let t2_data = T::extract_slice(&t2_full);
        let mut t2_proj_data = vec![T::zero_val(); t2_data.len()];
        t2_proj_data[1] = t2_data[1]; // [s1=0, s2=1]
        t2_proj_data[3] = t2_data[3]; // [s1=1, s2=1]

        let t2_proj = TensorDynLen::from_dense_data(vec![s1.clone(), s2.clone()], t2_proj_data);

        let expected = t1_proj.contract(&t2_proj);

        let contracted_data = T::extract_slice(&contracted_full);
        let expected_data = T::extract_slice(&expected);

        assert_eq!(contracted_data.len(), expected_data.len());
        for (i, (&actual, &exp)) in contracted_data.iter().zip(expected_data.iter()).enumerate() {
            assert!(
                T::abs_diff(actual, exp) < 1e-10,
                "Mismatch at index {}: actual={:?}, expected={:?}",
                i,
                actual,
                exp
            );
        }
    }

    #[test]
    fn test_proj_contract_numerical_correctness_f64() {
        test_proj_contract_numerical_correctness_generic::<f64>();
    }

    #[test]
    fn test_proj_contract_numerical_correctness_c64() {
        test_proj_contract_numerical_correctness_generic::<Complex64>();
    }
}
