use super::*;
use tensor4all_core::{DynId, Index};

/// Helper to create a simple tensor for testing
fn make_tensor(indices: Vec<DynIndex>) -> TensorDynLen {
    let dims: Vec<usize> = indices.iter().map(|i| i.size()).collect();
    let size: usize = dims.iter().product();
    let data: Vec<f64> = (0..size).map(|i| i as f64).collect();
    TensorDynLen::from_dense(indices, data).unwrap()
}

/// Helper to create a DynIndex
fn idx(id: u64, size: usize) -> DynIndex {
    Index::new_with_size(DynId(id), size)
}

#[test]
fn test_empty_tt() {
    let tt: TensorTrain = TensorTrain::new(vec![]).unwrap();
    assert!(tt.is_empty());
    assert_eq!(tt.len(), 0);
    assert_eq!(tt.llim(), -1);
    assert_eq!(tt.rlim(), 1);
    assert!(!tt.isortho());
}

#[test]
fn test_single_site_tt() {
    let tensor = make_tensor(vec![idx(0, 2)]);

    let tt = TensorTrain::new(vec![tensor]).unwrap();
    assert_eq!(tt.len(), 1);
    assert!(!tt.isortho());
    assert_eq!(tt.bond_dims(), Vec::<usize>::new());
}

#[test]
fn test_two_site_tt() {
    // Create two tensors with a shared link index
    let s0 = idx(0, 2); // site 0
    let l01 = idx(1, 3); // link 0-1
    let s1 = idx(2, 2); // site 1

    let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
    let t1 = make_tensor(vec![l01.clone(), s1.clone()]);

    let tt = TensorTrain::new(vec![t0, t1]).unwrap();
    assert_eq!(tt.len(), 2);
    assert_eq!(tt.bond_dims(), vec![3]);
    assert_eq!(tt.maxbonddim(), 3);

    // Check link index
    let link = tt.linkind(0).unwrap();
    assert_eq!(link.size(), 3);

    // Check site indices (nested vec)
    let site_inds = tt.siteinds();
    assert_eq!(site_inds.len(), 2);
    assert_eq!(site_inds[0].len(), 1);
    assert_eq!(site_inds[1].len(), 1);
    assert_eq!(site_inds[0][0].size(), 2);
    assert_eq!(site_inds[1][0].size(), 2);
}

#[test]
fn test_multi_site_indices() {
    // Test site with multiple physical indices
    let s0a = idx(0, 2); // site 0 index a
    let s0b = idx(1, 3); // site 0 index b
    let l01 = idx(2, 4); // link 0-1
    let s1 = idx(3, 2); // site 1

    let t0 = make_tensor(vec![s0a.clone(), s0b.clone(), l01.clone()]);
    let t1 = make_tensor(vec![l01.clone(), s1.clone()]);

    let tt = TensorTrain::new(vec![t0, t1]).unwrap();

    // Check site indices (nested vec)
    let site_inds = tt.siteinds();
    assert_eq!(site_inds.len(), 2);
    assert_eq!(site_inds[0].len(), 2); // site 0 has 2 indices
    assert_eq!(site_inds[1].len(), 1); // site 1 has 1 index
}

#[test]
fn test_ortho_tracking() {
    let s0 = idx(0, 2);
    let l01 = idx(1, 3);
    let s1 = idx(2, 2);

    let t0 = make_tensor(vec![s0, l01.clone()]);
    let t1 = make_tensor(vec![l01, s1]);

    // Create with specified orthogonality (ortho center at site 0)
    let tt = TensorTrain::with_ortho(
        vec![t0, t1],
        -1, // no left orthogonality
        1,  // right orthogonal from site 1
        Some(CanonicalForm::Unitary),
    )
    .unwrap();

    assert!(tt.isortho());
    assert_eq!(tt.orthocenter(), Some(0));
    assert_eq!(tt.canonical_form(), Some(CanonicalForm::Unitary));
}

#[test]
fn test_ortho_lims_range() {
    let s0 = idx(0, 2);
    let l01 = idx(1, 3);
    let l12 = idx(2, 3);
    let s1 = idx(3, 2);
    let s2 = idx(4, 2);

    let t0 = make_tensor(vec![s0, l01.clone()]);
    let t1 = make_tensor(vec![l01, s1, l12.clone()]);
    let t2 = make_tensor(vec![l12, s2]);

    // Create with partial orthogonality
    let tt = TensorTrain::with_ortho(vec![t0, t1, t2], 0, 2, None).unwrap();

    assert_eq!(tt.ortho_lims(), 1..2);
    assert!(tt.isortho());
    assert_eq!(tt.orthocenter(), Some(1));
}

#[test]
fn test_no_common_index_error() {
    let s0 = idx(0, 2);
    let s1 = idx(1, 2);

    let t0 = make_tensor(vec![s0]);
    let t1 = make_tensor(vec![s1]);

    let result = TensorTrain::new(vec![t0, t1]);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        TensorTrainError::InvalidStructure { .. }
    ));
}

#[test]
fn test_orthogonalize_two_site() {
    // Create a 2-site tensor train
    let s0 = idx(0, 2);
    let l01 = idx(1, 3);
    let s1 = idx(2, 2);

    let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
    let t1 = make_tensor(vec![l01.clone(), s1.clone()]);

    let mut tt = TensorTrain::new(vec![t0, t1]).unwrap();
    assert!(!tt.isortho());

    // Orthogonalize to site 0
    tt.orthogonalize(0).unwrap();
    assert!(tt.isortho());
    assert_eq!(tt.orthocenter(), Some(0));
    assert_eq!(tt.canonical_form(), Some(CanonicalForm::Unitary));

    // Orthogonalize to site 1
    tt.orthogonalize(1).unwrap();
    assert!(tt.isortho());
    assert_eq!(tt.orthocenter(), Some(1));
}

#[test]
fn test_orthogonalize_three_site() {
    // Create a 3-site tensor train
    let s0 = idx(0, 2);
    let l01 = idx(1, 3);
    let s1 = idx(2, 2);
    let l12 = idx(3, 3);
    let s2 = idx(4, 2);

    let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
    let t1 = make_tensor(vec![l01.clone(), s1.clone(), l12.clone()]);
    let t2 = make_tensor(vec![l12.clone(), s2.clone()]);

    let mut tt = TensorTrain::new(vec![t0, t1, t2]).unwrap();

    // Orthogonalize to middle site
    tt.orthogonalize(1).unwrap();
    assert!(tt.isortho());
    assert_eq!(tt.orthocenter(), Some(1));

    // Orthogonalize to left
    tt.orthogonalize(0).unwrap();
    assert!(tt.isortho());
    assert_eq!(tt.orthocenter(), Some(0));

    // Orthogonalize to right
    tt.orthogonalize(2).unwrap();
    assert!(tt.isortho());
    assert_eq!(tt.orthocenter(), Some(2));
}

#[test]
fn test_orthogonalize_with_lu() {
    let s0 = idx(0, 2);
    let l01 = idx(1, 3);
    let s1 = idx(2, 2);

    let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
    let t1 = make_tensor(vec![l01.clone(), s1.clone()]);

    let mut tt = TensorTrain::new(vec![t0, t1]).unwrap();

    tt.orthogonalize_with(0, CanonicalForm::LU).unwrap();
    assert!(tt.isortho());
    assert_eq!(tt.orthocenter(), Some(0));
    assert_eq!(tt.canonical_form(), Some(CanonicalForm::LU));
}

#[test]
fn test_orthogonalize_with_ci() {
    let s0 = idx(0, 2);
    let l01 = idx(1, 3);
    let s1 = idx(2, 2);

    let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
    let t1 = make_tensor(vec![l01.clone(), s1.clone()]);

    let mut tt = TensorTrain::new(vec![t0, t1]).unwrap();

    tt.orthogonalize_with(1, CanonicalForm::CI).unwrap();
    assert!(tt.isortho());
    assert_eq!(tt.orthocenter(), Some(1));
    assert_eq!(tt.canonical_form(), Some(CanonicalForm::CI));
}

#[test]
fn test_truncate_with_max_rank() {
    // Create a 3-site tensor train with large bond dimension
    let s0 = idx(0, 4);
    let l01 = idx(1, 8);
    let s1 = idx(2, 4);
    let l12 = idx(3, 8);
    let s2 = idx(4, 4);

    let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
    let t1 = make_tensor(vec![l01.clone(), s1.clone(), l12.clone()]);
    let t2 = make_tensor(vec![l12.clone(), s2.clone()]);

    let mut tt = TensorTrain::new(vec![t0, t1, t2]).unwrap();
    assert_eq!(tt.maxbonddim(), 8);

    // Truncate to max rank 4
    let options = TruncateOptions::svd().with_max_rank(4);
    tt.truncate(&options).unwrap();

    // Check that bond dimensions are reduced
    assert!(tt.maxbonddim() <= 4);
    assert_eq!(tt.canonical_form(), Some(CanonicalForm::Unitary));
}

#[test]
fn test_contract_with_fit_method() {
    use crate::ContractOptions;

    // Use two-site tensor trains with shared site indices to exercise contraction
    let s0 = idx(100, 2);
    let s1 = idx(101, 2);
    let l01_a = idx(102, 3);
    let l01_b = idx(103, 3);

    let t1_0 = make_tensor(vec![s0.clone(), l01_a.clone()]);
    let t1_1 = make_tensor(vec![l01_a.clone(), s1.clone()]);
    let tt1 = TensorTrain::new(vec![t1_0, t1_1]).unwrap();

    let t2_0 = make_tensor(vec![s0.clone(), l01_b.clone()]);
    let t2_1 = make_tensor(vec![l01_b.clone(), s1.clone()]);
    let tt2 = TensorTrain::new(vec![t2_0, t2_1]).unwrap();

    // Test contract with Fit method
    let options = ContractOptions::fit().with_max_rank(10).with_nhalfsweeps(4); // 4 half-sweeps = 2 full sweeps
    let result = tt1.contract(&tt2, &options);
    assert!(result.is_ok());
    let result_tt = result.unwrap();
    assert_eq!(result_tt.len(), 1);
}

#[test]
fn test_contract_with_naive_method() {
    use crate::ContractOptions;

    // Use single-site tensor trains with a shared site index
    let s0 = idx(200, 2);

    let t1 = make_tensor(vec![s0.clone()]);
    let tt1 = TensorTrain::new(vec![t1]).unwrap();

    let t2 = make_tensor(vec![s0.clone()]);
    let tt2 = TensorTrain::new(vec![t2]).unwrap();

    // Test contract with Naive method
    let options = ContractOptions::naive();
    let result = tt1.contract(&tt2, &options);
    assert!(result.is_ok());
    let result_tt = result.unwrap();
    assert_eq!(result_tt.len(), 1);
}

#[test]
fn test_contract_nhalfsweeps_conversion() {
    use crate::ContractOptions;

    // Use two-site tensor trains with shared site indices to exercise contraction
    let s0 = idx(300, 2);
    let s1 = idx(301, 2);
    let l01_a = idx(302, 3);
    let l01_b = idx(303, 3);

    let t1_0 = make_tensor(vec![s0.clone(), l01_a.clone()]);
    let t1_1 = make_tensor(vec![l01_a.clone(), s1.clone()]);
    let tt1 = TensorTrain::new(vec![t1_0, t1_1]).unwrap();

    let t2_0 = make_tensor(vec![s0.clone(), l01_b.clone()]);
    let t2_1 = make_tensor(vec![l01_b.clone(), s1.clone()]);
    let tt2 = TensorTrain::new(vec![t2_0, t2_1]).unwrap();

    // Test that nhalfsweeps is correctly converted to nfullsweeps
    // nhalfsweeps=6 should become nfullsweeps=3
    let options = ContractOptions::fit().with_nhalfsweeps(6).with_max_rank(10);
    let result = tt1.contract(&tt2, &options);
    assert!(result.is_ok());
    let result_tt = result.unwrap();
    assert_eq!(result_tt.len(), 1);
}

#[test]
fn test_contract_fit_odd_nhalfsweeps_errors() {
    use crate::ContractOptions;

    let s0 = idx(400, 2);
    let s1 = idx(401, 2);
    let l01_a = idx(402, 3);
    let l01_b = idx(403, 3);

    let t1_0 = make_tensor(vec![s0.clone(), l01_a.clone()]);
    let t1_1 = make_tensor(vec![l01_a.clone(), s1.clone()]);
    let tt1 = TensorTrain::new(vec![t1_0, t1_1]).unwrap();

    let t2_0 = make_tensor(vec![s0.clone(), l01_b.clone()]);
    let t2_1 = make_tensor(vec![l01_b.clone(), s1.clone()]);
    let tt2 = TensorTrain::new(vec![t2_0, t2_1]).unwrap();

    let options = ContractOptions::fit().with_nhalfsweeps(1).with_max_rank(10);
    let err = tt1.contract(&tt2, &options).unwrap_err();
    assert!(matches!(err, TensorTrainError::OperationError { .. }));
}

#[test]
fn test_truncate_invalid_rtol_errors() {
    let s0 = idx(500, 2);
    let l01 = idx(501, 3);
    let s1 = idx(502, 2);

    let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
    let t1 = make_tensor(vec![l01.clone(), s1.clone()]);

    let mut tt = TensorTrain::new(vec![t0, t1]).unwrap();
    let options = TruncateOptions::svd().with_rtol(-1.0);
    let err = tt.truncate(&options).unwrap_err();
    assert!(matches!(err, TensorTrainError::OperationError { .. }));
}

#[test]
fn test_inner_product() {
    let s0 = idx(0, 2);
    let l01 = idx(1, 3);
    let s1 = idx(2, 2);

    let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
    let t1 = make_tensor(vec![l01.clone(), s1.clone()]);

    let tt = TensorTrain::new(vec![t0, t1]).unwrap();

    // Compute norm squared
    let norm_sq = tt.norm_squared();
    assert!(norm_sq > 0.0);

    // Compute norm
    let norm = tt.norm();
    assert!((norm * norm - norm_sq).abs() < 1e-10);
}

#[test]
fn test_to_dense() {
    // Create a 2-site TT: s0 -- l01 -- s1
    let s0 = idx(0, 2);
    let l01 = idx(1, 3);
    let s1 = idx(2, 2);

    let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
    let t1 = make_tensor(vec![l01.clone(), s1.clone()]);

    let tt = TensorTrain::new(vec![t0.clone(), t1.clone()]).unwrap();

    // Convert to dense
    let dense = tt.to_dense().unwrap();

    // Expected: contract t0 and t1 along l01
    let expected = t0.contract(&t1);

    // Compare results
    let dense_data = dense.as_slice_f64().unwrap();
    let expected_data = expected.as_slice_f64().unwrap();

    assert_eq!(dense_data.len(), expected_data.len());
    for (i, (&d, &e)) in dense_data.iter().zip(expected_data.iter()).enumerate() {
        assert!(
            (d - e).abs() < 1e-10,
            "Mismatch at index {}: got {}, expected {}",
            i,
            d,
            e
        );
    }
}

#[test]
fn test_to_dense_single_site() {
    // Single site TT should return the tensor as-is
    let s0 = idx(0, 4);
    let t0 = make_tensor(vec![s0.clone()]);

    let tt = TensorTrain::new(vec![t0.clone()]).unwrap();
    let dense = tt.to_dense().unwrap();

    let dense_data = dense.as_slice_f64().unwrap();
    let expected_data = t0.as_slice_f64().unwrap();

    assert_eq!(dense_data.len(), expected_data.len());
    for (i, (&d, &e)) in dense_data.iter().zip(expected_data.iter()).enumerate() {
        assert!(
            (d - e).abs() < 1e-10,
            "Mismatch at index {}: got {}, expected {}",
            i,
            d,
            e
        );
    }
}

#[test]
fn test_to_dense_three_sites() {
    // 3-site TT: s0 -- l01 -- s1 -- l12 -- s2
    let s0 = idx(0, 2);
    let l01 = idx(1, 3);
    let s1 = idx(2, 2);
    let l12 = idx(3, 3);
    let s2 = idx(4, 2);

    let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
    let t1 = make_tensor(vec![l01.clone(), s1.clone(), l12.clone()]);
    let t2 = make_tensor(vec![l12.clone(), s2.clone()]);

    let tt = TensorTrain::new(vec![t0.clone(), t1.clone(), t2.clone()]).unwrap();
    let dense = tt.to_dense().unwrap();

    // Expected: contract t0, t1, t2 sequentially
    let expected = t0.contract(&t1).contract(&t2);

    let dense_data = dense.as_slice_f64().unwrap();
    let expected_data = expected.as_slice_f64().unwrap();

    assert_eq!(dense_data.len(), expected_data.len());
    for (i, (&d, &e)) in dense_data.iter().zip(expected_data.iter()).enumerate() {
        assert!(
            (d - e).abs() < 1e-10,
            "Mismatch at index {}: got {}, expected {}",
            i,
            d,
            e
        );
    }
}

#[test]
fn test_to_dense_empty() {
    let tt = TensorTrain::new(vec![]).unwrap();
    let result = tt.to_dense();
    assert!(result.is_err());
}

#[test]
fn test_add_simple() {
    // Create two TTs with the same structure
    // Both TTs must have the same site indices AND link indices
    let s0 = idx(0, 2);
    let l01 = idx(1, 3);
    let s1 = idx(2, 2);

    // First TT with data [1, 2, 3, ...]
    let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
    let t1 = make_tensor(vec![l01.clone(), s1.clone()]);
    let tt1 = TensorTrain::new(vec![t0, t1]).unwrap();

    // Second TT with same structure (clone and modify data for testing)
    let tt2 = tt1.clone();

    // Add them
    let sum = tt1.add(&tt2).unwrap();

    // Result should have double bond dimension
    assert_eq!(sum.len(), 2);
    assert_eq!(sum.bond_dims(), vec![6]); // 3 + 3

    // Verify numerically: sum.to_dense() == tt1.to_dense() + tt2.to_dense()
    let sum_dense = sum.to_dense().unwrap();
    let tt1_dense = tt1.to_dense().unwrap();
    let tt2_dense = tt2.to_dense().unwrap();

    let sum_data = sum_dense.as_slice_f64().unwrap();
    let tt1_data = tt1_dense.as_slice_f64().unwrap();
    let tt2_data = tt2_dense.as_slice_f64().unwrap();

    assert_eq!(sum_data.len(), tt1_data.len());
    for i in 0..sum_data.len() {
        let expected = tt1_data[i] + tt2_data[i];
        assert!(
            (sum_data[i] - expected).abs() < 1e-10,
            "Mismatch at {}: {} vs {}",
            i,
            sum_data[i],
            expected
        );
    }
}

#[test]
fn test_add_empty() {
    let empty = TensorTrain::new(vec![]).unwrap();

    let s0 = idx(0, 2);
    let l01 = idx(1, 3);
    let s1 = idx(2, 2);
    let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
    let t1 = make_tensor(vec![l01.clone(), s1.clone()]);
    let tt = TensorTrain::new(vec![t0, t1]).unwrap();

    // empty + tt = tt
    let result = empty.add(&tt).unwrap();
    assert_eq!(result.len(), tt.len());

    // tt + empty = tt
    let result = tt.add(&empty).unwrap();
    assert_eq!(result.len(), tt.len());

    // empty + empty = empty
    let result = empty.add(&empty).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_add_length_mismatch() {
    let s0 = idx(0, 2);
    let l01 = idx(1, 3);
    let s1 = idx(2, 2);
    let l12 = idx(3, 3);
    let s2 = idx(4, 2);

    let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
    let t1 = make_tensor(vec![l01.clone(), s1.clone()]);
    let tt1 = TensorTrain::new(vec![t0, t1]).unwrap();

    let t0_2 = make_tensor(vec![s0.clone(), l01.clone()]);
    let t1_2 = make_tensor(vec![l01.clone(), s1.clone(), l12.clone()]);
    let t2_2 = make_tensor(vec![l12.clone(), s2.clone()]);
    let tt2 = TensorTrain::new(vec![t0_2, t1_2, t2_2]).unwrap();

    // Length mismatch should fail
    let result = tt1.add(&tt2);
    assert!(result.is_err());
}

#[test]
fn test_set_llim_updates_canonical_region() {
    let s0 = idx(0, 2);
    let l01 = idx(1, 3);
    let s1 = idx(2, 2);

    let t0 = make_tensor(vec![s0, l01.clone()]);
    let t1 = make_tensor(vec![l01, s1]);

    let mut tt = TensorTrain::new(vec![t0, t1]).unwrap();

    // Set llim=-1, rlim already 1 => center at 0
    tt.set_llim(-1);
    // With rlim=1 (which is the default for non-ortho TT) and llim=-1, center should be at 0
    // But this depends on the rlim value, let's explicitly set both
    let mut tt2 = TensorTrain::with_ortho(
        vec![
            make_tensor(vec![idx(0, 2), idx(1, 3)]),
            make_tensor(vec![idx(1, 3), idx(2, 2)]),
        ],
        -1,
        1,
        Some(CanonicalForm::Unitary),
    )
    .unwrap();
    assert!(tt2.isortho());
    assert_eq!(tt2.orthocenter(), Some(0));

    // Setting llim to a value that breaks single-center should clear ortho
    tt2.set_llim(5);
    assert!(!tt2.isortho());
}

#[test]
fn test_set_rlim_updates_canonical_region() {
    let mut tt = TensorTrain::with_ortho(
        vec![
            make_tensor(vec![idx(0, 2), idx(1, 3)]),
            make_tensor(vec![idx(1, 3), idx(2, 2)]),
        ],
        -1,
        1,
        Some(CanonicalForm::Unitary),
    )
    .unwrap();
    assert!(tt.isortho());

    // Setting rlim to a value that breaks single-center should clear ortho
    tt.set_rlim(5);
    assert!(!tt.isortho());
}

#[test]
fn test_set_tensor_invalidates_ortho() {
    let s0 = idx(0, 2);
    let l01 = idx(1, 3);
    let s1 = idx(2, 2);

    let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
    let t1 = make_tensor(vec![l01.clone(), s1.clone()]);

    let mut tt =
        TensorTrain::with_ortho(vec![t0, t1], -1, 1, Some(CanonicalForm::Unitary)).unwrap();
    assert!(tt.isortho());

    // Replace tensor at site 0
    let new_tensor = make_tensor(vec![s0, l01]);
    tt.set_tensor(0, new_tensor);
    assert!(!tt.isortho());
}

#[test]
fn test_tensors_mut_returns_all_sites_in_order() {
    let s0 = idx(0, 2);
    let l01 = idx(1, 3);
    let s1 = idx(2, 2);

    let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
    let t1 = make_tensor(vec![l01.clone(), s1.clone()]);
    let mut tt = TensorTrain::new(vec![t0, t1]).unwrap();

    let replacement =
        TensorDynLen::from_dense(vec![s0, l01], vec![42.0; 6]).expect("valid replacement");

    {
        let mut tensors = tt.tensors_mut();
        assert_eq!(tensors.len(), 2);
        *tensors[0] = replacement.clone();
    }

    assert_eq!(tt.tensor(0).as_slice_f64().unwrap(), vec![42.0; 6]);
}

#[test]
fn test_scale() {
    let s0 = idx(0, 2);
    let l01 = idx(1, 3);
    let s1 = idx(2, 2);

    let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
    let t1 = make_tensor(vec![l01.clone(), s1.clone()]);

    let tt = TensorTrain::new(vec![t0, t1]).unwrap();

    // Scale by 2.0
    let scaled = tt.scale(AnyScalar::new_real(2.0)).unwrap();

    // Verify: norm of scaled should be 2 * norm of original
    let orig_norm = tt.norm();
    let scaled_norm = scaled.norm();
    assert!(
        (scaled_norm - 2.0 * orig_norm).abs() < 1e-10,
        "Expected scaled_norm = {}, got {}",
        2.0 * orig_norm,
        scaled_norm
    );
}

#[test]
fn test_scale_empty() {
    let tt = TensorTrain::new(vec![]).unwrap();
    let scaled = tt.scale(AnyScalar::new_real(2.0)).unwrap();
    assert!(scaled.is_empty());
}

#[test]
fn test_axpby() {
    let s0 = idx(0, 2);
    let l01 = idx(1, 3);
    let s1 = idx(2, 2);

    let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
    let t1 = make_tensor(vec![l01.clone(), s1.clone()]);

    let tt1 = TensorTrain::new(vec![t0.clone(), t1.clone()]).unwrap();
    let tt2 = tt1.clone();

    // Compute 2*tt1 + 3*tt2 = 5*tt1 (since tt1 == tt2)
    let result = tt1
        .axpby(AnyScalar::new_real(2.0), &tt2, AnyScalar::new_real(3.0))
        .unwrap();

    // Verify numerically via to_dense
    let result_dense = result.to_dense().unwrap();
    let tt1_dense = tt1.to_dense().unwrap();

    let result_data = result_dense.as_slice_f64().unwrap();
    let tt1_data = tt1_dense.as_slice_f64().unwrap();

    assert_eq!(result_data.len(), tt1_data.len());
    for i in 0..result_data.len() {
        let expected = 5.0 * tt1_data[i]; // 2*tt1 + 3*tt2 = 5*tt1
        assert!(
            (result_data[i] - expected).abs() < 1e-10,
            "Mismatch at {}: {} vs {}",
            i,
            result_data[i],
            expected
        );
    }

    // Bond dimension should be 6 (3 + 3)
    assert_eq!(result.bond_dims(), vec![6]);
}

#[test]
fn test_tensor_like_scale() {
    use tensor4all_core::TensorLike;

    let s0 = idx(0, 2);
    let l01 = idx(1, 3);
    let s1 = idx(2, 2);

    let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
    let t1 = make_tensor(vec![l01.clone(), s1.clone()]);

    let tt = TensorTrain::new(vec![t0, t1]).unwrap();

    // Use TensorLike::scale
    let scaled = TensorLike::scale(&tt, AnyScalar::new_real(2.0)).unwrap();

    let orig_norm = tt.norm();
    let scaled_norm = TensorLike::norm(&scaled);
    assert!(
        (scaled_norm - 2.0 * orig_norm).abs() < 1e-10,
        "Expected scaled_norm = {}, got {}",
        2.0 * orig_norm,
        scaled_norm
    );
}

#[test]
fn test_tensor_like_inner_product() {
    use tensor4all_core::TensorLike;

    let s0 = idx(0, 2);
    let l01 = idx(1, 3);
    let s1 = idx(2, 2);

    let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
    let t1 = make_tensor(vec![l01.clone(), s1.clone()]);

    let tt = TensorTrain::new(vec![t0, t1]).unwrap();

    // TensorLike::inner_product should equal TensorTrain::inner
    let inner_via_trait = TensorLike::inner_product(&tt, &tt).unwrap();
    let inner_direct = tt.inner(&tt);

    assert!(
        (inner_via_trait.real() - inner_direct.real()).abs() < 1e-10,
        "Inner product mismatch: {} vs {}",
        inner_via_trait.real(),
        inner_direct.real()
    );
}
