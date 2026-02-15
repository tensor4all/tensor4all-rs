//! Contraction operations for tensor trains.
//!
//! This module provides both a free function [`contract`] and an impl method
//! [`TensorTrain::contract`] for contracting two tensor trains.

use tensor4all_treetn::treetn::contraction::{
    contract as treetn_contract, ContractionMethod, ContractionOptions as TreeTNContractionOptions,
};
use tensor4all_treetn::CanonicalForm;

use crate::error::{Result, TensorTrainError};
use crate::options::{validate_truncation_params, ContractMethod, ContractOptions};
use crate::tensortrain::TensorTrain;
use tensor4all_core::truncation::HasTruncationParams;

/// Contract two tensor trains, returning a new tensor train.
///
/// This performs element-wise contraction of corresponding sites,
/// similar to MPO-MPO contraction in ITensor.
///
/// # Arguments
/// * `a` - The first tensor train
/// * `b` - The second tensor train
/// * `options` - Contraction options (method, max_rank, rtol, nhalfsweeps)
///
/// # Returns
/// A new tensor train resulting from the contraction.
///
/// # Errors
/// Returns an error if:
/// - Either tensor train is empty
/// - The tensor trains have different lengths
/// - The contraction algorithm fails
pub fn contract(
    a: &TensorTrain,
    b: &TensorTrain,
    options: &ContractOptions,
) -> Result<TensorTrain> {
    if a.is_empty() || b.is_empty() {
        return Err(TensorTrainError::InvalidStructure {
            message: "Cannot contract empty tensor trains".to_string(),
        });
    }

    if a.len() != b.len() {
        return Err(TensorTrainError::InvalidStructure {
            message: format!(
                "Tensor trains must have the same length for contraction: {} vs {}",
                a.len(),
                b.len()
            ),
        });
    }

    validate_truncation_params(options.truncation_params())?;

    if matches!(options.method(), ContractMethod::Fit) && !options.nhalfsweeps().is_multiple_of(2) {
        return Err(TensorTrainError::OperationError {
            message: format!(
                "nhalfsweeps must be a multiple of 2 for Fit method, got {}",
                options.nhalfsweeps()
            ),
        });
    }

    // Convert ContractOptions to TreeTN ContractionOptions
    let treetn_method = match options.method() {
        ContractMethod::Zipup => ContractionMethod::Zipup,
        ContractMethod::Fit => ContractionMethod::Fit,
        ContractMethod::Naive => ContractionMethod::Naive,
    };

    // Convert nhalfsweeps to nfullsweeps (nhalfsweeps / 2)
    let nfullsweeps = options.nhalfsweeps() / 2;
    let treetn_options = TreeTNContractionOptions::new(treetn_method).with_nfullsweeps(nfullsweeps);

    let treetn_options = if let Some(max_rank) = options.max_rank() {
        treetn_options.with_max_rank(max_rank)
    } else {
        treetn_options
    };

    let treetn_options = if let Some(rtol) = options.rtol() {
        treetn_options.with_rtol(rtol)
    } else {
        treetn_options
    };

    // Use the last site as the canonical center (consistent with existing behavior)
    let center = a.len() - 1;

    // For zip-up method, use contract_zipup_tree_accumulated
    let result_inner = if matches!(options.method(), ContractMethod::Zipup) {
        a.as_treetn()
            .contract_zipup_tree_accumulated(
                b.as_treetn(),
                &center,
                CanonicalForm::Unitary,
                options.rtol(),
                options.max_rank(),
            )
            .map_err(|e| TensorTrainError::InvalidStructure {
                message: format!("Zip-up contraction failed: {}", e),
            })?
    } else {
        treetn_contract(a.as_treetn(), b.as_treetn(), &center, treetn_options).map_err(|e| {
            TensorTrainError::InvalidStructure {
                message: format!("TreeTN contraction failed: {}", e),
            }
        })?
    };

    Ok(TensorTrain::from_inner(
        result_inner,
        Some(CanonicalForm::Unitary),
    ))
}

impl TensorTrain {
    /// Contract two tensor trains, returning a new tensor train.
    ///
    /// This performs element-wise contraction of corresponding sites,
    /// similar to MPO-MPO contraction in ITensor.
    ///
    /// # Arguments
    /// * `other` - The other tensor train to contract with
    /// * `options` - Contraction options (method, max_rank, rtol, nhalfsweeps)
    ///
    /// # Returns
    /// A new tensor train resulting from the contraction.
    ///
    /// # Errors
    /// Returns an error if:
    /// - Either tensor train is empty
    /// - The tensor trains have different lengths
    /// - The contraction algorithm fails
    pub fn contract(&self, other: &Self, options: &ContractOptions) -> Result<Self> {
        contract(self, other, options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TensorTrainError;
    use tensor4all_core::{DynId, DynIndex, Index, StorageScalar, TensorDynLen, TensorLike};

    /// Helper to create a simple tensor for testing
    fn make_tensor(indices: Vec<DynIndex>) -> TensorDynLen {
        let dims: Vec<usize> = indices.iter().map(|i| i.size()).collect();
        let size: usize = dims.iter().product();
        let data: Vec<f64> = (0..size).map(|i| (i + 1) as f64).collect();
        let storage = f64::dense_storage_with_shape(data, &dims);
        TensorDynLen::new(indices, storage)
    }

    /// Helper to create a DynIndex
    fn idx(id: u64, size: usize) -> DynIndex {
        Index::new_with_size(DynId(id), size)
    }

    /// Helper: compare TT contraction result against naive dense contraction.
    ///
    /// Converts both TTs to dense, contracts the dense tensors, then compares
    /// with the TT contraction result using `isapprox`.
    fn assert_matches_naive(tt1: &TensorTrain, tt2: &TensorTrain, result: &TensorTrain) {
        let naive_result = tt1.to_dense().unwrap().contract(&tt2.to_dense().unwrap());
        let result_dense = result.to_dense().unwrap();
        assert!(
            result_dense.isapprox(&naive_result, 1e-10, 0.0),
            "TT contraction result does not match naive: maxabs diff = {}",
            (&result_dense - &naive_result).maxabs()
        );
    }

    #[test]
    fn test_contract_free_fn_empty_first() {
        let empty = TensorTrain::new(vec![]).unwrap();

        let s0 = idx(1000, 2);
        let t0 = make_tensor(vec![s0.clone()]);
        let non_empty = TensorTrain::new(vec![t0]).unwrap();

        let options = ContractOptions::zipup();
        let err = contract(&empty, &non_empty, &options).unwrap_err();
        assert!(matches!(err, TensorTrainError::InvalidStructure { .. }));
    }

    #[test]
    fn test_contract_free_fn_empty_second() {
        let s0 = idx(1001, 2);
        let t0 = make_tensor(vec![s0.clone()]);
        let non_empty = TensorTrain::new(vec![t0]).unwrap();
        let empty = TensorTrain::new(vec![]).unwrap();

        let options = ContractOptions::zipup();
        let err = contract(&non_empty, &empty, &options).unwrap_err();
        assert!(matches!(err, TensorTrainError::InvalidStructure { .. }));
    }

    #[test]
    fn test_contract_free_fn_length_mismatch() {
        let s0 = idx(1002, 2);
        let l01 = idx(1003, 3);
        let s1 = idx(1004, 2);

        let t0 = make_tensor(vec![s0.clone()]);
        let tt1 = TensorTrain::new(vec![t0]).unwrap();

        let t0_2 = make_tensor(vec![s0.clone(), l01.clone()]);
        let t1_2 = make_tensor(vec![l01.clone(), s1.clone()]);
        let tt2 = TensorTrain::new(vec![t0_2, t1_2]).unwrap();

        let options = ContractOptions::zipup();
        let err = contract(&tt1, &tt2, &options).unwrap_err();
        assert!(matches!(err, TensorTrainError::InvalidStructure { .. }));
    }

    #[test]
    fn test_contract_free_fn_invalid_rtol() {
        let s0 = idx(1005, 2);
        let t0 = make_tensor(vec![s0.clone()]);
        let tt1 = TensorTrain::new(vec![t0.clone()]).unwrap();
        let tt2 = TensorTrain::new(vec![t0]).unwrap();

        let options = ContractOptions::zipup().with_rtol(-1.0);
        let err = contract(&tt1, &tt2, &options).unwrap_err();
        assert!(matches!(err, TensorTrainError::OperationError { .. }));
    }

    #[test]
    fn test_contract_free_fn_invalid_max_rank_zero() {
        let s0 = idx(1006, 2);
        let t0 = make_tensor(vec![s0.clone()]);
        let tt1 = TensorTrain::new(vec![t0.clone()]).unwrap();
        let tt2 = TensorTrain::new(vec![t0]).unwrap();

        let options = ContractOptions::zipup().with_max_rank(0);
        let err = contract(&tt1, &tt2, &options).unwrap_err();
        assert!(matches!(err, TensorTrainError::OperationError { .. }));
    }

    #[test]
    fn test_contract_zipup_single_site() {
        // Two single-site TTs with a shared site index
        let s0 = idx(1010, 3);
        let t1 = make_tensor(vec![s0.clone()]);
        let tt1 = TensorTrain::new(vec![t1]).unwrap();

        let t2 = make_tensor(vec![s0.clone()]);
        let tt2 = TensorTrain::new(vec![t2]).unwrap();

        let options = ContractOptions::zipup();
        let result = contract(&tt1, &tt2, &options).unwrap();
        assert_eq!(result.len(), 1);
        assert_matches_naive(&tt1, &tt2, &result);
    }

    #[test]
    fn test_contract_zipup_two_sites() {
        // Two 2-site TTs with shared site indices
        let s0 = idx(1020, 2);
        let s1 = idx(1021, 2);
        let l01_a = idx(1022, 3);
        let l01_b = idx(1023, 3);

        let t1_0 = make_tensor(vec![s0.clone(), l01_a.clone()]);
        let t1_1 = make_tensor(vec![l01_a.clone(), s1.clone()]);
        let tt1 = TensorTrain::new(vec![t1_0, t1_1]).unwrap();

        let t2_0 = make_tensor(vec![s0.clone(), l01_b.clone()]);
        let t2_1 = make_tensor(vec![l01_b.clone(), s1.clone()]);
        let tt2 = TensorTrain::new(vec![t2_0, t2_1]).unwrap();

        let options = ContractOptions::zipup().with_max_rank(10);
        let result = contract(&tt1, &tt2, &options).unwrap();
        // Result should contract over shared site indices
        assert_eq!(result.len(), 1);
        assert_matches_naive(&tt1, &tt2, &result);
    }

    #[test]
    fn test_contract_zipup_with_rtol() {
        let s0 = idx(1030, 2);
        let s1 = idx(1031, 2);
        let l01_a = idx(1032, 3);
        let l01_b = idx(1033, 3);

        let t1_0 = make_tensor(vec![s0.clone(), l01_a.clone()]);
        let t1_1 = make_tensor(vec![l01_a.clone(), s1.clone()]);
        let tt1 = TensorTrain::new(vec![t1_0, t1_1]).unwrap();

        let t2_0 = make_tensor(vec![s0.clone(), l01_b.clone()]);
        let t2_1 = make_tensor(vec![l01_b.clone(), s1.clone()]);
        let tt2 = TensorTrain::new(vec![t2_0, t2_1]).unwrap();

        let options = ContractOptions::zipup().with_rtol(1e-10);
        let result = contract(&tt1, &tt2, &options).unwrap();
        assert_eq!(result.len(), 1);
        assert_matches_naive(&tt1, &tt2, &result);
    }

    #[test]
    fn test_contract_naive_single_site() {
        // Naive method works for single-site TTs with a shared site index
        let s0 = idx(1040, 3);
        let t1 = make_tensor(vec![s0.clone()]);
        let tt1 = TensorTrain::new(vec![t1]).unwrap();

        let t2 = make_tensor(vec![s0.clone()]);
        let tt2 = TensorTrain::new(vec![t2]).unwrap();

        let options = ContractOptions::naive();
        let result = contract(&tt1, &tt2, &options).unwrap();
        assert_eq!(result.len(), 1);
        assert_matches_naive(&tt1, &tt2, &result);
    }

    #[test]
    fn test_contract_fit_two_sites() {
        let s0 = idx(1050, 2);
        let s1 = idx(1051, 2);
        let l01_a = idx(1052, 3);
        let l01_b = idx(1053, 3);

        let t1_0 = make_tensor(vec![s0.clone(), l01_a.clone()]);
        let t1_1 = make_tensor(vec![l01_a.clone(), s1.clone()]);
        let tt1 = TensorTrain::new(vec![t1_0, t1_1]).unwrap();

        let t2_0 = make_tensor(vec![s0.clone(), l01_b.clone()]);
        let t2_1 = make_tensor(vec![l01_b.clone(), s1.clone()]);
        let tt2 = TensorTrain::new(vec![t2_0, t2_1]).unwrap();

        let options = ContractOptions::fit().with_max_rank(10).with_nhalfsweeps(4);
        let result = contract(&tt1, &tt2, &options).unwrap();
        assert_eq!(result.len(), 1);
        assert_matches_naive(&tt1, &tt2, &result);
    }

    #[test]
    fn test_contract_fit_odd_nhalfsweeps() {
        let s0 = idx(1060, 2);
        let t0 = make_tensor(vec![s0.clone()]);
        let tt1 = TensorTrain::new(vec![t0.clone()]).unwrap();
        let tt2 = TensorTrain::new(vec![t0]).unwrap();

        // Odd nhalfsweeps should fail for Fit method
        let options = ContractOptions::fit().with_nhalfsweeps(3).with_max_rank(10);
        let err = contract(&tt1, &tt2, &options).unwrap_err();
        assert!(matches!(err, TensorTrainError::OperationError { .. }));
    }

    #[test]
    fn test_contract_fit_nhalfsweeps_zero_ok() {
        // nhalfsweeps=0 is a multiple of 2, should be accepted
        let s0 = idx(1070, 2);
        let t0 = make_tensor(vec![s0.clone()]);
        let tt1 = TensorTrain::new(vec![t0.clone()]).unwrap();
        let tt2 = TensorTrain::new(vec![t0]).unwrap();

        let options = ContractOptions::fit().with_nhalfsweeps(0).with_max_rank(10);
        let result = contract(&tt1, &tt2, &options).unwrap();
        assert_eq!(result.len(), 1);
        assert_matches_naive(&tt1, &tt2, &result);
    }

    #[test]
    fn test_contract_method_uses_tt_contract() {
        // Verify that TensorTrain::contract delegates to the free function
        let s0 = idx(1080, 2);
        let t0 = make_tensor(vec![s0.clone()]);
        let tt1 = TensorTrain::new(vec![t0.clone()]).unwrap();
        let tt2 = TensorTrain::new(vec![t0]).unwrap();

        let options = ContractOptions::zipup();
        let result_free = contract(&tt1, &tt2, &options).unwrap();
        let result_method = tt1.contract(&tt2, &options).unwrap();

        // Both should produce TTs with the same length
        assert_eq!(result_free.len(), result_method.len());

        // Both should match naive dense contraction
        assert_matches_naive(&tt1, &tt2, &result_free);
        assert_matches_naive(&tt1, &tt2, &result_method);

        // Both should produce identical dense results
        let free_data = result_free.to_dense().unwrap().to_vec_f64().unwrap();
        let method_data = result_method.to_dense().unwrap().to_vec_f64().unwrap();
        assert_eq!(free_data.len(), method_data.len());
        for (i, (&a, &b)) in free_data.iter().zip(method_data.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-10,
                "Element {} mismatch between free fn and method: {} vs {}",
                i,
                a,
                b
            );
        }
    }

    #[test]
    fn test_contract_zipup_with_truncation() {
        // 3-site TTs with shared site indices, bond dim 2
        // Contraction without truncation would produce bond dim up to 4.
        // Truncate to max_rank=1 and verify:
        //   (a) bond dims are actually truncated
        //   (b) result is approximately correct (not exact due to truncation)
        let s0 = idx(2000, 2);
        let s1 = idx(2001, 2);
        let s2 = idx(2002, 2);
        let l01_a = idx(2010, 2);
        let l12_a = idx(2011, 2);
        let l01_b = idx(2020, 2);
        let l12_b = idx(2021, 2);

        let tt1 = TensorTrain::new(vec![
            make_tensor(vec![s0.clone(), l01_a.clone()]),
            make_tensor(vec![l01_a.clone(), s1.clone(), l12_a.clone()]),
            make_tensor(vec![l12_a.clone(), s2.clone()]),
        ])
        .unwrap();

        let tt2 = TensorTrain::new(vec![
            make_tensor(vec![s0.clone(), l01_b.clone()]),
            make_tensor(vec![l01_b.clone(), s1.clone(), l12_b.clone()]),
            make_tensor(vec![l12_b.clone(), s2.clone()]),
        ])
        .unwrap();

        // Contract with truncation: max_rank=1
        let options = ContractOptions::zipup().with_max_rank(1);
        let result = contract(&tt1, &tt2, &options).unwrap();

        // Verify truncation actually happened: all bond dims should be <= 1
        for bd in result.bond_dims() {
            assert!(bd <= 1, "Bond dim {} exceeds max_rank=1", bd);
        }

        // Compare with naive (exact) result — should be approximate, not exact
        let naive_result = tt1.to_dense().unwrap().contract(&tt2.to_dense().unwrap());
        let naive_data = naive_result.to_vec_f64().unwrap();
        let result_data = result.to_dense().unwrap().to_vec_f64().unwrap();

        assert_eq!(result_data.len(), naive_data.len());
        let naive_norm: f64 = naive_data.iter().map(|x| x * x).sum::<f64>().sqrt();
        let error_norm: f64 = result_data
            .iter()
            .zip(naive_data.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        // Relative error should be bounded (not exact, but reasonable)
        let rel_error = error_norm / naive_norm;
        assert!(
            rel_error < 1.0,
            "Relative error {} is too large for truncated contraction",
            rel_error
        );
    }

    #[test]
    fn test_contract_free_fn_nan_rtol() {
        let s0 = idx(1090, 2);
        let t0 = make_tensor(vec![s0.clone()]);
        let tt1 = TensorTrain::new(vec![t0.clone()]).unwrap();
        let tt2 = TensorTrain::new(vec![t0]).unwrap();

        let options = ContractOptions::zipup().with_rtol(f64::NAN);
        let err = contract(&tt1, &tt2, &options).unwrap_err();
        assert!(matches!(err, TensorTrainError::OperationError { .. }));
    }

    #[test]
    fn test_contract_free_fn_inf_rtol() {
        let s0 = idx(1091, 2);
        let t0 = make_tensor(vec![s0.clone()]);
        let tt1 = TensorTrain::new(vec![t0.clone()]).unwrap();
        let tt2 = TensorTrain::new(vec![t0]).unwrap();

        let options = ContractOptions::zipup().with_rtol(f64::INFINITY);
        let err = contract(&tt1, &tt2, &options).unwrap_err();
        assert!(matches!(err, TensorTrainError::OperationError { .. }));
    }
}
