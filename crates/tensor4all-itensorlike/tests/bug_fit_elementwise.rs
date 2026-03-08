//! Bug: ContractOptions::fit() converges to wrong values for element-wise
//! MPS products via diagonal embedding.
//!
//! fit() converges but to WRONG values, while zipup() is exact.
//! More sweeps don't help — error stays the same.
//!
//! This blocks using fit() for bubble computations in quanticsnegf-rs.

use tensor4all_core::{factorize, DynIndex, FactorizeOptions, IndexLike, TensorDynLen};
use tensor4all_itensorlike::{ContractOptions, TensorTrain};

/// Convert matrix (row-major) to quantics interleaved bit ordering.
fn matrix_to_quantics(nbit: usize, data: &[f64]) -> Vec<f64> {
    let n = 1 << nbit;
    assert_eq!(data.len(), n * n);
    let mut quantics = vec![0.0; n * n];
    for row in 0..n {
        for col in 0..n {
            let mut q_idx = 0;
            for b in 0..nbit {
                let r_bit = (row >> (nbit - 1 - b)) & 1;
                let c_bit = (col >> (nbit - 1 - b)) & 1;
                q_idx = q_idx * 2 + r_bit;
                q_idx = q_idx * 2 + c_bit;
            }
            quantics[q_idx] = data[row * n + col];
        }
    }
    quantics
}

/// Create a QTT from a square matrix using QR factorization.
fn create_matrix_tt(
    row_indices: &[DynIndex],
    col_indices: &[DynIndex],
    data: &[f64],
) -> TensorTrain {
    let nbit = row_indices.len();
    let quantics_data = matrix_to_quantics(nbit, data);

    let mut site_indices = Vec::new();
    for i in 0..nbit {
        site_indices.push(row_indices[i].clone());
        site_indices.push(col_indices[i].clone());
    }
    let n_sites = site_indices.len();

    let mut remaining = TensorDynLen::from_dense_f64(site_indices.clone(), quantics_data);
    let mut tensors = Vec::with_capacity(n_sites);
    let opts = FactorizeOptions::qr().with_rtol(0.0);

    for site_idx in site_indices.iter().take(n_sites - 1) {
        let remaining_indices = remaining.indices().to_vec();
        let mut left_inds = Vec::new();
        for idx in &remaining_indices {
            if idx.id() == site_idx.id() {
                left_inds.push(idx.clone());
                break;
            }
            left_inds.push(idx.clone());
        }
        let result = factorize(&remaining, &left_inds, &opts).unwrap();
        tensors.push(result.left);
        remaining = result.right;
    }
    tensors.push(remaining);
    TensorTrain::new(tensors).unwrap()
}

/// Diagonalize: replace index `s` with [s_new1, s_new2], non-zero when s_new1==s_new2.
fn as_diagonal(
    tensor: &TensorDynLen,
    s: &DynIndex,
    s_new1: &DynIndex,
    s_new2: &DynIndex,
) -> TensorDynLen {
    let indices = tensor.indices();
    let dims: Vec<usize> = indices.iter().map(|idx| idx.dim()).collect();
    let dim = s.dim();
    let s_pos = indices.iter().position(|idx| idx.id() == s.id()).unwrap();

    let mut new_indices = indices.to_vec();
    new_indices.splice(s_pos..s_pos + 1, vec![s_new1.clone(), s_new2.clone()]);
    let mut new_dims = dims.clone();
    new_dims.splice(s_pos..s_pos + 1, vec![dim, dim]);

    let total_new: usize = new_dims.iter().product();
    let data = tensor.to_vec_f64().unwrap();
    let mut new_data = vec![0.0f64; total_new];

    for (flat, &val) in data.iter().enumerate() {
        let mut rem = flat;
        let mut old_idx = vec![0usize; dims.len()];
        for i in (0..dims.len()).rev() {
            old_idx[i] = rem % dims[i];
            rem /= dims[i];
        }
        let s_val = old_idx[s_pos];
        let mut new_idx = old_idx;
        new_idx.splice(s_pos..s_pos + 1, vec![s_val, s_val]);
        let mut nf = 0;
        for (i, &d) in new_dims.iter().enumerate() {
            nf = nf * d + new_idx[i];
        }
        new_data[nf] = val;
    }
    TensorDynLen::from_dense_f64(new_indices, new_data)
}

/// Extract diagonal: keep only s==s_result, remove s_result index.
fn extract_diagonal(tensor: &TensorDynLen, s: &DynIndex, s_result: &DynIndex) -> TensorDynLen {
    let indices = tensor.indices();
    let dims: Vec<usize> = indices.iter().map(|idx| idx.dim()).collect();
    let s_pos = indices.iter().position(|idx| idx.id() == s.id()).unwrap();
    let sr_pos = indices
        .iter()
        .position(|idx| idx.id() == s_result.id())
        .unwrap();

    let new_indices: Vec<DynIndex> = indices
        .iter()
        .filter(|idx| idx.id() != s_result.id())
        .cloned()
        .collect();
    let new_dims: Vec<usize> = new_indices.iter().map(|idx| idx.dim()).collect();
    let new_total: usize = new_dims.iter().product();
    let data = tensor.to_vec_f64().unwrap();
    let mut new_data = vec![0.0f64; new_total];

    for (flat, &val) in data.iter().enumerate() {
        let mut rem = flat;
        let mut idx = vec![0usize; dims.len()];
        for i in (0..dims.len()).rev() {
            idx[i] = rem % dims[i];
            rem /= dims[i];
        }
        if idx[s_pos] != idx[sr_pos] {
            continue;
        }
        let new_idx: Vec<usize> = idx
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != sr_pos)
            .map(|(_, &v)| v)
            .collect();
        let mut nf = 0;
        for (i, &d) in new_dims.iter().enumerate() {
            nf = nf * d + new_idx[i];
        }
        new_data[nf] = val;
    }
    TensorDynLen::from_dense_f64(new_indices, new_data)
}

/// Element-wise product via diagonal embedding + TT contraction.
fn elementwise_mul(
    m1: &TensorTrain,
    m2: &TensorTrain,
    sites: &[DynIndex],
    options: &ContractOptions,
) -> TensorTrain {
    let mut m1_prep = m1.clone();
    let mut m2_prep = m2.clone();
    let mut maps: Vec<(DynIndex, DynIndex)> = Vec::new();

    for s in sites {
        let s_contract = s.sim();
        let s_result = s.sim();

        let pos1 = m1_prep
            .siteinds()
            .iter()
            .position(|inds| inds.iter().any(|idx| idx.id() == s.id()))
            .unwrap();
        let pos2 = m2_prep
            .siteinds()
            .iter()
            .position(|inds| inds.iter().any(|idx| idx.id() == s.id()))
            .unwrap();

        let t1 = as_diagonal(m1_prep.tensor(pos1), s, &s_result, &s_contract);
        m1_prep.set_tensor(pos1, t1);

        let t2 = as_diagonal(m2_prep.tensor(pos2), s, &s_contract, s);
        m2_prep.set_tensor(pos2, t2);

        maps.push((s.clone(), s_result));
    }

    let mut result = m1_prep.contract(&m2_prep, options).unwrap();

    for (original, s_result) in &maps {
        let siteinds = result.siteinds();
        let pos = siteinds
            .iter()
            .position(|inds| inds.iter().any(|idx| idx.id() == original.id()))
            .unwrap();
        let t = result.tensor(pos);
        let extracted = extract_diagonal(t, original, s_result);
        result.set_tensor(pos, extracted);
    }
    result
}

/// Test: element-wise product of two NON-product-state MPS.
///
/// Product-state MPS (bond dim=1) work fine with both fit and zipup.
/// MPS with higher bond dim (e.g., from QR decomposition of a general
/// matrix) may fail with fit().
///
/// Use two different 4x4 matrices that produce non-trivial bond dims.
#[test]
fn test_fit_wrong_for_elementwise_nontrivial_bond() {
    let nbit = 2;
    let n = 1 << nbit;

    let row_inds: Vec<DynIndex> = (0..nbit)
        .map(|i| DynIndex::new_dyn_with_tag(2, &format!("z1={}", i + 1)).unwrap())
        .collect();
    let col_inds: Vec<DynIndex> = (0..nbit)
        .map(|i| DynIndex::new_dyn_with_tag(2, &format!("z2={}", i + 1)).unwrap())
        .collect();

    // A: sequential 1..16
    let a_data: Vec<f64> = (0..n * n).map(|k| (k + 1) as f64).collect();
    // B: sequential 51..66
    let b_data: Vec<f64> = (0..n * n).map(|k| (k + 51) as f64).collect();

    let m1 = create_matrix_tt(&row_inds, &col_inds, &a_data);
    let m2 = create_matrix_tt(&row_inds, &col_inds, &b_data);

    eprintln!(
        "m1 linkdims: {:?}",
        (0..m1.len().saturating_sub(1))
            .map(|s| m1.linkind(s).map(|l| l.dim()).unwrap_or(0))
            .collect::<Vec<_>>()
    );
    eprintln!(
        "m2 linkdims: {:?}",
        (0..m2.len().saturating_sub(1))
            .map(|s| m2.linkind(s).map(|l| l.dim()).unwrap_or(0))
            .collect::<Vec<_>>()
    );

    let sites: Vec<DynIndex> = row_inds.iter().chain(col_inds.iter()).cloned().collect();

    // Reference: element-wise product
    let expected_quantics = matrix_to_quantics(nbit, &{
        let mut e = vec![0.0; n * n];
        for i in 0..n * n {
            e[i] = a_data[i] * b_data[i];
        }
        e
    });
    let ref_norm: f64 = expected_quantics.iter().map(|x| x * x).sum::<f64>().sqrt();

    // zipup
    let result_zipup = elementwise_mul(&m1, &m2, &sites, &ContractOptions::zipup());
    let zipup_data = result_zipup.to_dense().unwrap().to_vec_f64().unwrap();
    let zipup_err: f64 = zipup_data
        .iter()
        .zip(expected_quantics.iter())
        .map(|(got, exp)| (got - exp).powi(2))
        .sum::<f64>()
        .sqrt()
        / ref_norm;

    // fit
    let result_fit = elementwise_mul(&m1, &m2, &sites, &ContractOptions::fit());
    let fit_data = result_fit.to_dense().unwrap().to_vec_f64().unwrap();
    let fit_err: f64 = fit_data
        .iter()
        .zip(expected_quantics.iter())
        .map(|(got, exp)| (got - exp).powi(2))
        .sum::<f64>()
        .sqrt()
        / ref_norm;

    // fit with 10 sweeps
    let result_fit10 = elementwise_mul(&m1, &m2, &sites, &ContractOptions::fit().with_nsweeps(10));
    let fit10_data = result_fit10.to_dense().unwrap().to_vec_f64().unwrap();
    let fit10_err: f64 = fit10_data
        .iter()
        .zip(expected_quantics.iter())
        .map(|(got, exp)| (got - exp).powi(2))
        .sum::<f64>()
        .sqrt()
        / ref_norm;

    eprintln!("zipup   rel_err = {:.6e}", zipup_err);
    eprintln!("fit     rel_err = {:.6e}", fit_err);
    eprintln!("fit(10) rel_err = {:.6e}", fit10_err);

    assert!(
        zipup_err < 1e-10,
        "zipup should be exact: {:.6e}",
        zipup_err
    );
    assert!(
        fit_err < 1e-6,
        "fit rel_err too large: {:.6e} (10 sweeps: {:.6e})",
        fit_err,
        fit10_err
    );
}
