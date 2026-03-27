use super::*;
use crate::defaults::Index;
use num_complex::Complex64;
use std::ffi::OsString;
use std::time::Duration;

struct ScopedEnvVar {
    key: &'static str,
    previous: Option<OsString>,
}

impl ScopedEnvVar {
    fn set(key: &'static str, value: &str) -> Self {
        let previous = std::env::var_os(key);
        unsafe {
            std::env::set_var(key, value);
        }
        Self { key, previous }
    }
}

impl Drop for ScopedEnvVar {
    fn drop(&mut self) {
        match &self.previous {
            Some(value) => unsafe {
                std::env::set_var(self.key, value);
            },
            None => unsafe {
                std::env::remove_var(self.key);
            },
        }
    }
}

fn make_test_tensor(shape: &[usize], ids: &[u64]) -> TensorDynLen {
    let indices: Vec<DynIndex> = ids
        .iter()
        .zip(shape.iter())
        .map(|(&id, &dim)| Index::new(DynId(id), dim))
        .collect();
    let total_size: usize = shape.iter().product();
    let data: Vec<Complex64> = (0..total_size)
        .map(|i| Complex64::new(i as f64, 0.0))
        .collect();
    TensorDynLen::from_dense(indices, data).unwrap()
}

// ========================================================================
// contract_multi tests
// ========================================================================

#[test]
fn test_contract_multi_empty() {
    let tensors: Vec<&TensorDynLen> = vec![];
    let result = contract_multi(&tensors, AllowedPairs::All);
    assert!(result.is_err());
}

#[test]
fn test_contract_multi_single() {
    let tensor = make_test_tensor(&[2, 3], &[1, 2]);
    let result = contract_multi(&[&tensor], AllowedPairs::All).unwrap();
    assert_eq!(result.dims(), tensor.dims());
}

#[test]
fn test_contract_multi_pair() {
    // A[i,j] * B[j,k] -> C[i,k]
    let a = make_test_tensor(&[2, 3], &[1, 2]); // i=1, j=2
    let b = make_test_tensor(&[3, 4], &[2, 3]); // j=2, k=3
    let result = contract_multi(&[&a, &b], AllowedPairs::All).unwrap();
    assert_eq!(result.dims(), vec![2, 4]); // i, k
}

#[test]
fn test_contract_multi_three() {
    // A[i,j] * B[j,k] * C[k,l] -> D[i,l]
    let a = make_test_tensor(&[2, 3], &[1, 2]); // i=1, j=2
    let b = make_test_tensor(&[3, 4], &[2, 3]); // j=2, k=3
    let c = make_test_tensor(&[4, 5], &[3, 4]); // k=3, l=4
    let result = contract_multi(&[&a, &b, &c], AllowedPairs::All).unwrap();
    let mut sorted_dims = result.dims();
    sorted_dims.sort();
    assert_eq!(sorted_dims, vec![2, 5]); // i=2, l=5
}

#[test]
fn test_contract_multi_four() {
    // A[i,j] * B[j,k] * C[k,l] * D[l,m] -> E[i,m]
    let a = make_test_tensor(&[2, 3], &[1, 2]);
    let b = make_test_tensor(&[3, 4], &[2, 3]);
    let c = make_test_tensor(&[4, 5], &[3, 4]);
    let d = make_test_tensor(&[5, 6], &[4, 5]);
    let result = contract_multi(&[&a, &b, &c, &d], AllowedPairs::All).unwrap();
    let mut sorted_dims = result.dims();
    sorted_dims.sort();
    assert_eq!(sorted_dims, vec![2, 6]); // i=2, m=6
}

#[test]
fn test_contract_multi_outer_product() {
    // A[i,j] * B[k,l] (no common indices) -> outer product C[i,j,k,l]
    let a = make_test_tensor(&[2, 3], &[1, 2]);
    let b = make_test_tensor(&[4, 5], &[3, 4]);
    let result = contract_multi(&[&a, &b], AllowedPairs::All).unwrap();
    let result_dims = result.dims();
    let total_elements: usize = result_dims.iter().product();
    assert_eq!(total_elements, 2 * 3 * 4 * 5);
    assert_eq!(result_dims.len(), 4);
}

#[test]
fn test_contract_multi_vector_outer_product() {
    // A[i] * B[j] (no common indices) -> outer product C[i,j]
    let a = make_test_tensor(&[2], &[1]); // i=1
    let b = make_test_tensor(&[3], &[2]); // j=2
    let result = contract_multi(&[&a, &b], AllowedPairs::All).unwrap();
    let result_dims = result.dims();
    let total_elements: usize = result_dims.iter().product();
    assert_eq!(total_elements, 2 * 3);
    assert_eq!(result.dims().len(), 2);
}

#[test]
fn test_contract_connected_disconnected_error() {
    let a = make_test_tensor(&[2, 3], &[1, 2]);
    let b = make_test_tensor(&[4, 5], &[3, 4]);
    let result = contract_connected(&[&a, &b], AllowedPairs::All);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .to_lowercase()
        .contains("disconnected"));
}

#[test]
fn test_contract_connected_specified_no_contractable_error() {
    let a = make_test_tensor(&[2, 3], &[1, 2]); // i=1, j=2
    let b = make_test_tensor(&[4, 5], &[3, 4]); // k=3, l=4 (no common with a)
    let result = contract_connected(&[&a, &b], AllowedPairs::Specified(&[(0, 1)]));
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string().to_lowercase();
    assert!(
        err_msg.contains("disconnected") || err_msg.contains("no contractable"),
        "Expected error about disconnected or no contractable indices, got: {}",
        err_msg
    );
}

#[test]
fn test_find_tensor_connected_components_trivial_cases() {
    let empty: Vec<&TensorDynLen> = Vec::new();
    assert!(find_tensor_connected_components(&empty, AllowedPairs::All).is_empty());

    let a = make_test_tensor(&[2, 3], &[1, 2]);
    assert_eq!(
        find_tensor_connected_components(&[&a], AllowedPairs::All),
        vec![vec![0]]
    );
}

#[test]
fn test_find_tensor_connected_components_multiple_components() {
    let a = make_test_tensor(&[2, 3], &[1, 2]);
    let b = make_test_tensor(&[3, 4], &[2, 3]);
    let c = make_test_tensor(&[5, 6], &[4, 5]);

    assert_eq!(
        find_tensor_connected_components(&[&a, &b, &c], AllowedPairs::All),
        vec![vec![0, 1], vec![2]]
    );
}

#[test]
fn test_remap_allowed_pairs_filters_pairs_outside_component() {
    let remapped = remap_allowed_pairs(AllowedPairs::Specified(&[(0, 1), (1, 3), (2, 3)]), &[1, 3]);
    match remapped {
        RemappedAllowedPairs::All => panic!("expected specified remapped pairs"),
        RemappedAllowedPairs::Specified(pairs) => assert_eq!(pairs, vec![(0, 1)]),
    }
}

#[test]
fn test_union_find_remap_and_tensor_id_helpers() {
    let i = DynId(1);
    let j = DynId(2);
    let k = DynId(3);

    let mut uf = AxisUnionFind::default();
    uf.union(i, j);

    assert_eq!(uf.remap(i), uf.remap(j));
    assert_ne!(uf.remap(i), uf.remap(k));
    assert_eq!(
        uf.remap_ids(&[i, j, k]),
        vec![uf.remap(i), uf.remap(j), uf.remap(k)]
    );

    let a = make_test_tensor(&[2, 3], &[1, 2]);
    let b = make_test_tensor(&[3, 4], &[2, 3]);
    let remapped = remap_tensor_ids(&[&a, &b], &mut uf);
    assert_eq!(remapped[0].len(), 2);
    assert_eq!(remapped[1].len(), 2);

    let output = vec![Index::new(DynId(1), 2), Index::new(DynId(3), 4)];
    let remapped_output = remap_output_ids(&output, &mut uf);
    assert_eq!(remapped_output.len(), 2);

    let sizes = collect_sizes(&[&a, &b], &mut uf);
    assert_eq!(sizes[&uf.remap(DynId(1))], 2);
    assert_eq!(sizes[&uf.remap(DynId(2))], 2);
    assert_eq!(sizes[&uf.remap(DynId(3))], 4);
}

#[test]
fn test_contract_profile_helpers() {
    let _profile_env = ScopedEnvVar::set("T4A_PROFILE_CONTRACT", "1");
    reset_contract_profile();

    let signature = ContractSignature {
        operands: vec![ContractOperandSignature {
            dims: vec![2, 3],
            ids: vec![1, 2],
            is_diag: false,
        }],
        output_ids: vec![1],
        output_dims: vec![2],
    };
    record_contract_profile(signature, Duration::from_micros(10));
    print_and_reset_contract_profile();
    reset_contract_profile();
}

// ========================================================================
// AllowedPairs::Specified tests
// ========================================================================

#[test]
fn test_contract_specified_pairs() {
    // A[i,j], B[j,k], C[i,l] - tensors 0, 1, 2
    let a = make_test_tensor(&[2, 3], &[1, 2]); // i=1, j=2
    let b = make_test_tensor(&[3, 4], &[2, 3]); // j=2, k=3
    let c = make_test_tensor(&[2, 5], &[1, 4]); // i=1, l=4
    let result = contract_multi(&[&a, &b, &c], AllowedPairs::Specified(&[(0, 1), (0, 2)])).unwrap();
    let mut sorted_dims = result.dims();
    sorted_dims.sort();
    assert_eq!(sorted_dims, vec![4, 5]); // k=4, l=5
}

#[test]
fn test_contract_specified_no_contractable_indices_error() {
    let a = make_test_tensor(&[2, 3], &[1, 2]); // i=1, j=2
    let b = make_test_tensor(&[3, 4], &[2, 3]); // j=2, k=3
    let c = make_test_tensor(&[6, 5], &[5, 4]); // m=5, l=4 (no common with B)
    let result = contract_multi(&[&a, &b, &c], AllowedPairs::Specified(&[(0, 1), (1, 2)]));
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("no contractable indices"));
}

#[test]
fn test_contract_specified_disconnected_outer_product() {
    let a = make_test_tensor(&[2, 3], &[1, 2]); // i=1, j=2
    let b = make_test_tensor(&[3, 4], &[2, 3]); // j=2, k=3
    let c = make_test_tensor(&[4, 5], &[4, 5]); // m=4, n=5
    let d = make_test_tensor(&[5, 6], &[5, 6]); // n=5, p=6
    let result = contract_multi(
        &[&a, &b, &c, &d],
        AllowedPairs::Specified(&[(0, 1), (2, 3)]),
    )
    .unwrap();
    assert_eq!(result.dims().len(), 4);
    let mut sorted_dims = result.dims();
    sorted_dims.sort();
    assert_eq!(sorted_dims, vec![2, 4, 4, 6]);
}

// ========================================================================
// Union-Find tests
// ========================================================================

#[test]
fn test_union_find_basic() {
    let mut uf = AxisUnionFind::new();

    let a = DynId(1);
    let b = DynId(2);
    let c = DynId(3);

    uf.make_set(a);
    uf.make_set(b);
    uf.make_set(c);

    assert_ne!(uf.find(a), uf.find(b));
    assert_ne!(uf.find(b), uf.find(c));

    uf.union(a, b);
    assert_eq!(uf.find(a), uf.find(b));
    assert_ne!(uf.find(a), uf.find(c));

    uf.union(b, c);
    assert_eq!(uf.find(a), uf.find(b));
    assert_eq!(uf.find(b), uf.find(c));
    assert_eq!(uf.find(a), uf.find(c));
}

#[test]
fn test_union_find_chain() {
    let mut uf = AxisUnionFind::new();

    let i = DynId(1);
    let j = DynId(2);
    let k = DynId(3);
    let l = DynId(4);

    uf.union(i, j);
    uf.union(j, k);
    uf.union(k, l);

    let rep = uf.find(i);
    assert_eq!(uf.find(j), rep);
    assert_eq!(uf.find(k), rep);
    assert_eq!(uf.find(l), rep);
}

#[test]
fn test_remap_ids() {
    let mut uf = AxisUnionFind::new();

    let i = DynId(1);
    let j = DynId(2);
    let k = DynId(3);

    uf.union(i, j);

    let ids = vec![i, j, k];
    let remapped = uf.remap_ids(&ids);

    assert_eq!(remapped[0], remapped[1]);
    assert_ne!(remapped[0], remapped[2]);
}

#[test]
fn test_three_diag_chain() {
    let mut uf = AxisUnionFind::new();

    let i = DynId(1);
    let j = DynId(2);
    let k = DynId(3);
    let l = DynId(4);

    uf.union(i, j);
    uf.union(j, k);
    uf.union(k, l);

    let rep = uf.find(i);
    assert_eq!(uf.find(j), rep);
    assert_eq!(uf.find(k), rep);
    assert_eq!(uf.find(l), rep);
}

#[test]
fn test_three_diag_star() {
    let mut uf = AxisUnionFind::new();

    let a = DynId(1);
    let b = DynId(2);
    let c = DynId(3);
    let d = DynId(4);

    uf.union(a, b);
    uf.union(a, c);
    uf.union(a, d);

    let rep = uf.find(a);
    assert_eq!(uf.find(b), rep);
    assert_eq!(uf.find(c), rep);
    assert_eq!(uf.find(d), rep);
}

#[test]
fn test_diag_with_three_axes() {
    let mut uf = AxisUnionFind::new();

    let i = DynId(1);
    let j = DynId(2);
    let k = DynId(3);
    let l = DynId(4);

    uf.union(i, j);
    uf.union(j, k);

    let rep = uf.find(i);
    assert_eq!(uf.find(j), rep);
    assert_eq!(uf.find(k), rep);
    assert_ne!(uf.find(l), rep);
}

#[test]
fn test_two_separate_diag_groups() {
    let mut uf = AxisUnionFind::new();

    let a = DynId(1);
    let b = DynId(2);
    let c = DynId(3);
    let d = DynId(4);

    uf.union(a, b);
    uf.union(c, d);

    assert_eq!(uf.find(a), uf.find(b));
    assert_eq!(uf.find(c), uf.find(d));
    assert_ne!(uf.find(a), uf.find(c));
}

#[test]
fn test_diag_and_dense_mixed() {
    let mut uf = AxisUnionFind::new();

    let i = DynId(1);
    let j = DynId(2);
    let k = DynId(3);

    uf.union(i, j);
    uf.make_set(k);

    assert_eq!(uf.find(i), uf.find(j));
    assert_ne!(uf.find(j), uf.find(k));
}

#[test]
fn test_complex_network() {
    let mut uf = AxisUnionFind::new();

    let a = DynId(1);
    let b = DynId(2);
    let c = DynId(3);
    let d = DynId(4);
    let e = DynId(5);
    let f = DynId(6);

    uf.union(a, b);
    uf.union(b, c);
    uf.make_set(d);
    uf.union(d, e);
    uf.union(e, f);

    let rep1 = uf.find(a);
    assert_eq!(uf.find(b), rep1);
    assert_eq!(uf.find(c), rep1);

    let rep2 = uf.find(d);
    assert_eq!(uf.find(e), rep2);
    assert_eq!(uf.find(f), rep2);

    assert_ne!(rep1, rep2);
}

#[test]
fn test_single_diag_tensor() {
    let mut uf = AxisUnionFind::new();

    let i = DynId(1);
    let j = DynId(2);

    uf.union(i, j);

    assert_eq!(uf.find(i), uf.find(j));
}

#[test]
fn test_empty_union_find() {
    let mut uf = AxisUnionFind::new();

    let x = DynId(42);
    uf.make_set(x);
    assert_eq!(uf.find(x), x);
}

#[test]
fn test_idempotent_union() {
    let mut uf = AxisUnionFind::new();

    let a = DynId(1);
    let b = DynId(2);

    uf.union(a, b);
    let rep1 = uf.find(a);

    uf.union(a, b);
    let rep2 = uf.find(a);

    uf.union(b, a);
    let rep3 = uf.find(a);

    assert_eq!(rep1, rep2);
    assert_eq!(rep2, rep3);
}

#[test]
fn test_self_union() {
    let mut uf = AxisUnionFind::new();

    let a = DynId(1);
    uf.union(a, a);

    assert_eq!(uf.find(a), a);
}

#[test]
fn test_four_diag_tensors_chain() {
    let mut uf = AxisUnionFind::new();

    let i = DynId(1);
    let j = DynId(2);
    let k = DynId(3);
    let l = DynId(4);
    let m = DynId(5);

    uf.union(i, j);
    uf.union(j, k);
    uf.union(k, l);
    uf.union(l, m);

    let rep = uf.find(i);
    assert_eq!(uf.find(j), rep);
    assert_eq!(uf.find(k), rep);
    assert_eq!(uf.find(l), rep);
    assert_eq!(uf.find(m), rep);
}

#[test]
fn test_diag_tensors_merge_two_chains() {
    let mut uf = AxisUnionFind::new();

    let a = DynId(1);
    let b = DynId(2);
    let c = DynId(3);
    let d = DynId(4);
    let e = DynId(5);

    uf.union(a, b);
    uf.union(b, c);
    uf.union(d, e);
    uf.union(e, c);

    let rep = uf.find(a);
    assert_eq!(uf.find(b), rep);
    assert_eq!(uf.find(c), rep);
    assert_eq!(uf.find(d), rep);
    assert_eq!(uf.find(e), rep);
}

#[test]
fn test_remap_preserves_order() {
    let mut uf = AxisUnionFind::new();

    let i = DynId(1);
    let j = DynId(2);
    let k = DynId(3);
    let l = DynId(4);

    uf.union(i, j);
    uf.union(k, l);

    let ids = vec![i, j, k, l, i, k];
    let remapped = uf.remap_ids(&ids);

    assert_eq!(remapped.len(), 6);
    assert_eq!(remapped[0], remapped[1]);
    assert_eq!(remapped[2], remapped[3]);
    assert_ne!(remapped[0], remapped[2]);
    assert_eq!(remapped[0], remapped[4]);
    assert_eq!(remapped[2], remapped[5]);
}

// ========================================================================
// contract_connected tests
// ========================================================================

fn make_dense_tensor(shape: &[usize], ids: &[u64]) -> TensorDynLen {
    let indices: Vec<DynIndex> = ids
        .iter()
        .zip(shape.iter())
        .map(|(&id, &dim)| Index::new(DynId(id), dim))
        .collect();
    let total_size: usize = shape.iter().product();
    let data: Vec<Complex64> = (0..total_size)
        .map(|i| Complex64::new((i + 1) as f64, 0.0))
        .collect();
    TensorDynLen::from_dense(indices, data).unwrap()
}

#[test]
fn test_contract_connected_empty() {
    let tensors: Vec<&TensorDynLen> = vec![];
    let result = contract_connected(&tensors, AllowedPairs::All);
    assert!(result.is_err());
}

#[test]
fn test_contract_connected_single() {
    let tensor = make_dense_tensor(&[2, 3], &[1, 2]);
    let result = contract_connected(&[&tensor], AllowedPairs::All).unwrap();
    assert_eq!(result.dims(), tensor.dims());
}

#[test]
fn test_contract_connected_pair_dense() {
    // A[i,j] * B[j,k] -> C[i,k]
    let a = make_dense_tensor(&[2, 3], &[1, 2]); // i=1, j=2
    let b = make_dense_tensor(&[3, 4], &[2, 3]); // j=2, k=3
    let result = contract_connected(&[&a, &b], AllowedPairs::All).unwrap();
    assert_eq!(result.dims(), vec![2, 4]); // i, k
}

#[test]
fn test_contract_connected_three_dense() {
    // A[i,j] * B[j,k] * C[k,l] -> D[i,l]
    let a = make_dense_tensor(&[2, 3], &[1, 2]); // i=1, j=2
    let b = make_dense_tensor(&[3, 4], &[2, 3]); // j=2, k=3
    let c = make_dense_tensor(&[4, 5], &[3, 4]); // k=3, l=4
    let result = contract_connected(&[&a, &b, &c], AllowedPairs::All).unwrap();
    let mut sorted_dims = result.dims();
    sorted_dims.sort();
    assert_eq!(sorted_dims, vec![2, 5]); // i=2, l=5
}
