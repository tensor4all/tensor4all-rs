use num_complex::Complex64;
use tensor4all_core::{DynIndex, TensorDynLen};

use crate::TreeTN;

fn make_three_node_chain() -> TreeTN<TensorDynLen, usize> {
    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let s2 = DynIndex::new_dyn(2);
    let bond01 = DynIndex::new_dyn(2);
    let bond12 = DynIndex::new_dyn(2);

    let t0 =
        TensorDynLen::from_dense(vec![s0, bond01.clone()], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let t1 = TensorDynLen::from_dense(
        vec![bond01, s1, bond12.clone()],
        vec![1.0_f64, -1.0, 0.5, 2.0, 1.5, 0.25, -0.75, 3.0],
    )
    .unwrap();
    let t2 = TensorDynLen::from_dense(vec![bond12, s2], vec![2.0_f64, 0.0, 1.0, 3.0]).unwrap();

    TreeTN::from_tensors(vec![t0, t1, t2], vec![0, 1, 2]).unwrap()
}

#[test]
fn test_inner_matches_norm_squared_for_three_node_chain() {
    let tn = make_three_node_chain();
    let inner = tn.inner(&tn).unwrap();

    let mut tn_for_norm = tn.clone();
    let norm_squared = tn_for_norm.norm_squared().unwrap();

    assert!((inner.real() - norm_squared).abs() < 1.0e-10);
    assert!(inner.imag().abs() < 1.0e-10);
}

#[test]
fn test_inner_preserves_complex_phase() {
    let site = DynIndex::new_dyn(2);
    let left = TensorDynLen::from_dense(
        vec![site.clone()],
        vec![Complex64::new(1.0, 1.0), Complex64::new(2.0, -1.0)],
    )
    .unwrap();
    let right = TensorDynLen::from_dense(
        vec![site],
        vec![Complex64::new(3.0, -2.0), Complex64::new(-1.0, 4.0)],
    )
    .unwrap();

    let left_tn = TreeTN::from_tensors(vec![left], vec![0usize]).unwrap();
    let right_tn = TreeTN::from_tensors(vec![right], vec![0usize]).unwrap();

    let inner = left_tn.inner(&right_tn).unwrap();

    assert!((inner.real() + 5.0).abs() < 1.0e-10);
    assert!((inner.imag() - 2.0).abs() < 1.0e-10);
}
