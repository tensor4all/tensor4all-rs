use super::*;

#[test]
fn test_shift_operator_construction() {
    let mut op: *mut t4a_linop = std::ptr::null_mut();

    // Periodic shift by +1 on 4-bit quantics
    let status = t4a_qtransform_shift(4, 1, t4a_boundary_condition::Periodic, &mut op);
    assert_eq!(status, T4A_SUCCESS);
    assert!(!op.is_null());
    t4a_linop_release(op);

    // Open shift
    let mut op2: *mut t4a_linop = std::ptr::null_mut();
    let status = t4a_qtransform_shift(4, -3, t4a_boundary_condition::Open, &mut op2);
    assert_eq!(status, T4A_SUCCESS);
    assert!(!op2.is_null());
    t4a_linop_release(op2);
}

#[test]
fn test_flip_operator_construction() {
    let mut op: *mut t4a_linop = std::ptr::null_mut();

    let status = t4a_qtransform_flip(4, t4a_boundary_condition::Periodic, &mut op);
    assert_eq!(status, T4A_SUCCESS);
    assert!(!op.is_null());
    t4a_linop_release(op);
}

#[test]
fn test_phase_rotation_construction() {
    let mut op: *mut t4a_linop = std::ptr::null_mut();

    let status = t4a_qtransform_phase_rotation(4, std::f64::consts::PI / 4.0, &mut op);
    assert_eq!(status, T4A_SUCCESS);
    assert!(!op.is_null());
    t4a_linop_release(op);
}

#[test]
fn test_cumsum_construction() {
    let mut op: *mut t4a_linop = std::ptr::null_mut();

    let status = t4a_qtransform_cumsum(4, &mut op);
    assert_eq!(status, T4A_SUCCESS);
    assert!(!op.is_null());
    t4a_linop_release(op);
}

#[test]
fn test_fourier_operator_construction() {
    // Forward Fourier
    let mut op: *mut t4a_linop = std::ptr::null_mut();
    let status = t4a_qtransform_fourier(4, 1, 0, 0.0, &mut op);
    assert_eq!(status, T4A_SUCCESS);
    assert!(!op.is_null());
    t4a_linop_release(op);

    // Inverse Fourier with custom params
    let mut op2: *mut t4a_linop = std::ptr::null_mut();
    let status = t4a_qtransform_fourier(4, 0, 16, 1e-12, &mut op2);
    assert_eq!(status, T4A_SUCCESS);
    assert!(!op2.is_null());
    t4a_linop_release(op2);
}

#[test]
fn test_shift_operator_multivar_construction() {
    let mut op: *mut t4a_linop = std::ptr::null_mut();

    let status =
        t4a_qtransform_shift_multivar(4, 1, t4a_boundary_condition::Periodic, 3, 1, &mut op);
    assert_eq!(status, T4A_SUCCESS);
    assert!(!op.is_null());
    t4a_linop_release(op);
}

#[test]
fn test_flip_operator_multivar_construction() {
    let mut op: *mut t4a_linop = std::ptr::null_mut();

    let status = t4a_qtransform_flip_multivar(4, t4a_boundary_condition::Open, 3, 2, &mut op);
    assert_eq!(status, T4A_SUCCESS);
    assert!(!op.is_null());
    t4a_linop_release(op);
}

#[test]
fn test_phase_rotation_multivar_construction() {
    let mut op: *mut t4a_linop = std::ptr::null_mut();

    let status =
        t4a_qtransform_phase_rotation_multivar(4, std::f64::consts::PI / 3.0, 3, 0, &mut op);
    assert_eq!(status, T4A_SUCCESS);
    assert!(!op.is_null());
    t4a_linop_release(op);
}

#[test]
fn test_affine_operator_construction() {
    let mut op: *mut t4a_linop = std::ptr::null_mut();

    // 3 output variables from 2 input variables:
    // y1 = x1
    // y2 = x1 + x2
    // y3 = x2
    let a_num = [
        1_i64, 1, 0, // first input column
        0, 1, 1, // second input column
    ];
    let a_den = [1_i64; 6];
    let b_num = [0_i64, 0, 0];
    let b_den = [1_i64; 3];
    let bc = [
        t4a_boundary_condition::Open,
        t4a_boundary_condition::Open,
        t4a_boundary_condition::Periodic,
    ];

    let status = t4a_qtransform_affine(
        4,
        a_num.as_ptr(),
        a_den.as_ptr(),
        b_num.as_ptr(),
        b_den.as_ptr(),
        3,
        2,
        bc.as_ptr(),
        &mut op,
    );
    assert_eq!(status, T4A_SUCCESS);
    assert!(!op.is_null());
    t4a_linop_release(op);
}

#[test]
fn test_binaryop_operator_construction() {
    let mut op: *mut t4a_linop = std::ptr::null_mut();

    let status = t4a_qtransform_binaryop(
        4,
        1,
        1,
        1,
        -1,
        t4a_boundary_condition::Open,
        t4a_boundary_condition::Periodic,
        &mut op,
    );
    assert_eq!(status, T4A_SUCCESS);
    assert!(!op.is_null());
    t4a_linop_release(op);
}

#[test]
fn test_linop_apply_shift() {
    use num_complex::Complex64;
    use tensor4all_core::index::{DynId, Index, TagSet};
    use tensor4all_core::TensorDynLen;
    use tensor4all_simplett::{types::tensor3_zeros, AbstractTensorTrain, Tensor3Ops, TensorTrain};
    use tensor4all_treetn::{apply_linear_operator, ApplyOptions, TreeTN};

    type DynIndex = Index<DynId, TagSet>;

    let r = 3;

    // Build the shift operator via Rust API
    let op =
        shift_operator(r, 1, BoundaryCondition::Periodic).expect("Failed to create shift operator");

    // Create a product state |0⟩ = |0⟩⊗|0⟩⊗|0⟩ (all bits zero)
    let mut tensors_mps: Vec<_> = Vec::with_capacity(r);
    for _ in 0..r {
        let mut t = tensor3_zeros::<Complex64>(1, 2, 1);
        *t.get3_mut(0, 0, 0) = Complex64::new(1.0, 0.0); // bit = 0
        tensors_mps.push(t);
    }
    let mps = TensorTrain::new(tensors_mps).expect("Failed to create MPS");

    // Convert MPS to TreeTN with indices matching operator's input (true_index)
    let n = mps.len();
    let mut bond_indices: Vec<DynIndex> = Vec::with_capacity(n + 1);
    for i in 0..=n {
        let dim = if i == 0 {
            1
        } else {
            mps.site_tensor(i - 1).right_dim()
        };
        bond_indices.push(Index::new_dyn(dim));
    }

    let mut tensors: Vec<TensorDynLen> = Vec::with_capacity(n);
    let node_names: Vec<usize> = (0..n).collect();

    for i in 0..n {
        let t = mps.site_tensor(i);
        let site_dim = t.site_dim();
        let right_dim = t.right_dim();
        let left_dim = t.left_dim();

        // Use operator's true_index as the state's site index
        let op_input = op
            .get_input_mapping(&i)
            .expect("input mapping")
            .true_index
            .clone();

        let mut indices: Vec<DynIndex> = Vec::new();
        let mut dims_vec: Vec<usize> = Vec::new();

        if i > 0 {
            indices.push(bond_indices[i].clone());
            dims_vec.push(left_dim);
        }
        indices.push(op_input);
        dims_vec.push(site_dim);
        if i < n - 1 {
            indices.push(bond_indices[i + 1].clone());
            dims_vec.push(right_dim);
        }

        let total_size: usize = dims_vec.iter().product();
        let mut data: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); total_size];

        #[allow(clippy::needless_range_loop)]
        if i == 0 && n == 1 {
            for s in 0..site_dim {
                data[s] = *t.get3(0, s, 0);
            }
        } else if i == 0 {
            for s in 0..site_dim {
                for rv in 0..right_dim {
                    data[s * right_dim + rv] = *t.get3(0, s, rv);
                }
            }
        } else if i == n - 1 {
            for l in 0..left_dim {
                for s in 0..site_dim {
                    data[l * site_dim + s] = *t.get3(l, s, 0);
                }
            }
        } else {
            for l in 0..left_dim {
                for s in 0..site_dim {
                    for rv in 0..right_dim {
                        data[(l * site_dim + s) * right_dim + rv] = *t.get3(l, s, rv);
                    }
                }
            }
        }

        tensors.push(TensorDynLen::from_dense(indices, data).unwrap());
    }

    let treetn = TreeTN::from_tensors(tensors, node_names).expect("Failed to create TreeTN");

    // Test via Rust API first to verify our TreeTN construction is correct
    let result =
        apply_linear_operator(&op, &treetn, ApplyOptions::naive()).expect("Rust apply failed");
    assert!(!result.node_names().is_empty());

    // Now test via C API: wrap in opaque types
    let c_op = Box::into_raw(Box::new(t4a_linop::new(op)));
    let c_state = Box::into_raw(Box::new(t4a_treetn::new(treetn)));

    let mut c_result: *mut t4a_treetn = std::ptr::null_mut();
    let status = t4a_linop_apply(
        c_op,
        c_state,
        0, // Naive
        0.0,
        0,
        &mut c_result,
    );
    assert_eq!(status, T4A_SUCCESS);
    assert!(!c_result.is_null());

    // Clean up
    unsafe {
        let _ = Box::from_raw(c_result);
        let _ = Box::from_raw(c_state);
        let _ = Box::from_raw(c_op);
    }
}

#[test]
fn test_linop_set_io_space_then_apply() {
    use num_complex::Complex64;
    use tensor4all_core::index::{DynId, Index, TagSet};
    use tensor4all_core::TensorDynLen;
    use tensor4all_simplett::{types::tensor3_zeros, AbstractTensorTrain, Tensor3Ops, TensorTrain};
    use tensor4all_treetn::TreeTN;

    type DynIndex = Index<DynId, TagSet>;

    let r = 3;
    let mut op: *mut t4a_linop = std::ptr::null_mut();
    let status = t4a_qtransform_shift(r, 1, t4a_boundary_condition::Periodic, &mut op);
    assert_eq!(status, T4A_SUCCESS);
    assert!(!op.is_null());

    let mut tensors_mps: Vec<_> = Vec::with_capacity(r);
    for _ in 0..r {
        let mut t = tensor3_zeros::<Complex64>(1, 2, 1);
        *t.get3_mut(0, 0, 0) = Complex64::new(1.0, 0.0);
        tensors_mps.push(t);
    }
    let mps = TensorTrain::new(tensors_mps).expect("Failed to create MPS");

    let n = mps.len();
    let mut bond_indices: Vec<DynIndex> = Vec::with_capacity(n + 1);
    for i in 0..=n {
        let dim = if i == 0 {
            1
        } else {
            mps.site_tensor(i - 1).right_dim()
        };
        bond_indices.push(Index::new_dyn(dim));
    }

    let mut tensors: Vec<TensorDynLen> = Vec::with_capacity(n);
    let node_names: Vec<usize> = (0..n).collect();
    let site_indices: Vec<DynIndex> = (0..n).map(|_| Index::new_dyn(2)).collect();

    for i in 0..n {
        let t = mps.site_tensor(i);
        let site_dim = t.site_dim();
        let right_dim = t.right_dim();
        let left_dim = t.left_dim();

        let mut indices: Vec<DynIndex> = Vec::new();

        if i > 0 {
            indices.push(bond_indices[i].clone());
        }
        indices.push(site_indices[i].clone());
        if i < n - 1 {
            indices.push(bond_indices[i + 1].clone());
        }

        let mut flat = vec![Complex64::new(0.0, 0.0); left_dim * site_dim * right_dim];
        if i == 0 {
            for s in 0..site_dim {
                for rv in 0..right_dim {
                    flat[s * right_dim + rv] = *t.get3(0, s, rv);
                }
            }
        } else if i == n - 1 {
            for l in 0..left_dim {
                for s in 0..site_dim {
                    flat[l * site_dim + s] = *t.get3(l, s, 0);
                }
            }
        } else {
            for l in 0..left_dim {
                for s in 0..site_dim {
                    for rv in 0..right_dim {
                        flat[(l * site_dim + s) * right_dim + rv] = *t.get3(l, s, rv);
                    }
                }
            }
        }

        tensors.push(TensorDynLen::from_dense(indices, flat).expect("site tensor"));
    }

    let state_tn = TreeTN::from_tensors(tensors, node_names).expect("state treetn");
    let state = Box::into_raw(Box::new(t4a_treetn::new(state_tn)));

    assert_eq!(t4a_linop_set_input_space(op, state), T4A_SUCCESS);
    assert_eq!(t4a_linop_set_output_space(op, state), T4A_SUCCESS);

    let mut out: *mut t4a_treetn = std::ptr::null_mut();
    let status = t4a_linop_apply(op, state, 0, 0.0, 0, &mut out);
    assert_eq!(status, T4A_SUCCESS);
    assert!(!out.is_null());

    unsafe {
        let _ = Box::from_raw(out);
        let _ = Box::from_raw(state);
        let _ = Box::from_raw(op);
    }
}

#[test]
fn test_null_pointer_guards() {
    let mut op: *mut t4a_linop = std::ptr::null_mut();

    // Null output pointer
    assert_eq!(
        t4a_qtransform_shift(4, 1, t4a_boundary_condition::Periodic, std::ptr::null_mut()),
        T4A_NULL_POINTER
    );
    assert_eq!(
        t4a_qtransform_flip(4, t4a_boundary_condition::Periodic, std::ptr::null_mut()),
        T4A_NULL_POINTER
    );
    assert_eq!(
        t4a_qtransform_phase_rotation(4, 1.0, std::ptr::null_mut()),
        T4A_NULL_POINTER
    );
    assert_eq!(
        t4a_qtransform_cumsum(4, std::ptr::null_mut()),
        T4A_NULL_POINTER
    );
    assert_eq!(
        t4a_qtransform_fourier(4, 1, 0, 0.0, std::ptr::null_mut()),
        T4A_NULL_POINTER
    );

    // Invalid r=0
    assert_eq!(
        t4a_qtransform_shift(0, 1, t4a_boundary_condition::Periodic, &mut op),
        T4A_INVALID_ARGUMENT
    );
    assert_eq!(
        t4a_qtransform_flip(0, t4a_boundary_condition::Periodic, &mut op),
        T4A_INVALID_ARGUMENT
    );
    assert_eq!(
        t4a_qtransform_phase_rotation(0, 1.0, &mut op),
        T4A_INVALID_ARGUMENT
    );
    assert_eq!(t4a_qtransform_cumsum(0, &mut op), T4A_INVALID_ARGUMENT);
    assert_eq!(
        t4a_qtransform_fourier(0, 1, 0, 0.0, &mut op),
        T4A_INVALID_ARGUMENT
    );

    // Apply null guards
    let mut out: *mut t4a_treetn = std::ptr::null_mut();
    assert_eq!(
        t4a_linop_apply(std::ptr::null(), std::ptr::null(), 1, 0.0, 0, &mut out),
        T4A_NULL_POINTER
    );

    // Invalid method
    // We need a valid op and state for this, so skip method check with nulls
    // (it returns NULL_POINTER first)

    // Release null should not crash
    t4a_linop_release(std::ptr::null_mut());
}
