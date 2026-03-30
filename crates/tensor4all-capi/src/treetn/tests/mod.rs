use super::*;
use tensor4all_core::{DynIndex, IndexLike, TensorDynLen};
use tensor4all_simplett::{tensor3_from_data, AbstractTensorTrain, Tensor3Ops, TensorTrain};
use tensor4all_treetn::TreeTN;

unsafe extern "C" {
    fn t4a_treetn_all_site_index_ids(
        ptr: *const t4a_treetn,
        out_index_ids: *mut u64,
        out_vertex_names: *mut libc::size_t,
        buf_len: libc::size_t,
        out_n_indices: *mut libc::size_t,
    ) -> StatusCode;

    fn t4a_treetn_evaluate(
        ptr: *const t4a_treetn,
        index_ids: *const u64,
        n_indices: libc::size_t,
        values: *const libc::size_t,
        n_points: libc::size_t,
        out_re: *mut libc::c_double,
        out_im: *mut libc::c_double,
    ) -> StatusCode;
}

// ========================================================================
// Helpers
// ========================================================================

/// Helper: create a 2-site MPS-like TreeTN with site dims (2, 2) and bond dim 3.
fn make_two_site_treetn() -> (*mut t4a_treetn, Vec<*mut t4a_tensor>, Vec<*mut t4a_index>) {
    use crate::{t4a_index_clone, t4a_index_new, t4a_tensor_new_dense_f64};

    let s0 = t4a_index_new(2);
    let l01 = t4a_index_new(3);
    let s1 = t4a_index_new(2);

    let data0: Vec<f64> = (0..6).map(|i| (i + 1) as f64).collect();
    let data1: Vec<f64> = (0..6).map(|i| (i + 1) as f64).collect();

    let inds0: [*const t4a_index; 2] = [s0, l01];
    let dims0: [libc::size_t; 2] = [2, 3];
    let t0 = t4a_tensor_new_dense_f64(2, inds0.as_ptr(), dims0.as_ptr(), data0.as_ptr(), 6);

    let l01_clone = t4a_index_clone(l01);
    let inds1: [*const t4a_index; 2] = [l01_clone, s1];
    let dims1: [libc::size_t; 2] = [3, 2];
    let t1 = t4a_tensor_new_dense_f64(2, inds1.as_ptr(), dims1.as_ptr(), data1.as_ptr(), 6);

    let tensors: [*const t4a_tensor; 2] = [t0, t1];
    let mut out: *mut t4a_treetn = std::ptr::null_mut();
    let status = t4a_treetn_new(tensors.as_ptr(), 2, &mut out);
    assert_eq!(status, T4A_SUCCESS);
    assert!(!out.is_null());

    (out, vec![t0, t1], vec![s0, l01, l01_clone, s1])
}

fn cleanup_treetn(
    tt: *mut t4a_treetn,
    tensors: Vec<*mut t4a_tensor>,
    indices: Vec<*mut t4a_index>,
) {
    t4a_treetn_release(tt);
    for t in tensors {
        crate::t4a_tensor_release(t);
    }
    for i in indices {
        crate::t4a_index_release(i);
    }
}

fn simplett_to_treetn(tt: &TensorTrain<f64>) -> *mut t4a_treetn {
    let n_sites = tt.len();
    assert!(n_sites > 0, "TensorTrain must have at least one site");

    let site_indices: Vec<DynIndex> = (0..n_sites)
        .map(|site| DynIndex::new_dyn(tt.site_tensor(site).site_dim()))
        .collect();
    let bond_indices: Vec<DynIndex> = (0..=n_sites)
        .map(|site| {
            let dim = if site == 0 {
                1
            } else {
                tt.site_tensor(site - 1).right_dim()
            };
            DynIndex::new_dyn(dim)
        })
        .collect();

    let mut tensors = Vec::with_capacity(n_sites);
    let node_names: Vec<usize> = (0..n_sites).collect();

    for site in 0..n_sites {
        let tensor = tt.site_tensor(site);
        let left_dim = tensor.left_dim();
        let site_dim = tensor.site_dim();
        let right_dim = tensor.right_dim();

        let mut indices = Vec::with_capacity(3);
        if site > 0 {
            indices.push(bond_indices[site].clone());
        }
        indices.push(site_indices[site].clone());
        if site + 1 < n_sites {
            indices.push(bond_indices[site + 1].clone());
        }

        let total_size = indices.iter().map(|idx| idx.dim()).product();
        let mut data = vec![0.0; total_size];

        if site == 0 && n_sites == 1 {
            for (s, slot) in data.iter_mut().enumerate() {
                *slot = *tensor.get3(0, s, 0);
            }
        } else if site == 0 {
            for s in 0..site_dim {
                for r in 0..right_dim {
                    let idx = s + site_dim * r;
                    data[idx] = *tensor.get3(0, s, r);
                }
            }
        } else if site + 1 == n_sites {
            for l in 0..left_dim {
                for s in 0..site_dim {
                    let idx = l + left_dim * s;
                    data[idx] = *tensor.get3(l, s, 0);
                }
            }
        } else {
            for l in 0..left_dim {
                for s in 0..site_dim {
                    for r in 0..right_dim {
                        let idx = l + left_dim * (s + site_dim * r);
                        data[idx] = *tensor.get3(l, s, r);
                    }
                }
            }
        }

        tensors.push(TensorDynLen::from_dense(indices, data).unwrap());
    }

    let treetn = TreeTN::from_tensors(tensors, node_names).unwrap();
    Box::into_raw(Box::new(t4a_treetn::new(treetn)))
}

fn make_product_treetn_from_simplett() -> *mut t4a_treetn {
    let tt = TensorTrain::new(vec![
        tensor3_from_data(vec![1.0, 2.0], 1, 2, 1),
        tensor3_from_data(vec![10.0, 20.0, 30.0], 1, 3, 1),
        tensor3_from_data(vec![100.0, 200.0], 1, 2, 1),
    ])
    .unwrap();

    simplett_to_treetn(&tt)
}

// ========================================================================
// Lifecycle tests
// ========================================================================

#[test]
fn test_treetn_new_and_release() {
    let (tt, tensors, indices) = make_two_site_treetn();

    // Check vertex count
    let mut n_vertices: libc::size_t = 0;
    let status = t4a_treetn_num_vertices(tt, &mut n_vertices);
    assert_eq!(status, T4A_SUCCESS);
    assert_eq!(n_vertices, 2);

    // Check edge count
    let mut n_edges: libc::size_t = 0;
    let status = t4a_treetn_num_edges(tt, &mut n_edges);
    assert_eq!(status, T4A_SUCCESS);
    assert_eq!(n_edges, 1);

    // Clone
    let tt2 = t4a_treetn_clone(tt);
    assert!(!tt2.is_null());
    t4a_treetn_release(tt2);

    cleanup_treetn(tt, tensors, indices);
}

#[test]
fn test_treetn_tensor_access() {
    let (tt, tensors, indices) = make_two_site_treetn();

    // Get tensor at vertex 0
    let mut tensor_out: *mut t4a_tensor = std::ptr::null_mut();
    let status = t4a_treetn_tensor(tt, 0, &mut tensor_out);
    assert_eq!(status, T4A_SUCCESS);
    assert!(!tensor_out.is_null());

    // Check tensor rank
    let mut rank: libc::size_t = 0;
    crate::t4a_tensor_get_rank(tensor_out, &mut rank);
    assert_eq!(rank, 2); // site + link
    crate::t4a_tensor_release(tensor_out);

    // Invalid vertex
    let mut tensor_out2: *mut t4a_tensor = std::ptr::null_mut();
    let status = t4a_treetn_tensor(tt, 99, &mut tensor_out2);
    assert_eq!(status, T4A_INVALID_ARGUMENT);

    cleanup_treetn(tt, tensors, indices);
}

#[test]
fn test_treetn_neighbors() {
    let (tt, tensors, indices) = make_two_site_treetn();

    let mut buf = [0usize; 4];
    let mut n_out: libc::size_t = 0;
    let status = t4a_treetn_neighbors(tt, 0, buf.as_mut_ptr(), 4, &mut n_out);
    assert_eq!(status, T4A_SUCCESS);
    assert_eq!(n_out, 1);
    assert_eq!(buf[0], 1);

    cleanup_treetn(tt, tensors, indices);
}

#[test]
fn test_treetn_linkind_and_bond_dim() {
    let (tt, tensors, indices) = make_two_site_treetn();

    // Link index between vertex 0 and 1
    let mut idx_out: *mut t4a_index = std::ptr::null_mut();
    let status = t4a_treetn_linkind(tt, 0, 1, &mut idx_out);
    assert_eq!(status, T4A_SUCCESS);
    assert!(!idx_out.is_null());
    crate::t4a_index_release(idx_out);

    // Bond dimension
    let mut dim: libc::size_t = 0;
    let status = t4a_treetn_bond_dim(tt, 0, 1, &mut dim);
    assert_eq!(status, T4A_SUCCESS);
    assert_eq!(dim, 3);

    // MPS convenience
    let mut dim2: libc::size_t = 0;
    let status = t4a_treetn_bond_dim_at(tt, 0, &mut dim2);
    assert_eq!(status, T4A_SUCCESS);
    assert_eq!(dim2, 3);

    // maxbonddim
    let mut max_dim: libc::size_t = 0;
    let status = t4a_treetn_maxbonddim(tt, &mut max_dim);
    assert_eq!(status, T4A_SUCCESS);
    assert_eq!(max_dim, 3);

    // bond_dims
    let mut dims = [0usize; 1];
    let status = t4a_treetn_bond_dims(tt, dims.as_mut_ptr(), 1);
    assert_eq!(status, T4A_SUCCESS);
    assert_eq!(dims[0], 3);

    cleanup_treetn(tt, tensors, indices);
}

#[test]
fn test_treetn_siteinds() {
    let (tt, tensors, indices) = make_two_site_treetn();

    let mut idx_buf: [*mut t4a_index; 4] = [std::ptr::null_mut(); 4];
    let mut n_out: libc::size_t = 0;
    let status = t4a_treetn_siteinds(tt, 0, idx_buf.as_mut_ptr(), 4, &mut n_out);
    assert_eq!(status, T4A_SUCCESS);
    assert_eq!(n_out, 1); // one site index at vertex 0

    // Clean up returned indices
    for idx in idx_buf.iter().take(n_out) {
        if !idx.is_null() {
            crate::t4a_index_release(*idx);
        }
    }

    cleanup_treetn(tt, tensors, indices);
}

// ========================================================================
// Orthogonalization tests
// ========================================================================

#[test]
fn test_treetn_orthogonalize() {
    let (tt, tensors, indices) = make_two_site_treetn();

    let status = t4a_treetn_orthogonalize(tt, 0);
    assert_eq!(status, T4A_SUCCESS);

    // Check canonical form
    let mut form: t4a_canonical_form = t4a_canonical_form::LU;
    let status = t4a_treetn_canonical_form(tt, &mut form);
    assert_eq!(status, T4A_SUCCESS);
    assert_eq!(form, t4a_canonical_form::Unitary);

    // Check ortho center
    let mut center_buf = [0usize; 4];
    let mut n_center: libc::size_t = 0;
    let status = t4a_treetn_ortho_center(tt, center_buf.as_mut_ptr(), 4, &mut n_center);
    assert_eq!(status, T4A_SUCCESS);
    assert_eq!(n_center, 1);
    assert_eq!(center_buf[0], 0);

    cleanup_treetn(tt, tensors, indices);
}

#[test]
fn test_treetn_orthogonalize_with() {
    let (tt, tensors, indices) = make_two_site_treetn();

    let status = t4a_treetn_orthogonalize_with(tt, 1, t4a_canonical_form::LU);
    assert_eq!(status, T4A_SUCCESS);

    let mut form: t4a_canonical_form = t4a_canonical_form::Unitary;
    let status = t4a_treetn_canonical_form(tt, &mut form);
    assert_eq!(status, T4A_SUCCESS);
    assert_eq!(form, t4a_canonical_form::LU);

    cleanup_treetn(tt, tensors, indices);
}

// ========================================================================
// Operation tests
// ========================================================================

#[test]
fn test_treetn_truncate() {
    let (tt, tensors, indices) = make_two_site_treetn();

    // Truncate with maxdim=1
    let status = t4a_treetn_truncate(tt, 0.0, 0.0, 1);
    assert_eq!(status, T4A_SUCCESS);

    let mut max_dim: libc::size_t = 0;
    t4a_treetn_maxbonddim(tt, &mut max_dim);
    assert_eq!(max_dim, 1);

    cleanup_treetn(tt, tensors, indices);
}

#[test]
fn test_treetn_truncate_with_cutoff() {
    let (tt, tensors, indices) = make_two_site_treetn();

    let status = t4a_treetn_truncate(tt, 0.0, 1e-10, 0);
    assert_eq!(status, T4A_SUCCESS);

    cleanup_treetn(tt, tensors, indices);
}

#[test]
fn test_treetn_truncate_with_rtol() {
    let (tt, tensors, indices) = make_two_site_treetn();

    let status = t4a_treetn_truncate(tt, 1e-5, 0.0, 0);
    assert_eq!(status, T4A_SUCCESS);

    cleanup_treetn(tt, tensors, indices);
}

#[test]
fn test_treetn_inner() {
    use crate::{t4a_index_clone, t4a_index_new, t4a_tensor_new_dense_f64};

    // Build two 1-site TreeTNs
    let s0 = t4a_index_new(2);
    let s0_clone = t4a_index_clone(s0);

    let data_a: Vec<f64> = vec![1.0, 0.0];
    let data_b: Vec<f64> = vec![0.0, 1.0];

    let inds_a: [*const t4a_index; 1] = [s0];
    let dims_a: [libc::size_t; 1] = [2];
    let t0 = t4a_tensor_new_dense_f64(1, inds_a.as_ptr(), dims_a.as_ptr(), data_a.as_ptr(), 2);

    let inds_b: [*const t4a_index; 1] = [s0_clone];
    let dims_b: [libc::size_t; 1] = [2];
    let t1 = t4a_tensor_new_dense_f64(1, inds_b.as_ptr(), dims_b.as_ptr(), data_b.as_ptr(), 2);

    let tensors0: [*const t4a_tensor; 1] = [t0];
    let tensors1: [*const t4a_tensor; 1] = [t1];

    let mut tt0: *mut t4a_treetn = std::ptr::null_mut();
    let mut tt1: *mut t4a_treetn = std::ptr::null_mut();
    t4a_treetn_new(tensors0.as_ptr(), 1, &mut tt0);
    t4a_treetn_new(tensors1.as_ptr(), 1, &mut tt1);

    let mut re: f64 = 0.0;
    let mut im: f64 = 0.0;
    let status = t4a_treetn_inner(tt0, tt1, &mut re, &mut im);
    assert_eq!(status, T4A_SUCCESS);
    assert!((re - 0.0).abs() < 1e-10);
    assert!((im - 0.0).abs() < 1e-10);

    t4a_treetn_release(tt0);
    t4a_treetn_release(tt1);
    crate::t4a_tensor_release(t0);
    crate::t4a_tensor_release(t1);
    crate::t4a_index_release(s0);
    crate::t4a_index_release(s0_clone);
}

#[test]
fn test_treetn_norm() {
    let (tt, tensors, indices) = make_two_site_treetn();

    let mut norm: f64 = 0.0;
    let status = t4a_treetn_norm(tt, &mut norm);
    assert_eq!(status, T4A_SUCCESS);
    assert!(norm > 0.0);

    cleanup_treetn(tt, tensors, indices);
}

#[test]
fn test_treetn_lognorm() {
    let (tt, tensors, indices) = make_two_site_treetn();

    let mut lognorm: f64 = 0.0;
    let status = t4a_treetn_lognorm(tt, &mut lognorm);
    assert_eq!(status, T4A_SUCCESS);
    assert!(lognorm.is_finite());

    cleanup_treetn(tt, tensors, indices);
}

#[test]
fn test_treetn_add() {
    let (tt1, tensors1, indices1) = make_two_site_treetn();
    let (tt2, tensors2, indices2) = make_two_site_treetn();

    let mut result: *mut t4a_treetn = std::ptr::null_mut();
    let status = t4a_treetn_add(tt1, tt2, &mut result);
    assert_eq!(status, T4A_SUCCESS);
    assert!(!result.is_null());

    // Result should have 2 vertices
    let mut n_vertices: libc::size_t = 0;
    t4a_treetn_num_vertices(result, &mut n_vertices);
    assert_eq!(n_vertices, 2);

    // Bond dimension should be sum (3 + 3 = 6)
    let mut max_bond: libc::size_t = 0;
    t4a_treetn_maxbonddim(result, &mut max_bond);
    assert_eq!(max_bond, 6);

    t4a_treetn_release(result);
    cleanup_treetn(tt1, tensors1, indices1);
    cleanup_treetn(tt2, tensors2, indices2);
}

#[test]
fn test_treetn_to_dense() {
    let (tt, tensors, indices) = make_two_site_treetn();

    let mut dense: *mut t4a_tensor = std::ptr::null_mut();
    let status = t4a_treetn_to_dense(tt, &mut dense);
    assert_eq!(status, T4A_SUCCESS);
    assert!(!dense.is_null());

    let mut rank: libc::size_t = 0;
    crate::t4a_tensor_get_rank(dense, &mut rank);
    assert_eq!(rank, 2); // two site indices

    crate::t4a_tensor_release(dense);
    cleanup_treetn(tt, tensors, indices);
}

#[test]
fn test_treetn_contract() {
    use crate::{t4a_index_clone, t4a_index_new, t4a_tensor_new_dense_f64};

    let s0 = t4a_index_new(2);
    let s0_clone = t4a_index_clone(s0);

    let data: Vec<f64> = vec![1.0, 2.0];

    let inds0: [*const t4a_index; 1] = [s0];
    let dims: [libc::size_t; 1] = [2];
    let t0 = t4a_tensor_new_dense_f64(1, inds0.as_ptr(), dims.as_ptr(), data.as_ptr(), 2);

    let inds1: [*const t4a_index; 1] = [s0_clone];
    let t1 = t4a_tensor_new_dense_f64(1, inds1.as_ptr(), dims.as_ptr(), data.as_ptr(), 2);

    let tensors0: [*const t4a_tensor; 1] = [t0];
    let tensors1: [*const t4a_tensor; 1] = [t1];

    let mut tt0: *mut t4a_treetn = std::ptr::null_mut();
    let mut tt1: *mut t4a_treetn = std::ptr::null_mut();
    t4a_treetn_new(tensors0.as_ptr(), 1, &mut tt0);
    t4a_treetn_new(tensors1.as_ptr(), 1, &mut tt1);

    let mut result: *mut t4a_treetn = std::ptr::null_mut();
    let status = t4a_treetn_contract(
        tt0,
        tt1,
        crate::types::t4a_contract_method::Zipup,
        0.0,
        0.0,
        0,
        &mut result,
    );
    assert_eq!(status, T4A_SUCCESS);
    assert!(!result.is_null());

    let mut n_vertices: libc::size_t = 0;
    t4a_treetn_num_vertices(result, &mut n_vertices);
    assert_eq!(n_vertices, 1);

    t4a_treetn_release(result);
    t4a_treetn_release(tt0);
    t4a_treetn_release(tt1);
    crate::t4a_tensor_release(t0);
    crate::t4a_tensor_release(t1);
    crate::t4a_index_release(s0);
    crate::t4a_index_release(s0_clone);
}

#[test]
fn test_treetn_linsolve() {
    use crate::{t4a_index_clone, t4a_index_new, t4a_tensor_new_dense_f64};

    // Build 1-site identity MPO and 1-site MPS for I * x = b
    let s0 = t4a_index_new(2);
    let s0p = t4a_index_new(2);

    let identity: Vec<f64> = vec![1.0, 0.0, 0.0, 1.0];
    let mpo_inds: [*const t4a_index; 2] = [s0, s0p];
    let mpo_dims: [libc::size_t; 2] = [2, 2];
    let mpo_t = t4a_tensor_new_dense_f64(
        2,
        mpo_inds.as_ptr(),
        mpo_dims.as_ptr(),
        identity.as_ptr(),
        4,
    );
    let mpo_tensors: [*const t4a_tensor; 1] = [mpo_t];
    let mut mpo: *mut t4a_treetn = std::ptr::null_mut();
    t4a_treetn_new(mpo_tensors.as_ptr(), 1, &mut mpo);
    assert!(!mpo.is_null());

    // RHS with index s0p
    let s0p_clone = t4a_index_clone(s0p);
    let rhs_data: Vec<f64> = vec![3.0, 4.0];
    let rhs_inds: [*const t4a_index; 1] = [s0p_clone];
    let rhs_dims: [libc::size_t; 1] = [2];
    let rhs_t = t4a_tensor_new_dense_f64(
        1,
        rhs_inds.as_ptr(),
        rhs_dims.as_ptr(),
        rhs_data.as_ptr(),
        2,
    );
    let rhs_tensors: [*const t4a_tensor; 1] = [rhs_t];
    let mut rhs: *mut t4a_treetn = std::ptr::null_mut();
    t4a_treetn_new(rhs_tensors.as_ptr(), 1, &mut rhs);
    assert!(!rhs.is_null());

    // Initial guess with index s0
    let s0_clone = t4a_index_clone(s0);
    let init_data: Vec<f64> = vec![1.0, 1.0];
    let init_inds: [*const t4a_index; 1] = [s0_clone];
    let init_dims: [libc::size_t; 1] = [2];
    let init_t = t4a_tensor_new_dense_f64(
        1,
        init_inds.as_ptr(),
        init_dims.as_ptr(),
        init_data.as_ptr(),
        2,
    );
    let init_tensors: [*const t4a_tensor; 1] = [init_t];
    let mut init: *mut t4a_treetn = std::ptr::null_mut();
    t4a_treetn_new(init_tensors.as_ptr(), 1, &mut init);
    assert!(!init.is_null());

    let mut result: *mut t4a_treetn = std::ptr::null_mut();
    let status = t4a_treetn_linsolve(
        mpo,
        rhs,
        init,
        0.0, // a0
        1.0, // a1
        4,   // nsweeps
        1e-10,
        0.0,
        10,
        &mut result,
    );
    assert_eq!(status, T4A_SUCCESS);
    assert!(!result.is_null());

    let mut n_vertices: libc::size_t = 0;
    t4a_treetn_num_vertices(result, &mut n_vertices);
    assert_eq!(n_vertices, 1);

    t4a_treetn_release(result);
    t4a_treetn_release(mpo);
    t4a_treetn_release(rhs);
    t4a_treetn_release(init);
    crate::t4a_tensor_release(mpo_t);
    crate::t4a_tensor_release(rhs_t);
    crate::t4a_tensor_release(init_t);
    crate::t4a_index_release(s0);
    crate::t4a_index_release(s0p);
    crate::t4a_index_release(s0p_clone);
    crate::t4a_index_release(s0_clone);
}

// ========================================================================
// IndexId-based evaluate tests
// ========================================================================

#[test]
fn test_treetn_all_site_index_ids_and_evaluate() {
    let tt = make_product_treetn_from_simplett();

    // Query size
    let mut n_indices: libc::size_t = 0;
    let status = unsafe {
        t4a_treetn_all_site_index_ids(
            tt,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
            &mut n_indices,
        )
    };
    assert_eq!(status, T4A_SUCCESS);
    assert_eq!(n_indices, 3); // 3 sites

    // Fill buffers
    let mut index_ids = vec![0u64; n_indices];
    let mut vertex_names = vec![0usize; n_indices];
    let status = unsafe {
        t4a_treetn_all_site_index_ids(
            tt,
            index_ids.as_mut_ptr(),
            vertex_names.as_mut_ptr(),
            n_indices,
            &mut n_indices,
        )
    };
    assert_eq!(status, T4A_SUCCESS);
    assert_eq!(n_indices, 3);

    // Build position map: vertex -> position in index_ids
    // Product TT: site 0 has dim 2, site 1 has dim 3, site 2 has dim 2
    // f(i, j, k) = (i+1) * (10*j + 10) * (100*k + 100)
    // Let's evaluate a single point using the new API
    // We need to arrange values in index_ids order

    // Single point: indices [1, 2, 1] in vertex order
    let mut values = vec![0usize; n_indices];
    for v in 0..3usize {
        let pos = vertex_names.iter().position(|&name| name == v).unwrap();
        values[pos] = [1, 2, 1][v];
    }

    let mut result_re = 0.0;
    let status = unsafe {
        t4a_treetn_evaluate(
            tt,
            index_ids.as_ptr(),
            n_indices,
            values.as_ptr(),
            1,
            &mut result_re,
            std::ptr::null_mut(),
        )
    };
    assert_eq!(status, T4A_SUCCESS);
    assert!((result_re - 12_000.0).abs() < 1e-10);

    // Multi-point evaluation: 3 points
    // point 0: [0, 0, 0], point 1: [1, 1, 0], point 2: [0, 2, 1]
    let points = [[0usize, 0, 0], [1, 1, 0], [0, 2, 1]];
    let mut flat_values = vec![0usize; n_indices * 3];
    for (p, point) in points.iter().enumerate() {
        for (v, &val) in point.iter().enumerate() {
            let pos = vertex_names.iter().position(|&name| name == v).unwrap();
            flat_values[pos + n_indices * p] = val;
        }
    }

    let mut results_re = [0.0; 3];
    let status = unsafe {
        t4a_treetn_evaluate(
            tt,
            index_ids.as_ptr(),
            n_indices,
            flat_values.as_ptr(),
            3,
            results_re.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    };
    assert_eq!(status, T4A_SUCCESS);
    assert_eq!(results_re, [1_000.0, 4_000.0, 6_000.0]);

    t4a_treetn_release(tt);
}

// ========================================================================
// Null pointer guard tests
// ========================================================================

#[test]
fn test_treetn_null_guards() {
    let mut dummy_out: *mut t4a_treetn = std::ptr::null_mut();
    let mut dummy_tensor: *mut t4a_tensor = std::ptr::null_mut();
    let mut dummy_index: *mut t4a_index = std::ptr::null_mut();
    let mut dummy_size: libc::size_t = 0;
    let mut dummy_double: f64 = 0.0;
    let mut dummy_form = t4a_canonical_form::Unitary;

    // t4a_treetn_new with null out
    assert_eq!(
        t4a_treetn_new(std::ptr::null(), 0, std::ptr::null_mut()),
        T4A_NULL_POINTER
    );

    // t4a_treetn_new with null tensors but non-zero count
    assert_eq!(
        t4a_treetn_new(std::ptr::null(), 3, &mut dummy_out),
        T4A_NULL_POINTER
    );

    // Accessors
    assert_eq!(
        t4a_treetn_num_vertices(std::ptr::null(), &mut dummy_size),
        T4A_NULL_POINTER
    );
    assert_eq!(
        t4a_treetn_num_edges(std::ptr::null(), &mut dummy_size),
        T4A_NULL_POINTER
    );
    assert_eq!(
        t4a_treetn_tensor(std::ptr::null(), 0, &mut dummy_tensor),
        T4A_NULL_POINTER
    );
    assert_eq!(
        t4a_treetn_set_tensor(std::ptr::null_mut(), 0, std::ptr::null()),
        T4A_NULL_POINTER
    );
    assert_eq!(
        t4a_treetn_neighbors(
            std::ptr::null(),
            0,
            std::ptr::null_mut(),
            0,
            &mut dummy_size
        ),
        T4A_NULL_POINTER
    );
    assert_eq!(
        t4a_treetn_linkind(std::ptr::null(), 0, 1, &mut dummy_index),
        T4A_NULL_POINTER
    );
    assert_eq!(
        t4a_treetn_siteinds(
            std::ptr::null(),
            0,
            std::ptr::null_mut(),
            0,
            &mut dummy_size
        ),
        T4A_NULL_POINTER
    );
    assert_eq!(
        t4a_treetn_bond_dim(std::ptr::null(), 0, 1, &mut dummy_size),
        T4A_NULL_POINTER
    );
    assert_eq!(
        t4a_treetn_maxbonddim(std::ptr::null(), &mut dummy_size),
        T4A_NULL_POINTER
    );
    assert_eq!(
        t4a_treetn_bond_dims(std::ptr::null(), std::ptr::null_mut(), 0),
        T4A_NULL_POINTER
    );

    // Orthogonalization
    assert_eq!(
        t4a_treetn_orthogonalize(std::ptr::null_mut(), 0),
        T4A_NULL_POINTER
    );
    assert_eq!(
        t4a_treetn_orthogonalize_with(std::ptr::null_mut(), 0, t4a_canonical_form::Unitary),
        T4A_NULL_POINTER
    );
    assert_eq!(
        t4a_treetn_ortho_center(std::ptr::null(), std::ptr::null_mut(), 0, &mut dummy_size),
        T4A_NULL_POINTER
    );
    assert_eq!(
        t4a_treetn_canonical_form(std::ptr::null(), &mut dummy_form),
        T4A_NULL_POINTER
    );

    // Operations
    assert_eq!(
        t4a_treetn_truncate(std::ptr::null_mut(), 0.0, 0.0, 0),
        T4A_NULL_POINTER
    );
    assert_eq!(
        t4a_treetn_inner(
            std::ptr::null(),
            std::ptr::null(),
            &mut dummy_double,
            &mut dummy_double
        ),
        T4A_NULL_POINTER
    );
    assert_eq!(
        t4a_treetn_norm(std::ptr::null_mut(), &mut dummy_double),
        T4A_NULL_POINTER
    );
    assert_eq!(
        t4a_treetn_lognorm(std::ptr::null_mut(), &mut dummy_double),
        T4A_NULL_POINTER
    );
    assert_eq!(
        t4a_treetn_add(std::ptr::null(), std::ptr::null(), &mut dummy_out),
        T4A_NULL_POINTER
    );
    assert_eq!(
        t4a_treetn_contract(
            std::ptr::null(),
            std::ptr::null(),
            crate::types::t4a_contract_method::Zipup,
            0.0,
            0.0,
            0,
            &mut dummy_out
        ),
        T4A_NULL_POINTER
    );
    assert_eq!(
        t4a_treetn_to_dense(std::ptr::null(), &mut dummy_tensor),
        T4A_NULL_POINTER
    );
    assert_eq!(
        t4a_treetn_linsolve(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            0.0,
            1.0,
            0,
            0.0,
            0.0,
            0,
            &mut dummy_out
        ),
        T4A_NULL_POINTER
    );
}
