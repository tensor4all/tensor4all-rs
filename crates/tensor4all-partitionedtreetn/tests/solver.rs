use std::collections::HashMap;

use num_complex::Complex64;
use tensor4all_core::{AnyScalar, DynIndex, IdxTensor, TensorVectorSpace};
use tensor4all_partitionedtreetn::{
    partitioned_linsolve, LinsolveRoute, PartitionedTreeTN, Projector, SubDomainTreeTN,
};
use tensor4all_treetn::{IndexMapping, LinsolveOptions, TreeTN, TruncationOptions};

/// Build a 2-site complex TreeTN on sites `(s0, s1)` with bond dimension 1
/// from a column-major dense array of length `s0.dim * s1.dim` where only
/// coordinate `p` on `s0` is nonzero (`vals` has length `s1.dim`).
fn make_state_patch(
    s0: &DynIndex,
    s1: &DynIndex,
    p: usize,
    vals: &[Complex64],
) -> SubDomainTreeTN<usize> {
    let mut v0 = vec![Complex64::new(0.0, 0.0); s0.dim];
    v0[p] = Complex64::new(1.0, 0.0);
    let bond = DynIndex::new_dyn(1);
    let t0 = IdxTensor::from_dense(vec![s0.clone(), bond.clone()], v0).unwrap();
    let t1 = IdxTensor::from_dense(vec![bond.clone(), s1.clone()], vals.to_vec()).unwrap();
    let mut tree = TreeTN::<IdxTensor, usize>::new();
    let n0 = tree.add_tensor(0, t0).unwrap();
    let n1 = tree.add_tensor(1, t1).unwrap();
    tree.connect(n0, &bond, n1, &bond).unwrap();
    let proj = Projector::from_pairs([(s0.clone(), p)]).unwrap();
    SubDomainTreeTN::new(tree, proj).unwrap()
}

/// Build a 2-site complex MPO block `A_{p_out, p_in}` acting on `(s0, s1)`
/// with a `2x2` complex matrix `m1` (column-major: `[m00, m10, m01, m11]`) on `s1`.
fn make_op_block(
    s0_out: &DynIndex,
    s0_in: &DynIndex,
    s1_out: &DynIndex,
    s1_in: &DynIndex,
    p_out: usize,
    p_in: usize,
    m1: &[Complex64; 4],
) -> SubDomainTreeTN<usize> {
    let mut m0 = vec![Complex64::new(0.0, 0.0); s0_out.dim * s0_in.dim];
    m0[p_out + s0_out.dim * p_in] = Complex64::new(1.0, 0.0);
    let bond = DynIndex::new_dyn(1);
    let t0 = IdxTensor::from_dense(vec![s0_out.clone(), s0_in.clone(), bond.clone()], m0).unwrap();
    let t1 = IdxTensor::from_dense(
        vec![bond.clone(), s1_out.clone(), s1_in.clone()],
        m1.to_vec(),
    )
    .unwrap();
    let mut tree = TreeTN::<IdxTensor, usize>::new();
    let n0 = tree.add_tensor(0, t0).unwrap();
    let n1 = tree.add_tensor(1, t1).unwrap();
    tree.connect(n0, &bond, n1, &bond).unwrap();
    let proj = Projector::from_pairs([(s0_out.clone(), p_out), (s0_in.clone(), p_in)]).unwrap();
    SubDomainTreeTN::new(tree, proj).unwrap()
}

fn mat_vec2(m: &[Complex64; 4], v: &[Complex64; 2]) -> [Complex64; 2] {
    [m[0] * v[0] + m[2] * v[1], m[1] * v[0] + m[3] * v[1]]
}

#[test]
fn routes_and_global_residual() {
    let s0 = DynIndex::new_dyn(3);
    let s1 = DynIndex::new_dyn(2);
    let s0_in = DynIndex::new_dyn(3);
    let s0_out = DynIndex::new_dyn(3);
    let s1_in = DynIndex::new_dyn(2);
    let s1_out = DynIndex::new_dyn(2);

    let p0 = Projector::from_pairs([(s0.clone(), 0)]).unwrap();
    let p1 = Projector::from_pairs([(s0.clone(), 1)]).unwrap();
    let p2 = Projector::from_pairs([(s0.clone(), 2)]).unwrap();

    // Manufactured complex solution on patches 0 and 1 (patch 2 is unsupported).
    let x0 = [Complex64::new(1.2, -0.4), Complex64::new(-0.3, 0.8)];
    let x1 = [Complex64::new(-0.5, 0.6), Complex64::new(0.9, 0.2)];
    let x_exact = PartitionedTreeTN::from_subdomains(vec![
        make_state_patch(&s0, &s1, 0, &x0),
        make_state_patch(&s0, &s1, 1, &x1),
    ])
    .unwrap();

    // 1. TensorVectorSpace exact inner product and axpby_truncated.
    let y0 = [Complex64::new(0.4, 0.5), Complex64::new(0.7, -0.1)];
    let y_part =
        PartitionedTreeTN::from_subdomains(vec![make_state_patch(&s0, &s1, 0, &y0)]).unwrap();
    let expected_inner = x0[0].conj() * y0[0] + x0[1].conj() * y0[1];
    let got_inner = x_exact.inner_product(&y_part).unwrap();
    assert!((got_inner.real() - expected_inner.re).abs() < 1e-12);
    assert!((got_inner.imag() - expected_inner.im).abs() < 1e-12);

    let a_coef = AnyScalar::new_complex(2.0, -0.5);
    let b_coef = AnyScalar::new_complex(-1.0, 0.25);
    let comb = x_exact
        .axpby_truncated(
            a_coef,
            &y_part,
            b_coef,
            &0usize,
            TruncationOptions::default(),
        )
        .unwrap();
    let expected_p0 = [
        Complex64::new(2.0, -0.5) * x0[0] + Complex64::new(-1.0, 0.25) * y0[0],
        Complex64::new(2.0, -0.5) * x0[1] + Complex64::new(-1.0, 0.25) * y0[1],
    ];
    let expected_p1 = [
        Complex64::new(2.0, -0.5) * x1[0],
        Complex64::new(2.0, -0.5) * x1[1],
    ];
    let expected_comb = PartitionedTreeTN::from_subdomains(vec![
        make_state_patch(&s0, &s1, 0, &expected_p0),
        make_state_patch(&s0, &s1, 1, &expected_p1),
    ])
    .unwrap();
    let comb_diff = comb
        .axpby(
            AnyScalar::new_real(1.0),
            &expected_comb,
            AnyScalar::new_real(-1.0),
        )
        .unwrap();
    assert!(comb_diff.norm().unwrap() < 1e-12);

    // Operator blocks on s1 (small norms so I - A is well-conditioned).
    let m00 = [
        Complex64::new(0.2, 0.05),
        Complex64::new(-0.04, 0.06),
        Complex64::new(0.03, -0.02),
        Complex64::new(0.15, -0.05),
    ];
    let m11 = [
        Complex64::new(0.18, -0.03),
        Complex64::new(0.05, 0.02),
        Complex64::new(-0.02, 0.04),
        Complex64::new(0.22, 0.01),
    ];
    let m01 = [
        Complex64::new(0.12, -0.04),
        Complex64::new(0.03, 0.05),
        Complex64::new(-0.04, 0.01),
        Complex64::new(0.09, 0.02),
    ];
    let m10 = [
        Complex64::new(-0.08, 0.03),
        Complex64::new(0.04, -0.02),
        Complex64::new(0.05, 0.02),
        Complex64::new(-0.11, -0.03),
    ];

    let mut input_mapping = HashMap::new();
    input_mapping.insert(
        0usize,
        IndexMapping {
            true_index: s0.clone(),
            internal_index: s0_in.clone(),
        },
    );
    input_mapping.insert(
        1usize,
        IndexMapping {
            true_index: s1.clone(),
            internal_index: s1_in.clone(),
        },
    );
    let mut output_mapping = HashMap::new();
    output_mapping.insert(
        0usize,
        IndexMapping {
            true_index: s0.clone(),
            internal_index: s0_out.clone(),
        },
    );
    output_mapping.insert(
        1usize,
        IndexMapping {
            true_index: s1.clone(),
            internal_index: s1_out.clone(),
        },
    );

    // Initial guess includes an unsupported patch p2 (no operator block, no rhs block).
    let init = PartitionedTreeTN::from_subdomains(vec![
        make_state_patch(&s0, &s1, 0, &[Complex64::new(0.0, 0.0); 2]),
        make_state_patch(&s0, &s1, 1, &[Complex64::new(0.0, 0.0); 2]),
        make_state_patch(
            &s0,
            &s1,
            2,
            &[Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
        ),
    ])
    .unwrap();

    let tolerance = 1e-8;
    let options = LinsolveOptions::default()
        .with_coefficients(AnyScalar::new_real(1.0), AnyScalar::new_real(-1.0))
        .with_nfullsweeps(6)
        .with_gmres_tol(tolerance)
        .with_gmres_restart_dim(15)
        .with_gmres_max_restarts(20)
        .with_convergence_tol(tolerance);

    // 2. Decoupled operator -> LinsolveRoute::PatchLocal.
    let op_diag = PartitionedTreeTN::from_subdomains(vec![
        make_op_block(&s0_out, &s0_in, &s1_out, &s1_in, 0, 0, &m00),
        make_op_block(&s0_out, &s0_in, &s1_out, &s1_in, 1, 1, &m11),
    ])
    .unwrap();
    let ax0_diag = mat_vec2(&m00, &x0);
    let ax1_diag = mat_vec2(&m11, &x1);
    let b_diag = PartitionedTreeTN::from_subdomains(vec![
        make_state_patch(&s0, &s1, 0, &[x0[0] - ax0_diag[0], x0[1] - ax0_diag[1]]),
        make_state_patch(&s0, &s1, 1, &[x1[0] - ax1_diag[0], x1[1] - ax1_diag[1]]),
    ])
    .unwrap();

    let local_res = partitioned_linsolve(
        &op_diag,
        &b_diag,
        init.clone(),
        &0usize,
        options.clone(),
        Some(input_mapping.clone()),
        Some(output_mapping.clone()),
    )
    .unwrap();
    assert_eq!(local_res.route, LinsolveRoute::PatchLocal);
    assert_eq!(local_res.krylov_iterations, 0);
    assert!(local_res.converged);
    assert!(local_res.residual <= 10.0 * tolerance);
    assert!(local_res.solution.contains(&p0));
    assert!(local_res.solution.contains(&p1));
    assert!(!local_res.solution.contains(&p2));
    let local_err = local_res
        .solution
        .axpby(
            AnyScalar::new_real(1.0),
            &x_exact,
            AnyScalar::new_real(-1.0),
        )
        .unwrap()
        .norm()
        .unwrap();
    assert!(local_err <= 10.0 * tolerance);

    // 3. Coupled operator -> LinsolveRoute::GlobalKrylov with whole-system residual.
    let op_coupled = PartitionedTreeTN::from_subdomains(vec![
        make_op_block(&s0_out, &s0_in, &s1_out, &s1_in, 0, 0, &m00),
        make_op_block(&s0_out, &s0_in, &s1_out, &s1_in, 1, 1, &m11),
        make_op_block(&s0_out, &s0_in, &s1_out, &s1_in, 0, 1, &m01),
        make_op_block(&s0_out, &s0_in, &s1_out, &s1_in, 1, 0, &m10),
    ])
    .unwrap();
    let ax0_coup = [
        mat_vec2(&m00, &x0)[0] + mat_vec2(&m01, &x1)[0],
        mat_vec2(&m00, &x0)[1] + mat_vec2(&m01, &x1)[1],
    ];
    let ax1_coup = [
        mat_vec2(&m10, &x0)[0] + mat_vec2(&m11, &x1)[0],
        mat_vec2(&m10, &x0)[1] + mat_vec2(&m11, &x1)[1],
    ];
    let b_coupled = PartitionedTreeTN::from_subdomains(vec![
        make_state_patch(&s0, &s1, 0, &[x0[0] - ax0_coup[0], x0[1] - ax0_coup[1]]),
        make_state_patch(&s0, &s1, 1, &[x1[0] - ax1_coup[0], x1[1] - ax1_coup[1]]),
    ])
    .unwrap();

    let coupled_res = partitioned_linsolve(
        &op_coupled,
        &b_coupled,
        init,
        &0usize,
        options,
        Some(input_mapping),
        Some(output_mapping),
    )
    .unwrap();
    assert_eq!(coupled_res.route, LinsolveRoute::GlobalKrylov);
    assert!(coupled_res.krylov_iterations > 0);
    assert!(coupled_res.converged);
    assert!(coupled_res.residual <= 10.0 * tolerance);
    assert!(coupled_res.solution.contains(&p0));
    assert!(coupled_res.solution.contains(&p1));
    assert!(!coupled_res.solution.contains(&p2));
    let coupled_err = coupled_res
        .solution
        .axpby(
            AnyScalar::new_real(1.0),
            &x_exact,
            AnyScalar::new_real(-1.0),
        )
        .unwrap()
        .norm()
        .unwrap();
    assert!(coupled_err <= 10.0 * tolerance);
}
