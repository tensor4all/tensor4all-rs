use super::*;
use crate::index::DefaultIndex as Index;
use crate::{
    Canonical, FactorizeOptions, SingularValueMeasure, SvdTruncationPolicy, ThresholdScale,
    TruncationRule,
};
use num_complex::Complex64;
use tenferro::Tensor as NativeTensor;

#[test]
fn compute_retained_rank_handles_edge_cases() {
    let per_value = SvdTruncationPolicy::new(1.0e-6);
    assert_eq!(compute_retained_rank(&[], &per_value), 1);
    assert_eq!(compute_retained_rank(&[0.0, 0.0], &per_value), 1);
    assert_eq!(compute_retained_rank(&[5.0, 1.0e-9], &per_value), 1);
    assert_eq!(
        compute_retained_rank(&[5.0, 1.0], &SvdTruncationPolicy::new(1.0e-12)),
        2
    );
}

#[test]
fn compute_retained_rank_supports_all_policy_axes() {
    let absolute_per_value = SvdTruncationPolicy::new(1.5).with_absolute();
    assert_eq!(compute_retained_rank(&[5.0, 1.0], &absolute_per_value), 1);

    let relative_squared_tail = SvdTruncationPolicy::new(0.05)
        .with_squared_values()
        .with_discarded_tail_sum();
    assert_eq!(
        compute_retained_rank(&[10.0, 1.0, 0.1], &relative_squared_tail),
        1
    );

    let absolute_squared_tail = SvdTruncationPolicy::new(0.02)
        .with_absolute()
        .with_squared_values()
        .with_discarded_tail_sum();
    assert_eq!(
        compute_retained_rank(&[1.0, 0.1, 0.1], &absolute_squared_tail),
        2
    );
}

#[test]
fn singular_values_from_native_accepts_real_and_complex_dense() {
    let dense = NativeTensor::from_vec_col_major(vec![2], vec![3.0_f64, 1.5]).unwrap();
    assert_eq!(singular_values_from_native(&dense).unwrap(), vec![3.0, 1.5]);

    let complex = NativeTensor::from_vec_col_major(
        vec![2],
        vec![Complex64::new(1.0, 2.0), Complex64::new(0.5, -4.0)],
    )
    .unwrap();
    assert_eq!(
        singular_values_from_native(&complex).unwrap(),
        vec![1.0, 0.5]
    );
}

#[test]
fn set_default_svd_truncation_policy_rejects_invalid_values() {
    let original = default_svd_truncation_policy();
    assert!(set_default_svd_truncation_policy(SvdTruncationPolicy::new(f64::NAN)).is_err());
    assert!(set_default_svd_truncation_policy(SvdTruncationPolicy::new(-1.0)).is_err());
    assert!(set_default_svd_truncation_policy(SvdTruncationPolicy::new(f64::INFINITY)).is_err());
    set_default_svd_truncation_policy(original).unwrap();
}

#[test]
fn svd_options_accessors_roundtrip() {
    let by_policy = SvdOptions::new().with_policy(
        SvdTruncationPolicy::new(1.0e-5)
            .with_absolute()
            .with_squared_values()
            .with_discarded_tail_sum(),
    );
    assert_eq!(
        by_policy.policy,
        Some(SvdTruncationPolicy {
            threshold: 1.0e-5,
            scale: ThresholdScale::Absolute,
            measure: SingularValueMeasure::SquaredValue,
            rule: TruncationRule::DiscardedTailSum,
        })
    );
    assert_eq!(by_policy.max_bond_dim, None);

    let by_rank = SvdOptions::new().with_max_bond_dim(7);
    assert_eq!(by_rank.policy, None);
    assert_eq!(by_rank.max_bond_dim, Some(7));
}

#[test]
fn singular_values_from_native_reads_narrow_real_parts_and_rejects_integers() {
    // A narrow real payload widens per value for the rank decision only.
    let tensor = NativeTensor::from_vec_col_major(vec![2], vec![1.0_f32, 2.0]).unwrap();
    assert_eq!(
        singular_values_from_native(&tensor).unwrap(),
        vec![1.0, 2.0]
    );

    let narrow_complex = NativeTensor::from_vec_col_major(
        vec![2],
        vec![
            num_complex::Complex32::new(1.0, 5.0),
            num_complex::Complex32::new(2.0, -5.0),
        ],
    )
    .unwrap();
    assert_eq!(
        singular_values_from_native(&narrow_complex).unwrap(),
        vec![1.0, 2.0]
    );

    let integer = NativeTensor::from_vec_col_major(vec![2], vec![1_i32, 2]).unwrap();
    let err = singular_values_from_native(&integer).unwrap_err();
    assert!(err
        .to_string()
        .contains("unsupported singular-value scalar type"));
}

#[test]
fn svd_with_max_bond_dim_truncates_native_outputs() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(2);
    let tensor =
        IdxTensor::from_dense(vec![i.clone(), j.clone()], vec![3.0, 0.0, 0.0, 1.0]).unwrap();

    let (u, s, v) = svd_with::<f64>(
        &tensor,
        std::slice::from_ref(&i),
        &SvdOptions::new().with_max_bond_dim(1),
    )
    .unwrap();

    assert_eq!(u.indices.last().unwrap().dim, 1);
    assert_eq!(s.indices[0].dim, 1);
    assert_eq!(s.indices[1].dim, 1);
    assert_eq!(v.indices.last().unwrap().dim, 1);
}

#[test]
fn svd_with_invalid_rtol_is_rejected_before_linalg() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(2);
    let tensor = IdxTensor::from_dense(vec![i.clone(), j.clone()], vec![0.0; 4]).unwrap();

    let nan = svd_with::<f64>(
        &tensor,
        std::slice::from_ref(&i),
        &SvdOptions::new().with_policy(SvdTruncationPolicy::new(f64::NAN)),
    );
    assert!(matches!(nan, Err(SvdError::InvalidThreshold(v)) if v.is_nan()));

    let negative = svd_with::<f64>(
        &tensor,
        std::slice::from_ref(&i),
        &SvdOptions::new().with_policy(SvdTruncationPolicy::new(-1.0)),
    );
    assert!(matches!(negative, Err(SvdError::InvalidThreshold(v)) if v == -1.0));
}

#[test]
fn svd_with_rejects_zero_max_bond_dim_and_infinite_thresholds() {
    let i = Index::new_dyn(3);
    let j = Index::new_dyn(3);
    let tensor = IdxTensor::from_dense(
        vec![i.clone(), j.clone()],
        (0..9).map(|x| x as f64).collect(),
    )
    .unwrap();

    // `max_bond_dim == 0` is a typed validation error, never a silent clamp to
    // rank one.
    assert!(matches!(
        svd_with::<f64>(
            &tensor,
            std::slice::from_ref(&i),
            &SvdOptions::new().with_max_bond_dim(0),
        ),
        Err(SvdError::InvalidMaxBondDim)
    ));

    // Positive/negative infinity and NaN thresholds are rejected before any
    // truncation decision.
    for threshold in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
        assert!(
            matches!(
                svd_with::<f64>(
                    &tensor,
                    std::slice::from_ref(&i),
                    &SvdOptions::new().with_policy(SvdTruncationPolicy::new(threshold)),
                ),
                Err(SvdError::InvalidThreshold(_))
            ),
            "svd_with must reject threshold {threshold}"
        );
    }
}

#[test]
fn svd_with_in_matches_host_path_on_explicit_cpu_context() {
    use std::sync::Arc;

    use tensor4all_tensorbackend::CpuBackend;
    use tensor4all_tensorbackend::CpuExecutionContext;

    use crate::ExecutionContext;

    let context = ExecutionContext::Cpu(Arc::new(CpuExecutionContext::from_backend(
        CpuBackend::new(),
    )));
    let i = DynIndex::new_dyn(4);
    let j = DynIndex::new_dyn(4);
    let mut data = vec![0.0_f64; 16];
    data[0] = 1.0;
    data[5] = 0.5;
    let tensor = IdxTensor::from_dense_in(&context, vec![i.clone(), j.clone()], data).unwrap();

    let options = SvdOptions::new().with_policy(SvdTruncationPolicy::new(1e-10));
    let (u_in, s_in, _v_in) =
        svd_with_in::<f64>(&tensor, std::slice::from_ref(&i), &options, &context).unwrap();
    let (u, s, _v) = svd_with::<f64>(&tensor, std::slice::from_ref(&i), &options).unwrap();

    assert_eq!(s_in.dims(), s.dims());
    assert_eq!(u_in.dims(), u.dims());
    u_in.validate_context(&context).unwrap();
    s_in.validate_context(&context).unwrap();
}

fn complex_matrix(rows: &DynIndex, cols: &DynIndex) -> IdxTensor {
    let len = rows.dim() * cols.dim();
    IdxTensor::from_dense(
        vec![rows.clone(), cols.clone()],
        (0..len)
            .map(|i| Complex64::new(i as f64 - 2.5, 0.5 * i as f64 - 1.0))
            .collect::<Vec<_>>(),
    )
    .unwrap()
}

/// `svd_for_factorize` gives `S` the factors' dtype, so `factorize` never
/// contracts a real `S` with a complex factor: that mixed-dtype pair takes the
/// einsum path that compiles a program per call (#890).
#[test]
fn svd_for_factorize_gives_s_the_factors_dtype() {
    let (rows, cols) = (Index::new_dyn(3), Index::new_dyn(4));
    let complex = complex_matrix(&rows, &cols);
    let result =
        svd_for_factorize(&complex, std::slice::from_ref(&rows), &SvdOptions::new()).unwrap();
    assert!(result.vh.is_complex());
    assert!(result.s.is_complex());

    let real = IdxTensor::from_dense(
        vec![rows.clone(), cols],
        (0..12).map(|i| i as f64 - 5.0).collect::<Vec<_>>(),
    )
    .unwrap();
    let result = svd_for_factorize(&real, &[rows], &SvdOptions::new()).unwrap();
    assert!(!result.s.is_complex());
}

/// With the complex `S`, both canonical forms still reconstruct the input.
#[test]
fn factorize_svd_reconstructs_complex_input_in_both_canonical_forms() {
    let (rows, cols) = (Index::new_dyn(3), Index::new_dyn(4));
    let complex = complex_matrix(&rows, &cols);
    for canonical in [Canonical::Left, Canonical::Right] {
        let options = FactorizeOptions {
            canonical,
            ..FactorizeOptions::svd()
        };
        let result = crate::factorize(&complex, std::slice::from_ref(&rows), &options).unwrap();
        let reconstructed = crate::contract_pair(&result.left, &result.right).unwrap();
        let error = reconstructed.sub(&complex).unwrap().norm().unwrap();
        assert!(
            error <= 1e-12 * complex.norm().unwrap(),
            "{canonical:?}: reconstruction error {error:e}"
        );
    }
}

/// The session SVD body is also a direct entry inside the crate; it must refuse a tracked operand
/// before reading its host payload, exactly as the public factorize session entry does.
#[test]
fn svd_for_factorize_on_session_refuses_a_tracked_operand() {
    use tensor4all_tensorbackend::CpuExecutionContext;

    let context = CpuExecutionContext::from_backend(
        tensor4all_tensorbackend::CpuBackend::with_threads(1).expect("1T CPU backend"),
    );
    let (rows, cols) = (Index::new_dyn(2), Index::new_dyn(2));
    let tracked = IdxTensor::from_dense(vec![rows.clone(), cols], vec![1.0_f64, 2.0, 3.0, 4.0])
        .unwrap()
        .enable_grad()
        .expect("tracked operand");
    let error = match context.with_concrete_session(|session| {
        svd_for_factorize_on_session::<f64>(
            session,
            &tracked,
            std::slice::from_ref(&rows),
            &SvdOptions::new(),
        )
    }) {
        Ok(Err(error)) => error,
        Ok(Ok(_)) => panic!("a tracked operand must be refused"),
        Err(error) => panic!("the session entry failed: {error:?}"),
    };
    match error {
        SvdError::UnsupportedStorage(message) => {
            assert!(message.contains("tracked"), "{message}");
        }
        other => panic!("expected an unsupported-storage refusal, got {other:?}"),
    }
    assert!(
        tracked.tracks_grad(),
        "the refusal must not detach the graph"
    );
}

#[test]
fn descending_order_is_none_for_ordered_values_and_sorts_the_rest() {
    assert_eq!(descending_order(&[]), None);
    assert_eq!(descending_order(&[3.0, 3.0, 1.0, 0.0]), None);
    assert_eq!(descending_order(&[0.5, 0.0, 1.0e-3]), Some(vec![0, 2, 1]));
    // Ties keep their backend order.
    assert_eq!(
        descending_order(&[0.0, 2.0, 1.0, 2.0]),
        Some(vec![1, 3, 2, 0])
    );
}

/// The #1028 input: a 128 x 128 complex matrix with three nonzero entries, whose only nonzero
/// singular values are 0.5968 and 6.3e-4. t4a-faer 0.25.0/0.25.1 return 6.3e-4 after a run of
/// zeros (tensor4all/faer#82).
fn unsorted_spectrum_matrix() -> (DynIndex, DynIndex, IdxTensor) {
    let n = 128;
    let entries = [
        (41, 80, -1.3368711811580607e-64, 3.48452997681476e-64),
        (67, 80, -5.871970588424443e-1, -1.065196368760547e-1),
        (5, 112, -2.8153454475366503e-4, 5.652970997533713e-4),
    ];
    let mut data = vec![Complex64::new(0.0, 0.0); n * n];
    for (row, col, re, im) in entries {
        data[row + n * col] = Complex64::new(re, im);
    }
    let (rows, cols) = (Index::new_dyn(n), Index::new_dyn(n));
    let tensor = IdxTensor::from_dense(vec![rows.clone(), cols.clone()], data).unwrap();
    (rows, cols, tensor)
}

fn unsorted_spectrum_policies() -> [SvdOptions; 2] {
    [
        SvdOptions::new(),
        SvdOptions::new().with_policy(
            SvdTruncationPolicy::new(1.0e-12)
                .with_squared_values()
                .with_discarded_tail_sum(),
        ),
    ]
}

/// Regression for #1028: the truncation keeps the two largest singular values and their vectors,
/// whatever order the backend returns them in.
#[test]
fn svd_with_keeps_a_larger_value_that_follows_a_smaller_one() {
    let (rows, _, tensor) = unsorted_spectrum_matrix();
    for options in unsorted_spectrum_policies() {
        let (u, s, v) =
            svd_with::<Complex64>(&tensor, std::slice::from_ref(&rows), &options).unwrap();
        assert_eq!(s.dims()[0], 2, "{options:?}");
        let vh = v.conj().permute(&[1, 0]).unwrap();
        let back = crate::contract(&[&u, &s, &vh]).unwrap();
        let error = back.sub(&tensor).unwrap().norm().unwrap();
        assert!(
            error <= 1e-12 * tensor.norm().unwrap(),
            "{options:?}: reconstruction error {error:e}"
        );
    }
}

/// The truncation cases of the #1028 input with their correct rank. The untruncated case keeps every
/// value, so it checks that the factors are reordered even when no value is dropped.
fn unsorted_spectrum_cases() -> Vec<(SvdOptions, usize)> {
    let [default, tail] = unsorted_spectrum_policies();
    vec![(default, 2), (tail, 2), (SvdOptions::full_rank(), 128)]
}

/// `result` holds the nonincreasing singular values in its `S` factor and reconstructs `tensor`.
fn assert_sorted_factorization(result: &SvdFactorizeResult, tensor: &IdxTensor, rank: usize) {
    assert_eq!(result.rank, rank);
    let s_dense: Vec<Complex64> = result.s.to_vec().unwrap();
    let diagonal: Vec<f64> = (0..rank).map(|p| s_dense[p * (rank + 1)].re).collect();
    assert_eq!(diagonal, result.singular_values, "rank {rank}");
    assert!(
        diagonal.windows(2).all(|pair| pair[0] >= pair[1]),
        "rank {rank}: S is not nonincreasing"
    );
    // `V^H` shares the bond leg with `U`; move it onto the `sim` leg of `S` to contract.
    let vh = result
        .vh
        .replaceind(&result.bond_index, &result.s.indices[1])
        .unwrap();
    let back = crate::contract(&[&result.u, &result.s, &vh]).unwrap();
    let error = back.sub(tensor).unwrap().norm().unwrap();
    assert!(
        error <= 1e-12 * tensor.norm().unwrap(),
        "rank {rank}: reconstruction error {error:e}"
    );
}

/// Regression for #1028 on the eager factorization route.
#[test]
fn svd_for_factorize_orders_an_unsorted_spectrum() {
    let (rows, _, tensor) = unsorted_spectrum_matrix();
    for (options, rank) in unsorted_spectrum_cases() {
        let result = svd_for_factorize(&tensor, std::slice::from_ref(&rows), &options).unwrap();
        assert_sorted_factorization(&result, &tensor, rank);
    }
}

/// Regression for #1028 on the explicit-session factorization route.
#[test]
fn svd_for_factorize_on_session_orders_an_unsorted_spectrum() {
    use tensor4all_tensorbackend::CpuExecutionContext;

    let context = CpuExecutionContext::from_backend(
        tensor4all_tensorbackend::CpuBackend::with_threads(1).expect("1T CPU backend"),
    );
    let (rows, _, tensor) = unsorted_spectrum_matrix();
    for (options, rank) in unsorted_spectrum_cases() {
        let result = context
            .with_concrete_session(|session| {
                svd_for_factorize_on_session::<Complex64>(
                    session,
                    &tensor,
                    std::slice::from_ref(&rows),
                    &options,
                )
            })
            .unwrap()
            .unwrap();
        assert_sorted_factorization(&result, &tensor, rank);
    }
}
