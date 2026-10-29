//! The held-session contraction runs one `dot_general` instead of planning an einsum, and passes
//! Eager operands as borrowed reads. The result must stay bit-for-bit the global route's result.

use std::sync::Arc;

use num_complex::Complex64;

use tensor4all_core::index::{DefaultIndex as Index, DynId};
use tensor4all_core::{
    contract_pair, contract_pair_on_session, ExecutionContext, IdxTensor,
    PairwiseContractionOptions,
};
use tensor4all_tensorbackend::{CpuBackend, CpuExecutionContext, Storage};

/// Deterministic pseudo-random values in [-1, 1).
fn values(len: usize, seed: u64) -> Vec<f64> {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
        })
        .collect()
}

fn complex_values(len: usize, seed: u64) -> Vec<Complex64> {
    let re = values(len, seed);
    let im = values(len, seed + 1000);
    re.into_iter()
        .zip(im)
        .map(|(r, i)| Complex64::new(r, i))
        .collect()
}

fn bits_f64(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn bits_c64(values: &[Complex64]) -> Vec<(u64, u64)> {
    values
        .iter()
        .map(|v| (v.re.to_bits(), v.im.to_bits()))
        .collect()
}

#[derive(Clone, Copy)]
enum Repr {
    Eager,
    Materialized,
}

#[derive(Clone, Copy)]
enum Scalar {
    F64,
    C64,
}

/// Build one operand in the requested representation and dtype.
fn operand(
    handle: &ExecutionContext,
    indices: &[Index<DynId>],
    scalar: Scalar,
    repr: Repr,
    seed: u64,
) -> IdxTensor {
    let dims: Vec<usize> = indices.iter().map(|i| i.dim).collect();
    let len: usize = dims.iter().product();
    match (scalar, repr) {
        (Scalar::F64, Repr::Eager) => {
            IdxTensor::from_dense_in(handle, indices.to_vec(), values(len, seed)).unwrap()
        }
        (Scalar::C64, Repr::Eager) => {
            IdxTensor::from_dense_in(handle, indices.to_vec(), complex_values(len, seed)).unwrap()
        }
        (Scalar::F64, Repr::Materialized) => {
            let storage = Storage::from_dense_col_major(values(len, seed), &dims).unwrap();
            IdxTensor::from_storage(indices.to_vec(), Arc::new(storage)).unwrap()
        }
        (Scalar::C64, Repr::Materialized) => {
            let storage = Storage::from_dense_col_major(complex_values(len, seed), &dims).unwrap();
            IdxTensor::from_storage(indices.to_vec(), Arc::new(storage)).unwrap()
        }
    }
}

/// The same operand data on the process-global context, for the global-route oracle.
fn global_operand(indices: &[Index<DynId>], scalar: Scalar, seed: u64) -> IdxTensor {
    let len: usize = indices.iter().map(|i| i.dim).product();
    match scalar {
        Scalar::F64 => IdxTensor::from_dense(indices.to_vec(), values(len, seed)).unwrap(),
        Scalar::C64 => IdxTensor::from_dense(indices.to_vec(), complex_values(len, seed)).unwrap(),
    }
}

fn assert_bitwise(label: &str, scalar: Scalar, actual: &IdxTensor, expected: &IdxTensor) {
    assert_eq!(actual.indices(), expected.indices(), "{label}: indices");
    match scalar {
        Scalar::F64 => assert_eq!(
            bits_f64(&actual.to_vec::<f64>().unwrap()),
            bits_f64(&expected.to_vec::<f64>().unwrap()),
            "{label}: f64 bits"
        ),
        Scalar::C64 => assert_eq!(
            bits_c64(&actual.to_vec::<Complex64>().unwrap()),
            bits_c64(&expected.to_vec::<Complex64>().unwrap()),
            "{label}: c64 bits"
        ),
    }
}

/// Run every dtype/representation combination of one index layout against the global route.
fn check_layout(name: &str, lhs_indices: &[Index<DynId>], rhs_indices: &[Index<DynId>]) {
    let context = Arc::new(CpuExecutionContext::from_backend(
        CpuBackend::with_threads(1).unwrap(),
    ));
    let handle = ExecutionContext::Cpu(Arc::clone(&context));
    for scalar in [Scalar::F64, Scalar::C64] {
        for (lhs_repr, rhs_repr, repr_name) in [
            (Repr::Eager, Repr::Eager, "eager/eager"),
            (
                Repr::Materialized,
                Repr::Materialized,
                "materialized/materialized",
            ),
            (Repr::Eager, Repr::Materialized, "eager/materialized"),
            (Repr::Materialized, Repr::Eager, "materialized/eager"),
        ] {
            let lhs = operand(&handle, lhs_indices, scalar, lhs_repr, 1);
            let rhs = operand(&handle, rhs_indices, scalar, rhs_repr, 2);
            // The oracle: the same data as global-context operands, through the global route.
            let expected = contract_pair(
                &global_operand(lhs_indices, scalar, 1),
                &global_operand(rhs_indices, scalar, 2),
            )
            .unwrap();
            // Two calls on one held session: the second one hits every plan cache.
            for call in 0..2 {
                let actual = context
                    .with_concrete_session(|session| {
                        contract_pair_on_session(
                            session,
                            &lhs,
                            &rhs,
                            PairwiseContractionOptions::new(),
                        )
                    })
                    .unwrap()
                    .unwrap();
                let label = format!("{name} {repr_name} call {call}");
                assert_bitwise(&label, scalar, &actual, &expected);
            }
        }
    }
}

#[test]
fn rank2_matmul_matches_the_global_route_bitwise() {
    let (i, k, j) = (Index::new_dyn(2), Index::new_dyn(2), Index::new_dyn(2));
    check_layout("2x2", &[i, k.clone()], &[k, j]);
}

#[test]
fn three_leg_single_contracted_axis_matches_the_global_route_bitwise() {
    // A[2,4,2] . B[2,4,4] over one leg (abc,cef->abef shape): lhs axis 2 with rhs axis 0.
    let (a, b, c) = (Index::new_dyn(2), Index::new_dyn(4), Index::new_dyn(2));
    let (e, f) = (Index::new_dyn(4), Index::new_dyn(4));
    check_layout("3-leg", &[a, b, c.clone()], &[c, e, f]);
}

#[test]
fn outer_product_matches_the_global_route_bitwise() {
    let (i, j, k) = (Index::new_dyn(3), Index::new_dyn(4), Index::new_dyn(2));
    check_layout("outer", &[i, k], &[j]);
}

#[test]
fn full_contraction_to_scalar_matches_the_global_route_bitwise() {
    let (i, j) = (Index::new_dyn(3), Index::new_dyn(5));
    check_layout("scalar", &[i.clone(), j.clone()], &[i, j]);
}

#[test]
fn permuted_and_non_leading_contracted_axes_match_the_global_route_bitwise() {
    // The shared indices sit in the middle / trailing positions and in different orders.
    let (a, b, c, d) = (
        Index::new_dyn(3),
        Index::new_dyn(4),
        Index::new_dyn(2),
        Index::new_dyn(5),
    );
    check_layout(
        "permuted",
        &[a.clone(), b.clone(), c.clone(), d.clone()],
        &[d.clone(), Index::new_dyn(3), b.clone()],
    );
    check_layout(
        "permuted-two-shared",
        &[a.clone(), b.clone(), c.clone()],
        &[c, a, d],
    );
}

#[test]
fn larger_sizes_match_the_global_route_bitwise() {
    let (i, k, j) = (Index::new_dyn(64), Index::new_dyn(64), Index::new_dyn(64));
    check_layout("64x64", &[i, k.clone()], &[k, j]);
    let (a, b, c) = (Index::new_dyn(16), Index::new_dyn(24), Index::new_dyn(32));
    let (e, f) = (Index::new_dyn(8), Index::new_dyn(16));
    check_layout("3-leg large", &[a, b, c.clone()], &[c, e, f]);
}

#[test]
fn conjugated_and_mixed_dtype_operands_still_match_the_global_route() {
    let context = Arc::new(CpuExecutionContext::from_backend(
        CpuBackend::with_threads(1).unwrap(),
    ));
    let handle = ExecutionContext::Cpu(Arc::clone(&context));
    let (i, k, j) = (Index::new_dyn(4), Index::new_dyn(3), Index::new_dyn(5));
    let lhs_c = operand(
        &handle,
        &[i.clone(), k.clone()],
        Scalar::C64,
        Repr::Eager,
        1,
    );
    let rhs_c = operand(
        &handle,
        &[k.clone(), j.clone()],
        Scalar::C64,
        Repr::Eager,
        2,
    );
    let rhs_f = operand(
        &handle,
        &[k.clone(), j.clone()],
        Scalar::F64,
        Repr::Eager,
        3,
    );
    for (label, lhs, rhs, options) in [
        (
            "lhs_conj",
            &lhs_c,
            &rhs_c,
            PairwiseContractionOptions::new().with_lhs_conj(true),
        ),
        (
            "rhs_conj",
            &lhs_c,
            &rhs_c,
            PairwiseContractionOptions::new().with_rhs_conj(true),
        ),
        ("mixed", &lhs_c, &rhs_f, PairwiseContractionOptions::new()),
    ] {
        let expected = {
            let l = if options.lhs_conj {
                lhs.conj()
            } else {
                lhs.clone()
            };
            let r = if options.rhs_conj {
                rhs.conj()
            } else {
                rhs.clone()
            };
            contract_pair(&l, &r).unwrap()
        };
        let actual = context
            .with_concrete_session(|session| contract_pair_on_session(session, lhs, rhs, options))
            .unwrap()
            .unwrap();
        assert_bitwise(label, Scalar::C64, &actual, &expected);
    }
}
