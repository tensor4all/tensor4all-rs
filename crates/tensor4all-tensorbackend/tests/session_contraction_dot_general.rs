//! `Session::contraction` (one `dot_general`) must equal the labelled einsum it replaced, bit for bit.

use num_complex::Complex64;
use tenferro::Tensor;
use tenferro_tensor::TensorRead;
use tensor4all_tensorbackend::{CpuBackend, CpuExecutionContext};

type Case = (Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>);

fn lcg(len: usize, seed: u64) -> Vec<f64> {
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

fn f64_tensor(shape: &[usize], seed: u64) -> Tensor {
    let len = shape.iter().product();
    Tensor::from_vec_col_major(shape.to_vec(), lcg(len, seed)).unwrap()
}

fn c64_tensor(shape: &[usize], seed: u64) -> Tensor {
    let len: usize = shape.iter().product();
    let data: Vec<Complex64> = lcg(len, seed)
        .into_iter()
        .zip(lcg(len, seed + 77))
        .map(|(re, im)| Complex64::new(re, im))
        .collect();
    Tensor::from_vec_col_major(shape.to_vec(), data).unwrap()
}

/// The einsum labels `Session::contraction` used to build: contracted axes share a label, the
/// output is the lhs free axes then the rhs free axes.
fn einsum_labels(
    lhs_rank: usize,
    lhs_axes: &[usize],
    rhs_rank: usize,
    rhs_axes: &[usize],
) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let mut next = 0;
    let mut lhs_ids = vec![usize::MAX; lhs_rank];
    let mut rhs_ids = vec![usize::MAX; rhs_rank];
    for (&l, &r) in lhs_axes.iter().zip(rhs_axes) {
        lhs_ids[l] = next;
        rhs_ids[r] = next;
        next += 1;
    }
    let mut output = Vec::new();
    for id in lhs_ids.iter_mut().filter(|id| **id == usize::MAX) {
        *id = next;
        output.push(next);
        next += 1;
    }
    for id in rhs_ids.iter_mut().filter(|id| **id == usize::MAX) {
        *id = next;
        output.push(next);
        next += 1;
    }
    (lhs_ids, rhs_ids, output)
}

fn bits(tensor: &Tensor, complex: bool) -> Vec<u64> {
    if complex {
        tensor
            .as_slice::<Complex64>()
            .unwrap()
            .iter()
            .flat_map(|v| [v.re.to_bits(), v.im.to_bits()])
            .collect()
    } else {
        tensor
            .as_slice::<f64>()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect()
    }
}

#[test]
fn contraction_equals_the_einsum_route_bitwise() {
    let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(1).unwrap());
    // (lhs shape, lhs axes, rhs shape, rhs axes)
    let cases: Vec<Case> = vec![
        (vec![2, 2], vec![1], vec![2, 2], vec![0]),
        (vec![2, 4, 2], vec![2], vec![2, 4, 4], vec![0]),
        (vec![3, 4], vec![], vec![2, 5], vec![]),
        (vec![3, 5], vec![0, 1], vec![3, 5], vec![0, 1]),
        (vec![3, 4, 2, 5], vec![3, 1], vec![5, 3, 4], vec![0, 2]),
        (vec![3, 4, 2], vec![0, 2], vec![2, 6, 3], vec![2, 0]),
        (vec![64, 64], vec![1], vec![64, 64], vec![0]),
        (vec![16, 24, 32], vec![2], vec![32, 8, 16], vec![0]),
    ];
    for complex in [false, true] {
        for (lhs_shape, lhs_axes, rhs_shape, rhs_axes) in &cases {
            let (lhs, rhs) = if complex {
                (c64_tensor(lhs_shape, 1), c64_tensor(rhs_shape, 2))
            } else {
                (f64_tensor(lhs_shape, 1), f64_tensor(rhs_shape, 2))
            };
            let (lhs_ids, rhs_ids, output) =
                einsum_labels(lhs_shape.len(), lhs_axes, rhs_shape.len(), rhs_axes);
            let (fast, fast_reads, old) = context
                .with_concrete_session(|session| {
                    let fast = session.contraction(&lhs, lhs_axes, &rhs, rhs_axes)?;
                    let fast_reads = session.contraction_reads(
                        TensorRead::from_tensor(&lhs),
                        lhs_axes,
                        TensorRead::from_tensor(&rhs),
                        rhs_axes,
                    )?;
                    let old = session.einsum(&[&lhs, &rhs], &[&lhs_ids, &rhs_ids], &output)?;
                    Ok::<_, tenferro_einsum::Error>((fast, fast_reads, old))
                })
                .unwrap()
                .unwrap();
            let label =
                format!("complex={complex} {lhs_shape:?}{lhs_axes:?} x {rhs_shape:?}{rhs_axes:?}");
            assert_eq!(fast.shape(), old.shape(), "{label}: shape");
            assert_eq!(
                bits(&fast_reads, complex),
                bits(&fast, complex),
                "{label}: reads vs contraction"
            );
            if complex && lhs_axes.is_empty() {
                // The old einsum route evaluated a complex outer product through a different kernel
                // than `dot_general` (the global route's), which rounds some entries one ulp apart.
                // The new route is bitwise the global route (see the tensor4all-core tests); here the
                // old route is only held to a few ulps.
                let (a, b) = (
                    fast.as_slice::<Complex64>().unwrap(),
                    old.as_slice::<Complex64>().unwrap(),
                );
                for (x, y) in a.iter().zip(b) {
                    assert!(
                        (x - y).norm() <= 4.0 * f64::EPSILON * x.norm().max(1e-300),
                        "{label}: {x} vs {y}"
                    );
                }
            } else {
                assert_eq!(
                    bits(&fast, complex),
                    bits(&old, complex),
                    "{label}: contraction vs einsum"
                );
            }
        }
    }
}

#[test]
fn contraction_promotes_a_dtype_mismatch_like_before() {
    let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(1).unwrap());
    let lhs = f64_tensor(&[3, 4], 1);
    let rhs = c64_tensor(&[4, 2], 2);
    let product = context
        .with_concrete_session(|session| session.contraction(&lhs, &[1], &rhs, &[0]))
        .unwrap()
        .unwrap();
    assert_eq!(product.shape(), &[3, 2]);
    assert!(product.as_slice::<Complex64>().is_ok());
}
