//! Which thread-count determinism the dense products deliver, per scalar type
//! (#1192).
//!
//! `docs/design/tree-patching-error-contract.md` promises one thing about
//! numerics, **Tier 1**: bitwise identical on the same machine and build at a
//! **fixed thread count**, across fresh threads, processes and repeated runs.
//! Bitwise identity **across** thread counts is not promised for any operation
//! (owner decision after #1202), so nothing here fails when a thread count moves
//! the bits.
//!
//! `mat_mul` and `IdxTensor::contract_pair` are bitwise identical across thread
//! counts for `f64` and are not for `Complex64`: one worker and a team take
//! different kernels (a fused complex gemm against the packed driver's real-`f64`
//! planar kernels), so their rounding differs. Both are observations under Tier 1,
//! recorded by the second test instead of asserted, so it passes wherever another
//! host's split starts.
//!
//! The thread count comes from an explicit `CpuBackend::with_threads`, never from
//! `RAYON_NUM_THREADS`, so the environment cannot change what a cell means. A
//! provider that schedules its own threads can still ignore that width; the
//! recorded cells come from the platform's faer path, where it does not.
//! `contract_pair` runs on the engine its operands' eager runtime carries, and
//! that table holds weak handles, so the context is kept alive across the call:
//! a dropped context falls back to the process-global engine.
//!
//! Run with `--nocapture` to see the record.

use std::sync::Arc;

use num_complex::Complex64;
use tensor4all_core::{DynIndex, ExecutionContext, IdxTensor, TensorContractionLike};
use tensor4all_tensorbackend::{BlasMul, CpuBackend, CpuExecutionContext, Matrix};

const THREAD_COUNTS: [usize; 3] = [1, 2, 8];

/// The shapes #1192 reported the divergence at: a cube, a tall-skinny pair, two
/// uneven pairs, and two large cubes.
const SHAPES: [(usize, usize, usize); 6] = [
    (64, 64, 64),
    (58, 58, 128),
    (145, 145, 128),
    (165, 151, 128),
    (256, 256, 256),
    (500, 500, 500),
];

/// The Tier 1 half runs every cell twice per thread count, so it stays on the
/// cheapest shapes and leaves the reported ones to the recorded half.
const REPEATED_SHAPES: [(usize, usize, usize); 3] = [(64, 64, 64), (58, 58, 128), (256, 256, 256)];

#[derive(Clone, Copy)]
enum Op {
    MatMul,
    ContractPair,
}

impl Op {
    fn name(self) -> &'static str {
        match self {
            Op::MatMul => "mat_mul",
            Op::ContractPair => "contract_pair",
        }
    }
}

#[derive(Clone, Copy)]
enum Scalar {
    F64,
    C64,
}

impl Scalar {
    fn name(self) -> &'static str {
        match self {
            Scalar::F64 => "f64",
            Scalar::C64 => "c64",
        }
    }
}

/// A backend of exactly `threads`, so the cell means one thread count only.
fn context(threads: usize) -> Arc<CpuExecutionContext> {
    let backend = CpuBackend::with_threads(threads).expect("CPU backend");
    // A pool that cannot reserve its workers within the provider's retry budget
    // runs serially instead, which would silently turn this cell into a different
    // worker count: the digest would then compare two widths and could fail on a
    // busy host. Assert the effective width the provider documents for exactly
    // this case instead of trusting the request.
    assert_eq!(
        (backend.engine().has_team(), backend.engine().num_threads()),
        (threads > 1, threads),
        "the {threads}-worker backend did not take effect; a fallback to serial \
         would change what this cell measures"
    );
    Arc::new(CpuExecutionContext::from_backend(backend))
}

/// Deterministic complex input in `[-0.5, 0.5)`, column-major.
fn complex_input(len: usize, seed: u64) -> Vec<Complex64> {
    let mut state = seed | 1;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    (0..len).map(|_| Complex64::new(next(), next())).collect()
}

/// The real part of the same stream, so a real cell and a complex cell see the
/// same numbers and their reference accumulations stay comparable.
fn real_input(len: usize, seed: u64) -> Vec<f64> {
    complex_input(len, seed).into_iter().map(|v| v.re).collect()
}

/// One `mat_mul` on the cell's own engine. The dtype-specific part is only the
/// operand payload and the read-back.
fn mat_mul_payload<T: BlasMul>(
    threads: usize,
    m: usize,
    k: usize,
    n: usize,
    left: Vec<T>,
    right: Vec<T>,
) -> Matrix<T> {
    let context = context(threads);
    let a = Matrix::from_col_major_vec(m, k, left);
    let b = Matrix::from_col_major_vec(k, n, right);
    let c = context
        .with_concrete_session(|session| session.mat_mul(&a, &b))
        .expect("session")
        .expect("mat_mul");
    assert_eq!(
        (c.nrows(), c.ncols()),
        (m, n),
        "mat_mul returned the wrong shape"
    );
    c
}

fn mat_mul_f64(threads: usize, m: usize, k: usize, n: usize) -> Vec<Complex64> {
    mat_mul_payload(
        threads,
        m,
        k,
        n,
        real_input(m * k, 7),
        real_input(k * n, 11),
    )
    .as_col_major_slice()
    .iter()
    .map(|v| Complex64::from(*v))
    .collect()
}

fn mat_mul_c64(threads: usize, m: usize, k: usize, n: usize) -> Vec<Complex64> {
    mat_mul_payload(
        threads,
        m,
        k,
        n,
        complex_input(m * k, 7),
        complex_input(k * n, 11),
    )
    .as_col_major_slice()
    .to_vec()
}

/// `[m, k]` and `[k, n]` over distinct indices, built in the cell's own context so
/// the contraction runs on that engine rather than the process-global one.
fn operand_pair<T: tensor4all_core::TensorElement>(
    context: &Arc<CpuExecutionContext>,
    m: usize,
    k: usize,
    n: usize,
    left: Vec<T>,
    right: Vec<T>,
) -> (IdxTensor, IdxTensor) {
    let execution = ExecutionContext::Cpu(context.clone());
    let (i, j, l) = (
        DynIndex::new_dyn(m),
        DynIndex::new_dyn(k),
        DynIndex::new_dyn(n),
    );
    let a = IdxTensor::from_dense_in(&execution, vec![i, j.clone()], left).expect("left operand");
    let b = IdxTensor::from_dense_in(&execution, vec![j, l], right).expect("right operand");
    (a, b)
}

/// One `contract_pair` on the cell's own engine.
fn contract_product<T: tensor4all_core::TensorElement>(
    threads: usize,
    m: usize,
    k: usize,
    n: usize,
    left: Vec<T>,
    right: Vec<T>,
) -> IdxTensor {
    // The context has to outlive the contraction: the operands' eager runtime is
    // bound to it weakly, so a dropped context falls back to the process-global
    // engine and the cell stops meaning one thread count.
    let context = context(threads);
    let (a, b) = operand_pair(&context, m, k, n, left, right);
    let product = a.contract_pair(&b).expect("contract_pair");
    // The index shape is part of the product, not just the payload: a result
    // carrying swapped axes and the same flat values must fail here.
    assert_eq!(
        product.dims(),
        [m, n],
        "contract_pair returned the wrong index shape"
    );
    product
}

fn contract_f64(threads: usize, m: usize, k: usize, n: usize) -> Vec<Complex64> {
    contract_product(
        threads,
        m,
        k,
        n,
        real_input(m * k, 7),
        real_input(k * n, 11),
    )
    .to_vec::<f64>()
    .expect("host payload")
    .into_iter()
    .map(Complex64::from)
    .collect()
}

fn contract_c64(threads: usize, m: usize, k: usize, n: usize) -> Vec<Complex64> {
    contract_product(
        threads,
        m,
        k,
        n,
        complex_input(m * k, 7),
        complex_input(k * n, 11),
    )
    .to_vec::<Complex64>()
    .expect("host payload")
}

fn product(op: Op, scalar: Scalar, threads: usize, m: usize, k: usize, n: usize) -> Vec<Complex64> {
    match (op, scalar) {
        (Op::MatMul, Scalar::F64) => mat_mul_f64(threads, m, k, n),
        (Op::MatMul, Scalar::C64) => mat_mul_c64(threads, m, k, n),
        (Op::ContractPair, Scalar::F64) => contract_f64(threads, m, k, n),
        (Op::ContractPair, Scalar::C64) => contract_c64(threads, m, k, n),
    }
}

/// The input pair of one cell, widened to complex so a real and a complex cell
/// share the reference below.
fn inputs(scalar: Scalar, m: usize, k: usize, n: usize) -> (Vec<Complex64>, Vec<Complex64>) {
    match scalar {
        Scalar::F64 => (
            real_input(m * k, 7)
                .into_iter()
                .map(Complex64::from)
                .collect(),
            real_input(k * n, 11)
                .into_iter()
                .map(Complex64::from)
                .collect(),
        ),
        Scalar::C64 => (complex_input(m * k, 7), complex_input(k * n, 11)),
    }
}

/// One output entry, accumulated sequentially over the whole contracted extent in
/// the input's own order: the plainest possible reference. Sampled entries keep
/// the check a value check without an `m*n*k` pass per cell.
fn reference_entry(
    (a, b): (&[Complex64], &[Complex64]),
    (m, k): (usize, usize),
    (row, column): (usize, usize),
) -> Complex64 {
    let mut sum = Complex64::new(0.0, 0.0);
    for inner in 0..k {
        sum += a[row + inner * m] * b[inner + column * k];
    }
    sum
}

/// FNV-1a, byte by byte, over every payload value's bits in flat order.
///
/// The per-byte step is what makes a sign flip visible: xor-then-multiply once per
/// 64-bit word cancels an even number of sign-bit flips exactly.
fn digest(values: &[Complex64]) -> u64 {
    values
        .iter()
        .flat_map(|v| [v.re.to_bits(), v.im.to_bits()])
        .flat_map(u64::to_le_bytes)
        .fold(0xcbf29ce484222325, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
        })
}

/// `(first differing element index, largest absolute difference)` over two equal
/// length payloads; `None` when they are bitwise identical.
fn difference(a: &[Complex64], b: &[Complex64]) -> Option<(usize, f64)> {
    assert_eq!(
        a.len(),
        b.len(),
        "different payload lengths ({}/{}): a shortened result must fail here, not compare short",
        a.len(),
        b.len()
    );
    let mut first = None;
    let mut largest = 0.0_f64;
    for (index, (x, y)) in a.iter().zip(b).enumerate() {
        if x.re.to_bits() != y.re.to_bits() || x.im.to_bits() != y.im.to_bits() {
            first.get_or_insert(index);
            largest = largest.max((x - y).norm());
        }
    }
    first.map(|index| (index, largest))
}

fn label(op: Op, scalar: Scalar, m: usize, k: usize, n: usize, threads: usize) -> String {
    format!("{} {} {m}x{k}x{n} @ {threads}T", op.name(), scalar.name())
}

/// Tier 1: a fixed thread count repeats bitwise, so the digest is a legal key.
///
/// Two calls with fresh pools, fresh indices and fresh payloads must agree
/// exactly, at every thread count. This is within-process repeatability at a fixed
/// thread count; fresh caller threads and a second process are not covered here,
/// so no cross-process claim is made.
#[test]
fn a_fixed_thread_count_repeats_the_product_bits() {
    for op in [Op::MatMul, Op::ContractPair] {
        for scalar in [Scalar::F64, Scalar::C64] {
            for (m, k, n) in REPEATED_SHAPES {
                for threads in THREAD_COUNTS {
                    let first = product(op, scalar, threads, m, k, n);
                    let second = product(op, scalar, threads, m, k, n);
                    assert_eq!(
                        digest(&first),
                        digest(&second),
                        "{} changed between two runs at the same thread count",
                        label(op, scalar, m, k, n, threads)
                    );
                }
            }
        }
    }
}

/// Recorded rather than promised: today the `f64` digests agree across
/// thread counts and the `Complex64` ones need not.
///
/// The direction is not asserted, because neither direction is a promise on
/// another host. What is asserted holds whichever way it goes: the shape does not
/// move with the thread count, every payload is finite, and four sampled entries
/// of each thread count's product still match a plain sequential accumulation to
/// round-off.
#[test]
fn thread_count_may_change_the_complex_product_bits_today() {
    for op in [Op::MatMul, Op::ContractPair] {
        for (m, k, n) in SHAPES {
            for scalar in [Scalar::F64, Scalar::C64] {
                let reference_inputs = inputs(scalar, m, k, n);
                let samples = [(0, 0), (m - 1, n - 1), (m / 2, n / 3), (m / 3, n / 2)];
                let runs: Vec<Vec<Complex64>> = THREAD_COUNTS
                    .iter()
                    .map(|&threads| product(op, scalar, threads, m, k, n))
                    .collect();
                for (&threads, payload) in THREAD_COUNTS.iter().zip(&runs) {
                    assert_eq!(
                        payload.len(),
                        m * n,
                        "{}: the product changed shape with the thread count",
                        label(op, scalar, m, k, n, threads)
                    );
                    assert!(
                        payload.iter().all(|v| v.re.is_finite() && v.im.is_finite()),
                        "{}: the product is not finite",
                        label(op, scalar, m, k, n, threads)
                    );
                    for (row, column) in samples {
                        let expected = reference_entry(
                            (&reference_inputs.0, &reference_inputs.1),
                            (m, k),
                            (row, column),
                        );
                        let got = payload[row + column * m];
                        // The allowance comes from the reference, not from the
                        // payload under test: a corrupted output must not raise the
                        // tolerance it is checked against.
                        assert!(
                            (got - expected).norm() <= 1e-12 * expected.norm().max(1.0),
                            "{}: entry ({row}, {column}) is {got:?}, but the sequential \
                             accumulation gives {expected:?}",
                            label(op, scalar, m, k, n, threads)
                        );
                    }
                }
                println!(
                    "{} {} {m}x{k}x{n}: {}",
                    op.name(),
                    scalar.name(),
                    THREAD_COUNTS
                        .iter()
                        .zip(&runs)
                        .map(|(&threads, payload)| format!(
                            "{threads}T {:#018x}{}",
                            digest(payload),
                            match difference(payload, &runs[0]) {
                                None => String::new(),
                                Some((index, largest)) => format!(
                                    " differs from 1T: first differing element {index}, \
                                     max absolute difference {largest:e}"
                                ),
                            }
                        ))
                        .collect::<Vec<_>>()
                        .join(" | ")
                );
            }
        }
    }
}
