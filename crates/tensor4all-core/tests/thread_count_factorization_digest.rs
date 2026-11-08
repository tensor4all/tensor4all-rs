//! Which determinism `qr` and `svd` actually deliver, measured (#1162).
//!
//! `docs/design/tree-patching-error-contract.md` splits the promise in two:
//!
//! - **Tier 1** (enforced): bitwise identical on the same machine and build at a
//!   **fixed thread count**, across fresh threads, processes and repeated runs.
//! - **Tier 2** (observed where it happens, not promised): identical **across
//!   thread counts**.
//!
//! The two tests here are the two halves. The first one checks within-process
//! repeatability at a fixed thread count for `qr` and `svd`; that is the Tier 1
//! half this file can exercise, and it must always pass. Fresh caller threads and
//! a second process are not covered here, so no cross-process claim is made. The
//! second test records the current Tier 2 state instead of asserting a direction,
//! so it passes whether the factorizations agree across thread counts or not;
//! what it does assert is what holds either way: the factor shapes do not move
//! with the thread count, and every factorization reconstructs the input to
//! round-off. It prints the digest verdict and the largest payload difference per
//! cell, so the current state is read from the run rather than guessed.
//!
//! The thread count comes from an explicit `CpuBackend::with_threads`, not from
//! `RAYON_NUM_THREADS`, so the environment cannot change what a cell means. A
//! provider that owns its own threads could still ignore that width; the numbers
//! recorded for these cells come from the platform's faer path, where it does
//! not.
//!
//! Run with `--nocapture` to see the per-cell record.

use std::sync::Arc;

use num_complex::Complex64;
use tensor4all_core::qr::{qr_with_in, QrOptions};
use tensor4all_core::svd::{svd_with_in, SvdOptions};
use tensor4all_core::{
    DynIndex, ExecutionContext, IdxTensor, SvdTruncationPolicy, TensorContractionLike,
};
use tensor4all_tensorbackend::{CpuBackend, CpuExecutionContext};

const THREAD_COUNTS: [usize; 3] = [1, 4, 8];

/// Small shapes, plus two above the `m*n ~ 16k` threshold at which the reported
/// divergence started.
const SHAPES: [(usize, usize); 6] = [(8, 8), (16, 16), (32, 32), (64, 64), (224, 112), (448, 224)];

#[derive(Clone, Copy)]
enum Kind {
    Qr,
    Svd,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Kind::Qr => "qr",
            Kind::Svd => "svd",
        }
    }
}

/// A backend of exactly `threads`, so the cell means one thread count only.
fn context(threads: usize) -> ExecutionContext {
    ExecutionContext::Cpu(Arc::new(CpuExecutionContext::from_backend(
        CpuBackend::with_threads(threads).expect("CPU backend"),
    )))
}

/// Deterministic complex input in `[-0.5, 0.5)`, column-major.
fn seeded_values(len: usize, seed: u64) -> Vec<Complex64> {
    let mut state = seed | 1;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    (0..len).map(|_| Complex64::new(next(), next())).collect()
}

fn matrix(context: &ExecutionContext, m: usize, n: usize) -> (DynIndex, IdxTensor) {
    let (i, j) = (DynIndex::new_dyn(m), DynIndex::new_dyn(n));
    let data = seeded_values(m * n, 7);
    let t = IdxTensor::from_dense_in(context, vec![i.clone(), j], data).expect("seeded matrix");
    (i, t)
}

fn factorize(
    kind: Kind,
    t: &IdxTensor,
    left: &DynIndex,
    context: &ExecutionContext,
) -> Vec<IdxTensor> {
    let left_inds = std::slice::from_ref(left);
    match kind {
        Kind::Qr => {
            let (q, r) =
                qr_with_in::<Complex64>(t, left_inds, &QrOptions::new(), context).expect("QR");
            vec![q, r]
        }
        Kind::Svd => {
            // A tiny cutoff keeps the retained rank at `min(m, n)` for these
            // well-conditioned inputs, so every thread count returns the same
            // shapes and the payloads stay comparable elementwise.
            let options = SvdOptions::new().with_policy(SvdTruncationPolicy::new(1e-14));
            let (u, s, v) = svd_with_in::<Complex64>(t, left_inds, &options, context).expect("SVD");
            vec![u, s, v]
        }
    }
}

/// One factorization with a fresh backend, fresh indices and a fresh payload, so
/// two calls exercise two thread pools rather than two passes over one plan.
fn run(kind: Kind, threads: usize, m: usize, n: usize) -> (IdxTensor, Vec<IdxTensor>) {
    let context = context(threads);
    let (left, t) = matrix(&context, m, n);
    let factors = factorize(kind, &t, &left, &context);
    (t, factors)
}

fn label(kind: Kind, m: usize, n: usize, threads: usize) -> String {
    format!("{} {m}x{n} @ {threads}T", kind.name())
}

/// The factor payload as complex values; the SVD's `S` is `f64` even for a
/// complex input, so the real case is widened here rather than duplicating the
/// digest and comparison per dtype.
fn payload(factor: &IdxTensor) -> Vec<Complex64> {
    factor
        .to_vec::<Complex64>()
        .or_else(|_| {
            factor
                .to_vec::<f64>()
                .map(|v| v.into_iter().map(Complex64::from).collect())
        })
        .expect("host factor payload")
}

/// FNV-1a, byte by byte, over every payload value's bits in flat order.
///
/// The per-byte final step is what makes a sign flip visible: xor-then-multiply
/// once per 64-bit word cancels an even number of sign-bit flips exactly, so a
/// factored matrix and its negation would share a digest.
///
/// `IdxTensor::content_digest` is not usable here: the SVD's third factor is a
/// non-contiguous view, which that accessor reports as `None`, so a digest over
/// the read payload is the one that covers all three factors.
fn payload_digest(factor: &IdxTensor) -> u64 {
    payload(factor)
        .iter()
        .flat_map(|v| [v.re.to_bits(), v.im.to_bits()])
        .flat_map(u64::to_le_bytes)
        .fold(0xcbf29ce484222325, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
        })
}

fn digests(factors: &[IdxTensor]) -> Vec<u64> {
    factors.iter().map(payload_digest).collect()
}

/// Largest finite value of `values`, or `NaN` as soon as one value is `NaN`.
///
/// `f64::max` and `total_cmp` both hide NaN here: `max` ignores it, and
/// `total_cmp` orders the *negative* NaN pattern below `0.0`, so a NaN payload
/// would report a zero difference and pass the reconstruction check. Only an
/// explicit `is_nan` separates "no difference" from "not a number", and the
/// returned NaN then fails every `<=` tolerance comparison.
fn largest(values: impl Iterator<Item = f64>) -> f64 {
    let mut largest = 0.0_f64;
    for value in values {
        if value.is_nan() {
            return f64::NAN;
        }
        if value > largest {
            largest = value;
        }
    }
    largest
}

/// `(largest entry, largest |a - b|)` over two host payloads.
///
/// The comparison is on payload values, not on tensors: the factors of two calls
/// carry different index identities and eager contexts, so a tensor subtraction
/// would reject them (or answer `false`) before comparing anything.
fn payload_difference(a: &IdxTensor, b: &IdxTensor) -> (f64, f64) {
    let (a_values, b_values) = (payload(a), payload(b));
    assert_eq!(
        a_values.len(),
        b_values.len(),
        "different payload lengths ({}/{}): a shortened result must fail here, not compare short",
        a_values.len(),
        b_values.len()
    );
    // The same values in the same order can still describe another matrix: a
    // `[2, 3]` and a `[3, 2]` payload have one length. Compare the shape too, so a
    // wrong unfold or a transposed result fails instead of comparing flat.
    assert_eq!(
        a.dims(),
        b.dims(),
        "different payload shapes ({:?}/{:?}, {} elements each)",
        a.dims(),
        b.dims(),
        a_values.len()
    );
    let magnitude = largest(a_values.iter().map(|v| v.norm()));
    let difference = largest(a_values.iter().zip(&b_values).map(|(x, y)| (x - y).norm()));
    (magnitude, difference)
}

/// The largest of [`payload_difference`] over the returned factors.
fn max_abs_difference(reference: &[IdxTensor], candidate: &[IdxTensor]) -> (f64, f64) {
    assert_eq!(
        reference.len(),
        candidate.len(),
        "different factor counts ({}/{})",
        reference.len(),
        candidate.len()
    );
    let pairs: Vec<(f64, f64)> = reference
        .iter()
        .zip(candidate)
        .map(|(a, b)| payload_difference(a, b))
        .collect();
    (
        largest(pairs.iter().map(|(magnitude, _)| *magnitude)),
        largest(pairs.iter().map(|(_, difference)| *difference)),
    )
}

/// `A` rebuilt from one factorization: `Q R`, or `U S V^H`.
///
/// This is what a factorization keeps under a thread count even when its bits
/// move, so it is the one property the recording test asserts. The measured
/// deviation per operation and size is in `docs/design/tree-patching-error-contract.md`
/// (Determinism), not here.
fn reconstruction(kind: Kind, factors: &[IdxTensor]) -> IdxTensor {
    match kind {
        Kind::Qr => factors[0].contract_pair(&factors[1]).expect("Q R"),
        Kind::Svd => factors[0]
            .contract_pair(&factors[1])
            .expect("U S")
            .contract_pair(&factors[2].conj())
            .expect("U S V^H"),
    }
}

/// Tier 1: a fixed thread count repeats bitwise, so the digest is a legal key.
///
/// Two calls with fresh pools, fresh indices and fresh payloads must agree
/// exactly. This is within-process repeatability at a fixed thread count; fresh
/// caller threads and a second process are not covered here.
#[test]
fn a_fixed_thread_count_repeats_the_factorization_bits() {
    for kind in [Kind::Qr, Kind::Svd] {
        for (m, n) in SHAPES {
            for threads in THREAD_COUNTS {
                let (_, first) = run(kind, threads, m, n);
                let (_, second) = run(kind, threads, m, n);
                assert_eq!(
                    digests(&first),
                    digests(&second),
                    "{} changed between two runs at the same thread count",
                    label(kind, m, n, threads)
                );
            }
        }
    }
}

/// Tier 2, recorded rather than promised: today the digests may or may not agree
/// between 1, 4 and 8 threads.
///
/// The direction is not asserted, because neither direction is a promise. What is
/// asserted holds whichever way it goes: the factor shapes do not move with the
/// thread count, and each thread count still factorizes the input (to round-off).
#[test]
fn thread_count_may_change_the_last_bits_today() {
    for kind in [Kind::Qr, Kind::Svd] {
        for (m, n) in SHAPES {
            let runs: Vec<(IdxTensor, Vec<IdxTensor>)> = THREAD_COUNTS
                .iter()
                .map(|&threads| run(kind, threads, m, n))
                .collect();
            let cells: Vec<Vec<u64>> = runs.iter().map(|(_, factors)| digests(factors)).collect();
            for (&threads, (input, factors)) in THREAD_COUNTS.iter().zip(&runs) {
                let rebuilt = reconstruction(kind, factors);
                // `payload_difference` asserts that the rebuilt payload has the
                // input's element count **and shape**, so a truncated or
                // mis-shaped result fails here instead of comparing short or flat.
                // The input, not another thread count, is the reference.
                let (magnitude, difference) = payload_difference(input, &rebuilt);
                assert!(
                    difference.is_finite() && magnitude.is_finite(),
                    "{}: the reconstruction is not finite: max|diff| = {difference}, \
                     largest entry = {magnitude}",
                    label(kind, m, n, threads)
                );
                assert!(
                    difference <= 1e-12 * magnitude.max(1.0),
                    "{}: the factorization does not rebuild the input: max|diff| = \
                     {difference:.3e}, largest entry = {magnitude:.3e}",
                    label(kind, m, n, threads)
                );
            }
            for a in 0..THREAD_COUNTS.len() {
                for b in a + 1..THREAD_COUNTS.len() {
                    let (left, right) = (THREAD_COUNTS[a], THREAD_COUNTS[b]);
                    let cell = format!("{} {left}T vs {right}T", label(kind, m, n, left));
                    for (x, y) in runs[a].1.iter().zip(&runs[b].1) {
                        assert_eq!(
                            x.dims(),
                            y.dims(),
                            "{cell}: a factor shape moved with the thread count"
                        );
                    }
                    if cells[a] == cells[b] {
                        println!("{cell}: identical ({:?})", cells[a]);
                    } else {
                        let (magnitude, difference) = max_abs_difference(&runs[a].1, &runs[b].1);
                        println!(
                            "{cell}: differ, max|diff| = {difference:.3e}, largest entry = \
                             {magnitude:.3e} ({:?} / {:?})",
                            cells[a], cells[b]
                        );
                    }
                }
            }
        }
    }
}

/// The digest has to see a sign flip, or the two tests above cannot tell "the
/// same bits" from "the same bits with a sign moved".
///
/// Two flips, not one: a word-wise FNV-1a cancels an even number of sign-bit
/// flips exactly, which is why [`payload_digest`] multiplies after each byte.
#[test]
fn the_digest_sees_a_sign_flip() {
    let index = DynIndex::new_dyn(2);
    let one_way = IdxTensor::from_dense(vec![index.clone()], vec![1.0_f64, -2.0]).expect("dense");
    let other = IdxTensor::from_dense(vec![index], vec![-1.0_f64, 2.0]).expect("dense");
    assert_ne!(payload_digest(&one_way), payload_digest(&other));
}

/// A NaN payload must not read as "no difference".
///
/// Both bit patterns are checked because the negative NaN sorts below `0.0`
/// under `total_cmp`, which is exactly how a `total_cmp`-based maximum loses it.
#[test]
fn a_nan_payload_cannot_pass_the_reconstruction_tolerance() {
    let index = DynIndex::new_dyn(1);
    // One pattern per call: a tensor holding both would let the positive NaN,
    // which `total_cmp` sorts highest, mask the negative one.
    for bits in [0x7ff8_0000_0000_0000_u64, 0xfff8_0000_0000_0000] {
        let nan =
            IdxTensor::from_dense(vec![index.clone()], vec![f64::from_bits(bits)]).expect("dense");
        let (magnitude, difference) = payload_difference(&nan, &nan);
        assert!(
            difference.is_nan(),
            "NaN {bits:#x} reported a finite difference"
        );
        assert!(magnitude.is_nan(), "NaN {bits:#x} reported a finite entry");
        // The tolerance check in the reconstruction assertion is a `<=`, which is
        // false for an incomparable operand, so a NaN difference is rejected
        // there rather than passing as zero.
        assert!(
            difference
                .partial_cmp(&(1e-12 * magnitude.max(1.0)))
                .is_none(),
            "the tolerance comparison accepted NaN ({bits:#x})"
        );
    }
}

/// A payload shorter than its reference must fail instead of comparing short.
#[test]
#[should_panic(expected = "different payload lengths")]
fn a_short_payload_is_rejected() {
    let (short, long) = (DynIndex::new_dyn(1), DynIndex::new_dyn(2));
    let one = IdxTensor::from_dense(vec![short], vec![1.0_f64]).expect("dense");
    let two = IdxTensor::from_dense(vec![long], vec![1.0_f64, 2.0]).expect("dense");
    let _ = payload_difference(&one, &two);
}

/// The same values in the same order, but a different shape, must fail: an
/// element count alone does not say which matrix a payload is.
#[test]
#[should_panic(expected = "different payload shapes")]
fn a_payload_with_a_different_shape_is_rejected() {
    let (two, three) = (DynIndex::new_dyn(2), DynIndex::new_dyn(3));
    let values = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let two_by_three =
        IdxTensor::from_dense(vec![two.clone(), three.clone()], values.clone()).expect("dense");
    let three_by_two = IdxTensor::from_dense(vec![three, two], values).expect("dense");
    assert_eq!(two_by_three.dims(), vec![2, 3]);
    assert_eq!(three_by_two.dims(), vec![3, 2]);
    let _ = payload_difference(&two_by_three, &three_by_two);
}
