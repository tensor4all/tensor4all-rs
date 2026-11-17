//! Construction cost of piecewise Chebyshev patches against the QTT patches.
//!
//! Report only: both constructors see the same 2D localized function and the
//! same tolerance. Run with `cargo bench -p tensor4all-interpolativeqtt --bench chebyshev_patches`.

use std::convert::Infallible;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use tensor4all_core::ColMajorArrayRef;
use tensor4all_interpolativeqtt::{
    interpolate_chebyshev_patches, interpolate_patched_nd, ChebyshevPatchOptions,
    PatchedInterpolativeOptions,
};
use tensor4all_tensorbackend::{CpuBackend, CpuExecutionContext, Exec};

const DEGREE: usize = 8;
const NUM_BITS: usize = 10;
const TOLERANCE: f64 = 1e-4;

fn function(point: &[f64]) -> f64 {
    1.0 / ((point[0] - 0.3).powi(2) + (point[1] - 0.6).powi(2) + 1e-2)
}

fn evaluate(points: ColMajorArrayRef<'_, f64>, _exec: Exec<'_>) -> Result<Vec<f64>, Infallible> {
    Ok(points
        .data()
        .chunks_exact(points.shape()[0])
        .map(function)
        .collect())
}

fn counting<'a>(
    counted: &'a AtomicUsize,
) -> impl Fn(ColMajorArrayRef<'_, f64>, Exec<'_>) -> Result<Vec<f64>, Infallible> + 'a {
    move |points, exec| {
        counted.fetch_add(points.shape()[1], Ordering::Relaxed);
        evaluate(points, exec)
    }
}

fn time<T>(run: impl FnOnce() -> T) -> (T, Duration) {
    let start = Instant::now();
    let value = run();
    (value, start.elapsed())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(1)?);
    let lower = [0.0, 0.0];
    let upper = [1.0, 1.0];

    let counted = AtomicUsize::new(0);
    let (patches, elapsed) = time(|| {
        interpolate_chebyshev_patches(
            &context,
            counting(&counted),
            &lower,
            &upper,
            DEGREE,
            &ChebyshevPatchOptions {
                tolerance: TOLERANCE,
                ..Default::default()
            },
        )
    });
    let patches = patches?;
    let valuations = patches.num_evaluations();
    println!(
        "chebyshev_patches: {} patches, {valuations} evaluations ({} counted), {:?}",
        patches.patches().len(),
        counted.load(Ordering::Relaxed),
        elapsed
    );

    let counted = AtomicUsize::new(0);
    let (qtt, elapsed) = time(|| {
        interpolate_patched_nd(
            &context,
            counting(&counted),
            &lower,
            &upper,
            NUM_BITS,
            DEGREE,
            &PatchedInterpolativeOptions {
                tolerance: TOLERANCE,
                ..Default::default()
            },
        )
    });
    let qtt = qtt?;
    println!(
        "patched_nd:        {} patches, {} evaluations, {:?}",
        qtt.len(),
        counted.load(Ordering::Relaxed),
        elapsed
    );
    Ok(())
}
