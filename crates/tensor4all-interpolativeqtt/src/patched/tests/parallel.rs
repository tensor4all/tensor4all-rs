use super::*;
use hataori_tenferro::TenferroLanes;
use rayon::prelude::*;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    OnceLock,
};

fn lanes(count: usize, threads: usize) -> TenferroLanes {
    TenferroLanes::new(
        (0..count)
            .map(|_| CpuBackend::with_threads(threads).unwrap().into_engine())
            .collect(),
    )
    .unwrap()
}

fn compare<T: InterpolativeScalar + Send + Sync + PartialEq + std::fmt::Debug>(
    f: impl Fn(&[f64]) -> T + Sync,
) {
    for dims in [1, 2] {
        let options = PatchedInterpolativeOptions {
            tolerance: 1e-4,
            ..Default::default()
        };
        let serial = interpolate_patched_nd(
            |p, _| batch(p, &f),
            &vec![0.0; dims],
            &vec![1.0; dims],
            5,
            5,
            &options,
        )
        .unwrap();
        for (p, b) in [(1, 1), (2, 1), (4, 1), (1, 2), (2, 2), (3, 3)] {
            let mut executor = lanes(p, b);
            let parallel = interpolate_patched_nd_on_lanes(
                &mut executor,
                |p, exec| {
                    assert_eq!(exec.budget().get(), b);
                    batch(p, &f)
                },
                &vec![0.0; dims],
                &vec![1.0; dims],
                5,
                5,
                &options,
            )
            .unwrap();
            assert_eq!(serial.len(), parallel.len(), "P={p}, B={b}");
            for (a, z) in serial.iter().zip(&parallel) {
                assert_eq!(a.prefix, z.prefix, "P={p}, B={b}");
                assert_eq!(a.error_estimate, z.error_estimate, "P={p}, B={b}");
                assert_eq!(a.tt.site_tensors(), z.tt.site_tensors(), "P={p}, B={b}");
            }
        }
    }
}

#[test]
fn lane_and_thread_counts_are_bitwise_deterministic() {
    compare(lorentzian);
    compare(|x| Complex64::new(lorentzian(x), 0.5 * lorentzian(x)));
}

#[test]
fn serial_ranges_and_external_evaluators_agree() {
    let mut executor = lanes(2, 3);
    let options = PatchedInterpolativeOptions {
        tolerance: 1e-4,
        ..Default::default()
    };
    let widths = AtomicUsize::new(0);
    let ranges = interpolate_patched_nd_on_lanes(
        &mut executor,
        |points, exec| {
            let outputs: Vec<OnceLock<f64>> =
                (0..points.shape()[1]).map(|_| OnceLock::new()).collect();
            widths.fetch_max(exec.budget().get(), Ordering::Relaxed);
            exec.for_ranges(outputs.len(), 1.try_into().unwrap(), |range| {
                for i in range {
                    outputs[i]
                        .set(lorentzian(&points.data()[i..i + 1]))
                        .unwrap();
                }
            });
            Ok::<_, Infallible>(
                outputs
                    .into_iter()
                    .map(|slot| slot.into_inner().unwrap())
                    .collect(),
            )
        },
        &[0.0],
        &[1.0],
        5,
        5,
        &options,
    )
    .unwrap();
    let external = interpolate_patched_nd_on_lanes(
        &mut executor,
        |points, exec| {
            exec.external(|width| {
                widths.fetch_max(width.get(), Ordering::Relaxed);
                Ok::<_, Infallible>(
                    points
                        .data()
                        .par_iter()
                        .map(|&x| lorentzian(&[x]))
                        .collect(),
                )
            })
        },
        &[0.0],
        &[1.0],
        5,
        5,
        &options,
    )
    .unwrap();
    assert_eq!(widths.load(Ordering::Relaxed), 3);
    assert_eq!(ranges.len(), external.len());
    for (a, b) in ranges.iter().zip(&external) {
        assert_eq!(a.prefix, b.prefix);
        assert_eq!(a.tt.site_tensors(), b.tt.site_tensors());
    }
}

#[derive(Debug, thiserror::Error)]
#[error("failed at {at}")]
struct LaneError {
    at: f64,
}

#[test]
fn simultaneous_errors_preserve_the_first_box_source() {
    let mut executor = lanes(3, 1);
    let error = interpolate_patched_nd_on_lanes(
        &mut executor,
        |points, _| {
            let at = points.data()[0];
            let width = points.data().last().unwrap() - at;
            if width < 0.3 {
                if at == 0.0 {
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
                return Err(LaneError { at });
            }
            Ok(points.data().iter().map(|&x| lorentzian(&[x])).collect())
        },
        &[0.0],
        &[1.0],
        5,
        5,
        &Default::default(),
    )
    .unwrap_err();
    match error {
        InterpolativeQttError::Evaluation { source } => {
            assert_eq!(source.downcast_ref::<LaneError>().unwrap().at, 0.0)
        }
        e => panic!("{e}"),
    }
}
