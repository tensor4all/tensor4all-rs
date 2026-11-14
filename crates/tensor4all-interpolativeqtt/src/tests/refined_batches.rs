//! The multiscale assembly names the kind of every batch it requests.
use crate::get_chebyshev_grid;
use crate::interpolation::{build_refined_cores, SampleBatch};
use crate::interval::NInterval;

#[test]
fn batches_name_their_chebyshev_interval_or_exact_grid_points() {
    let degree = 3;
    let basis = get_chebyshev_grid(degree).unwrap();
    let domain = NInterval::new(&[0.0], &[1.0]).unwrap();
    // Refine the leftmost box down to the last core, so the exact-sampling path
    // runs alongside the Chebyshev batches.
    let is_safe = |interval: &NInterval, _: usize| interval.start()[0] > 0.0;
    let (mut nodes, mut grid) = (0, 0);
    let cores = build_refined_cores(
        &mut |batch: SampleBatch<'_>| {
            match batch {
                SampleBatch::Nodes { interval, points } => {
                    nodes += 1;
                    let (a, len) = (interval.start()[0], interval.lengths()[0]);
                    let expected: Vec<Vec<f64>> =
                        basis.grid().iter().map(|&x| vec![a + len * x]).collect();
                    assert_eq!(points, expected.as_slice());
                }
                SampleBatch::Grid { points } => {
                    grid += 1;
                    assert_eq!(points.len(), 2);
                    assert_eq!(points[0], vec![0.0]);
                    assert!(points[1][0] > 0.0);
                }
            }
            Ok(batch
                .points()
                .iter()
                .map(|p| 1.0 + p[0])
                .collect::<Vec<f64>>())
        },
        &domain,
        4,
        &basis,
        is_safe,
    )
    .unwrap();
    assert_eq!(cores.len(), 4);
    // One unsafe box per level reaches the last core; every other box is a
    // Chebyshev batch: level 0 has one safe child, levels 1 and 2 one each.
    assert_eq!(grid, 1);
    assert_eq!(nodes, 3);
}
