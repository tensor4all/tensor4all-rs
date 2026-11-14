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
                    assert_eq!(points.data(), expected.concat());
                }
                SampleBatch::Grid { points } => {
                    grid += 1;
                    assert_eq!(points.shape(), &[1, 2]);
                    assert_eq!(points.data()[0], 0.0);
                    assert!(points.data()[1] > 0.0);
                }
            }
            Ok(batch
                .points()
                .data()
                .chunks_exact(domain.ndims())
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

#[test]
fn coordinate_buffer_preserves_tensor_product_order_and_rounding() {
    use crate::interpolation::interval_points;
    let interval = NInterval::new(&[-2.3, 0.7, -0.125], &[1.1, 2.2, 0.875]).unwrap();
    let grid = get_chebyshev_grid(4).unwrap();
    let points = interval_points(&interval, grid.grid()).unwrap();
    let lengths = interval.lengths();
    let mut expected = Vec::new();
    for &z in grid.grid() {
        for &y in grid.grid() {
            for &x in grid.grid() {
                expected.extend_from_slice(&[
                    interval.start()[0] + lengths[0] * x,
                    interval.start()[1] + lengths[1] * y,
                    interval.start()[2] + lengths[2] * z,
                ]);
            }
        }
    }
    assert_eq!(
        points.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
    assert!(interval_points(&interval, &[]).unwrap().is_empty());
    assert_eq!(
        interval_points(&interval, &[0.0]).unwrap(),
        interval.start()
    );
}

#[test]
fn coordinate_buffer_checks_element_and_byte_products_before_allocation() {
    use crate::interpolation::interval_points;
    for dims in [usize::BITS as usize - 4, usize::BITS as usize] {
        let domain = NInterval::new(&vec![0.0; dims], &vec![1.0; dims]).unwrap();
        assert!(matches!(
            interval_points(&domain, &[0.0, 1.0]),
            Err(crate::InterpolativeQttError::InvalidArgument { .. })
        ));
    }
}
