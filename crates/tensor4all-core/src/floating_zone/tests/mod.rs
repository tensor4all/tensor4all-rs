use super::super::{floating_zone_walk, floating_zone_walk_with_initial_error, FloatingZoneError};

fn rejected_without_evaluation(
    dimensions: &[usize],
    start: Vec<usize>,
    error: f64,
    sweeps: usize,
    tolerance: f64,
) -> FloatingZoneError<std::convert::Infallible> {
    floating_zone_walk_with_initial_error::<_, std::convert::Infallible>(
        dimensions,
        &start,
        error,
        sweeps,
        tolerance,
        |_, _| panic!("invalid arguments reached the evaluator"),
    )
    .unwrap_err()
}

#[test]
fn invalid_starting_points_are_rejected_even_with_zero_sweeps() {
    for sweeps in [0, 1] {
        for start in [vec![], vec![0, 0]] {
            let actual = start.len();
            assert_eq!(
                rejected_without_evaluation(&[2], start, 0.0, sweeps, 1.0),
                FloatingZoneError::StartingPointLength {
                    expected: 1,
                    actual
                },
            );
        }
        assert_eq!(
            rejected_without_evaluation(&[2, 0], vec![0, 0], 0.0, sweeps, 1.0),
            FloatingZoneError::InvalidLocalDimension { site: 1 },
        );
        for coordinate in [2, 3, usize::MAX] {
            assert_eq!(
                rejected_without_evaluation(&[1, 2], vec![0, coordinate], 0.0, sweeps, 1.0),
                FloatingZoneError::StartingCoordinate {
                    site: 1,
                    coordinate,
                    dimension: 2
                },
            );
        }
    }
}

#[test]
fn initial_error_and_stopping_tolerance_are_checked_before_evaluation() {
    for sweeps in [0, 1] {
        for value in [-1.0, f64::NEG_INFINITY, f64::INFINITY, f64::NAN] {
            match rejected_without_evaluation(&[2], vec![0], value, sweeps, 1.0) {
                FloatingZoneError::InvalidInitialError { value: actual } => {
                    assert_eq!(actual.to_bits(), value.to_bits());
                }
                error => panic!("unexpected error: {error:?}"),
            }
        }
        for value in [-1.0, f64::NEG_INFINITY, f64::NAN] {
            match rejected_without_evaluation(&[2], vec![0], 0.0, sweeps, value) {
                FloatingZoneError::InvalidTolerance { value: actual } => {
                    assert_eq!(actual.to_bits(), value.to_bits());
                }
                error => panic!("unexpected error: {error:?}"),
            }
        }
    }
}

#[test]
fn candidate_storage_overflow_is_rejected_before_allocation() {
    let vector_bytes = std::mem::size_of::<Vec<usize>>();
    for dimensions in [
        vec![usize::MAX],
        vec![usize::MAX / vector_bytes],
        vec![usize::MAX / vector_bytes, 1, 1, 1],
        vec![isize::MAX as usize / vector_bytes + 1],
    ] {
        assert_eq!(
            rejected_without_evaluation(&dimensions, vec![0; dimensions.len()], 0.0, 1, 1.0),
            FloatingZoneError::AllocationSizeOverflow { site: 0 },
        );
    }
    let result = floating_zone_walk_with_initial_error::<_, std::convert::Infallible>(
        &[usize::MAX],
        &vec![0],
        2.0,
        0,
        f64::INFINITY,
        |_, _| panic!("zero sweeps must not allocate a scan batch"),
    )
    .unwrap();
    assert_eq!(result, (vec![0], 2.0));
}

#[test]
fn empty_and_singleton_domains_need_no_callback() {
    for (dimensions, start) in [(vec![], vec![]), (vec![1, 1], vec![0, 0])] {
        for tolerance in [0.0, f64::INFINITY] {
            let result = floating_zone_walk_with_initial_error::<_, std::convert::Infallible>(
                &dimensions,
                &start,
                0.5,
                4,
                tolerance,
                |_, _| panic!("this domain has no alternate coordinates"),
            )
            .unwrap();
            assert_eq!(result, (start.clone(), 0.5));
        }
    }
}

#[test]
fn callback_failure_retains_its_error_source() {
    use std::error::Error;
    let result = floating_zone_walk_with_initial_error(
        &[2],
        &vec![0],
        0.0,
        1,
        1.0,
        |_, _| -> Result<Vec<f64>, std::io::Error> {
            Err(std::io::Error::other("evaluation failed"))
        },
    );
    let error = result.unwrap_err();
    assert_eq!(error.source().unwrap().to_string(), "evaluation failed");
    match error {
        FloatingZoneError::Evaluation(error) => {
            assert_eq!(error.kind(), std::io::ErrorKind::Other);
        }
        error => panic!("unexpected error: {error:?}"),
    }
}

#[test]
fn supplied_start_error_preserves_every_scan_and_skips_only_the_seed() {
    for dims in [vec![], vec![1], vec![1, 3, 2], vec![2, 4, 3, 2]] {
        for sweeps in [0, 1, 8] {
            for tolerance in [0.0, 0.5, f64::INFINITY] {
                for last in [false, true] {
                    let start = dims.iter().map(|&d| if last { d - 1 } else { 0 }).collect();
                    for error_at in [rough_error as fn(&[usize]) -> f64, |_| 0.0, |_| 1.0] {
                        let mut reference_calls = Vec::new();
                        let expected = floating_zone_walk::<_, std::convert::Infallible>(
                            &dims,
                            &start,
                            sweeps,
                            tolerance,
                            |site, points| {
                                reference_calls.push((site, points.to_vec()));
                                Ok(points.iter().map(|p| error_at(p)).collect())
                            },
                        )
                        .unwrap();
                        let mut actual_calls = Vec::new();
                        let actual =
                            floating_zone_walk_with_initial_error::<_, std::convert::Infallible>(
                                &dims,
                                &start,
                                error_at(&start),
                                sweeps,
                                tolerance,
                                |site, points| {
                                    actual_calls.push((site, points.to_vec()));
                                    Ok(points.iter().map(|p| error_at(p)).collect())
                                },
                            )
                            .unwrap();
                        assert_eq!(actual, expected);
                        assert_eq!(reference_calls[0], (None, vec![start.clone()]));
                        assert_eq!(actual_calls, reference_calls[1..]);
                    }
                }
            }
        }
    }
}

#[test]
fn supplied_start_error_propagates_scan_failure() {
    let result = floating_zone_walk_with_initial_error(
        &[2],
        &vec![0],
        0.0,
        1,
        1.0,
        |site, _| -> Result<Vec<f64>, &str> {
            assert_eq!(site, Some(0));
            Err("scan failed")
        },
    );
    assert_eq!(result, Err(FloatingZoneError::Evaluation("scan failed")));

    // The callback error type need not implement Display or std::error::Error.
    #[derive(Debug, PartialEq)]
    struct Token(u8);
    let result = floating_zone_walk_with_initial_error(
        &[2],
        &vec![0],
        0.0,
        1,
        1.0,
        |_, _| -> Result<Vec<f64>, Token> { Err(Token(7)) },
    );
    assert_eq!(result, Err(FloatingZoneError::Evaluation(Token(7))));
}

/// The reference implementation this walk replaces: every candidate at a
/// site is evaluated, including the one the pivot already holds.
fn full_batch_walk(
    local_dims: &[usize],
    init_p: &[usize],
    max_sweeps: usize,
    early_stop_tol: f64,
    error_at: &dyn Fn(&[usize]) -> f64,
) -> (Vec<usize>, f64, usize) {
    let mut pivot = init_p.to_vec();
    let mut evaluated = 1usize;
    let mut max_error = error_at(&pivot);
    for _ in 0..max_sweeps {
        let previous = max_error;
        for site in 0..local_dims.len() {
            let mut best_index = pivot[site];
            let mut best_error = 0.0f64;
            for value in 0..local_dims[site] {
                let mut point = pivot.clone();
                point[site] = value;
                evaluated += 1;
                let error = error_at(&point);
                if error > best_error {
                    best_error = error;
                    best_index = value;
                }
            }
            pivot[site] = best_index;
            max_error = max_error.max(best_error);
        }
        if max_error == previous || max_error > early_stop_tol {
            break;
        }
    }
    (pivot, max_error, evaluated)
}

fn rough_error(point: &[usize]) -> f64 {
    let mut value = 0.0;
    for (site, &coordinate) in point.iter().enumerate() {
        value += ((site as f64 + 1.7) * (coordinate as f64 + 0.3)).sin();
    }
    value.abs()
}

/// Skipping the held candidate must not change the trajectory, the pivot,
/// or the reported error, only the number of points evaluated.
#[test]
fn skipping_the_held_candidate_matches_the_full_batch_walk() {
    for local_dims in [
        vec![2usize, 2, 2, 2, 2],
        vec![3usize, 2, 4],
        vec![2usize, 5, 3, 2],
    ] {
        for start in [
            vec![0usize; local_dims.len()],
            vec![1usize; local_dims.len()],
        ] {
            let start: Vec<usize> = start
                .iter()
                .zip(&local_dims)
                .map(|(&coordinate, &dim)| coordinate % dim)
                .collect();
            let (expected_pivot, expected_error, full_points) =
                full_batch_walk(&local_dims, &start, 8, f64::INFINITY, &rough_error);

            let mut evaluated = 0usize;
            let mut scan_sites = Vec::new();
            let (pivot, error) = floating_zone_walk::<_, std::convert::Infallible>(
                &local_dims,
                &start,
                8,
                f64::INFINITY,
                |site, points| {
                    scan_sites.push(site);
                    evaluated += points.len();
                    Ok(points.iter().map(|point| rough_error(point)).collect())
                },
            )
            .unwrap();

            assert_eq!(pivot, expected_pivot);
            assert_eq!(error, expected_error);
            assert!(
                evaluated < full_points,
                "the skip must evaluate fewer points: {evaluated} against {full_points}"
            );
            // Every scan declares its site, and only the seed does not.
            assert_eq!(scan_sites.first().copied(), Some(None));
            assert!(scan_sites[1..].iter().all(Option::is_some));
            // Every batch a scan asks for excludes exactly one candidate.
            let sweeps = (scan_sites.len() - 1) / local_dims.len();
            let per_sweep: usize = local_dims.iter().map(|dim| dim - 1).sum();
            assert_eq!(evaluated, 1 + sweeps * per_sweep);
        }
    }
}

/// A site of local dimension one has no other candidate, so its scan asks
/// for nothing at all and the pivot keeps its only value.
#[test]
fn a_singleton_site_is_never_evaluated() {
    let local_dims = [1usize, 2];
    let mut batches = Vec::new();
    let (pivot, error) = floating_zone_walk::<_, std::convert::Infallible>(
        &local_dims,
        &vec![0usize, 0],
        4,
        f64::INFINITY,
        |site, points| {
            batches.push((site, points.len()));
            Ok(points.iter().map(|point| point[1] as f64).collect())
        },
    )
    .unwrap();

    assert_eq!(pivot, vec![0, 1]);
    assert_eq!(error, 1.0);
    assert!(!batches.contains(&(Some(0), 1)));
    assert!(batches.contains(&(Some(1), 1)));
}

/// The error the seed call reports is the pivot's own error, so a start
/// that is already the maximum is not lost by the first scan.
#[test]
fn a_maximal_start_is_kept() {
    let local_dims = [2usize, 2];
    let (pivot, error) = floating_zone_walk::<_, std::convert::Infallible>(
        &local_dims,
        &vec![1usize, 1],
        4,
        f64::INFINITY,
        |_site, points| {
            Ok(points
                .iter()
                .map(|point| if point == &vec![1usize, 1] { 5.0 } else { 1.0 })
                .collect())
        },
    )
    .unwrap();

    assert_eq!(pivot, vec![1, 1]);
    assert_eq!(error, 5.0);
}
