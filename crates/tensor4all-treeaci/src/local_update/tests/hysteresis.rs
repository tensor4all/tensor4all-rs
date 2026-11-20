//! The hysteresis floor of a local update (#784): a pivot between the lowered
//! threshold and the threshold is kept only when the edge already had it.
use super::*;
use crate::local_update::LocalUpdateResult;
use num_complex::{Complex32, Complex64};
use tensor4all_core::Scalar;
use tensor4all_tensorbackend::mat_mul;

/// Rank-one target plus a 0.01 perturbation at (1, 1): after normalization
/// by the sampled scale 9, the second pivot is about 1.1e-3, between the
/// threshold 0.02 / 9 and that threshold divided by the hysteresis factor.
fn verify<T: crate::TreeAciScalar + std::fmt::Debug>(phase: T, rounding: f64) {
    let bond = DynIndex::new_dyn(1);
    let input = TreeTN::from_tensors(
        vec![
            IdxTensor::from_dense(
                vec![DynIndex::new_dyn(3), bond.clone()],
                vec![T::from_f64(1.0); 3],
            )
            .unwrap(),
            IdxTensor::from_dense(vec![bond, DynIndex::new_dyn(3)], vec![T::from_f64(1.0); 3])
                .unwrap(),
        ],
        vec![0, 1],
    )
    .unwrap();
    let inputs = vec![input];
    let target = [1.0, 2.0, 3.0, 2.0, 4.01, 6.0, 3.0, 6.0, 9.0].map(|x| T::from_f64(x) * phase);
    let options = TreeAciOptions {
        tolerance: 0.02 / 9.0,
        scale_tolerance: true,
        ..TreeAciOptions::default()
    };
    let problem = prepare_problem::<T, _>(&inputs, &options).unwrap();
    let (arena, active) =
        SampleArena::from_global_seeds(&problem, &[vec![0, 0], vec![1, 1], vec![2, 2]]).unwrap();
    let frames =
        InputFrameStore::from_samples(&inputs, &problem, &arena, &mut crate::exec::Exec::Global)
            .unwrap();
    for left_orthogonal in [false, true] {
        let update = |previous_rank: Option<usize>| {
            materialize_and_factor_edge(
                &inputs,
                &problem,
                &active,
                &frames,
                0,
                &options,
                left_orthogonal,
                previous_rank,
                &mut |_: crate::TreeElementwiseBatch<'_, T>, output: &mut [T]| {
                    output.copy_from_slice(&target);
                    Ok(())
                },
                &mut crate::exec::Exec::Global,
            )
            .unwrap()
        };
        let residual = |result: &LocalUpdateResult<T>| {
            let rebuilt = mat_mul(&result.left, &result.right).unwrap();
            target
                .iter()
                .zip(rebuilt.as_col_major_slice())
                .map(|(&a, &b)| Scalar::abs_val(a - b))
                .fold(0.0_f64, f64::max)
        };

        let plain = update(None);
        assert_eq!(plain.left.ncols(), 1);
        assert!((residual(&plain) - 0.01 * phase.abs_val()).abs() < rounding);

        // A previous rank of one gives the plain result: no floor to hold.
        let held_at_one = update(Some(1));
        assert_eq!(held_at_one.left.ncols(), 1);
        assert_eq!(held_at_one.pivot_errors, plain.pivot_errors);
        assert_eq!(
            held_at_one.left.as_col_major_slice(),
            plain.left.as_col_major_slice()
        );
        assert_eq!(
            held_at_one.right.as_col_major_slice(),
            plain.right.as_col_major_slice()
        );

        // The edge had two pivots: the sub-threshold one stays, and the
        // perturbation is reconstructed.
        let held = update(Some(2));
        assert_eq!(held.left.ncols(), 2);
        assert_eq!(held.row_samples.len(), 2);
        assert_eq!(held.col_samples.len(), 2);
        assert!(residual(&held) < rounding);
        let threshold = options
            .tolerance_policy()
            .absolute_threshold(held.sampled_scale);
        assert!(held.pivot_errors[1] < threshold);
        assert!(held.pivot_errors[1] >= threshold / crate::rank_hysteresis::RANK_HYSTERESIS_FACTOR);

        // The floor never reaches below the lowered threshold.
        let capped = update(Some(5));
        assert_eq!(capped.left.ncols(), 2);
        assert_eq!(capped.pivot_errors, held.pivot_errors);
    }
}

#[test]
fn hysteresis_keeps_a_sub_threshold_pivot_only_up_to_the_previous_rank() {
    verify::<f32>(1.0, 2e-5);
    verify::<f64>(1.0, 1e-12);
    verify::<Complex32>(Complex32::new(1.0, 0.25), 2e-5);
    verify::<Complex64>(Complex64::new(1.0, 0.25), 1e-12);
}
