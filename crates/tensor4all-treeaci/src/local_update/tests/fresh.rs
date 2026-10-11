//! Fresh LUCI uses its own cross even when several near-threshold crosses fit.
use super::*;
use num_complex::{Complex32, Complex64};
use tensor4all_core::Scalar;
use tensor4all_tensorbackend::mat_mul;

fn verify<T: crate::TreeAciScalar>(phase: T, rounding: f64) {
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
    for relative in [false, true] {
        let options = TreeAciOptions {
            tolerance: if relative {
                0.02 / 9.0
            } else {
                0.02 * phase.abs_val()
            },
            scale_tolerance: relative,
            ..TreeAciOptions::default()
        };
        let problem = prepare_problem::<T, _>(&inputs, &options).unwrap();
        let (arena, active) =
            SampleArena::from_global_seeds(&problem, &[vec![0, 0], vec![1, 1], vec![2, 2]])
                .unwrap();
        let frames = InputFrameStore::from_samples(&inputs, &problem, &arena).unwrap();
        for forward in [0, 1] {
            for left_orthogonal in [false, true] {
                let result = materialize_and_factor_edge(
                    &inputs,
                    &problem,
                    &active,
                    &frames,
                    forward,
                    &options,
                    left_orthogonal,
                    &mut |_: crate::TreeElementwiseBatch<'_, T>, output: &mut [T]| {
                        output.copy_from_slice(&target);
                        Ok(())
                    },
                )
                .unwrap();
                assert_eq!(result.row_samples[0].local_coordinate, 2);
                assert_eq!(result.col_samples[0].local_coordinate, 2);
                let reconstructed = mat_mul(&result.left, &result.right).unwrap();
                let residual = target
                    .iter()
                    .zip(reconstructed.as_col_major_slice())
                    .map(|(&a, &b)| Scalar::abs_val(a - b))
                    .fold(0.0_f64, f64::max);
                assert!((residual - 0.01 * phase.abs_val()).abs() < rounding);
                assert!(!options
                    .tolerance_policy()
                    .exceeds(residual, result.sampled_scale));
            }
        }
    }
}

#[test]
fn fresh_cross_reconstructs_the_same_known_perturbation_for_all_scalars_and_orientations() {
    verify::<f32>(1.0, 2e-5);
    verify::<f64>(1.0, 1e-12);
    verify::<Complex32>(Complex32::new(1.0, 0.25), 2e-5);
    verify::<Complex64>(Complex64::new(1.0, 0.25), 1e-12);
}
