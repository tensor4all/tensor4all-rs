//! Public stopping diagnostics, including limit precedence and dense accuracy.

use anyhow::Result;
use num_complex::Complex64;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use tensor4all_core::{CommonScalar, IdxTensor, MatrixLuciScalar, TensorElement};
use tensor4all_tensorbackend::FullPivLuScalar;
use tensor4all_treetci::{
    crossinterpolate2, crossinterpolate2_with_rng, globalpivot::ScalarParts, optimize_default,
    optimize_with_proposer, optimize_with_proposer_with_rng, to_treetn, DefaultProposer,
    GlobalIndexBatch, TreeTCI2, TreeTciGraph, TreeTciOptions, TreeTciRunResult, TreeTciTermination,
};

#[derive(Clone, Copy, Debug)]
enum EntryPoint {
    Default,
    Proposer,
    ProposerWithRng,
    CrossInterpolate,
    CrossInterpolateWithRng,
}

const ENTRY_POINTS: [EntryPoint; 5] = [
    EntryPoint::Default,
    EntryPoint::Proposer,
    EntryPoint::ProposerWithRng,
    EntryPoint::CrossInterpolate,
    EntryPoint::CrossInterpolateWithRng,
];

fn dense_reference<T: TensorElement>(
    result: &TreeTciRunResult,
    n_sites: usize,
    values: Vec<T>,
) -> IdxTensor {
    let indices = (0..n_sites)
        .map(|site| {
            result
                .treetn
                .site_space(&site)
                .unwrap()
                .iter()
                .next()
                .unwrap()
                .clone()
        })
        .collect();
    IdxTensor::from_dense(indices, values).unwrap()
}

fn run_identity<T>(scale: T, options: TreeTciOptions, entry: EntryPoint) -> TreeTciRunResult
where
    T: MatrixLuciScalar + CommonScalar + FullPivLuScalar + TensorElement + ScalarParts,
{
    let graph = TreeTciGraph::linear_chain(3).unwrap();
    // The first two sites form a rank-3 identity; the third is constant.
    let evaluate = |batch: GlobalIndexBatch<'_>| -> Result<Vec<T>> {
        Ok((0..batch.n_points())
            .map(|point| {
                if batch.get(0, point).unwrap() == batch.get(1, point).unwrap() {
                    scale
                } else {
                    T::zero()
                }
            })
            .collect())
    };
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    match entry {
        EntryPoint::CrossInterpolate => crossinterpolate2(
            evaluate,
            vec![3, 3, 2],
            graph,
            vec![vec![0; 3]],
            options,
            None,
            &DefaultProposer,
        )
        .unwrap(),
        EntryPoint::CrossInterpolateWithRng => crossinterpolate2_with_rng(
            evaluate,
            vec![3, 3, 2],
            graph,
            vec![vec![0; 3]],
            options,
            None,
            &DefaultProposer,
            &mut rng,
        )
        .unwrap(),
        _ => {
            let mut state = TreeTCI2::<T>::new(vec![3, 3, 2], graph).unwrap();
            state.add_global_pivots(&[vec![0; 3]]).unwrap();
            state.max_sample_value = CommonScalar::abs_val(scale);
            let result = match entry {
                EntryPoint::Default => optimize_default(&mut state, evaluate, &options),
                EntryPoint::Proposer => {
                    optimize_with_proposer(&mut state, evaluate, &options, &DefaultProposer)
                }
                EntryPoint::ProposerWithRng => optimize_with_proposer_with_rng(
                    &mut state,
                    evaluate,
                    &options,
                    &DefaultProposer,
                    &mut rng,
                ),
                _ => unreachable!("high-level entries handled above"),
            }
            .unwrap();
            TreeTciRunResult {
                treetn: to_treetn(&state, evaluate, None).unwrap(),
                ranks: result.ranks,
                errors: result.errors,
                termination: result.termination,
            }
        }
    }
}

fn check_reasons<T>(scale: T)
where
    T: MatrixLuciScalar + CommonScalar + FullPivLuScalar + TensorElement + ScalarParts,
{
    for entry in ENTRY_POINTS {
        for enable_global_pivots in [false, true] {
            for (max_iter, cap, reason, iterations) in [
                (20, None, TreeTciTermination::Converged, 3),
                // A complete confirmation window may converge on the last sweep.
                (3, None, TreeTciTermination::Converged, 3),
                (20, Some(1), TreeTciTermination::MaxBondDimension, 3),
                // Saturation takes precedence on the final sweep as well.
                (3, Some(1), TreeTciTermination::MaxBondDimension, 3),
                (1, None, TreeTciTermination::MaxIterations, 1),
                // Reaching the cap once does not complete its confirmation window.
                (1, Some(1), TreeTciTermination::MaxIterations, 1),
                // Even an exact approximation needs the convergence window.
                (2, None, TreeTciTermination::MaxIterations, 2),
                // A cap at the exact rank still reports saturation first.
                (20, Some(3), TreeTciTermination::MaxBondDimension, 3),
            ] {
                let result = run_identity(
                    scale,
                    TreeTciOptions {
                        tolerance: 1e-12,
                        max_iter,
                        max_bond_dim: cap,
                        enable_global_pivots,
                        seed: Some(7),
                        ..Default::default()
                    },
                    entry,
                );
                assert_eq!(result.termination, reason, "{entry:?}, cap {cap:?}");
                assert_eq!(result.ranks, vec![cap.unwrap_or(3); iterations]);
                assert_eq!(result.errors.len(), iterations);

                // Column-major dense reference using the network's own indices.
                let values = (0..18)
                    .map(|flat| {
                        if flat % 3 == (flat / 3) % 3 {
                            scale
                        } else {
                            T::zero()
                        }
                    })
                    .collect();
                let expected = dense_reference(&result, 3, values);
                let dense = result.treetn.to_dense().unwrap();
                let residual = dense.sub(&expected).unwrap().maxabs().unwrap();
                let max_value = CommonScalar::abs_val(scale);
                if cap == Some(1) {
                    assert!(residual > 0.5 * max_value, "residual {residual:e}");
                } else {
                    assert!(
                        residual <= 1e-12 * max_value,
                        "{entry:?}, {reason:?}: residual {residual:e}, scale {max_value:e}"
                    );
                }
            }
        }
    }
}

#[test]
fn termination_reasons_real() {
    check_reasons(2.0_f64);
}

#[test]
fn termination_reasons_complex() {
    check_reasons(Complex64::new(2.0, -0.5));
}

#[test]
fn iteration_limit_on_a_function_requiring_more_sweeps() {
    const N_SITES: usize = 7;
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let values: Vec<f64> = (0..1 << N_SITES)
        .map(|_| rng.random_range(1.0..2.0))
        .collect();
    let evaluate = |batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> {
        Ok((0..batch.n_points())
            .map(|point| {
                let flat = (0..N_SITES).fold(0, |flat, site| {
                    flat | (batch.get(site, point).unwrap() << site)
                });
                values[flat]
            })
            .collect())
    };
    for max_iter in [1, 20] {
        let result = crossinterpolate2(
            evaluate,
            vec![2; N_SITES],
            TreeTciGraph::linear_chain(N_SITES).unwrap(),
            vec![vec![0; N_SITES]],
            TreeTciOptions {
                tolerance: 1e-12,
                max_iter,
                seed: Some(7),
                ..Default::default()
            },
            None,
            &DefaultProposer,
        )
        .unwrap();
        let expected = dense_reference(&result, N_SITES, values.clone());
        let residual = result
            .treetn
            .to_dense()
            .unwrap()
            .sub(&expected)
            .unwrap()
            .maxabs()
            .unwrap();
        if max_iter == 1 {
            assert_eq!(result.termination, TreeTciTermination::MaxIterations);
            assert_eq!(result.ranks.len(), 1);
            assert!(residual > 1e-3, "residual {residual:e}");
        } else {
            assert_eq!(result.termination, TreeTciTermination::Converged);
            let scale = expected.maxabs().unwrap();
            assert!(residual <= 1e-12 * scale, "residual {residual:e}");
        }
    }
}

#[test]
fn zero_tolerance_requires_strictly_smaller_error() {
    let result = run_identity(
        1.0,
        TreeTciOptions {
            tolerance: 0.0,
            max_iter: 4,
            enable_global_pivots: false,
            ..Default::default()
        },
        EntryPoint::Default,
    );
    assert_eq!(result.errors, vec![0.0; 4]);
    assert_eq!(result.termination, TreeTciTermination::MaxIterations);
}
