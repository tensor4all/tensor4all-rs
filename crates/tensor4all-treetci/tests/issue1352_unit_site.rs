//! Exhausted candidates at a dimension-one site must not certify convergence.

use anyhow::Result;
use num_complex::Complex64;
use tensor4all_core::{ColMajorArrayRef, CommonScalar, IdxTensor, MatrixLuciScalar, TensorElement};
use tensor4all_tensorbackend::FullPivLuScalar;
use tensor4all_treetci::{
    crossinterpolate2, globalpivot::ScalarParts, DefaultProposer, GlobalIndexBatch, TreeTciGraph,
    TreeTciOptions, TreeTciTermination,
};
use tensor4all_treetn::{EvaluationHint, TreeTNCachedEvaluator};

fn check_unit_site<T>(scale: T, tie_free: bool) -> Result<()>
where
    T: MatrixLuciScalar + CommonScalar + FullPivLuScalar + TensorElement + ScalarParts,
{
    let dims = vec![2, 2, 2, 2, 1, 2, 2, 2, 2];
    let target = |point: &[usize]| {
        let a = point[..4].iter().fold(0, |x, &bit| 2 * x + bit) as f64;
        let c = point[5..].iter().fold(0, |x, &bit| 2 * x + bit) as f64;
        let weight = if tie_free {
            1.0 + 0.1 * (1.0 + a).sqrt() + 0.05 * (2.0 + c).ln()
        } else {
            1.0
        };
        scale * T::from_f64((-((a - c) / 1.5).powi(2)).exp() * weight)
    };
    // Column-major grid and reference; the same 256 points serve every run.
    let points: Vec<usize> = (0..256)
        .flat_map(|flat| {
            let mut remainder = flat;
            dims.iter()
                .map(|&dim| {
                    let coordinate = remainder % dim;
                    remainder /= dim;
                    coordinate
                })
                .collect::<Vec<_>>()
        })
        .collect();
    let values: Vec<T> = points.chunks_exact(dims.len()).map(target).collect();
    let target_scale = values
        .iter()
        .copied()
        .map(CommonScalar::abs_val)
        .fold(0.0, f64::max);
    let seeds: &[u64] = if tie_free {
        &[2, 5]
    } else {
        &[4, 2, 5, 1, 3, 8]
    };
    for enable_global_pivots in [true, false] {
        for &seed in seeds {
            let run = crossinterpolate2(
                |batch: GlobalIndexBatch<'_>| -> Result<Vec<T>> {
                    Ok(batch
                        .data()
                        .chunks_exact(batch.n_sites())
                        .map(target)
                        .collect())
                },
                dims.clone(),
                TreeTciGraph::linear_chain(dims.len())?,
                vec![vec![1, 1, 1, 1, 0, 1, 1, 1, 1]],
                TreeTciOptions {
                    seed: Some(seed),
                    enable_global_pivots,
                    ..Default::default()
                },
                Some(0),
                &DefaultProposer,
                &mut (),
            )?;
            let indices: Vec<_> = (0..dims.len())
                .map(|site| {
                    run.treetn
                        .site_space(&site)
                        .unwrap()
                        .iter()
                        .next()
                        .unwrap()
                        .clone()
                })
                .collect();
            let reference = IdxTensor::from_dense(indices.clone(), values.clone())?;
            let dense = run.treetn.to_dense()?;
            let residual = dense.sub(&reference)?.maxabs()?;
            let relative_error = residual / target_scale;

            // Check the same cached readout used by global search against a
            // single dense contraction, including hints around the unit site.
            let mut cache =
                TreeTNCachedEvaluator::<usize>::new(&run.treetn, &indices, Default::default())?;
            let shape = [dims.len(), 256];
            let batch = ColMajorArrayRef::new(&points, &shape)?;
            for site in [3, 4, 5] {
                let cached_values = cache
                    .evaluate_batched_with_hint(batch, EvaluationHint::around(site))?
                    .into_iter()
                    .map(|v| Complex64::new(v.real(), v.imag()))
                    .collect();
                let cached_dense = IdxTensor::from_dense(indices.clone(), cached_values)?;
                let readout_error = dense.sub(&cached_dense)?.maxabs()? / target_scale;
                assert!(readout_error <= 1e-12,
                    "site {site}, seed {seed}, global {enable_global_pivots}: cached residual {readout_error:e}");
            }
            // Floating-point pivot ties can change the trajectory by scalar
            // type. Exact full-rank results may converge; inaccurate exhausted
            // candidates must be reported as unmeasured.
            if run.termination == TreeTciTermination::Converged {
                assert!(
                    relative_error <= 1e-12,
                    "seed {seed}, global {enable_global_pivots}: residual {relative_error:e}"
                );
            } else {
                assert_eq!(
                    run.termination,
                    TreeTciTermination::DegenerateCandidateSet,
                    "seed {seed}, global {enable_global_pivots}: residual {relative_error:e}"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn gaussian_with_unit_site_real() -> Result<()> {
    check_unit_site(1.0_f64, false)?;
    check_unit_site(1.0_f64, true)
}

#[test]
fn gaussian_with_unit_site_complex() -> Result<()> {
    check_unit_site(Complex64::new(1.0, 0.5), false)?;
    check_unit_site(Complex64::new(1.0, 0.5), true)
}
