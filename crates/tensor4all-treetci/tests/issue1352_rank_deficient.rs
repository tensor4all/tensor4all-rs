//! Non-exhausted candidates at a unit junction can miss independent directions.
//!
//! Frozen target data from Simon's reproducer:
//! https://github.com/tensor4all/tensor4all-rs/issues/1352#issuecomment-6939597508
//! The fixture retains its exact f64 values without relying on StdRng in tests.

use anyhow::Result;
use num_complex::Complex64;
use tensor4all_core::{CommonScalar, IdxTensor, MatrixLuciScalar, TensorElement};
use tensor4all_tensorbackend::FullPivLuScalar;
use tensor4all_treetci::{
    globalpivot::ScalarParts, optimize_default, to_treetn, GlobalIndexBatch, TreeTCI2, TreeTciEdge,
    TreeTciGraph, TreeTciOptions, TreeTciTermination,
};

const R: usize = 5;
const N: usize = 1 << R;
const CENTRE: usize = 3 * R;

fn axes(point: &[usize]) -> [usize; 3] {
    [0, 1, 2].map(|branch| (0..R).fold(0, |n, bit| 2 * n + point[3 * bit + branch]))
}

fn graph() -> Result<TreeTciGraph> {
    let mut edges: Vec<_> = (0..3)
        .flat_map(|branch| {
            (0..R - 1)
                .map(move |level| TreeTciEdge::new(3 * level + branch, 3 * (level + 1) + branch))
        })
        .collect();
    edges.extend((0..3).map(|branch| TreeTciEdge::new(3 * (R - 1) + branch, CENTRE)));
    Ok(TreeTciGraph::new(CENTRE + 1, &edges)?)
}

fn check<T>(fixture: &str, seeds: &[u64], scale: T) -> Result<()>
where
    T: MatrixLuciScalar + CommonScalar + FullPivLuScalar + TensorElement + ScalarParts,
{
    let matrix: Vec<_> = fixture
        .split_whitespace()
        .map(|word| f64::from_bits(u64::from_str_radix(word, 16).unwrap()))
        .collect();
    assert_eq!(matrix.len(), N * N);
    let (largest, &max_value) = matrix
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.abs().total_cmp(&b.abs()))
        .unwrap();
    let start_axes = [largest % N, 0, largest / N];
    let start: Vec<_> = (0..=CENTRE)
        .map(|site| {
            if site == CENTRE {
                0
            } else {
                (start_axes[site % 3] >> (R - 1 - site / 3)) & 1
            }
        })
        .collect();
    let value = |point: &[usize]| {
        let [a, b, c] = axes(point);
        if b == 0 {
            scale * T::from_f64(matrix[a + N * c])
        } else {
            T::zero()
        }
    };
    for &seed in seeds {
        let mut dims = vec![2; CENTRE];
        dims.push(1);
        let mut state = TreeTCI2::<T>::new(dims, graph()?)?;
        state.add_global_pivots(std::slice::from_ref(&start))?;
        state.set_max_sample_value(max_value.abs() * CommonScalar::abs_val(scale));
        let evaluate = |batch: GlobalIndexBatch<'_>| -> Result<Vec<T>> {
            Ok(batch
                .data()
                .chunks_exact(batch.n_sites())
                .map(value)
                .collect())
        };
        let options = TreeTciOptions {
            seed: Some(seed),
            ..Default::default()
        };
        let run = optimize_default(&mut state, evaluate, &options)?;
        let tn = to_treetn(&state, evaluate, Some(0))?;
        let indices: Vec<_> = (0..=CENTRE)
            .map(|site| tn.site_space(&site).unwrap().iter().next().unwrap().clone())
            .collect();
        let values: Vec<_> = (0..1 << CENTRE)
            .map(|flat| {
                let mut point = [0; CENTRE + 1];
                for (site, bit) in point[..CENTRE].iter_mut().enumerate() {
                    *bit = (flat >> site) & 1;
                }
                value(&point)
            })
            .collect();
        let reference = IdxTensor::from_dense(indices.clone(), values)?;
        let difference = tn.to_dense()?.sub(&reference)?;
        let error = difference.maxabs()? / reference.maxabs()?;
        // Every listed seed converges, and the dense comparison confirms it.
        // Requiring `Converged` also catches a regression that gives up with
        // a bad approximation instead of reporting a false convergence.
        assert!(
            run.termination == TreeTciTermination::Converged && error <= options.tolerance,
            "seed {seed}: {:?} with reported {:?}, true residual {error:e}",
            run.termination,
            run.errors.last()
        );
    }
    Ok(())
}

#[test]
fn non_exhausted_three_branch_real() -> Result<()> {
    check(
        include_str!("fixtures/issue1352_m12_bits.txt"),
        &[1, 2],
        1.0_f64,
    )?;
    check(
        include_str!("fixtures/issue1352_m14_bits.txt"),
        &[1, 2, 6],
        1.0_f64,
    )
}

#[test]
fn non_exhausted_three_branch_complex() -> Result<()> {
    check(
        include_str!("fixtures/issue1352_m12_bits.txt"),
        &[1, 2],
        Complex64::new(1.0, 0.5),
    )?;
    check(
        include_str!("fixtures/issue1352_m14_bits.txt"),
        &[1, 2, 6],
        Complex64::new(1.0, 0.5),
    )
}
