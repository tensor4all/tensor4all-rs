//! Non-exhausted candidates at a unit junction can miss independent directions.
//!
//! Frozen target data from Simon's reproducers:
//! https://github.com/tensor4all/tensor4all-rs/issues/1352#issuecomment-6939597508
//! The fixture retains its exact f64 values without relying on StdRng in tests.

use anyhow::Result;
use num_complex::Complex64;
use tensor4all_core::{CommonScalar, IdxTensor, MatrixLuciScalar, TensorElement};
use tensor4all_tensorbackend::FullPivLuScalar;
use tensor4all_treetci::{
    globalpivot::ScalarParts, optimize_default, to_treetn, GlobalIndexBatch, StopRule, TreeTCI2,
    TreeTciEdge, TreeTciGraph, TreeTciOptions, TreeTciTermination,
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
    check_plane(
        fixture,
        seeds,
        scale,
        0,
        &[],
        TreeTciOptions::default().tolerance,
    )
}

/// Run each seed on the table `fixture` placed on the plane `y = plane` (zero
/// elsewhere) and compare the materialized tree with the dense target.
///
/// With no `starts`, the run starts at the table's largest entry and is
/// normalized by it; otherwise it starts at `starts` (axis coordinates) and
/// is normalized by the value at the first one.
fn check_plane<T>(
    fixture: &str,
    seeds: &[u64],
    scale: T,
    plane: usize,
    starts: &[[usize; 3]],
    tolerance: f64,
) -> Result<()>
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
    let default_start = [[largest % N, plane, largest / N]];
    let normalize_by_maximum = starts.is_empty();
    let start_axes = if normalize_by_maximum {
        &default_start[..]
    } else {
        starts
    };
    let starts: Vec<Vec<_>> = start_axes
        .iter()
        .map(|axes| {
            (0..=CENTRE)
                .map(|site| {
                    if site == CENTRE {
                        0
                    } else {
                        (axes[site % 3] >> (R - 1 - site / 3)) & 1
                    }
                })
                .collect()
        })
        .collect();
    let value = |point: &[usize]| {
        let [a, b, c] = axes(point);
        if b == plane {
            scale * T::from_f64(matrix[a + N * c])
        } else {
            T::zero()
        }
    };
    let initial_scale = if normalize_by_maximum {
        max_value.abs() * CommonScalar::abs_val(scale)
    } else {
        CommonScalar::abs_val(value(&starts[0]))
    };
    // The dense target is built once and relabelled for each run's indices.
    let mut values = Some(
        (0..1 << CENTRE)
            .map(|flat| {
                let mut point = [0; CENTRE + 1];
                for (site, bit) in point[..CENTRE].iter_mut().enumerate() {
                    *bit = (flat >> site) & 1;
                }
                value(&point)
            })
            .collect::<Vec<_>>(),
    );
    let mut reference: Option<IdxTensor> = None;
    // Both stop rules must avoid a false `Converged` here (#1405): the fix
    // is structural (#1434), not a property of the sampled stop.
    let runs = seeds
        .iter()
        .flat_map(|&seed| [StopRule::Sampled, StopRule::PerBond].map(|rule| (seed, rule)));
    for (seed, stop_rule) in runs {
        let mut dims = vec![2; CENTRE];
        dims.push(1);
        let mut state = TreeTCI2::<T>::new(dims, graph()?)?;
        state.add_global_pivots(&starts)?;
        state.set_max_sample_value(initial_scale);
        let evaluate = |batch: GlobalIndexBatch<'_>| -> Result<Vec<T>> {
            Ok(batch
                .data()
                .chunks_exact(batch.n_sites())
                .map(value)
                .collect())
        };
        let options = TreeTciOptions {
            tolerance,
            stop_rule,
            seed: Some(seed),
            ..Default::default()
        };
        let run = optimize_default(&mut state, evaluate, &options)?;
        let tn = to_treetn(&state, evaluate, Some(0))?;
        let indices: Vec<_> = (0..=CENTRE)
            .map(|site| tn.site_space(&site).unwrap().iter().next().unwrap().clone())
            .collect();
        if reference.is_none() {
            reference = Some(IdxTensor::from_dense(
                indices.clone(),
                values.take().unwrap(),
            )?);
        }
        let stored = reference.as_ref().unwrap();
        let target = stored.replace_indices(stored.indices(), &indices)?;
        let difference = tn.to_dense()?.sub(&target)?;
        let error = difference.maxabs()? / target.maxabs()?;
        // Every listed seed converges, and the dense comparison confirms it.
        // Requiring `Converged` also catches a regression that gives up with
        // a bad approximation instead of reporting a false convergence.
        assert!(
            run.termination == TreeTciTermination::Converged && error <= tolerance,
            "seed {seed}, {stop_rule:?}: {:?} with reported {:?}, true residual {error:e}",
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

// Frozen f64 bits of Simon's attached g[z][x] on the plane y = 16, preserving
// exact zeros and decimal-to-f64 rounding:
// https://github.com/tensor4all/tensor4all-rs/issues/1352#issuecomment-6967309986
// Original table: https://github.com/user-attachments/files/35004742/g.txt
// Before the unit-junction expansion, seeds 4 and 6 stopped `Converged` at a
// dense relative error of 6.33e-10 and 2.52e-8 (tolerance 1e-10): the x and
// z bonds at the centre capped each other at rank 11 and 10 below the true
// rank 12.
fn check_reopened<T>(scale: T) -> Result<()>
where
    T: MatrixLuciScalar + CommonScalar + FullPivLuScalar + TensorElement + ScalarParts,
{
    check_plane(
        include_str!("fixtures/issue1352_reopened_bits.txt"),
        &[4, 6],
        scale,
        N / 2,
        &[[N / 2, N / 2, N / 2], [N / 2 - 1, N / 2, N / 2 - 1]],
        1e-10,
    )
}

#[test]
fn locked_unit_junction_real() -> Result<()> {
    check_reopened(1.0_f64)
}

#[test]
fn locked_unit_junction_complex() -> Result<()> {
    check_reopened(Complex64::new(1.0, 0.5))
}
