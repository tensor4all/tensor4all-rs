use crate::g0;

use num_complex::Complex64;
use tensor4all_core::IdxTensor;
use tensor4all_treeaci::{tree_elementwise_batched, TreeAciTermination};

/// Dense G0 on the full `2^(3r)` grid, in the fixture's site order.
fn expected_g0(r: usize, sites: &[tensor4all_core::DynIndex]) -> g0::TestResult<IdxTensor> {
    let values: Vec<_> = (0..1usize << (3 * r))
        .map(|flat| {
            let mut point = [0; 3];
            for bit in 0..r {
                for (var, x) in point.iter_mut().enumerate() {
                    *x = 2 * *x + ((flat >> (3 * bit + var)) & 1);
                }
            }
            g0::exact(r, point[0], point[1], point[2])
        })
        .collect();
    Ok(IdxTensor::from_dense(sites.to_vec(), values)?)
}

#[test]
fn low_temperature_g0_full_grid() -> g0::TestResult<()> {
    for r in [3, 4, 5] {
        for mode in ["cttn", "swap", "nblock"] {
            let (sites, inputs) = g0::fixture(r, mode)?;
            let expected = expected_g0(r, &sites)?;
            let dense_inputs: Vec<Vec<Complex64>> = inputs
                .iter()
                .map(|tree| {
                    Ok(tree
                        .to_dense()?
                        .permute_indices(&sites)?
                        .to_vec::<Complex64>()?)
                })
                .collect::<g0::TestResult<_>>()?;
            let direct = IdxTensor::from_dense(
                sites.clone(),
                (0..dense_inputs[0].len())
                    .map(|i| {
                        1.0 / (Complex64::new(0.5, 0.0)
                            + 2.0 * dense_inputs[0][i]
                            + 2.0 * dense_inputs[1][i]
                            + Complex64::i() * dense_inputs[2][i]
                            - dense_inputs[3][i])
                    })
                    .collect::<Vec<_>>(),
            )?;
            assert!(direct.sub(&expected)?.maxabs()? < 1e-10);
            let options = g0::options();
            let result = tree_elementwise_batched(g0::operator, &inputs, &options)?;
            let error = result.tree.to_dense()?.sub(&expected)?.maxabs()?;
            assert!(
                error <= options.tolerance * options.global_tolerance_margin,
                "r={r}, mode={mode}, max absolute residual={error:e}"
            );
        }
    }
    Ok(())
}

/// #1451: with `min_sweeps = 1` a stop rests on the residuals recorded in
/// one pass. Each was recorded when its edge was committed, and the later
/// commits of the same pass rewrote the output, so before the fix these
/// runs stopped `Converged` with a dense error of 4.73 (about 4,700 times the
/// tolerance): the wrong plane x = 16 lay outside every local matrix the
/// recorded residuals had seen. The stop now re-measures those residuals on
/// the current output and the current local matrices. Seed 1 is the run the
/// diagnosis traced (on `main` before #1450 it stopped after pass 3 at 4.73;
/// since #1450 it no longer reaches that state). Seed 50 (cttn) and seed 16
/// (swap) reach it on the current code: the re-measured pass-3 residual is
/// the wrong plane's error, the stop is refused, and pass 4 converges.
#[test]
fn one_pass_stop_rereads_residuals_the_later_commits_made_stale() -> g0::TestResult<()> {
    let r = 5;
    // (mode, seed, re-measured pass-3 residual if that stop was refused,
    // dense error of the returned tree)
    let cases: [(&str, u64, Option<f64>, f64); 3] = [
        ("cttn", 1, None, 9.450715517665529e-4),
        ("cttn", 50, Some(4.730515133088), 9.450715517665529e-4),
        ("swap", 16, Some(4.730515133088), 6.017484583886214e-4),
    ];
    for (mode, seed, refused, dense_error) in cases {
        let (sites, inputs) = g0::fixture(r, mode)?;
        let expected = expected_g0(r, &sites)?;
        let mut options = g0::options();
        options.tolerance = 1e-3;
        options.rng_seed = seed;
        options.min_sweeps = 1;
        let result = tree_elementwise_batched(g0::operator, &inputs, &options)?;
        let error = result.tree.to_dense()?.sub(&expected)?.maxabs()?;
        let context = format!(
            "{mode} seed {seed}: {:?} after {} passes, max_errors {:?}, Guard {:?}",
            result.termination,
            result.max_ranks.len(),
            result.max_errors,
            result.global_pivots_found
        );
        assert_eq!(
            result.termination,
            TreeAciTermination::Converged,
            "{context}"
        );
        assert_eq!(result.max_ranks.len(), 4, "{context}");
        assert!(
            (error - dense_error).abs() <= 1e-6 * dense_error,
            "{context}: dense error {error:e}"
        );
        assert!(error < options.tolerance * options.global_tolerance_margin);
        if let Some(refused) = refused {
            // Pass 3 met every other stop condition: the Guard found nothing.
            assert_eq!(result.global_pivots_found[2], 0, "{context}");
            assert!(
                (result.max_errors[2] - refused).abs() <= 1e-6 * refused,
                "{context}"
            );
        }
        assert!(result.max_errors[3] <= options.tolerance, "{context}");
    }
    Ok(())
}

/// #1451, nblock: seed 196 stopped `Converged` with dense error 3.8e-2
/// before the fix. The re-measured residual that refuses the stop sits on an
/// edge already at its algebraic rank bound; re-fitting that edge on its next
/// commit removes it, so the refusal must continue the sweep rather than
/// report `RankLimited` (recomputing the rank-limit flag from the re-measured
/// residuals ended this run `RankLimited` after 7 passes at 1.4e-3).
#[test]
fn refused_stop_at_a_saturated_edge_keeps_sweeping() -> g0::TestResult<()> {
    let r = 5;
    let (sites, inputs) = g0::fixture(r, "nblock")?;
    let expected = expected_g0(r, &sites)?;
    let mut options = g0::options();
    options.tolerance = 1e-3;
    options.rng_seed = 196;
    options.min_sweeps = 1;
    let result = tree_elementwise_batched(g0::operator, &inputs, &options)?;
    let error = result.tree.to_dense()?.sub(&expected)?.maxabs()?;
    assert_eq!(result.termination, TreeAciTermination::Converged);
    assert_eq!(result.max_ranks.len(), 9);
    let dense_error = 1.0964638964815038e-3;
    assert!(
        (error - dense_error).abs() <= 1e-6 * dense_error,
        "dense error {error:e}"
    );
    Ok(())
}

// The r=9 sibling witness of this file's regression lives in
// `heavy_g0_convergence.rs`: it needs ~23 minutes in the `ci` profile, so it runs
// from the scheduled heavy-tests workflow instead of on every push.
