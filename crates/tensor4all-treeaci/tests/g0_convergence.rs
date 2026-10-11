#[path = "support/g0.rs"]
mod g0;

use num_complex::Complex64;
use tensor4all_core::IdxTensor;
use tensor4all_treeaci::tree_elementwise_batched;

#[test]
fn low_temperature_g0_full_grid() -> g0::TestResult<()> {
    for r in [3, 4, 5] {
        for mode in ["cttn", "swap", "nblock"] {
            let (sites, inputs) = g0::fixture(r, mode)?;
            let expected_values: Vec<_> = (0..1usize << (3 * r))
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
            let expected = IdxTensor::from_dense(sites.clone(), expected_values)?;
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

// The r=9 sibling witness of this file's regression lives in
// `heavy_g0_convergence.rs`: it needs ~23 minutes in the `ci` profile, so it runs
// from the scheduled heavy-tests workflow instead of on every push.
