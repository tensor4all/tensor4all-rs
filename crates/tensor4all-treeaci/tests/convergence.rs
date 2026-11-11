//! Small tolerance-boundary witness for #784, independent of external data.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use tensor4all_core::{ColMajorArrayRef, DynIndex, IdxTensor};
use tensor4all_treeaci::{hadamard_many, TreeAciOptions, TreeAciTermination};
use tensor4all_treetn::{CachedEvaluatorOptions, EvaluationHint, TreeTN, TreeTNCachedEvaluator};

#[test]
fn tolerance_boundary_crosses_converge_without_relaxing_rank_stability() {
    let n = 16;
    let chi = 12;
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    let sites: Vec<_> = (0..n).map(|_| DynIndex::new_dyn(3)).collect();
    let bonds: Vec<_> = (1..n).map(|_| DynIndex::new_dyn(chi)).collect();
    let cores = (0..n)
        .map(|site| {
            let left = if site == 0 { 1 } else { chi };
            let right = if site + 1 == n { 1 } else { chi };
            let mut values = Vec::with_capacity(left * 3 * right);
            for j in 0..right {
                for _ in 0..3 * left {
                    values.push(rng.random_range(-1.0..1.0) * 0.8_f64.powi(j as i32));
                }
            }
            let mut indices = Vec::new();
            if site > 0 {
                indices.push(bonds[site - 1].clone());
            }
            indices.push(sites[site].clone());
            if site + 1 < n {
                indices.push(bonds[site].clone());
            }
            IdxTensor::from_dense(indices, values).unwrap()
        })
        .collect();
    let input = TreeTN::from_tensors(cores, (0..n).collect::<Vec<_>>()).unwrap();
    let options = TreeAciOptions {
        tolerance: 1e-3,
        max_bond_dim: Some(400),
        rng_seed: 1,
        ..Default::default()
    };
    let result = hadamard_many::<f64, _>(&[input.clone(), input.clone()], &options).unwrap();
    // Fresh complete pivoting cycles for all 20 passes on this witness: the
    // maximum rank stays 46 while smaller cuts repeatedly regrow.
    assert_eq!(result.termination, TreeAciTermination::Converged);
    assert!(result.max_ranks.len() < options.max_sweeps);
    assert!(result
        .max_errors
        .iter()
        .all(|&error| error <= options.tolerance));
    assert!(result.global_pivots_found.iter().all(|&found| found == 0));

    // Independent held-out points; never expand the 3^16 physical grid.
    let mut sample_rng = ChaCha8Rng::seed_from_u64(784);
    let mut points: Vec<usize> = (0..n * 256)
        .map(|_| sample_rng.random_range(0..3))
        .collect();
    // Include a fixed reference peak as well as the held-out stream. This
    // high-dynamic-range target's random samples underestimate its scale;
    // local-scale stopping never promises pointwise relative error.
    points.extend([2, 1, 0, 2, 0, 2, 0, 1, 2, 0, 1, 2, 1, 2, 0, 1]);
    let point_shape = [n, 257];
    let points = ColMajorArrayRef::new(&points, &point_shape).unwrap();
    let mut expected =
        TreeTNCachedEvaluator::new(&input, &sites, CachedEvaluatorOptions::default()).unwrap();
    let mut actual =
        TreeTNCachedEvaluator::new(&result.tree, &sites, CachedEvaluatorOptions::default())
            .unwrap();
    let expected = expected
        .evaluate_batched_typed::<f64>(points, EvaluationHint::default())
        .unwrap();
    let actual = actual
        .evaluate_batched_typed::<f64>(points, EvaluationHint::default())
        .unwrap();
    let scale = expected.iter().map(|&x| x * x).fold(0.0_f64, f64::max);
    let error = expected
        .iter()
        .zip(actual)
        .map(|(&x, y)| (x * x - y).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        error <= 10.0 * options.tolerance * scale,
        "held-out normalized max residual={:e}",
        error / scale
    );
}
