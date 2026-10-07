//! Continued optimization must recover after a bond cap is raised (#833).

use anyhow::Result;
use num_complex::Complex64;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use tensor4all_core::{DynIndex, IdxTensor, Scalar, TensorElement};
use tensor4all_treetci::{
    optimize_with_proposer_with_rng, to_treetn, DefaultProposer, GlobalIndexBatch, TreeTCI2,
    TreeTciEdge, TreeTciGraph, TreeTciOptions, TreeTciTermination,
};
use tensor4all_treetn::TreeTN;

fn graph() -> TreeTciGraph {
    TreeTciGraph::new(
        7,
        &[
            TreeTciEdge::new(0, 1),
            TreeTciEdge::new(1, 2),
            TreeTciEdge::new(0, 3),
            TreeTciEdge::new(3, 4),
            TreeTciEdge::new(0, 5),
            TreeTciEdge::new(5, 6),
        ],
    )
    .unwrap()
}

fn reference<T: TensorElement>(tn: &TreeTN<IdxTensor, usize>, values: Vec<T>) -> IdxTensor {
    let indices: Vec<DynIndex> = (0..7)
        .map(|site| tn.site_space(&site).unwrap().iter().next().unwrap().clone())
        .collect();
    IdxTensor::from_dense(indices, values).unwrap()
}

macro_rules! continued_schedule {
    ($name:ident, $scalar:ty, $phase:expr) => {
        #[test]
        fn $name() {
            for global in [false, true] {
                let dims = vec![1, 4, 2, 4, 2, 4, 2];
                let value = |point: &[usize]| {
                    let x = point[1] + 4 * point[2];
                    let y = point[3] + 4 * point[4];
                    let z = point[5] + 4 * point[6];
                    let s = (x + y + z) % 8;
                    <$scalar as Scalar>::from_f64(
                        [2.0, 0.125, 0.5, -0.25, 0.875, 0.375, -0.625, 1.25][s],
                    ) * $phase
                };
                let evaluate = |batch: GlobalIndexBatch<'_>| -> Result<Vec<$scalar>> {
                    let mut point = vec![0; 7];
                    let mut values = Vec::with_capacity(batch.n_points());
                    for p in 0..batch.n_points() {
                        for (site, slot) in point.iter_mut().enumerate() {
                            *slot = batch.get(site, p).unwrap();
                        }
                        values.push(value(&point));
                    }
                    Ok(values)
                };
                let make_state = || {
                    let mut state = TreeTCI2::<$scalar>::new(dims.clone(), graph()).unwrap();
                    // Seed independent variations on each arm so local-only sweeps
                    // can discover the junction rank without global search.
                    let pivots: Vec<_> = (0..4)
                        .flat_map(|x| {
                            (0..4).flat_map(move |y| (0..4).map(move |z| vec![0, x, 0, y, 0, z, 0]))
                        })
                        .collect();
                    state.add_global_pivots(&pivots).unwrap();
                    state.max_sample_value = value(&[0; 7]).abs_val();
                    state
                };
                let mut state = make_state();
                let mut rng = ChaCha8Rng::seed_from_u64(7);
                let options = TreeTciOptions {
                    tolerance: 1e-13,
                    max_iter: 12,
                    max_bond_dim: Some(4),
                    enable_global_pivots: global,
                    nsearch: 8,
                    ..Default::default()
                };
                let capped = optimize_with_proposer_with_rng(
                    &mut state,
                    evaluate,
                    &options,
                    &DefaultProposer,
                    &mut rng,
                )
                .unwrap();
                assert_eq!(capped.termination, TreeTciTermination::MaxBondDimension);
                assert_eq!(state.max_bond_dim(), 4);
                let capped_tn = to_treetn(&state, evaluate, None).unwrap();
                let capped_dense = capped_tn.to_dense().unwrap();
                let mut points = vec![0; 7];
                let mut values = Vec::with_capacity(512);
                for _ in 0..512 {
                    values.push(value(&points));
                    for (slot, &dim) in points.iter_mut().zip(&dims) {
                        *slot += 1;
                        if *slot < dim {
                            break;
                        }
                        *slot = 0;
                    }
                }
                let capped_ref = reference(&capped_tn, values.clone());
                let scale = capped_ref.maxabs().unwrap();
                assert!(capped_dense.sub(&capped_ref).unwrap().maxabs().unwrap() > 1e-12 * scale);
                let options = TreeTciOptions {
                    max_bond_dim: None,
                    ..options
                };
                let continued = optimize_with_proposer_with_rng(
                    &mut state,
                    evaluate,
                    &options,
                    &DefaultProposer,
                    &mut rng,
                )
                .unwrap();
                assert_eq!(continued.termination, TreeTciTermination::Converged);
                assert_eq!(state.max_bond_dim(), 8);
                let warm = to_treetn(&state, evaluate, None).unwrap();
                let warm_dense = warm.to_dense().unwrap();
                let warm_ref = reference(&warm, values.clone());
                let warm_error = warm_dense.sub(&warm_ref).unwrap().maxabs().unwrap();
                assert!(
                    warm_error <= 1e-12 * scale,
                    "continued residual {warm_error:e}"
                );

                let mut fresh_state = make_state();
                let mut fresh_rng = ChaCha8Rng::seed_from_u64(7);
                let fresh_result = optimize_with_proposer_with_rng(
                    &mut fresh_state,
                    evaluate,
                    &options,
                    &DefaultProposer,
                    &mut fresh_rng,
                )
                .unwrap();
                assert_eq!(fresh_result.termination, TreeTciTermination::Converged);
                let fresh = to_treetn(&fresh_state, evaluate, None).unwrap();
                let mut fresh_dense = fresh.to_dense().unwrap();
                let fresh_ref = reference(&fresh, values);
                let fresh_error = fresh_dense.sub(&fresh_ref).unwrap().maxabs().unwrap();
                assert!(
                    fresh_error <= 1e-12 * scale,
                    "fresh residual {fresh_error:e}"
                );
                for site in 0..7 {
                    let old = fresh.site_space(&site).unwrap().iter().next().unwrap();
                    let new = warm.site_space(&site).unwrap().iter().next().unwrap();
                    fresh_dense = fresh_dense.replaceind(old, new).unwrap();
                }
                let difference = warm_dense.sub(&fresh_dense).unwrap().maxabs().unwrap();
                assert!(
                    difference <= 1e-12 * scale,
                    "continued/fresh difference {difference:e}"
                );
            }
        }
    };
}
continued_schedule!(continued_real_comb_recovers_from_a_cap, f64, 1.0_f64);
continued_schedule!(
    continued_complex_comb_recovers_from_a_cap,
    Complex64,
    Complex64::new(1.0, -0.25)
);
