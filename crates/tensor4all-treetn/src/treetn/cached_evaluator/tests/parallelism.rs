//! The batch entrance hands the caller's `Parallelism` to the per-point centre
//! kernel, which shards its points over that team and nowhere else.
//!
//! `point_shards_for_test` is set by the thread that calls the kernel from
//! the value it was given, so these tests observe the argument's effect; no
//! global, thread-local or ambient pool is read to decide the shard count.

use super::*;

fn pool(threads: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
}

fn three_node_chain() -> (Vec<DynIndex>, TreeTN<IdxTensor, usize>) {
    let sites: Vec<DynIndex> = [2usize, 8, 2]
        .iter()
        .map(|&d| DynIndex::new_dyn(d))
        .collect();
    let b0 = DynIndex::new_dyn(6);
    let b1 = DynIndex::new_dyn(6);
    let value = |k: usize| ((k * 37 % 101) as f64 - 50.0) / 17.0;
    let take = |start: usize, n: usize| (start..start + n).map(value).collect::<Vec<_>>();
    let n0 = IdxTensor::from_dense(vec![sites[0].clone(), b0.clone()], take(0, 12)).unwrap();
    let n1 = IdxTensor::from_dense(vec![b0, sites[1].clone(), b1.clone()], take(100, 6 * 8 * 6))
        .unwrap();
    let n2 = IdxTensor::from_dense(vec![b1, sites[2].clone()], take(500, 12)).unwrap();
    let tree = TreeTN::from_tensors(vec![n0, n1, n2], vec![0usize, 1, 2]).unwrap();
    (sites, tree)
}

fn kernel_inputs(points: usize) -> (Vec<f64>, Vec<usize>, Vec<RawCenterComponent<f64>>) {
    let core: Vec<f64> = (0..25).map(|k| (k as f64) * 0.25 - 3.0).collect();
    let components = [2usize, 1]
        .into_iter()
        .map(|axis| RawCenterComponent {
            axis,
            dim: 5,
            point_to_assignment: (0..points).collect(),
            values: (0..points * 5)
                .map(|k| ((k * 7 + axis) % 11) as f64 - 5.0)
                .collect(),
        })
        .collect();
    (core, vec![0; points], components)
}

#[test]
fn raw_center_kernel_shards_by_the_budget_it_is_given() {
    let (core, physical, components) = kernel_inputs(23); // 6 lane blocks
    let run = |parallelism| {
        contract_raw_center(&core, &[1, 5, 5], 0, &physical, &components, parallelism).unwrap()
    };
    let serial = run(Parallelism::Serial);
    assert_eq!(point_shards_for_test(), 1);
    let pool = pool(8);
    for (budget, shards) in [(1, 1), (2, 2), (3, 3), (4, 3), (6, 6), (8, 6)] {
        let values = run(Parallelism::Pool {
            pool: &pool,
            budget,
        });
        assert_eq!(point_shards_for_test(), shards, "budget {budget}");
        assert_eq!(
            values.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            serial.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "budget {budget}"
        );
    }
}

/// Both centre routes of the entrance: the raw internal-centre fold (centre
/// configured, no hint) and the hinted edge-cut assembly. Each shards its
/// points by exactly the budget of the `Parallelism` handed to the entrance,
/// and each returns the bits of the sequential call.
#[test]
fn batch_entrance_passes_the_callers_parallelism_to_the_per_point_kernels() {
    let (sites, tree) = three_node_chain();
    let n = 64usize; // 16 lane blocks
    let mut data = Vec::with_capacity(3 * n);
    for p in 0..n {
        data.extend([p % 2, (p * 3) % 8, (p / 3) % 2]);
    }
    let shape = [3, n];
    let points = ColMajorArrayRef::new(&data, &shape).unwrap();
    let pool = pool(4);
    for hinted in [false, true] {
        let hint = if hinted {
            EvaluationHint::around(1)
        } else {
            EvaluationHint::default()
        };
        let new = || {
            let options = CachedEvaluatorOptions::<usize> {
                center: Some(1),
                ..Default::default()
            };
            TreeTNCachedEvaluator::new(&tree, &sites, options).unwrap()
        };
        let bits = |values: Vec<f64>| values.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        let serial = new()
            .evaluate_batched_typed_with_parallelism::<f64>(
                points,
                hint.clone(),
                Parallelism::Serial,
            )
            .unwrap();
        assert_eq!(point_shards_for_test(), 1, "hinted {hinted}");
        let default_entrance = new()
            .evaluate_batched_typed::<f64>(points, hint.clone())
            .unwrap();
        assert_eq!(point_shards_for_test(), 1, "the plain entrance is Serial");
        assert_eq!(bits(default_entrance), bits(serial.clone()));
        for budget in [2usize, 4] {
            let values = new()
                .evaluate_batched_typed_with_parallelism::<f64>(
                    points,
                    hint.clone(),
                    Parallelism::Pool {
                        pool: &pool,
                        budget,
                    },
                )
                .unwrap();
            assert_eq!(
                point_shards_for_test(),
                budget,
                "hinted {hinted} budget {budget}"
            );
            assert_eq!(
                bits(values),
                bits(serial.clone()),
                "hinted {hinted} budget {budget}"
            );
        }
    }
}
