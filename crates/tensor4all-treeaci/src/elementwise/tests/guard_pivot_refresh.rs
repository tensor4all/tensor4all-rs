//! Guard discoveries must be able to change a previously admissible cross.
use super::*;
const FACTORS: [[[f64; 3]; 5]; 3] = [
    [
        [-0.5251262723109726, 0.08853644648949333, 0.0],
        [-0.4005388404334278, 0.6310539453808466, -0.5025368962179297],
        [
            -0.010589030524726706,
            -0.95196326747172,
            0.38646437445934145,
        ],
        [
            -0.25432596503013194,
            -0.37682367771999914,
            0.10558933118980507,
        ],
        [0.38766731646406427, 0.9190846535281776, 0.5651649242239709],
    ],
    [
        [0.35589692804801576, -0.6289203263646921, 0.0],
        [0.6808427001086033, 0.05435054812483919, 0.48058722348566096],
        [-0.7255733952096364, 0.5477719167363593, -0.6751157134138754],
        [
            -0.11123756279465002,
            0.8857090804796508,
            -0.46083133292051626,
        ],
        [
            -0.33892511711408035,
            -0.8126641770231315,
            -0.496073490572706,
        ],
    ],
    [
        [0.7066786207519287, -0.2096997745029765, 0.0],
        [0.7663707041695349, 0.8652062436469272, -0.6606531670872284],
        [-0.6110741273524618, 0.8803300319431235, 0.17551104823835662],
        [0.12105134623218161, 0.8642451062138901, 0.7645082687759834],
        [0.8500149907724852, -0.7873493142635386, 0.43566220127547606],
    ],
];

fn check_guard_refresh<T: crate::TreeAciScalar>() {
    let dims = [2, 3, 3, 3, 3];
    let edges = [(0, 1), (1, 2), (1, 3), (3, 4)];
    let physical = dims.map(DynIndex::new_dyn);
    let inputs = (0..5)
        .map(|coordinate| {
            let bonds = edges.map(|_| DynIndex::new_dyn(1));
            let tensors = (0..5)
                .map(|node| {
                    let mut indices = vec![physical[node].clone()];
                    for (edge, &(a, b)) in edges.iter().enumerate() {
                        if a == node || b == node {
                            indices.push(bonds[edge].clone());
                        }
                    }
                    let values = (0..dims[node])
                        .map(|x| T::from_f64(if node == coordinate { x as f64 } else { 1.0 }))
                        .collect();
                    IdxTensor::from_dense(indices, values).unwrap()
                })
                .collect();
            TreeTN::from_tensors(tensors, (0..5).collect()).unwrap()
        })
        .collect::<Vec<_>>();
    let target = |point: &[usize]| {
        FACTORS
            .iter()
            .enumerate()
            .map(|(term, factors)| {
                (if term == 2 { 0.9e-9 } else { 1.0 })
                    * factors
                        .iter()
                        .zip(point)
                        .map(|(f, &x)| f[x])
                        .product::<f64>()
            })
            .sum::<f64>()
    };
    let options = TreeAciOptions {
        tolerance: 1e-9,
        max_bond_dim: Some(3),
        rng_seed: 17,
        min_sweeps: 2,
        max_sweeps: 20,
        ..TreeAciOptions::default()
    };
    let result = tree_elementwise_batched::<T, _, _>(
        |batch, output| {
            for (p, value) in output.iter_mut().enumerate() {
                let point = (0..5)
                    .map(|i| batch.get(i, p).map(|x| x.abs_val().round() as usize))
                    .collect::<crate::Result<Vec<_>>>()?;
                *value = T::from_f64(target(&point));
            }
            Ok(())
        },
        &inputs,
        &options,
    )
    .unwrap();
    assert_eq!(result.termination, TreeAciTermination::Converged);
    assert!(
        result.max_ranks.len() <= 6,
        "passes: {}",
        result.max_ranks.len()
    );
    assert!(result.global_pivots_found.iter().any(|&n| n > 0));
    assert!(result
        .global_pivots_found
        .iter()
        .rev()
        .take(2)
        .all(|&n| n == 0));
    assert!(
        result.diagnostics.evaluated_points <= 2000,
        "points: {}",
        result.diagnostics.evaluated_points
    );
    assert!(result
        .diagnostics
        .edge_ranks
        .iter()
        .all(|(_, _, rank)| *rank <= 3));
    let values = (0..dims.iter().product())
        .map(|mut flat| {
            let point = dims.map(|d| {
                let x = flat % d;
                flat /= d;
                x
            });
            T::from_f64(target(&point))
        })
        .collect();
    let expected = IdxTensor::from_dense(physical.to_vec(), values).unwrap();
    let residual = result
        .tree
        .to_dense()
        .unwrap()
        .sub(&expected)
        .unwrap()
        .maxabs()
        .unwrap()
        / expected.maxabs().unwrap();
    assert!(
        residual <= options.tolerance,
        "relative dense max residual: {residual}"
    );
}
#[test]
fn guard_discoveries_refresh_preferred_pivots_on_a_mixed_capacity_tree() {
    check_guard_refresh::<f64>();
}
#[test]
fn complex_guard_discoveries_refresh_preferred_pivots_on_a_mixed_capacity_tree() {
    check_guard_refresh::<Complex64>();
}
