//! SRC regression for #1472: nonunit bonds and one-sided output leaves.
use num_complex::Complex64;
use tensor4all_core::{
    DynIndex, IdxTensor, SvdTruncationPolicy, TensorContractionLike, TensorElement,
};
use tensor4all_treetn::{
    contraction::{contract, ContractionOptions, SrcOptions},
    TreeTN,
};

fn network<S: TensorElement>(
    sites: Vec<Vec<DynIndex>>,
    edges: &[(usize, usize)],
    offset: f64,
    sample: &impl Fn(f64) -> S,
) -> TreeTN<IdxTensor, usize> {
    let mut indices = sites;
    for &(a, b) in edges {
        let bond = DynIndex::new_dyn(2);
        indices[a].push(bond.clone());
        indices[b].push(bond);
    }
    let tensors = indices
        .into_iter()
        .enumerate()
        .map(|(node, indices)| {
            let len = indices.iter().map(|i| i.dim).product();
            let values = (0..len)
                .map(|i| sample(i as f64 + offset + node as f64))
                .collect();
            IdxTensor::from_dense(indices, values).unwrap()
        })
        .collect();
    TreeTN::from_tensors(tensors, vec![0, 1, 2, 3]).unwrap()
}

fn check_one_sided_outputs<S: TensorElement>(sample: impl Fn(f64) -> S) {
    for star in [true, false] {
        let mid = DynIndex::new_dyn(2);
        let shared = DynIndex::new_dyn(2);
        // Full index identity matters even with equal logical IDs.
        let row = mid.prime();
        let x = DynIndex::new_dyn(2);
        let y = x.prime();
        let edges = if star {
            vec![(0, 1), (0, 2), (0, 3)]
        } else {
            vec![(2, 1), (1, 0), (0, 3)]
        };
        let left = network(
            vec![
                vec![mid.clone()],
                vec![shared.clone()],
                vec![x.clone()],
                vec![y.clone()],
            ],
            &edges,
            0.2,
            &sample,
        );
        let right = network(
            vec![vec![mid, row.clone()], vec![shared], vec![], vec![]],
            &edges,
            1.4,
            &sample,
        );
        let reference = left
            .to_dense()
            .unwrap()
            .contract_pair(&right.to_dense().unwrap())
            .unwrap();
        assert_eq!(reference.dims().iter().product::<usize>(), 8);
        for (a, b) in [(&left, &right), (&right, &left)] {
            // Endpoint centers use the chain specialization; an interior center
            // uses the rooted-tree recurrence. A star always uses that recurrence.
            let centers = if star { vec![0] } else { vec![2, 3, 0] };
            for center in centers {
                for adaptive in [false, true] {
                    let src = if adaptive {
                        SrcOptions::adaptive(1e-12, 8)
                            .with_min_rank(2)
                            .with_rank_increment(3)
                    } else {
                        SrcOptions::fixed()
                    }
                    .with_seed(1472);
                    let options = ContractionOptions::src()
                        .with_max_bond_dim(8)
                        .with_svd_policy(SvdTruncationPolicy::new(0.0))
                        .with_src_options(src);
                    let result = contract(a, b, &center, options).unwrap();
                    let dense = result.to_dense().unwrap();
                    for index in [&row, &x, &y] {
                        assert!(dense.indices().contains(index));
                    }
                    let residual = dense.sub(&reference).unwrap().maxabs().unwrap();
                    assert!(
                        residual < 1e-11,
                        "star={star}, center={center}, adaptive={adaptive}: residual {residual:e}"
                    );
                }
            }
        }
    }
}

#[test]
fn src_one_sided_outputs_real() {
    check_one_sided_outputs(f64::cos);
}

#[test]
fn src_one_sided_outputs_complex() {
    check_one_sided_outputs(|v| Complex64::new(v.cos(), (v * 0.7).sin()));
}
