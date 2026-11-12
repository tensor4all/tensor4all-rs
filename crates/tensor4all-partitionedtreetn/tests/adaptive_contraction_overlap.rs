//! Overlapping pair outputs retain original-task selection and exact values.
//!
//! The two original groups are `{x=0}` and `{y=0}`, although the operand
//! partitions are disjoint along the contracted `k`. The analytical product is
//! `(3 [x=0] + 2 [y=0]) * (z+1)`, with an optional common complex scale.

use num_complex::Complex64;
use tensor4all_core::{DynIndex, IdxTensor, TensorElement};
use tensor4all_partitionedtreetn::{
    contract_adaptive, contract_adaptive_selected, BudgetAllocation, PartitionedTreeTN,
    PartitionedTreeTNError, PatchSplitStrategy, PatchingOptions, Projector, SubDomainTreeTN,
    AUTOMATIC_REFINEMENT_LIMIT,
};
use tensor4all_treetn::{contraction::ContractionOptions, SumEngine, TreeTN};

struct Fixture {
    left: PartitionedTreeTN,
    right: PartitionedTreeTN,
    x: DynIndex,
    y: DynIndex,
    z: DynIndex,
    output: Vec<DynIndex>,
}

impl Fixture {
    fn new<T: TensorElement + From<f64>>(
        branched: bool,
        x_dim: usize,
        scale: impl Fn(f64) -> T,
    ) -> Self {
        let k = DynIndex::new_dyn(2);
        let x = DynIndex::new_dyn(x_dim);
        // Full identity matters: x and y share an ID but have different primes.
        let y = x.prime();
        let z = DynIndex::new_dyn(2);
        let (left, right, output) = if branched {
            let mut trees = Vec::new();
            let u = DynIndex::new_dyn(1);
            let v = DynIndex::new_dyn(1);
            let w = DynIndex::new_dyn(1);
            for is_left in [true, false] {
                let bonds: Vec<_> = (0..3).map(|_| DynIndex::new_bond(1).unwrap()).collect();
                let mut root_indices = vec![k.clone()];
                root_indices.extend(bonds.iter().cloned());
                let mut tensors = vec![IdxTensor::from_dense(
                    root_indices,
                    if is_left {
                        vec![scale(1.0); 2]
                    } else {
                        vec![T::from(2.0), T::from(3.0)]
                    },
                )
                .unwrap()];
                let sites = if is_left { [&x, &u, &z] } else { [&v, &y, &w] };
                for (bond, site) in bonds.into_iter().zip(sites) {
                    let values = (0..site.dim)
                        .map(|coordinate| {
                            T::from(if site == &z {
                                (coordinate + 1) as f64
                            } else {
                                1.0
                            })
                        })
                        .collect();
                    tensors.push(IdxTensor::from_dense(vec![bond, site.clone()], values).unwrap());
                }
                trees.push(TreeTN::from_tensors(tensors, vec![0usize, 1, 2, 3]).unwrap());
            }
            (
                trees.remove(0),
                trees.remove(0),
                vec![x.clone(), y.clone(), z.clone(), u, v, w],
            )
        } else {
            let left_values = (0..2)
                .flat_map(|z| {
                    let scale = &scale;
                    (0..x_dim).flat_map(move |_| [scale((z + 1) as f64); 2])
                })
                .collect();
            let right_values = (0..x_dim)
                .flat_map(|_| [T::from(2.0), T::from(3.0)])
                .collect();
            (
                TreeTN::from_tensors(
                    vec![
                        IdxTensor::from_dense(vec![k.clone(), x.clone(), z.clone()], left_values)
                            .unwrap(),
                    ],
                    vec![0usize],
                )
                .unwrap(),
                TreeTN::from_tensors(
                    vec![IdxTensor::from_dense(vec![k.clone(), y.clone()], right_values).unwrap()],
                    vec![0usize],
                )
                .unwrap(),
                vec![x.clone(), y.clone(), z.clone()],
            )
        };
        let patch = |tree: &TreeTN<IdxTensor, usize>, entries| {
            SubDomainTreeTN::new(tree.clone(), Projector::from_pairs(entries).unwrap()).unwrap()
        };
        let left = PartitionedTreeTN::from_subdomains(vec![
            patch(&left, vec![(k.clone(), 0)]),
            patch(&left, vec![(k.clone(), 1), (x.clone(), 0)]),
        ])
        .unwrap();
        let right = PartitionedTreeTN::from_subdomains(vec![
            patch(&right, vec![(k.clone(), 0), (y.clone(), 0)]),
            patch(&right, vec![(k, 1)]),
        ])
        .unwrap();
        Self {
            left,
            right,
            x,
            y,
            z,
            output,
        }
    }

    fn key(&self, is_x: bool) -> Projector {
        Projector::from_pairs([(if is_x { &self.x } else { &self.y }.clone(), 0)]).unwrap()
    }

    fn expected<T: TensorElement>(
        &self,
        scale: impl Fn(f64) -> T,
        x_term: bool,
        y_term: bool,
    ) -> IdxTensor {
        let values = (1..=2)
            .flat_map(|z| {
                let scale = &scale;
                (0..self.y.dim).flat_map(move |y| {
                    (0..self.x.dim).map(move |x| {
                        scale(
                            z as f64
                                * (if x_term && x == 0 { 3.0 } else { 0.0 }
                                    + if y_term && y == 0 { 2.0 } else { 0.0 }),
                        )
                    })
                })
            })
            .collect();
        IdxTensor::from_dense(self.output.clone(), values).unwrap()
    }
}

fn options(fixture: &Fixture, cap: Option<usize>, engine: SumEngine) -> PatchingOptions {
    PatchingOptions {
        cutoff: 0.0,
        allocation: BudgetAllocation::Volume,
        max_bond_dim: cap,
        patch_order: vec![fixture.z.clone()],
        split_strategy: PatchSplitStrategy::Sequential,
        sum_engine: engine,
    }
}

fn assert_values(actual: &PartitionedTreeTN, expected: &IdxTensor) {
    let dense = actual.to_treetn().unwrap().to_dense().unwrap();
    let error = dense.sub(expected).unwrap().maxabs().unwrap();
    assert!(error < 1e-10, "dense residual {error:e}");
}

fn check_overlap<T: TensorElement + From<f64>>(scale: impl Fn(f64) -> T + Copy) {
    for branched in [false, true] {
        let fixture = Fixture::new(branched, 2, scale);
        let method = ContractionOptions::zipup();
        for engine in [SumEngine::Binary, SumEngine::Nary] {
            for cap in [None, Some(1)] {
                let options = options(&fixture, cap, engine);
                let full = contract_adaptive(&fixture.left, &fixture.right, &0, &method, &options)
                    .unwrap();
                assert_values(&full, &fixture.expected(scale, true, true));
                assert_eq!(full.len(), if cap.is_some() { 6 } else { 3 });
                assert!(full
                    .iter()
                    .all(|(key, _)| key.is_projected_at(&fixture.x)
                        && key.is_projected_at(&fixture.y)));

                let wanted = [fixture.key(true), fixture.key(false)];
                let all = contract_adaptive_selected(
                    &fixture.left,
                    &fixture.right,
                    &0,
                    &method,
                    &options,
                    &wanted,
                )
                .unwrap();
                assert_values(&all, &fixture.expected(scale, true, true));
                assert_eq!(
                    full.iter()
                        .map(|(key, _)| key)
                        .collect::<std::collections::HashSet<_>>(),
                    all.iter().map(|(key, _)| key).collect()
                );

                for select_x in [true, false] {
                    let key = fixture.key(select_x);
                    let selected = contract_adaptive_selected(
                        &fixture.left,
                        &fixture.right,
                        &0,
                        &method,
                        &options,
                        &[key.clone(), key.clone()],
                    )
                    .unwrap();
                    assert_values(&selected, &fixture.expected(scale, select_x, !select_x));
                    assert_eq!(selected.len(), if cap.is_some() { 2 } else { 1 });
                    assert!(selected.iter().all(|(child, _)| child.is_subset_of(&key)));
                }
            }
        }

        let reverse = |partition: &PartitionedTreeTN| {
            let mut patches: Vec<_> = partition.values().cloned().collect();
            patches.reverse();
            PartitionedTreeTN::from_subdomains(patches).unwrap()
        };
        let reversed_left = reverse(&fixture.left);
        let reversed_right = reverse(&fixture.right);
        let options = options(&fixture, None, SumEngine::Binary);
        let reversed =
            contract_adaptive(&reversed_left, &reversed_right, &0, &method, &options).unwrap();
        assert_values(&reversed, &fixture.expected(scale, true, true));
        // A descendant of an original key is not itself an original selected key.
        let descendant = fixture.key(true).intersection(&fixture.key(false)).unwrap();
        let absent = contract_adaptive_selected(
            &fixture.left,
            &fixture.right,
            &0,
            &method,
            &options,
            &[descendant],
        )
        .unwrap();
        assert!(absent.is_empty());
    }
}

#[test]
fn real_overlap_and_selection_match_analytical_values_on_branched_and_edgeless_trees() {
    check_overlap(|value| value);
}

#[test]
fn complex_overlap_and_selection_match_analytical_values_on_branched_and_edgeless_trees() {
    check_overlap(|value| Complex64::new(value, 0.5 * value));
}

#[test]
fn oversized_overlap_grid_returns_typed_error_but_single_selected_key_needs_no_refinement() {
    let fixture = Fixture::new(false, 65, |value| value);
    assert!(fixture.x.dim * fixture.y.dim > AUTOMATIC_REFINEMENT_LIMIT);
    let method = ContractionOptions::zipup();
    let options = options(&fixture, None, SumEngine::Binary);
    assert!(matches!(
        contract_adaptive(&fixture.left, &fixture.right, &0, &method, &options),
        Err(PartitionedTreeTNError::RefinementTooLarge { atoms, limit })
            if atoms == fixture.x.dim * fixture.y.dim && limit == AUTOMATIC_REFINEMENT_LIMIT
    ));
    // y has the same dimension as x. Selecting x=0 leaves y free and must
    // exclude the original y=0 group's contribution at x=y=0.
    let selected = contract_adaptive_selected(
        &fixture.left,
        &fixture.right,
        &0,
        &method,
        &options,
        &[fixture.key(true)],
    )
    .unwrap();
    assert_eq!(selected.len(), 1);
    let key = fixture.key(true);
    let patch = selected.get(&key).unwrap();
    assert!((patch.norm_squared().unwrap() - 45.0 * fixture.y.dim as f64).abs() < 1e-7);
}
