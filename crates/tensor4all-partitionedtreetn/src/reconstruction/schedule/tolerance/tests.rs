use super::*;
use crate::TreeTN;

#[test]
fn forecast_truncates_depth_and_skips_inactive_levels() {
    let keys: BTreeSet<_> = [0usize, 2, 8, 10]
        .into_iter()
        .map(|value| vec![(4, value)])
        .collect();
    let cells = BTreeMap::from([(0, keys)]);
    let observed = |depth| {
        forecast(cells.clone(), depth)
            .unwrap()
            .into_iter()
            .map(|p| (p.active, p.merging_levels_left))
            .collect::<Vec<_>>()
    };
    assert_eq!(
        observed(4),
        vec![(false, 2), (true, 2), (false, 1), (true, 1)]
    );
    assert_eq!(observed(3), vec![(false, 1), (true, 1), (false, 0)]);
    assert!(observed(0).is_empty());
    let axes = BTreeMap::from([(
        0,
        [0usize, 1]
            .into_iter()
            .flat_map(|a| (0..2).map(move |b| vec![(1, a), (1, b)]))
            .collect(),
    )]);
    let plans = forecast(axes, 2).unwrap();
    assert!(plans[0].active);
    assert!(!plans[1].active);
    assert_eq!(plans[0].merging_levels_left, 1);
}

#[test]
fn candidate_order_chooses_cheaper_singles_and_deterministic_ties() {
    assert!(!prefers(80, Kind::Joint, 40, Kind::Single(0)));
    assert!(prefers(40, Kind::Joint, 40, Kind::Single(0)));
    assert!(prefers(40, Kind::Single(0), 40, Kind::Single(1)));
    assert!(!prefers(40, Kind::Single(1), 40, Kind::Single(0)));
    assert!(prefers(100, Kind::Single(0), 100, Kind::Unsplit));
    assert!(!prefers(100, Kind::Unsplit, 100, Kind::Single(0)));
    assert!(prefers(20, Kind::Single(1), 40, Kind::Joint));
}

fn geometry(lengths: &[usize]) -> ScheduleGeometry {
    let indices: Vec<_> = (0..lengths.iter().sum())
        .map(|_| DynIndex::new_dyn(2))
        .collect();
    let mut offset = 0;
    let groups: Vec<_> = lengths
        .iter()
        .map(|&n| {
            let axis = CoordinateGroup::new(indices[offset..offset + n].to_vec());
            offset += n;
            axis
        })
        .collect();
    ScheduleGeometry::new(&indices, Some(&groups)).unwrap()
}

#[test]
fn child_planning_bounds_allocation_and_prunes_before_expansion() {
    let geom = geometry(&vec![1; 64]);
    let root = geom.root_output(0);
    let axes = geom.refining_axes(&root, 1);
    let error = geom
        .child_keys(&root, &axes, None, (2, "max_work_items"))
        .unwrap_err();
    assert!(matches!(
        error,
        PartitionedTreeTNError::ResourceLimit {
            limit: "max_work_items",
            value: 3,
            ..
        }
    ));
    let wanted = vec![vec![(1, 0); 64]];
    let children = geom
        .child_keys(&root, &axes, Some(&wanted), (1, "max_work_items"))
        .unwrap();
    assert_eq!(
        children
            .iter()
            .map(|key| key.prefix.clone())
            .collect::<Vec<_>>(),
        wanted
    );
    assert!(geom
        .child_keys(&root, &[], None, (0, "max_work_items"))
        .is_err());
    // The private key planner rejects packed-coordinate overflow rather than wrapping.
    let geom = geometry(&[usize::BITS as usize]);
    let mut key = geom.root_output(0);
    key.prefix[0] = (64, usize::MAX);
    assert!(geom
        .child_keys(&key, &[0], None, (2, "max_work_items"))
        .unwrap_err()
        .to_string()
        .contains("usize coordinate capacity"));
}

#[test]
fn candidate_limits_distinguish_planned_keys_from_retained_work() {
    let geom = geometry(&[2]);
    let mut options = MergeRefineOptions::<usize> {
        merge_policy: MergePolicy::Tolerance,
        max_work_items: 8,
        max_terms: 1,
        ..Default::default()
    };
    let context = |options| ProbeContext {
        geometry: &geom,
        center: &0,
        options,
        share: 0.0,
    };
    let shape = MergeShape {
        parents: 2,
        pairs: 1,
    };
    assert_eq!(
        child_limit(&context(&options), shape).unwrap(),
        (8, "max_work_items")
    );
    assert!(child_limit(
        &context(&options),
        MergeShape {
            parents: 0,
            pairs: 0
        }
    )
    .is_err());

    let site = DynIndex::new_dyn(2);
    let term = SubDomainTreeTN::from_treetn(
        TreeTN::from_tensors(
            vec![IdxTensor::from_dense(vec![site], vec![1.0, 2.0]).unwrap()],
            vec![0usize],
        )
        .unwrap(),
    )
    .unwrap();
    let items = BTreeMap::from([
        (
            vec![(1, 0)],
            Item {
                terms: vec![term.clone()],
            },
        ),
        (
            vec![(1, 1)],
            Item {
                terms: vec![term.clone()],
            },
        ),
    ]);
    let multiple_terms = BTreeMap::from([(
        vec![(0, 0)],
        Item {
            terms: vec![term.clone(), term],
        },
    )]);
    let error = RetainedWork::default()
        .include(&multiple_terms, &options)
        .unwrap_err();
    assert!(matches!(
        error,
        PartitionedTreeTNError::ResourceLimit {
            limit: "max_terms",
            value: 2,
            ..
        }
    ));
    let error = RetainedWork::default()
        .include(&items, &options)
        .unwrap_err();
    assert!(matches!(
        error,
        PartitionedTreeTNError::ResourceLimit {
            limit: "max_terms",
            value: 2,
            ..
        }
    ));
    options.max_terms = 8;
    options.max_work_items = 1;
    let error = RetainedWork::default()
        .include(&items, &options)
        .unwrap_err();
    assert!(matches!(
        error,
        PartitionedTreeTNError::ResourceLimit {
            limit: "max_work_items",
            value: 2,
            ..
        }
    ));
    options.max_work_items = 3;
    options.max_terms = 3;
    let mut used = RetainedWork { items: 1, terms: 1 };
    used.include(&items, &options).unwrap();
    assert_eq!((used.items, used.terms), (3, 3));
    let error = used.include(&items, &options).unwrap_err();
    assert!(matches!(
        error,
        PartitionedTreeTNError::ResourceLimit {
            limit: "max_work_items",
            value: 5,
            ..
        }
    ));
}

#[test]
fn candidate_counts_only_children_retained_after_restriction() {
    let geom = geometry(&[1]);
    let site = geom.axes[0].inputs[0].clone();
    let tensor = IdxTensor::from_dense(vec![site.clone()], vec![2.0, 3.0]).unwrap();
    let term = SubDomainTreeTN::new(
        TreeTN::from_tensors(vec![tensor], vec![0usize]).unwrap(),
        Projector::from_pairs([(site, 0)]).unwrap(),
    )
    .unwrap();
    let items = BTreeMap::from([(vec![(0, 0)], Item { terms: vec![term] })]);
    let options = MergeRefineOptions {
        merge_policy: MergePolicy::Tolerance,
        max_terms: 1,
        ..Default::default()
    };
    let context = ProbeContext {
        geometry: &geom,
        center: &0,
        options: &options,
        share: 0.0,
    };
    let root = geom.root_output(0);
    let shape = merge_shape(&items).unwrap();
    let trial = RegionTrial {
        context: &context,
        region: &root,
        items: &items,
        wanted: None,
        budget: 0.0,
        shape,
        max_children: child_limit(&context, shape).unwrap(),
        retained: RetainedWork::default(),
    };
    let chosen = trial.choose(1, &mut MergeCounters::default()).unwrap();
    assert_eq!(chosen.kind, Kind::Single(0));
    assert_eq!((chosen.retained.items, chosen.retained.terms), (1, 1));
    assert_eq!(chosen.regions.len(), 2);
    assert_eq!(
        chosen
            .regions
            .iter()
            .filter(|(_, items, _)| !items.is_empty())
            .count(),
        1
    );
    let value = &chosen
        .regions
        .iter()
        .find(|(_, items, _)| !items.is_empty())
        .unwrap()
        .1;
    let dense = value.values().next().unwrap().terms[0]
        .data()
        .to_dense()
        .unwrap();
    assert_eq!(dense.to_vec::<f64>().unwrap(), vec![2.0, 0.0]);
}

#[test]
fn over_limit_split_preserves_feasible_nonuniform_unsplit_candidate() {
    fn check<T: tensor4all_core::TensorElement + From<f64>>() {
        let mut geom = geometry(&[2]);
        geom.spectators = vec![Projector::new()];
        let root = geom.root_output(0);
        let term = SubDomainTreeTN::from_treetn(
            TreeTN::from_tensors(
                vec![IdxTensor::from_dense(
                    geom.axes[0].outputs.clone(),
                    [1.0, 2.0, 3.0, 4.0].into_iter().map(T::from).collect(),
                )
                .unwrap()],
                vec![0usize],
            )
            .unwrap(),
        )
        .unwrap();
        // The reviewer's exact reproduction: a two-bit axis, nonuniform leaves
        // (1, 0) and (2, 2), two distinct parents and no sibling pair.
        let items = BTreeMap::from([
            (vec![(1, 0)], Item::single(term.clone())),
            (vec![(2, 2)], Item::single(term.clone())),
        ]);
        let shape = merge_shape(&items).unwrap();
        assert_eq!((shape.parents, shape.pairs), (2, 0));
        for (max_work_items, max_terms, previous) in [(3, 8, 0), (8, 3, 0), (4, 4, 1)] {
            let options = MergeRefineOptions {
                merge_policy: MergePolicy::Tolerance,
                max_work_items,
                max_terms,
                ..Default::default()
            };
            let context = ProbeContext {
                geometry: &geom,
                center: &0,
                options: &options,
                share: 0.0,
            };
            let trial = RegionTrial {
                context: &context,
                region: &root,
                items: &items,
                wanted: None,
                budget: 0.0,
                shape,
                max_children: child_limit(&context, shape).unwrap(),
                retained: RetainedWork {
                    items: previous,
                    terms: previous,
                },
            };
            let error = trial
                .candidate(&[0], Kind::Single(0), &mut MergeCounters::default())
                .err()
                .unwrap();
            assert!(matches!(
                error,
                PartitionedTreeTNError::ResourceLimit { .. }
            ));
            let mut counters = MergeCounters::default();
            let chosen = trial.choose(1, &mut counters).unwrap();
            assert_eq!(chosen.kind, Kind::Unsplit);
            assert_eq!(chosen.regions.len(), 1);
            assert_eq!(chosen.regions[0].0, root);
            assert_eq!(chosen.regions[0].2, 0.0);
            assert_eq!(
                (chosen.retained.items, chosen.retained.terms),
                (previous + 2, previous + 2)
            );
            assert_eq!(counters.refined_regions, 0);
            assert_eq!(counters.stopped_regions, 1);
            assert_eq!(counters.pruned_regions, 0);
            // The rejected probe must not replace or alter either retained term.
            let expected = term.data().to_dense().unwrap();
            for item in chosen.regions[0].1.values() {
                assert_eq!(item.terms.len(), 1);
                let actual = item.terms[0].data().to_dense().unwrap();
                assert_eq!(actual.sub(&expected).unwrap().maxabs().unwrap(), 0.0);
            }
        }
    }
    check::<f64>();
    check::<num_complex::Complex64>();
}

#[test]
fn serial_joint_probes_keep_the_chosen_partition_and_its_ledger() {
    let mut geom = geometry(&[2, 2]);
    geom.spectators = vec![Projector::new()];
    let root = geom.root_output(0);
    // Empty items are reachable after exact cancellation. Their zero storage
    // makes the documented tie order observable without backend factorization.
    let items: BTreeMap<_, _> = (0..2)
        .flat_map(|a| {
            (0..2).map(move |b| (vec![(1, a), (1, b)], Item::<usize> { terms: Vec::new() }))
        })
        .collect();
    let shape = merge_shape(&items).unwrap();
    assert_eq!((shape.parents, shape.pairs), (1, 3));
    let options = MergeRefineOptions {
        merge_policy: MergePolicy::Tolerance,
        ..Default::default()
    };
    let context = ProbeContext {
        geometry: &geom,
        center: &0,
        options: &options,
        share: 0.0,
    };
    let mut counters = MergeCounters::default();
    let trial = RegionTrial {
        context: &context,
        region: &root,
        items: &items,
        wanted: None,
        budget: 1.0,
        shape,
        max_children: (4096, "max_work_items"),
        retained: RetainedWork::default(),
    };
    let selected = trial.choose(1, &mut counters).unwrap();
    assert_eq!(selected.kind, Kind::Joint);
    assert_eq!(
        selected
            .regions
            .iter()
            .map(|(key, _, _)| key.prefix.clone())
            .collect::<BTreeSet<_>>(),
        (0..2)
            .flat_map(|a| (0..2).map(move |b| vec![(1, a), (1, b)]))
            .collect()
    );
    assert!(selected.regions.iter().all(|(_, _, error)| *error == 0.0));
    assert_eq!(counters.additions, 27);
    assert_eq!(counters.refined_regions, 1);
    assert_eq!(counters.stopped_regions, 0);
    assert_eq!(counters.pruned_regions, 0);
    let mut counters = MergeCounters::default();
    let stopped = trial.choose(2, &mut counters).unwrap();
    assert_eq!(stopped.kind, Kind::Unsplit);
    assert_eq!(counters.additions, 3);
    let mut counters = MergeCounters::default();
    let (next, components) = advance(
        &context,
        BTreeMap::from([(root.clone(), items.clone())]),
        1,
        None,
        1.0,
        &mut counters,
    )
    .unwrap();
    assert!(next.is_empty());
    assert!(components.is_empty());
    assert_eq!(counters.additions, 27);

    // Each single-axis candidate fits, but planning four joint children does
    // not. Keep the first single on the deterministic tie, without aborting.
    let mut counters = MergeCounters::default();
    let selected = RegionTrial {
        max_children: (2, "max_work_items"),
        ..trial
    }
    .choose(1, &mut counters)
    .unwrap();
    assert_eq!(selected.kind, Kind::Single(0));
    assert_eq!(selected.regions.len(), 2);
    assert_eq!(counters.additions, 15);
    assert_eq!(counters.refined_regions, 1);
    assert_eq!(counters.stopped_regions, 1);
    assert_eq!(counters.pruned_regions, 0);

    // Single-axis probes each exclude a sibling, but only the selected joint
    // split contributes its three cut children. Empty retained tensors do not
    // turn a coverage count into an output-result count.
    let wanted = vec![vec![(1, 0), (1, 0)]];
    let mut counters = MergeCounters::default();
    let selected = RegionTrial {
        wanted: Some(&wanted),
        ..trial
    }
    .choose(1, &mut counters)
    .unwrap();
    assert_eq!(selected.kind, Kind::Joint);
    assert_eq!(selected.regions.len(), 1);
    assert_eq!(counters.pruned_regions, 3);
    assert_eq!(counters.refined_regions, 1);
}

#[test]
fn joint_retained_limits_reject_only_resource_failures() {
    let mut geom = geometry(&[2, 2]);
    geom.spectators = vec![Projector::new()];
    let root = geom.root_output(0);
    let term = SubDomainTreeTN::from_treetn(
        TreeTN::from_tensors(
            vec![IdxTensor::from_dense(
                geom.axes
                    .iter()
                    .flat_map(|axis| axis.outputs.clone())
                    .collect(),
                (1..=16).map(f64::from).collect(),
            )
            .unwrap()],
            vec![0usize],
        )
        .unwrap(),
    )
    .unwrap();
    let items = BTreeMap::from([
        (vec![(1, 0), (0, 0)], Item::single(term.clone())),
        (vec![(2, 2), (0, 0)], Item::single(term)),
    ]);
    let shape = merge_shape(&items).unwrap();
    for (max_work_items, max_terms) in [(5, 16), (16, 5)] {
        let options = MergeRefineOptions {
            merge_policy: MergePolicy::Tolerance,
            max_work_items,
            max_terms,
            ..Default::default()
        };
        let context = ProbeContext {
            geometry: &geom,
            center: &0,
            options: &options,
            share: 0.0,
        };
        let trial = RegionTrial {
            context: &context,
            region: &root,
            items: &items,
            wanted: None,
            budget: 0.0,
            shape,
            max_children: child_limit(&context, shape).unwrap(),
            retained: RetainedWork::default(),
        };
        let error = trial
            .candidate(&[0, 1], Kind::Joint, &mut MergeCounters::default())
            .err()
            .unwrap();
        assert!(matches!(
            error,
            PartitionedTreeTNError::ResourceLimit { value: 6, .. }
        ));
        assert!(trial
            .probe(&[0, 1], Kind::Joint, &mut MergeCounters::default())
            .unwrap()
            .is_none());
        let chosen = trial.choose(1, &mut MergeCounters::default()).unwrap();
        // Dense single-node projections retain their full payload, so these
        // splits do not pay. Resource handling must not change that decision.
        assert_eq!(chosen.kind, Kind::Unsplit);
        assert_eq!((chosen.retained.items, chosen.retained.terms), (2, 2));
        assert_eq!(chosen.regions.len(), 1);
        assert_eq!(chosen.regions[0].0.prefix, vec![(0, 0), (0, 0)]);
        // Coordinate overflow is a validation failure, not a rejected resource
        // probe. It must remain observable even when an incumbent exists.
        let mut invalid_root = root.clone();
        invalid_root.prefix[0] = (usize::BITS as usize, usize::MAX);
        let error = RegionTrial {
            region: &invalid_root,
            ..trial
        }
        .probe(&[0], Kind::Single(0), &mut MergeCounters::default())
        .err()
        .unwrap();
        assert!(matches!(
            error,
            PartitionedTreeTNError::InvalidOptions { .. }
        ));
    }
}

/// The middle cut really needs rank four, while the end cuts contain redundant
/// rank-four spaces. Tolerance mode must reduce those spaces even though the
/// largest bond is unchanged.
#[test]
fn tolerance_compression_reduces_other_bonds_when_the_maximum_stays_fixed() {
    fn check<T: tensor4all_core::TensorElement + From<f64>>() {
        let sites: Vec<_> = (0..4).map(|_| DynIndex::new_dyn(2)).collect();
        let bonds: Vec<_> = (0..3).map(|_| DynIndex::new_dyn(4)).collect();
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1130);
        let tensors = sites
            .iter()
            .enumerate()
            .map(|(i, site)| {
                let mut indices = Vec::new();
                if i > 0 {
                    indices.push(bonds[i - 1].clone());
                }
                indices.push(site.clone());
                if i < 3 {
                    indices.push(bonds[i].clone());
                }
                let size = indices.iter().map(|index| index.dim).product();
                let values: Vec<T> = (0..size)
                    .map(|_| T::from(rng.random_range(-1.0..1.0)))
                    .collect();
                IdxTensor::from_dense(indices, values).unwrap()
            })
            .collect();
        let source = SubDomainTreeTN::from_treetn(
            TreeTN::from_tensors(tensors, (0usize..4).collect()).unwrap(),
        )
        .unwrap();
        let mut counters = MergeCounters::default();
        let (candidate, residual) = compress_item(&source, &0, None, 1e-6, &mut counters).unwrap();
        assert_eq!(candidate.max_bond_dim(), source.max_bond_dim());
        assert!(
            crate::patching::logical_parameter_count(&candidate).unwrap()
                < crate::patching::logical_parameter_count(&source).unwrap()
        );
        assert_eq!(counters.compressions, 1);
        let expected = source.data().to_dense().unwrap();
        let actual = candidate.data().to_dense().unwrap();
        let error = actual.sub(&expected).unwrap().norm().unwrap();
        assert!(
            error <= residual + 1e-12,
            "error={error:e}, residual={residual:e}"
        );
        assert!(residual <= 1e-6);
    }
    check::<f64>();
    check::<num_complex::Complex64>();
}
