use super::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Bounded-rank complex input, including three untransformed spectator sites.
fn random_input(n_fourier: usize) -> (Vec<DynIndex>, TreeTN<IdxTensor, usize>) {
    let n = n_fourier + 3;
    let mut sites: Vec<_> = (0..n).map(|_| DynIndex::new_dyn(2)).collect();
    // Full identity must distinguish the selected site from this spectator.
    sites[n_fourier] = sites[0].prime();
    let bonds: Vec<_> = (1..n)
        .map(|i| DynIndex::new_dyn(4.min(1 << i).min(1 << (n - i))))
        .collect();
    let mut rng = ChaCha8Rng::seed_from_u64(0x5eed_0000 + n_fourier as u64);
    let tensors = sites
        .iter()
        .enumerate()
        .map(|(i, site)| {
            let mut indices = Vec::new();
            if i > 0 {
                indices.push(bonds[i - 1].clone());
            }
            indices.push(site.clone());
            if i + 1 < n {
                indices.push(bonds[i].clone());
            }
            let size = indices.iter().map(|index| index.dim).product();
            let data: Vec<_> = (0..size)
                .map(|_| Complex64::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)))
                .collect();
            IdxTensor::from_dense(indices, data).unwrap()
        })
        .collect();
    (
        sites,
        TreeTN::from_tensors(tensors, (0..n).collect()).unwrap(),
    )
}

pub(super) fn check_levels(report: &MergeRefineReport) {
    assert_eq!(report.levels.len(), report.level_count);
    assert_eq!(report.merge_policy, MergePolicy::Tolerance);
    let mut remaining = report.absolute_tolerance - report.application_error_bound;
    for (position, level) in report.levels.iter().enumerate() {
        assert_eq!(level.level, position + 1);
        assert!(level.merging_levels_left.is_some());
        assert_eq!(level.remaining_before, remaining);
        assert!(level.error_bound <= level.allocated_error, "{level:?}");
        assert_eq!(level.budget_spent, level.error_bound);
        assert_eq!(
            level.unused_allocation,
            (level.allocated_error - level.error_bound).max(0.0)
        );
        remaining = (remaining - level.error_bound).max(0.0);
        assert_eq!(level.remaining_after, remaining);
    }
    assert_eq!(
        report
            .levels
            .iter()
            .fold(report.application_error_bound, |total, level| total
                + level.error_bound),
        report.error_bound
    );
    assert!(
        report.error_bound <= report.absolute_tolerance,
        "{report:?}"
    );
}

fn check_dft(
    result: &MergeRefineResult<usize>,
    tree: &TreeTN<IdxTensor, usize>,
    sites: &[DynIndex],
    selected: &[DynIndex],
) {
    let input = tree.to_dense().unwrap();
    let msb_order: Vec<_> = sites.iter().rev().cloned().collect();
    let values = reorder(
        &input.to_vec::<Complex64>().unwrap(),
        input.indices(),
        &msb_order,
    );
    let (indices, actual) = dense_of_regions(result);
    let expected = dft_oracle(&indices, sites, selected, &values);
    let error = l2_error(&actual, &expected);
    assert!(
        error <= result.report().error_bound + 1e-10,
        "L2 error={error:e}, report={:?}",
        result.report()
    );
    assert!(max_error(&actual, &expected) <= result.report().error_bound + 1e-10);
}

#[test]
fn tolerance_merges_compress_without_a_rank_goal() {
    for bits in [3, 4] {
        let (sites, tree) = random_input(bits);
        let selected = &sites[..bits];
        let preimage = dyadic_input_leaves(&tree, selected);
        let operator = quantics_fourier_operator(bits, FourierOptions::default()).unwrap();
        let exact = schedule_exact(&preimage, &operator, selected, None, 4096);
        let result = schedule_merge_refine(
            &preimage,
            &0,
            &operator,
            selected,
            &SubsetOperatorOptions { unitary: true },
            ReconstructionTolerance {
                rtol: 1e-8,
                atol: 0.0,
            },
            &MergeRefineOptions {
                merge_policy: MergePolicy::Tolerance,
                ..Default::default()
            },
        )
        .unwrap();
        let report = result.report();
        check_levels(report);
        assert!(report.compressions > 0, "{report:?}");
        assert!(
            report.logical_parameters < exact.report().logical_parameters,
            "{report:?}"
        );
        assert_eq!(report.term_count, report.region_count);
        assert!(Projector::are_disjoint(
            &result.regions().map(|(p, _)| p.clone()).collect::<Vec<_>>()
        ));
        assert!(result.clone().into_partition().is_ok());
        check_dft(&result, &tree, &sites, selected);
    }
}

#[test]
fn tolerance_merges_bound_cancellation_and_charge_only_kept_candidates() {
    let sites: Vec<_> = (0..5).map(|_| DynIndex::new_dyn(2)).collect();
    let values: Vec<_> = (0..32)
        .map(|x| {
            let k = (x >> 2) as f64;
            let spectator = (x & 3) as f64;
            // An independent, much smaller spectator component makes the charged
            // approximation observable while the larger cancelled signal is needed.
            let parity = if x & 3 == 0 || x & 3 == 3 { 1.0 } else { -1.0 };
            Complex64::new(
                1.0 + 2e-3 * (k * 0.7).sin() * (spectator + 1.0) + 1e-6 * (k * 1.7).sin() * parity,
                1e-3 * (k * 1.1).cos() * (spectator - 1.0),
            )
        })
        .collect();
    let tree = mps(&sites, &values);
    let selected = &sites[..3];
    let preimage = dyadic_input_leaves(&tree, selected);
    let operator = quantics_fourier_operator(3, FourierOptions::default()).unwrap();
    let result = schedule_merge_refine(
        &preimage,
        &0,
        &operator,
        selected,
        &SubsetOperatorOptions { unitary: true },
        ReconstructionTolerance {
            rtol: 1e-4,
            atol: 0.0,
        },
        &MergeRefineOptions {
            merge_policy: MergePolicy::Tolerance,
            ..Default::default()
        },
    )
    .unwrap();
    check_levels(result.report());
    assert!(result.report().compressions > 0);
    assert!(
        result.report().error_bound > 1e-10,
        "the case must charge an approximation: {:?}",
        result.report()
    );
    // On nonzero frequencies the constant cancels; check every coordinate once
    // against the independently constructed dense DFT, including those small outputs.
    let (indices, actual) = dense_of_regions(&result);
    let expected = dft_oracle(&indices, &sites, selected, &values);
    let error = max_error(&actual, &expected);
    assert!(
        error <= result.report().error_bound + 1e-12,
        "pointwise error={error:e}, {:?}",
        result.report()
    );
    assert!(l2_error(&actual, &expected) <= result.report().error_bound + 1e-12);
}

#[test]
fn tolerance_merges_report_inactive_levels_early_stops_and_zero_budget() {
    let (sites, tree) = random_input(3);
    let selected = &sites[..3];
    let full = SubDomainTreeTN::from_treetn(tree.clone()).unwrap();
    let leaves = [(1, 0), (2, 2), (3, 6), (3, 7)]
        .into_iter()
        .map(|(depth, bits)| {
            let projector = input_leaf_projector(&selected[..depth], bits);
            full.project(&projector).unwrap().unwrap()
        })
        .collect();
    let uneven =
        ReconstructionTarget::from_partition(&PartitionedTreeTN::from_subdomains(leaves).unwrap())
            .unwrap();
    let whole = target_of(tree.clone());
    let operator = quantics_fourier_operator(3, FourierOptions::default()).unwrap();
    for (target, depth, tolerance, inactive) in [
        (&uneven, Some(0), 1e-8, false),
        (&uneven, Some(1), 1e-8, false),
        (&uneven, None, 0.0, false),
        (&whole, None, 1e-8, true),
    ] {
        let result = schedule_merge_refine(
            target,
            &0,
            &operator,
            selected,
            &SubsetOperatorOptions { unitary: true },
            ReconstructionTolerance {
                rtol: tolerance,
                atol: 0.0,
            },
            &MergeRefineOptions {
                merge_policy: MergePolicy::Tolerance,
                output_depth: depth,
                ..Default::default()
            },
        )
        .unwrap();
        check_levels(result.report());
        if depth == Some(0) {
            assert!(result.report().levels.is_empty());
        }
        if depth == Some(1) {
            assert_eq!(result.report().levels[0].merging_levels_left, Some(1));
            assert_eq!(
                result.report().levels[0].allocated_error,
                result.report().absolute_tolerance
            );
            assert!(result.clone().into_partition().is_err());
        }
        if inactive {
            assert!(result
                .report()
                .levels
                .iter()
                .all(|level| level.merging_levels_left == Some(0) && level.allocated_error == 0.0));
        }
        if tolerance == 0.0 {
            assert_eq!(result.report().error_bound, 0.0);
            assert_eq!(result.report().compressions, 0);
        }
        check_dft(&result, &tree, &sites, selected);
    }
}

#[test]
fn tolerance_merges_handle_selected_multiaxis_sparse_spectator_cells() {
    let (sites, tree) = random_input(4);
    let selected = &sites[..4];
    let full = SubDomainTreeTN::from_treetn(tree.clone()).unwrap();
    let mut leaves = Vec::new();
    for leaf in 0..8 {
        let projector = input_leaf_projector(selected, leaf);
        if leaf < 4 {
            leaves.push(full.project(&projector).unwrap().unwrap());
        } else {
            for bit in 0..2 {
                let mut cell = projector.clone();
                cell.insert(sites[4].clone(), bit).unwrap();
                leaves.push(full.project(&cell).unwrap().unwrap());
            }
        }
    }
    let target =
        ReconstructionTarget::from_partition(&PartitionedTreeTN::from_subdomains(leaves).unwrap())
            .unwrap();
    let groups = axis_groups(&[2, 2], selected);
    let operator = chain_axis_fourier_operator(&[2, 2], selected);
    let result = schedule_merge_refine(
        &target,
        &0,
        &operator,
        selected,
        &SubsetOperatorOptions { unitary: true },
        ReconstructionTolerance {
            rtol: 1e-8,
            atol: 0.0,
        },
        &MergeRefineOptions {
            merge_policy: MergePolicy::Tolerance,
            coverage: CoverageContract::ZeroForMissingLeaves,
            coordinate_groups: Some(groups),
            output_regions: Some(vec![OutputRegion::per_axis([(1, 0), (0, 0)])]),
            ..Default::default()
        },
    )
    .unwrap();
    check_levels(result.report());
    assert_eq!(result.report().applied_operator_count, 12);
    assert_eq!(result.report().work_items_per_level[0], 16);
    assert!(result.report().compression_attempts > 0);
    assert_eq!(result.report().term_count, result.report().region_count);
    let projectors: Vec<_> = result.regions().map(|(p, _)| p.clone()).collect();
    assert!(Projector::are_disjoint(&projectors));
    let masked = full
        .project(&Projector::from_pairs([(sites[0].clone(), 0)]).unwrap())
        .unwrap()
        .unwrap();
    let input = masked.data().to_dense().unwrap();
    let msb: Vec<_> = sites.iter().rev().cloned().collect();
    let mut values = reorder(&input.to_vec::<Complex64>().unwrap(), input.indices(), &msb);
    let (indices, actual) = dense_of_regions(&result);
    for axis in selected.chunks(2) {
        let transformed = dft_oracle(&indices, &sites, axis, &values);
        values = reorder(&transformed, &indices, &msb);
    }
    let mut expected = reorder(&values, &msb, &indices);
    for (storage, value) in expected.iter_mut().enumerate() {
        if !projectors.iter().any(|p| {
            p.iter()
                .all(|(index, bit)| dense_bit(&indices, storage, index) == *bit)
        }) {
            *value = Complex64::new(0.0, 0.0);
        }
    }
    let error = l2_error(&actual, &expected);
    assert!(
        error <= result.report().error_bound + 1e-10,
        "L2 error={error:e}, {:?}",
        result.report()
    );
}

#[test]
fn tolerance_merges_charge_application_before_the_level_budget() {
    let sites: Vec<_> = (0..3).map(|_| DynIndex::new_dyn(2)).collect();
    let values: Vec<_> = (0..8)
        .map(|i| Complex64::new(1.0 + i as f64 * 0.1, 0.1 * (i as f64).sin()))
        .collect();
    let tree = mps(&sites, &values);
    let target = dyadic_input_leaves(&tree, &sites);
    let operator = identity_plus_scaled_x_tensor(&sites, 0.1);
    let exact = schedule_exact(&target, &operator, &sites, None, 64);
    let (reference_indices, reference) = dense_of_regions(&exact);
    for policy in [
        MergePolicy::Tolerance,
        MergePolicy::RankGoal(1_usize.try_into().unwrap()),
    ] {
        let result = schedule_merge_refine(
            &target,
            &0,
            &operator,
            &sites,
            &SubsetOperatorOptions { unitary: false },
            ReconstructionTolerance {
                rtol: 0.2,
                atol: 0.0,
            },
            &MergeRefineOptions {
                merge_policy: policy,
                application: OperatorApplication::Measured(
                    ApplyOptions::zipup().with_max_bond_dim(1),
                ),
                ..Default::default()
            },
        )
        .unwrap();
        let report = result.report();
        assert!(report.application_error_bound > 0.0);
        if policy == MergePolicy::Tolerance {
            check_levels(report);
        } else {
            assert!(report
                .levels
                .iter()
                .all(|level| level.merging_levels_left.is_none()));
        }
        assert_eq!(
            report.levels[0].remaining_before,
            report.absolute_tolerance - report.application_error_bound
        );
        for level in &report.levels {
            assert_eq!(
                level.remaining_after,
                (level.remaining_before - level.budget_spent).max(0.0)
            );
            assert!(level.budget_spent >= level.error_bound);
        }
        let (indices, actual) = dense_of_regions(&result);
        let error = l2_error(&actual, &reorder(&reference, &reference_indices, &indices));
        assert!(
            error <= report.error_bound + 1e-12,
            "L2 error={error:e}, {report:?}"
        );
    }
}
