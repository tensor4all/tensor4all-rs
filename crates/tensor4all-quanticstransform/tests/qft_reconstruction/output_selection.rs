use super::*;
use tensor4all_quanticstransform::{patched_fourier_transform, PatchedFourierOptions};

/// A sum of two Fourier modes transforms to two deltas whose bits differ on
/// every axis. A tight rank goal cannot keep their sum at rank one, but either
/// selected child can keep one delta exactly. This drives genuine split decisions.
fn check_selected_modes(lengths: &[usize]) {
    let bits: usize = lengths.iter().sum();
    let n = 1usize << bits;
    let sites: Vec<_> = (0..bits).map(|_| DynIndex::new_dyn(2)).collect();
    let c = Complex64::new(0.0, 0.6);
    let values: Vec<_> = (0..n)
        .map(|value| {
            let mut trailing = bits;
            let phase: f64 = lengths
                .iter()
                .map(|&length| {
                    trailing -= length;
                    let size = 1usize << length;
                    ((value >> trailing) & (size - 1)) as f64 / size as f64
                })
                .sum();
            Complex64::new(1.0, 0.0)
                + c * Complex64::from_polar(1.0, -2.0 * std::f64::consts::PI * phase)
        })
        .collect();
    let tree = mps(&sites, &values);
    let full = SubDomainTreeTN::from_treetn(tree.clone()).unwrap();
    let partition = PartitionedTreeTN::from_subdomains(
        (0..2)
            .map(|bit| {
                full.project(&Projector::from_pairs([(sites[0].clone(), bit)]).unwrap())
                    .unwrap()
                    .unwrap()
            })
            .collect(),
    )
    .unwrap();
    let preimage = ReconstructionTarget::from_partition(&partition).unwrap();
    let operator = chain_axis_fourier_operator(lengths, &sites);
    let groups = axis_groups(lengths, &sites);
    let axes: Vec<_> = groups.iter().map(|group| group.inputs.clone()).collect();
    let selection = OutputRegion::per_axis(lengths.iter().map(|_| (1, 0)));
    let expected_projector = groups.iter().fold(Projector::new(), |projector, group| {
        projector
            .intersection(&output_prefix_projector(group, 1, 0))
            .unwrap()
    });
    for policy in [
        MergePolicy::Exact,
        MergePolicy::RankGoal(1usize.try_into().unwrap()),
        MergePolicy::Tolerance,
    ] {
        for selected in [false, true] {
            for frontend in [false, true] {
                let mut options = PatchedFourierOptions::default();
                options.tolerance.rtol = 1e-8;
                options.merge_refine.merge_policy = policy;
                options.merge_refine.output_regions = selected.then(|| vec![selection.clone()]);
                let result = if frontend {
                    patched_fourier_transform(
                        &partition,
                        &0,
                        &axes,
                        FourierOptions::default(),
                        &options,
                    )
                    .unwrap()
                } else {
                    schedule_merge_refine(
                        &preimage,
                        &0,
                        &operator,
                        &sites,
                        &SubsetOperatorOptions { unitary: true },
                        ReconstructionTolerance {
                            rtol: 1e-8,
                            atol: 0.0,
                        },
                        &MergeRefineOptions {
                            merge_policy: policy,
                            coordinate_groups: Some(groups.clone()),
                            output_regions: selected.then(|| vec![selection.clone()]),
                            ..Default::default()
                        },
                    )
                    .unwrap()
                };
                let report = result.report();
                // At the first split, one of 2^axes children is selected. The
                // selection is already satisfied thereafter; it cuts no descendants.
                assert_eq!(
                    report.pruned_regions,
                    if selected {
                        (1usize << lengths.len()) - 1
                    } else {
                        0
                    },
                    "{policy:?}: {report:?}"
                );
                assert_eq!(report.merge_policy, policy);
                assert!(report.error_bound <= report.absolute_tolerance);
                assert_eq!(report.applied_operator_count, 2);
                if policy == MergePolicy::Tolerance {
                    super::tolerance_merges::check_levels(report);
                }
                if selected {
                    assert!(report.refined_regions > 0);
                    for (projector, _) in result.regions() {
                        assert_eq!(
                            projector.intersection(&expected_projector).as_ref(),
                            Some(projector)
                        );
                    }
                }
                let (_, actual) = dense_of_regions(&result);
                let mut expected = vec![Complex64::new(0.0, 0.0); n];
                expected[0] = Complex64::new((n as f64).sqrt(), 0.0);
                if !selected {
                    expected[n - 1] = c * (n as f64).sqrt();
                }
                let error = l2_error(&actual, &expected);
                assert!(
                    error <= report.error_bound + 1e-9,
                    "{policy:?}: error {error:e}, {report:?}"
                );
            }
        }
    }
}

#[test]
fn all_policies_count_only_selected_single_axis_cuts() {
    check_selected_modes(&[3]);
}

#[test]
fn all_policies_count_only_selected_joint_cuts() {
    check_selected_modes(&[2, 2]);
}

/// One input leaf means each incoming region contains exactly one work item.
/// Thus this fixed case also pins the distinction between incoming region jobs
/// (one, then two) and four final output regions.
#[test]
fn minimal_two_bit_exact_fanout_has_two_second_level_region_jobs() {
    let sites: Vec<_> = (0..2).map(|_| DynIndex::new_dyn(2)).collect();
    let values = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
    ];
    let preimage = target_of(mps(&sites, &values));
    let operator = quantics_fourier_operator(2, FourierOptions::default()).unwrap();
    let result = schedule_merge_refine(
        &preimage,
        &0,
        &operator,
        &sites,
        &SubsetOperatorOptions { unitary: true },
        ReconstructionTolerance::default(),
        &MergeRefineOptions::default(),
    )
    .unwrap();
    assert_eq!(result.report().applied_operator_count, 1);
    assert_eq!(result.report().work_items_per_level, vec![1, 2, 4]);
    assert_eq!(result.report().region_count, 4);
    assert_eq!(result.report().pruned_regions, 0);
    assert_eq!(result.report().error_bound, 0.0);
    let (_, actual) = dense_of_regions(&result);
    let expected = vec![Complex64::new(0.5, 0.0); 4];
    assert!(l2_error(&actual, &expected) < TOL);
}
