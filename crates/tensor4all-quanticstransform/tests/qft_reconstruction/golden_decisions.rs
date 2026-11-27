//! Golden regression for the `Exact` and `RankGoal` merge-refine decisions.
//!
//! The other exact and rank-goal tests compare against dense oracles, so they
//! would accept a different but equally accurate trajectory. These fixed
//! schedules pin the trajectory itself: every counter, every level record and
//! every retained region key, term input prefix and bond dimension. The values
//! were captured on `origin/main` at 9ddffd17 (the #1407 merge), before the
//! #1407 review follow-ups, so a later change to tolerance-only code that alters
//! an exact or rank-goal decision fails here.
//!
//! Integer decisions are compared exactly. Floating-point report fields come
//! from backend factorizations, whose last bits may depend on the host's SIMD
//! dispatch, so they are compared to `1e-10` relative to the reference scale;
//! values that are zero by construction must be exactly zero.

use super::*;

/// One run's decisions in a stable text form, and its floating-point fields.
fn snapshot(result: &MergeRefineResult<usize>, groups: &[CoordinateGroup]) -> (String, Vec<f64>) {
    let report = result.report();
    let key = |projector: &Projector, inputs: bool| -> Vec<(usize, usize)> {
        groups
            .iter()
            .map(|group| {
                let order = if inputs {
                    &group.inputs
                } else {
                    &group.outputs
                };
                order
                    .iter()
                    .map_while(|index| projector.get(index))
                    .fold((0, 0), |(depth, prefix), bit| (depth + 1, 2 * prefix + bit))
            })
            .collect()
    };
    let mut text = format!(
        "policy {:?}; levels {} applied {} additions {} projections {} dropped {} attempts {} \
         compressions {} peak {}; work {:?}; refined {} stopped {} pruned {} regions {} \
         terms {} bond {} transient {} parameters {}",
        report.merge_policy,
        report.level_count,
        report.applied_operator_count,
        report.additions,
        report.projections,
        report.dropped_terms,
        report.compression_attempts,
        report.compressions,
        report.peak_work_items,
        report.work_items_per_level,
        report.refined_regions,
        report.stopped_regions,
        report.pruned_regions,
        report.region_count,
        report.term_count,
        report.max_bond_dim,
        report.max_transient_bond_dim,
        report.logical_parameters,
    );
    let mut floats = vec![
        report.reference_scale,
        report.absolute_tolerance,
        report.error_bound,
        report.application_error_bound,
        report.dropped_error,
    ];
    for level in &report.levels {
        text += &format!(
            "\nlevel {} left {:?}: additions {} attempts {} compressions {} dropped {}",
            level.level,
            level.merging_levels_left,
            level.additions,
            level.compression_attempts,
            level.compressions,
            level.dropped_terms,
        );
        floats.extend([
            level.allocated_error,
            level.error_bound,
            level.remaining_before,
            level.remaining_after,
            level.budget_spent,
            level.unused_allocation,
        ]);
    }
    for (output, _) in result.regions() {
        let mut terms: Vec<_> = result
            .items()
            .filter(|(_, region, _)| *region == output)
            .map(|(input, _, term)| (key(input, true), term.max_bond_dim()))
            .collect();
        terms.sort();
        text += &format!("\nregion {:?}: {:?}", key(output, false), terms);
    }
    (text, floats)
}

fn check(
    result: &MergeRefineResult<usize>,
    groups: &[CoordinateGroup],
    expected_text: &str,
    expected_floats: &[f64],
) {
    let (text, floats) = snapshot(result, groups);
    let report = result.report();
    assert_eq!(text, expected_text, "\n{text}\n{floats:?}");
    assert_eq!(floats.len(), expected_floats.len(), "{floats:?}");
    let scale = report.reference_scale;
    for (position, (&actual, &expected)) in floats.iter().zip(expected_floats).enumerate() {
        if expected == 0.0 && report.merge_policy == MergePolicy::Exact {
            assert_eq!(actual, 0.0, "float {position}: {floats:?}");
        }
        assert!(
            (actual - expected).abs() <= 1e-10 * scale.max(expected.abs()),
            "float {position}: {actual:?} != {expected:?}\n{floats:?}"
        );
    }
}

/// The two-mode, two-axis input of `output_selection`: two input leaves split
/// on the leading input bit, so rank-goal one must split while exact refines
/// uniformly, each with and without an output selection.
#[test]
fn golden_two_axis_exact_and_rank_goal_decisions() {
    let lengths = [2usize, 2];
    let bits = 4usize;
    let n = 1usize << bits;
    let sites: Vec<_> = (0..bits).map(|_| DynIndex::new_dyn(2)).collect();
    let c = Complex64::new(0.0, 0.6);
    let values: Vec<_> = (0..n)
        .map(|value| {
            let phase = (value >> 2) as f64 / 4.0 + (value & 3) as f64 / 4.0;
            Complex64::new(1.0, 0.0)
                + c * Complex64::from_polar(1.0, -2.0 * std::f64::consts::PI * phase)
        })
        .collect();
    let tree = mps(&sites, &values);
    let preimage = leaf_target(
        &tree,
        vec![
            prefix_projector(&sites, 1, 0),
            prefix_projector(&sites, 1, 1),
        ],
    );
    let operator = chain_axis_fourier_operator(&lengths, &sites);
    let groups = axis_groups(&lengths, &sites);
    let selection = OutputRegion::per_axis([(1, 0), (1, 0)]);
    let cases: [(MergePolicy, bool, &str, &[f64]); 4] = [
        (
            MergePolicy::Exact,
            false,
            concat!(
                "policy Exact; levels 2 applied 2 additions 4 projections 48 dropped 0 attempts 0 compressions 0 peak 16; work [2, 4, 16]; refined 5 stopped 0 pruned 0 regions 16 terms 16 bond 8 transient 8 parameters 2304\n",
                "level 1 left None: additions 4 attempts 0 compressions 0 dropped 0\n",
                "level 2 left None: additions 0 attempts 0 compressions 0 dropped 0\n",
                "region [(2, 0), (2, 0)]: [([(0, 0), (0, 0)], 8)]\n",
                "region [(2, 0), (2, 1)]: [([(0, 0), (0, 0)], 8)]\n",
                "region [(2, 0), (2, 2)]: [([(0, 0), (0, 0)], 8)]\n",
                "region [(2, 0), (2, 3)]: [([(0, 0), (0, 0)], 8)]\n",
                "region [(2, 1), (2, 0)]: [([(0, 0), (0, 0)], 8)]\n",
                "region [(2, 1), (2, 1)]: [([(0, 0), (0, 0)], 8)]\n",
                "region [(2, 1), (2, 2)]: [([(0, 0), (0, 0)], 8)]\n",
                "region [(2, 1), (2, 3)]: [([(0, 0), (0, 0)], 8)]\n",
                "region [(2, 2), (2, 0)]: [([(0, 0), (0, 0)], 8)]\n",
                "region [(2, 2), (2, 1)]: [([(0, 0), (0, 0)], 8)]\n",
                "region [(2, 2), (2, 2)]: [([(0, 0), (0, 0)], 8)]\n",
                "region [(2, 2), (2, 3)]: [([(0, 0), (0, 0)], 8)]\n",
                "region [(2, 3), (2, 0)]: [([(0, 0), (0, 0)], 8)]\n",
                "region [(2, 3), (2, 1)]: [([(0, 0), (0, 0)], 8)]\n",
                "region [(2, 3), (2, 2)]: [([(0, 0), (0, 0)], 8)]\n",
                "region [(2, 3), (2, 3)]: [([(0, 0), (0, 0)], 8)]",
            ),
            &[
                4.66476151587624,
                4.66476151587624e-8,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                4.66476151587624e-8,
                4.66476151587624e-8,
                0.0,
                0.0,
                0.0,
                0.0,
                4.66476151587624e-8,
                4.66476151587624e-8,
                0.0,
                0.0,
            ],
        ),
        (
            MergePolicy::Exact,
            true,
            concat!(
                "policy Exact; levels 2 applied 2 additions 1 projections 12 dropped 0 attempts 0 compressions 0 peak 4; work [2, 1, 4]; refined 2 stopped 0 pruned 3 regions 4 terms 4 bond 8 transient 8 parameters 576\n",
                "level 1 left None: additions 1 attempts 0 compressions 0 dropped 0\n",
                "level 2 left None: additions 0 attempts 0 compressions 0 dropped 0\n",
                "region [(2, 0), (2, 0)]: [([(0, 0), (0, 0)], 8)]\n",
                "region [(2, 0), (2, 1)]: [([(0, 0), (0, 0)], 8)]\n",
                "region [(2, 1), (2, 0)]: [([(0, 0), (0, 0)], 8)]\n",
                "region [(2, 1), (2, 1)]: [([(0, 0), (0, 0)], 8)]",
            ),
            &[
                4.66476151587624,
                4.66476151587624e-8,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                4.66476151587624e-8,
                4.66476151587624e-8,
                0.0,
                0.0,
                0.0,
                0.0,
                4.66476151587624e-8,
                4.66476151587624e-8,
                0.0,
                0.0,
            ],
        ),
        (
            MergePolicy::RankGoal(1usize.try_into().unwrap()),
            false,
            concat!(
                "policy RankGoal(1); levels 2 applied 2 additions 9 projections 44 dropped 2 attempts 7 compressions 7 peak 4; work [2, 4, 4]; refined 1 stopped 8 pruned 0 regions 2 terms 2 bond 1 transient 8 parameters 16\n",
                "level 1 left None: additions 9 attempts 7 compressions 7 dropped 2\n",
                "level 2 left None: additions 0 attempts 0 compressions 0 dropped 0\n",
                "region [(1, 0), (1, 0)]: [([(0, 0), (0, 0)], 1)]\n",
                "region [(1, 1), (1, 1)]: [([(0, 0), (0, 0)], 1)]",
            ),
            &[
                4.66476151587624,
                4.66476151587624e-8,
                5.2019892419737916e-15,
                0.0,
                6.774967307654806e-15,
                2.33238075793812e-8,
                5.2019892419737916e-15,
                4.66476151587624e-8,
                4.66475852813369e-8,
                2.9877425497130466e-14,
                2.3323802377391957e-8,
                0.0,
                0.0,
                4.66475852813369e-8,
                4.66475852813369e-8,
                0.0,
                0.0,
            ],
        ),
        (
            MergePolicy::RankGoal(1usize.try_into().unwrap()),
            true,
            concat!(
                "policy RankGoal(1); levels 2 applied 2 additions 4 projections 18 dropped 0 attempts 4 compressions 4 peak 2; work [2, 1, 1]; refined 1 stopped 2 pruned 3 regions 1 terms 1 bond 1 transient 8 parameters 8\n",
                "level 1 left None: additions 4 attempts 4 compressions 4 dropped 0\n",
                "level 2 left None: additions 0 attempts 0 compressions 0 dropped 0\n",
                "region [(1, 0), (1, 0)]: [([(0, 0), (0, 0)], 1)]",
            ),
            &[
                4.66476151587624,
                4.66476151587624e-8,
                3.4486677906042526e-16,
                0.0,
                0.0,
                2.33238075793812e-8,
                3.4486677906042526e-16,
                4.66476151587624e-8,
                4.66476004559908e-8,
                1.4702771598244828e-14,
                2.332380723451442e-8,
                0.0,
                0.0,
                4.66476004559908e-8,
                4.66476004559908e-8,
                0.0,
                0.0,
            ],
        ),
    ];
    for (policy, selected, expected_text, expected_floats) in cases {
        let result = schedule_merge_refine(
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
        .unwrap();
        check(&result, &groups, expected_text, expected_floats);
    }
}

/// A smooth single-axis input on the asymmetric dyadic leaves `{0, 10, 110,
/// 111}` of four bits. Exact refines this nonuniform input tree to sixteen
/// regions; rank goal two stops at the root, because one-axis children keep
/// every site, and compresses each merge there.
#[test]
fn golden_single_axis_exact_and_rank_goal_decisions() {
    let r = 4usize;
    let sites: Vec<DynIndex> = (0..r).map(|_| DynIndex::new_dyn(2)).collect();
    let values: Vec<Complex64> = (0..1usize << r)
        .map(|m| {
            let x = m as f64 / 16.0;
            Complex64::new((-8.0 * (x - 0.4).powi(2)).exp(), 0.3 * (3.0 * x).sin())
        })
        .collect();
    let tree = mps(&sites, &values);
    let preimage = leaf_target(
        &tree,
        vec![
            prefix_projector(&sites, 1, 0),
            prefix_projector(&sites, 2, 2),
            prefix_projector(&sites, 3, 6),
            prefix_projector(&sites, 3, 7),
        ],
    );
    let operator = quantics_fourier_operator(r, FourierOptions::default()).expect("fourier");
    let groups = vec![CoordinateGroup::new(sites.clone())];
    let cases: [(MergePolicy, &str, &[f64]); 2] = [
        (
            MergePolicy::Exact,
            concat!(
                "policy Exact; levels 4 applied 4 additions 14 projections 104 dropped 0 attempts 0 compressions 0 peak 16; work [4, 6, 8, 8, 16]; refined 15 stopped 0 pruned 0 regions 16 terms 16 bond 16 transient 16 parameters 8704\n",
                "level 1 left None: additions 2 attempts 0 compressions 0 dropped 0\n",
                "level 2 left None: additions 4 attempts 0 compressions 0 dropped 0\n",
                "level 3 left None: additions 8 attempts 0 compressions 0 dropped 0\n",
                "level 4 left None: additions 0 attempts 0 compressions 0 dropped 0\n",
                "region [(4, 0)]: [([(0, 0)], 16)]\n",
                "region [(4, 1)]: [([(0, 0)], 16)]\n",
                "region [(4, 2)]: [([(0, 0)], 16)]\n",
                "region [(4, 3)]: [([(0, 0)], 16)]\n",
                "region [(4, 4)]: [([(0, 0)], 16)]\n",
                "region [(4, 5)]: [([(0, 0)], 16)]\n",
                "region [(4, 6)]: [([(0, 0)], 16)]\n",
                "region [(4, 7)]: [([(0, 0)], 16)]\n",
                "region [(4, 8)]: [([(0, 0)], 16)]\n",
                "region [(4, 9)]: [([(0, 0)], 16)]\n",
                "region [(4, 10)]: [([(0, 0)], 16)]\n",
                "region [(4, 11)]: [([(0, 0)], 16)]\n",
                "region [(4, 12)]: [([(0, 0)], 16)]\n",
                "region [(4, 13)]: [([(0, 0)], 16)]\n",
                "region [(4, 14)]: [([(0, 0)], 16)]\n",
                "region [(4, 15)]: [([(0, 0)], 16)]",
            ),
            &[
                2.790592838770849,
                0.002790592838770849,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.002790592838770849,
                0.002790592838770849,
                0.0,
                0.0,
                0.0,
                0.0,
                0.002790592838770849,
                0.002790592838770849,
                0.0,
                0.0,
                0.0,
                0.0,
                0.002790592838770849,
                0.002790592838770849,
                0.0,
                0.0,
                0.0,
                0.0,
                0.002790592838770849,
                0.002790592838770849,
                0.0,
                0.0,
            ],
        ),
        (
            MergePolicy::RankGoal(2usize.try_into().unwrap()),
            concat!(
                "policy RankGoal(2); levels 4 applied 4 additions 5 projections 36 dropped 0 attempts 5 compressions 5 peak 4; work [4, 3, 2, 1, 1]; refined 0 stopped 1 pruned 0 regions 1 terms 1 bond 4 transient 8 parameters 40\n",
                "level 1 left None: additions 3 attempts 3 compressions 3 dropped 0\n",
                "level 2 left None: additions 1 attempts 1 compressions 1 dropped 0\n",
                "level 3 left None: additions 1 attempts 1 compressions 1 dropped 0\n",
                "level 4 left None: additions 0 attempts 0 compressions 0 dropped 0\n",
                "region [(0, 0)]: [([(0, 0)], 4)]",
            ),
            &[
                2.790592838770849,
                0.002790592838770849,
                6.937156590665905e-15,
                0.0,
                0.0,
                0.0006976482096927122,
                5.427767603594427e-16,
                0.002790592838770849,
                0.002790592838768672,
                2.17698419432187e-15,
                0.0006976482096921695,
                0.0009301976129228906,
                4.8178877021171925e-15,
                0.002790592838768672,
                0.002790592838763854,
                4.8178877021171925e-15,
                0.0009301976129180728,
                0.001395296419381927,
                1.5764921281892703e-15,
                0.002790592838763854,
                0.0027905928387622777,
                1.5764921281892703e-15,
                0.0013952964193803506,
                0.0,
                0.0,
                0.0027905928387622777,
                0.0027905928387622777,
                0.0,
                0.0,
            ],
        ),
    ];
    for (policy, expected_text, expected_floats) in cases {
        let result = schedule_merge_refine(
            &preimage,
            &0,
            &operator,
            &sites,
            &SubsetOperatorOptions { unitary: true },
            ReconstructionTolerance {
                rtol: 1e-3,
                atol: 0.0,
            },
            &MergeRefineOptions {
                merge_policy: policy,
                ..Default::default()
            },
        )
        .unwrap();
        check(&result, &groups, expected_text, expected_floats);
    }
}
