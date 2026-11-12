#![cfg(feature = "partitioned")]
mod support;
use num_complex::Complex64;
use std::collections::HashSet;
use support::*;
use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_partitionedtreetn::{
    reconstruction::{ReconstructionTarget, SubsetOperatorOptions},
    PartitionedTreeTN, PartitionedTreeTNError, Projector,
};
use tensor4all_quanticstransform::{
    patched_fourier_transform, quantics_fourier_operator, quantics_fourier_operator_multiaxis,
    AxisBit, CoordinateGroup, FourierOptions, PatchedFourierOptions, QuanticsTransformError,
};
use tensor4all_treetn::{
    apply_linear_operator_to_indices, ApplyOptions, RestructureOptions, SiteIndexNetwork, TreeTN,
};

fn asym(axes: &[Vec<DynIndex>]) -> Vec<Vec<(DynIndex, usize)>> {
    let mut prefixes = vec![vec![]];
    for axis in axes {
        let mut local = Vec::new();
        for depth in 1..=axis.len() {
            let mut key: Vec<_> = axis
                .iter()
                .take(depth - 1)
                .cloned()
                .map(|s| (s, 1))
                .collect();
            key.push((axis[depth - 1].clone(), 0));
            local.push(key);
        }
        local.push(axis.iter().cloned().map(|s| (s, 1)).collect());
        prefixes = prefixes
            .iter()
            .flat_map(|prefix| {
                local
                    .iter()
                    .map(move |key| prefix.iter().chain(key).cloned().collect())
            })
            .collect();
    }
    prefixes
}
fn logical_layout(sites: &[DynIndex], axes: &[Vec<DynIndex>]) -> Vec<Option<AxisBit>> {
    sites
        .iter()
        .map(|site| {
            axes.iter().enumerate().find_map(|(axis, bits)| {
                bits.iter()
                    .position(|s| s == site)
                    .map(|bit| AxisBit { axis, bit })
            })
        })
        .collect()
}
fn constructed_reference(
    tree: &TreeTN<IdxTensor, usize>,
    sites: &[DynIndex],
    axes: &[Vec<DynIndex>],
    options: FourierOptions,
) -> IdxTensor {
    let selected: Vec<_> = logical_layout(sites, axes)
        .into_iter()
        .enumerate()
        .filter_map(|(j, b)| b.map(|b| (j, b)))
        .collect();
    let operator = quantics_fourier_operator_multiaxis(
        &selected.iter().map(|(_, b)| *b).collect::<Vec<_>>(),
        options,
    )
    .unwrap();
    let operator = operator
        .rename_nodes(
            &selected
                .iter()
                .enumerate()
                .map(|(j, (physical, _))| (j, *physical))
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let inputs: Vec<_> = selected
        .iter()
        .map(|(j, _)| {
            (
                operator.get_input_mapping(j).unwrap().true_index.clone(),
                sites[*j].clone(),
            )
        })
        .collect();
    let outputs: Vec<_> = selected
        .iter()
        .map(|(j, _)| {
            (
                operator.get_output_mapping(j).unwrap().true_index.clone(),
                sites[*j].clone(),
            )
        })
        .collect();
    apply_linear_operator_to_indices(&operator, tree, &inputs, &outputs, ApplyOptions::naive())
        .unwrap()
        .to_dense()
        .unwrap()
}
#[test]
fn nonuniform_patches_match_constructed_operator_and_dft() {
    for n_axes in [1, 2] {
        let n = 2 * n_axes;
        let sites: Vec<_> = (0..n).map(|_| DynIndex::new_dyn(2)).collect();
        let axes: Vec<_> = (0..n_axes)
            .map(|a| vec![sites[a].clone(), sites[a + n_axes].clone()])
            .collect();
        let input = values(n);
        let tree = chain(&sites, &input, &(0..n).collect::<Vec<_>>());
        let state = partition(&tree, &asym(&axes));
        for normalize in [true, false] {
            let options = FourierOptions {
                normalize,
                ..Default::default()
            };
            let result = patched_fourier_transform(
                &state,
                &0,
                &axes,
                options.clone(),
                &PatchedFourierOptions::default(),
            )
            .unwrap();
            let actual = result_dense(&result);
            let reference = constructed_reference(&tree, &sites, &axes, options);
            let error = actual.sub(&reference).unwrap().maxabs().unwrap();
            assert!(
                error <= result.report().error_bound + ROUND_OFF,
                "{error:e}"
            );
            assert_eq!(result.report().applied_operator_count, state.len());
            assert_eq!(result.report().error_bound, 0.0);
            let input_norm = state.norm().unwrap();
            let amplification = if normalize { 1.0 } else { (1usize << n) as f64 };
            assert!(
                (result.report().reference_scale - amplification * input_norm).abs() < ROUND_OFF
            );
            assert_dense(
                &reference,
                &sites,
                &dft(&input, &logical_layout(&sites, &axes), -1.0, normalize),
                ROUND_OFF,
            );
        }
    }
}
#[test]
fn spectator_and_single_patch_preserve_dense_application() {
    let sites: Vec<_> = (0..5).map(|_| DynIndex::new_dyn(2)).collect();
    let axes = vec![
        vec![sites[0].clone(), sites[3].clone()],
        vec![sites[1].clone(), sites[4].clone()],
    ];
    let input = values(5);
    let tree = chain(&sites, &input, &[0, 1, 2, 3, 4]);
    for prefixes in [vec![vec![]], asym(&axes)] {
        let state = partition(&tree, &prefixes);
        let result = patched_fourier_transform(
            &state,
            &0,
            &axes,
            FourierOptions::forward(),
            &PatchedFourierOptions::default(),
        )
        .unwrap();
        let reference = constructed_reference(&tree, &sites, &axes, FourierOptions::forward());
        assert!(
            result_dense(&result)
                .sub(&reference)
                .unwrap()
                .maxabs()
                .unwrap()
                < ROUND_OFF
        );
        assert_dense(
            &reference,
            &sites,
            &dft(&input, &logical_layout(&sites, &axes), -1.0, true),
            ROUND_OFF,
        );
        assert!(result.regions().all(|(p, terms)| p.get(&sites[2]).is_none()
            && terms
                .iter()
                .all(|t| t.data().site_index_network().find_node_by_index(&sites[2]) == Some(&2))));
    }
}
#[test]
fn reversed_axes_inverse_restores_input_and_same_order_does_not() {
    let sites: Vec<_> = (0..4).map(|_| DynIndex::new_dyn(2)).collect();
    let axes = vec![
        vec![sites[0].clone(), sites[2].clone()],
        vec![sites[1].clone(), sites[3].clone()],
    ];
    let input = values(4);
    let tree = chain(&sites, &input, &[0, 1, 2, 3]);
    let state = partition(&tree, &asym(&axes));
    let mut bounded = PatchedFourierOptions::default();
    bounded.tolerance.rtol = 1e-12;
    bounded.merge_refine.target_bond_dim = Some(4);
    let forward = patched_fourier_transform(&state, &0, &axes, FourierOptions::forward(), &bounded)
        .unwrap()
        .into_partition()
        .unwrap();
    let inverse_axes: Vec<_> = axes
        .iter()
        .map(|axis| axis.iter().rev().cloned().collect())
        .collect();
    let restored = patched_fourier_transform(
        &forward,
        &0,
        &inverse_axes,
        FourierOptions::inverse(),
        &bounded,
    )
    .unwrap();
    assert_dense(&result_dense(&restored), &sites, &input, ROUND_OFF);
    let wrong = patched_fourier_transform(&forward, &0, &axes, FourierOptions::inverse(), &bounded)
        .unwrap();
    let oracle = IdxTensor::from_dense(sites, input).unwrap();
    assert!(result_dense(&wrong).sub(&oracle).unwrap().maxabs().unwrap() > 0.1);
}
#[test]
fn output_regions_restrict_the_returned_transform() {
    let sites: Vec<_> = (0..3).map(|_| DynIndex::new_dyn(2)).collect();
    let axes = vec![sites.clone()];
    let input = values(3);
    let tree = chain(&sites, &input, &[0, 1, 2]);
    let state = partition(&tree, &asym(&axes));
    let mut options = PatchedFourierOptions::default();
    options.merge_refine.output_regions =
        Some(vec![Projector::from_pairs([(sites[2].clone(), 1)]).unwrap()]);
    let result =
        patched_fourier_transform(&state, &0, &axes, FourierOptions::forward(), &options).unwrap();
    assert_eq!(result.regions().count(), 4);
    assert!(result.regions().all(|(p, _)| p.get(&sites[2]) == Some(1)));
    let mut expected = dft(&input, &logical_layout(&sites, &axes), -1.0, true);
    for (j, value) in expected.iter_mut().enumerate() {
        if (j >> 2) & 1 == 0 {
            *value = Complex64::new(0.0, 0.0);
        }
    }
    assert_dense(&result_dense(&result), &sites, &expected, ROUND_OFF);
}
#[test]
fn adaptive_bound_covers_constructed_operator_error() {
    let sites: Vec<_> = (0..4).map(|_| DynIndex::new_dyn(2)).collect();
    let axes = vec![sites.clone()];
    let input = values(4);
    let tree = chain(&sites, &input, &[0, 1, 2, 3]);
    let prefixes = (0..16)
        .map(|j| {
            sites
                .iter()
                .enumerate()
                .map(|(bit, s)| (s.clone(), (j >> bit) & 1))
                .collect()
        })
        .collect::<Vec<_>>();
    let state = partition(&tree, &prefixes);
    let mut options = PatchedFourierOptions::default();
    options.tolerance.rtol = 0.3;
    options.merge_refine.target_bond_dim = Some(2);
    let result =
        patched_fourier_transform(&state, &0, &axes, FourierOptions::forward(), &options).unwrap();
    let report = result.report();
    assert!(report.compressions > 0, "{report:?}");
    assert!(report.error_bound > 0.0, "{report:?}");
    assert!(report.error_bound <= report.absolute_tolerance);
    let reference = constructed_reference(&tree, &sites, &axes, FourierOptions::forward());
    let error = result_dense(&result)
        .sub(&reference)
        .unwrap()
        .maxabs()
        .unwrap();
    assert!(
        error <= report.error_bound + ROUND_OFF,
        "{error:e} > {}",
        report.error_bound
    );
}
#[test]
fn generic_names_and_multiple_bits_on_one_node() {
    let sites: Vec<_> = (0..3).map(|_| DynIndex::new_dyn(2)).collect();
    let input = values(3);
    let axes = vec![sites[..2].to_vec()];
    let tree = TreeTN::from_tensors(
        vec![IdxTensor::from_dense(sites.clone(), input.clone()).unwrap()],
        vec![()],
    )
    .unwrap();
    let state = partition(&tree, &asym(&axes));
    let result = patched_fourier_transform(
        &state,
        &(),
        &axes,
        FourierOptions::forward(),
        &PatchedFourierOptions::default(),
    )
    .unwrap();
    assert_dense(
        &result_dense(&result),
        &sites,
        &dft(&input, &logical_layout(&sites, &axes), -1.0, true),
        ROUND_OFF,
    );
    // The canonical reconstruction entry point accepts independent node types too.
    let target = ReconstructionTarget::from_partition(&state).unwrap();
    let operator = quantics_fourier_operator(2, FourierOptions::forward()).unwrap();
    let image = ReconstructionTarget::from_subset_operator(
        &target,
        &(),
        &operator,
        &axes[0],
        &SubsetOperatorOptions { unitary: true },
    )
    .unwrap();
    assert!((image.reference_scale() - target.reference_scale()).abs() < ROUND_OFF);
    let tree = chain(&sites, &input, &["zeta", "alpha", "mu"]);
    let result = patched_fourier_transform(
        &partition(&tree, &[vec![]]),
        &"alpha",
        &axes,
        FourierOptions::forward(),
        &PatchedFourierOptions::default(),
    )
    .unwrap();
    assert_dense(
        &result_dense(&result),
        &sites,
        &dft(&input, &logical_layout(&sites, &axes), -1.0, true),
        ROUND_OFF,
    );
}
#[test]
fn path_on_branched_tree_is_accepted_but_branching_selection_is_rejected() {
    let sites: Vec<_> = (0..4).map(|_| DynIndex::new_dyn(2)).collect();
    let input = values(4);
    let source = TreeTN::from_tensors(
        vec![IdxTensor::from_dense(sites.clone(), input.clone()).unwrap()],
        vec![0usize],
    )
    .unwrap();
    let mut topology = SiteIndexNetwork::new();
    for (j, site) in sites.iter().enumerate() {
        topology.add_node(j, HashSet::from([site.clone()])).unwrap();
    }
    for j in 1..4 {
        topology.add_edge(&0, &j).unwrap();
    }
    let tree = source
        .restructure_to(&topology, &RestructureOptions::default())
        .unwrap();
    let state = partition(&tree, &[vec![]]);
    let axes = vec![vec![sites[1].clone(), sites[2].clone()]];
    let result = patched_fourier_transform(
        &state,
        &0,
        &axes,
        FourierOptions::forward(),
        &PatchedFourierOptions::default(),
    )
    .unwrap();
    assert_dense(
        &result_dense(&result),
        &sites,
        &dft(&input, &logical_layout(&sites, &axes), -1.0, true),
        ROUND_OFF,
    );
    let error = patched_fourier_transform(
        &state,
        &0,
        &[sites[1..].to_vec()],
        FourierOptions::forward(),
        &PatchedFourierOptions::default(),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        QuanticsTransformError::InvalidConfiguration { .. }
    ));
    assert!(error.to_string().contains("one tree path"));
}
#[test]
fn validation_and_scheduler_sources_are_preserved() {
    let sites: Vec<_> = (0..3).map(|_| DynIndex::new_dyn(2)).collect();
    let tree = chain(&sites, &values(3), &[0, 1, 2]);
    let state = partition(&tree, &[vec![]]);
    let axes = vec![sites.clone()];
    for invalid_axes in [
        vec![],
        vec![vec![sites[0].clone()]],
        vec![vec![sites[0].clone(); 2]],
        vec![vec![sites[0].clone(), DynIndex::new_dyn(2)]],
        vec![vec![sites[0].clone(), DynIndex::new_dyn(3)]],
    ] {
        assert!(matches!(
            patched_fourier_transform(
                &state,
                &0,
                &invalid_axes,
                FourierOptions::forward(),
                &PatchedFourierOptions::default()
            ),
            Err(QuanticsTransformError::InvalidConfiguration { .. })
        ));
    }
    let mut options = PatchedFourierOptions::default();
    options.merge_refine.coordinate_groups = Some(vec![CoordinateGroup::new(sites.clone())]);
    assert!(
        patched_fourier_transform(&state, &0, &axes, FourierOptions::forward(), &options)
            .unwrap_err()
            .to_string()
            .contains("leave it unset")
    );
    assert!(patched_fourier_transform(
        &state,
        &99,
        &axes,
        FourierOptions::forward(),
        &PatchedFourierOptions::default()
    )
    .is_err());
    assert!(patched_fourier_transform(
        &PartitionedTreeTN::<usize>::new(),
        &0,
        &axes,
        FourierOptions::forward(),
        &PatchedFourierOptions::default()
    )
    .is_err());
    let mut options = PatchedFourierOptions::default();
    options.merge_refine.max_work_items = 0;
    let error = patched_fourier_transform(&state, &0, &axes, FourierOptions::forward(), &options)
        .unwrap_err();
    assert!(matches!(
        error,
        QuanticsTransformError::Reconstruction {
            source: PartitionedTreeTNError::InvalidOptions { .. }
        }
    ));
    assert!(std::error::Error::source(&error).is_some());
    let sparse = partition(&tree, &[vec![(sites[0].clone(), 0)]]);
    assert!(matches!(
        patched_fourier_transform(
            &sparse,
            &0,
            &axes,
            FourierOptions::forward(),
            &PatchedFourierOptions::default()
        ),
        Err(QuanticsTransformError::Reconstruction { .. })
    ));
}
