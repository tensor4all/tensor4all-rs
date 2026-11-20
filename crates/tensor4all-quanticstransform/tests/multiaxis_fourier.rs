mod support;
use support::*;
use tensor4all_core::DynIndex;
use tensor4all_quanticstransform::{
    quantics_fourier_operator, quantics_fourier_operator_multiaxis, AxisBit, FourierOptions,
    QuanticsTransformError,
};

fn check(layout: Vec<AxisBit>) {
    check_with_reference(layout, true);
}

/// `sequential_reference` applies each one-axis operator along its bit order,
/// which is a state path only when every axis is monotone along the chain.
fn check_with_reference(layout: Vec<AxisBit>, sequential_reference: bool) {
    let sites: Vec<_> = layout.iter().map(|_| DynIndex::new_dyn(2)).collect();
    let input = values(sites.len());
    let tree = chain(&sites, &input, &(0..sites.len()).collect::<Vec<_>>());
    for (sign, normalize) in [(-1.0, true), (1.0, false)] {
        let operator = quantics_fourier_operator_multiaxis(
            &layout,
            FourierOptions {
                sign,
                normalize,
                ..Default::default()
            },
        )
        .unwrap();
        let actual = apply(&operator, &tree, &sites).to_dense().unwrap();
        assert_dense(
            &actual,
            &sites,
            &dft(
                &input,
                &layout.iter().copied().map(Some).collect::<Vec<_>>(),
                sign,
                normalize,
            ),
            ROUND_OFF,
        );
        eprintln!(
            "layout {layout:?}, restructured max bond {}",
            operator.mpo().link_dims().into_iter().max().unwrap_or(1)
        );
        assert!(operator.mpo().node_names().iter().all(|j| operator
            .get_input_mappings(j)
            .unwrap()
            .len()
            == 1
            && operator.get_output_mappings(j).unwrap().len() == 1));
        if !sequential_reference {
            continue;
        }
        // Independent reference: apply the one-axis operators sequentially.
        let mut sequential = tree.clone();
        let mut axes = std::collections::BTreeMap::<usize, Vec<(usize, usize)>>::new();
        for (position, entry) in layout.iter().enumerate() {
            axes.entry(entry.axis)
                .or_default()
                .push((entry.bit, position));
        }
        for positions in axes.values_mut() {
            positions.sort();
            let one = quantics_fourier_operator(
                positions.len(),
                FourierOptions {
                    sign,
                    normalize,
                    ..Default::default()
                },
            )
            .unwrap();
            let one = one
                .rename_nodes(
                    &positions
                        .iter()
                        .enumerate()
                        .map(|(j, (_, physical))| (j, *physical))
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let inputs = positions
                .iter()
                .map(|(_, j)| {
                    (
                        one.get_input_mapping(j).unwrap().true_index.clone(),
                        sites[*j].clone(),
                    )
                })
                .collect::<Vec<_>>();
            let outputs = positions
                .iter()
                .map(|(_, j)| {
                    (
                        one.get_output_mapping(j).unwrap().true_index.clone(),
                        sites[*j].clone(),
                    )
                })
                .collect::<Vec<_>>();
            sequential = tensor4all_treetn::apply_linear_operator_to_indices(
                &one,
                &sequential,
                &inputs,
                &outputs,
                tensor4all_treetn::ApplyOptions::naive(),
            )
            .unwrap();
        }
        assert!(
            actual
                .sub(&sequential.to_dense().unwrap())
                .unwrap()
                .maxabs()
                .unwrap()
                < ROUND_OFF
        );
    }
}
#[test]
fn single_axis_matches_existing_operator() {
    let layout = (0..3)
        .map(|bit| AxisBit { axis: 0, bit })
        .collect::<Vec<_>>();
    let sites = (0..3).map(|_| DynIndex::new_dyn(2)).collect::<Vec<_>>();
    let tree = chain(&sites, &values(3), &[0, 1, 2]);
    let new = quantics_fourier_operator_multiaxis(&layout, FourierOptions::forward()).unwrap();
    let old = quantics_fourier_operator(3, FourierOptions::forward()).unwrap();
    let error = apply(&new, &tree, &sites)
        .to_dense()
        .unwrap()
        .sub(&apply(&old, &tree, &sites).to_dense().unwrap())
        .unwrap()
        .maxabs()
        .unwrap();
    assert!(error < ROUND_OFF, "{error:e}");
    check(layout);
}
#[test]
fn grouped_axes() {
    check(
        (0..2)
            .flat_map(|axis| (0..2).map(move |bit| AxisBit { axis, bit }))
            .collect(),
    );
}
#[test]
fn interleaved_axes() {
    check(
        (0..2)
            .flat_map(|bit| (0..2).map(move |axis| AxisBit { axis, bit }))
            .collect(),
    );
}
#[test]
fn rejects_invalid_layouts_and_options() {
    for layout in [
        vec![],
        vec![AxisBit { axis: 0, bit: 0 }],
        vec![AxisBit { axis: 0, bit: 0 }; 2],
        vec![AxisBit { axis: 0, bit: 0 }, AxisBit { axis: 0, bit: 2 }],
        vec![
            AxisBit {
                axis: usize::MAX,
                bit: 0,
            },
            AxisBit {
                axis: usize::MAX,
                bit: 1,
            },
        ],
    ] {
        assert!(matches!(
            quantics_fourier_operator_multiaxis(&layout, FourierOptions::forward()),
            Err(QuanticsTransformError::InvalidConfiguration { .. })
        ));
    }
    let layout = [AxisBit { axis: 0, bit: 0 }, AxisBit { axis: 0, bit: 1 }];
    assert!(matches!(
        quantics_fourier_operator_multiaxis(
            &layout,
            FourierOptions {
                sign: 0.0,
                ..Default::default()
            }
        ),
        Err(QuanticsTransformError::InvalidConfiguration { .. })
    ));
}

#[test]
fn reversed_and_mixed_direction_axes() {
    // Axis 0 runs MSB-to-LSB, axis 1 LSB-to-MSB along the chain.
    check(vec![
        AxisBit { axis: 0, bit: 0 },
        AxisBit { axis: 1, bit: 2 },
        AxisBit { axis: 0, bit: 1 },
        AxisBit { axis: 1, bit: 1 },
        AxisBit { axis: 0, bit: 2 },
        AxisBit { axis: 1, bit: 0 },
    ]);
}

#[test]
fn non_monotone_axis_uses_exact_restructure() {
    check_with_reference(
        vec![
            AxisBit { axis: 0, bit: 1 },
            AxisBit { axis: 1, bit: 0 },
            AxisBit { axis: 0, bit: 0 },
            AxisBit { axis: 1, bit: 1 },
            AxisBit { axis: 0, bit: 2 },
        ],
        false,
    );
}

#[test]
fn interleaved_bonds_are_products_of_axis_bonds() {
    // Direct embedding is exact without swaps: at every cut the bond is the
    // product of the two axes' one-axis bonds there, and the norm is N.
    let r = 10;
    let one = quantics_fourier_operator(r, FourierOptions::forward()).unwrap();
    let axis_bonds = one.mpo().link_dims();
    let bond = |cut: usize| {
        if cut == 0 || cut == r {
            1
        } else {
            axis_bonds[cut - 1]
        }
    };
    let layout = (0..r)
        .flat_map(|bit| (0..2).map(move |axis| AxisBit { axis, bit }))
        .collect::<Vec<_>>();
    let operator = quantics_fourier_operator_multiaxis(&layout, FourierOptions::forward()).unwrap();
    // After site 2k+1 both axes consumed k+1 bits; after 2k, axis 0 is one ahead.
    let expected: Vec<_> = (1..2 * r)
        .map(|sites| bond(sites.div_ceil(2)) * bond(sites / 2))
        .collect();
    assert_eq!(operator.mpo().link_dims(), expected);
    let norm = operator.mpo().clone().norm().unwrap();
    assert!((norm / f64::from(1u32 << r) - 1.0).abs() < 1e-8, "{norm}");
}
