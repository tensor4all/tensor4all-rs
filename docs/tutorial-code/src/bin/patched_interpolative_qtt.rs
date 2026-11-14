//! Construct complex 2D Chebyshev patches and feed the patched Fourier transform.
//! The dense DFT oracle is explicitly limited to sixteen grid points.
use num_complex::Complex64;
use std::{collections::HashSet, convert::Infallible, error::Error};
use tensor4all_core::{ColMajorArrayRef, DynIndex, IdxTensor, LinearizationOrder};
use tensor4all_interpolativeqtt::{interpolate_patched_nd, PatchedInterpolativeOptions};
use tensor4all_partitionedtreetn::{PartitionedTreeTN, Projector, SubDomainTreeTN};
use tensor4all_quanticstransform::{
    patched_fourier_transform, FourierOptions, PatchedFourierOptions,
};
use tensor4all_tensorbackend::CpuBackend;
use tensor4all_treetn::{
    insert_onehot_site_in_treetn_chain, tensor_train_to_treetn_with_names, SiteIndexNetwork,
    SplitOptions,
};

fn green(point: &[f64]) -> Complex64 {
    let energy = (2.0 * std::f64::consts::PI * point[0]).cos()
        + (2.0 * std::f64::consts::PI * point[1]).cos();
    Complex64::new(1.0, 0.0) / Complex64::new(0.2 - energy, 0.1)
}

fn main() -> Result<(), Box<dyn Error>> {
    // ANCHOR: construction
    let options = PatchedInterpolativeOptions {
        tolerance: 1e-8,
        max_bond_dim: Some(1),
        ..Default::default()
    };
    // The caller owns the CPU engine; construction never creates a backend.
    let mut backend = CpuBackend::with_threads(1)?;
    let patches = interpolate_patched_nd(
        backend.engine_mut(),
        |points, _exec| Ok::<_, Infallible>(points.data().chunks(2).map(green).collect()),
        &[0.0, 0.0],
        &[1.0, 1.0],
        2,
        3,
        &options,
    )?;
    assert_eq!(patches.len(), 4);
    assert!(patches
        .iter()
        .all(|patch| patch.prefix.iter().all(|axis| axis.len() == 1)));
    // ANCHOR_END: construction
    let patch_count = patches.len();
    let sites: Vec<_> = (0..4).map(|_| DynIndex::new_dyn(2)).collect();
    let mut target = SiteIndexNetwork::new();
    for (node, site) in sites.iter().enumerate() {
        target.add_node(node, HashSet::from([site.clone()]))?;
        if node > 0 {
            target.add_edge(&(node - 1), &node)?;
        }
    }
    let mut subdomains = Vec::new();
    for patch in patches {
        let depth = patch.prefix[0].len();
        let (mut tree, mut fused) =
            tensor_train_to_treetn_with_names(&patch.tt, (0..(2 - depth)).collect())?;
        // All physical axes must exist before SubDomainTreeTN applies its mask.
        // Insert one-hot fused prefix sites through the existing high-level bridge.
        for level in (0..depth).rev() {
            let site = DynIndex::new_dyn(4);
            let value = patch.prefix[0][level] + 2 * patch.prefix[1][level];
            tree = insert_onehot_site_in_treetn_chain::<Complex64>(tree, 0, site.clone(), value)?;
            fused.insert(0, site);
        }
        for (level, site) in fused.iter().enumerate() {
            tree = tree.replace_site_index_with_indices(
                site,
                &sites[2 * level..2 * level + 2],
                LinearizationOrder::ColumnMajor,
            )?;
        }
        let tree = tree.split_to(&target, &SplitOptions::default())?;
        let projector =
            Projector::from_pairs(patch.prefix.iter().enumerate().flat_map(|(axis, bits)| {
                let sites = &sites;
                bits.iter()
                    .enumerate()
                    .map(move |(level, &bit)| (sites[2 * level + axis].clone(), bit))
            }))?;
        subdomains.push(SubDomainTreeTN::new(tree, projector)?);
    }
    let state = PartitionedTreeTN::from_subdomains(subdomains)?;
    // Exhaustive validation is intentionally restricted to this 4 x 4 reference.
    let points: Vec<_> = (0..16)
        .flat_map(|flat| (0..4).map(move |axis| (flat >> axis) & 1))
        .collect();
    let values = state.evaluate_batched(&sites, ColMajorArrayRef::new(&points, &[4, 16])?)?;
    let input: Vec<_> = (0..16)
        .map(|flat| {
            let kx = 2 * (flat & 1) + ((flat >> 2) & 1);
            let ky = 2 * ((flat >> 1) & 1) + ((flat >> 3) & 1);
            green(&[kx as f64 / 4.0, ky as f64 / 4.0])
        })
        .collect();
    let mut input_error = 0.0_f64;
    for (actual, &expected) in values.iter().zip(&input) {
        input_error = input_error.max((actual.real() - expected.re).abs());
        input_error = input_error.max((actual.imag() - expected.im).abs());
    }
    assert!(input_error < 1e-8, "input residual {input_error}");

    let axes = vec![
        vec![sites[0].clone(), sites[2].clone()],
        vec![sites[1].clone(), sites[3].clone()],
    ];
    let reconstruction = PatchedFourierOptions::default();
    let transformed = patched_fourier_transform(
        &state,
        &0,
        &axes,
        FourierOptions::forward(),
        &reconstruction,
    )?;
    let mut dense: Option<IdxTensor> = None;
    for (_, terms) in transformed.regions() {
        for term in terms {
            let value = term.data().to_dense()?;
            dense = Some(match dense {
                None => value,
                Some(total) => total.add(&value)?,
            });
        }
    }
    let expected: Vec<Complex64> = (0..16)
        .map(|r| {
            // Fourier output reverses significance within each physical axis.
            let rx = 2 * ((r >> 2) & 1) + (r & 1);
            let ry = 2 * ((r >> 3) & 1) + ((r >> 1) & 1);
            input
                .iter()
                .enumerate()
                .map(|(k, &value)| {
                    let kx = 2 * (k & 1) + ((k >> 2) & 1);
                    let ky = 2 * ((k >> 1) & 1) + ((k >> 3) & 1);
                    value
                        * Complex64::from_polar(
                            0.25,
                            -2.0 * std::f64::consts::PI * (kx * rx + ky * ry) as f64 / 4.0,
                        )
                })
                .sum()
        })
        .collect();
    let oracle = IdxTensor::from_dense(sites, expected)?;
    let fourier_error = dense
        .ok_or("expected Fourier output")?
        .sub(&oracle)?
        .maxabs()?;
    assert!(fourier_error < 2e-9, "Fourier residual {fourier_error}");
    println!("patches={patch_count}, input residual={input_error:.3e}, Fourier residual={fourier_error:.3e}");
    Ok(())
}
