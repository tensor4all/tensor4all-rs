//! Small 2D k-space patches to r-space regions, checked against a direct DFT.
//! The dense input and reference are deliberately limited to sixteen points.
use num_complex::Complex64;
use std::collections::HashSet;
use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_partitionedtreetn::{PartitionedTreeTN, Projector, SubDomainTreeTN};
use tensor4all_quanticstransform::{
    patched_fourier_transform, FourierOptions, PatchedFourierOptions,
};
use tensor4all_treetn::{RestructureOptions, SiteIndexNetwork, TreeTN};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Physical path: kx_MSB, ky_MSB, kx_LSB, ky_LSB.
    let sites: Vec<_> = (0..4).map(|_| DynIndex::new_dyn(2)).collect();
    let axes = vec![
        vec![sites[0].clone(), sites[2].clone()],
        vec![sites[1].clone(), sites[3].clone()],
    ];
    // Column-major flat input: physical site 0 varies fastest.
    let input: Vec<_> = (0..16)
        .map(|j| Complex64::new(1.0 + (j as f64 * 0.7).sin(), (j as f64 * 0.3).cos()))
        .collect();
    let source = TreeTN::from_tensors(
        vec![IdxTensor::from_dense(sites.clone(), input.clone())?],
        vec![0usize],
    )?;
    let mut topology = SiteIndexNetwork::new();
    for (j, site) in sites.iter().enumerate() {
        topology.add_node(j, HashSet::from([site.clone()]))?;
        if j > 0 {
            topology.add_edge(&(j - 1), &j)?;
        }
    }
    let tree = source.restructure_to(&topology, &RestructureOptions::default())?;
    // Unequal x-prefix depths: {0, 10, 11}, crossed with y-prefixes {0, 1}.
    let x_prefixes = [
        vec![(sites[0].clone(), 0)],
        vec![(sites[0].clone(), 1), (sites[2].clone(), 0)],
        vec![(sites[0].clone(), 1), (sites[2].clone(), 1)],
    ];
    let mut patches = Vec::new();
    for x in &x_prefixes {
        for y in 0..2 {
            let mut prefix = x.clone();
            prefix.push((sites[1].clone(), y));
            patches.push(SubDomainTreeTN::new(
                tree.clone(),
                Projector::from_pairs(prefix)?,
            )?);
        }
    }
    let state = PartitionedTreeTN::from_subdomains(patches)?;
    let mut reconstruction = PatchedFourierOptions::default();
    reconstruction.tolerance.rtol = 1e-12;
    reconstruction.merge_refine.target_bond_dim = Some(4);
    let result = patched_fourier_transform(
        &state,
        &0,
        &axes,
        FourierOptions::forward(),
        &reconstruction,
    )?;
    assert_eq!(result.report().applied_operator_count, 6);
    assert!(result.report().error_bound <= result.report().absolute_tolerance); // Reconstruction only.
                                                                                // Only this small reference check sums the returned regions densely.
    let mut dense: Option<IdxTensor> = None;
    for (_, terms) in result.regions() {
        for term in terms {
            let value = term.data().to_dense()?;
            dense = Some(match dense {
                None => value,
                Some(total) => total.add(&value)?,
            });
        }
    }
    let dense = dense.ok_or("expected nonzero output")?;
    let expected: Vec<Complex64> = (0..16)
        .map(|r| {
            // Output significance is reversed within each axis.
            let rx = 2 * ((r >> 2) & 1) + (r & 1);
            let ry = 2 * ((r >> 3) & 1) + ((r >> 1) & 1);
            input
                .iter()
                .enumerate()
                .map(|(k, value)| {
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
    let oracle = IdxTensor::from_dense(sites.clone(), expected)?;
    // A separate allowance covers Fourier construction error and roundoff.
    assert!(dense.sub(&oracle)?.maxabs()? < 2e-10);

    // Reversing each axis is required for the normalized inverse round trip.
    let transformed = result.into_partition()?;
    let inverse_axes = axes
        .iter()
        .map(|axis| axis.iter().rev().cloned().collect())
        .collect::<Vec<_>>();
    let restored = patched_fourier_transform(
        &transformed,
        &0,
        &inverse_axes,
        FourierOptions::inverse(),
        &reconstruction,
    )?;
    let mut restored_dense: Option<IdxTensor> = None;
    for (_, terms) in restored.regions() {
        for term in terms {
            let value = term.data().to_dense()?;
            restored_dense = Some(match restored_dense {
                None => value,
                Some(total) => total.add(&value)?,
            });
        }
    }
    let original = IdxTensor::from_dense(sites, input)?;
    assert!(
        restored_dense
            .ok_or("expected restored output")?
            .sub(&original)?
            .maxabs()?
            < 2e-10
    );
    Ok(())
}
