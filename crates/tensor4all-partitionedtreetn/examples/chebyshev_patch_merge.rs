//! Piecewise Chebyshev patches turned into a patched interleaved QTT and merged
//! recursively under an L2 budget.
//!
//! The function is the U = 0 Green's function of the 2D square-lattice Hubbard
//! model at tau = beta / 2, `G(k) = -1 / (2 cosh(beta xi_k / 2))` with
//! `xi_k = -2 (cos kx + cos ky) - mu`, on the quadrant `[0, pi)^2` sampled on a
//! `2^R x 2^R` grid. Each accepted Chebyshev box becomes a QTT on its own bits,
//! embedded as a projected network on the interleaved chain
//! `x0, y0, x1, y1, ...` (most significant first), and
//! [`merge_sibling_patches`] merges complete sibling groups while the global RMS
//! error stays below `rtol * rms(G)`.
//!
//! Usage:
//! `cargo run --release -p tensor4all-partitionedtreetn --example chebyshev_patch_merge -- \
//!   [degree=8] [rtol=1e-5] [chis=16,32,64] [betas=10,20,40,80,160] [R=16]`

use std::collections::HashSet;
use std::convert::Infallible;
use std::error::Error;
use std::time::Instant;

use rand::{Rng, SeedableRng};
use tensor4all_core::{outer_product, ColMajorArrayRef, DynIndex, IdxTensor, LinearizationOrder};
use tensor4all_interpolativeqtt::{
    get_chebyshev_grid, interpolate_chebyshev_patches, interpolate_single_scale_nd, ChebyshevPatch,
    ChebyshevPatchOptions, InterpolativeQttOptions, LagrangePolynomials,
};
use tensor4all_partitionedtreetn::adaptive_interpolation::{
    merge_sibling_patches, PatchErrorBound, SiblingPatchMergeOptions,
};
use tensor4all_partitionedtreetn::{PartitionedTreeTN, Projector, SubDomainTreeTN};
use tensor4all_tensorbackend::{CpuBackend, CpuExecutionContext};
use tensor4all_treetn::{SiteIndexNetwork, SplitOptions};

const MU: f64 = -1.0;
const PI: f64 = std::f64::consts::PI;
/// Fraction of the budget the Chebyshev patches may use; the rest is merge slack.
const CHEBYSHEV_SHARE: f64 = 0.1;
/// Share of each patch's own slack spent compressing it before merging.
const LEAF_SHARE: f64 = 0.5;

fn green(point: &[f64], beta: f64) -> f64 {
    let xi = -2.0 * (point[0].cos() + point[1].cos()) - MU;
    -1.0 / (2.0 * (0.5 * beta * xi).cosh())
}

/// The patch's own tensor-product polynomial at a physical point.
fn patch_polynomial(patch: &ChebyshevPatch<f64>, basis: &LagrangePolynomials, x: &[f64]) -> f64 {
    let n = basis.len();
    let weights: Vec<Vec<f64>> = (0..2)
        .map(|axis| {
            let t = (x[axis] - patch.lower[axis]) / (patch.upper[axis] - patch.lower[axis]);
            (0..n)
                .map(|a| basis.evaluate(a, t).unwrap_or(f64::NAN))
                .collect()
        })
        .collect();
    let mut value = 0.0;
    for b in 0..n {
        for a in 0..n {
            value += patch.values[a + n * b] * weights[0][a] * weights[1][b];
        }
    }
    value
}

/// Free parameters of a projected network: tensor sizes, counting a projected
/// site's one-hot tensor once per fixed value.
fn parameters(subdomain: &SubDomainTreeTN<usize>, sites: &[DynIndex]) -> usize {
    let data = subdomain.data();
    (0..sites.len())
        .map(|node| {
            let size: usize = data
                .node_index(&node)
                .and_then(|index| data.tensor(index))
                .map_or(0, |tensor| tensor.dims().iter().product());
            if subdomain.is_projected_at(&sites[node]) {
                size / sites[node].dim
            } else {
                size
            }
        })
        .sum()
}

fn embed(
    patch: &ChebyshevPatch<f64>,
    basis: &LagrangePolynomials,
    degree: usize,
    bits: usize,
    sites: &[DynIndex],
    target: &SiteIndexNetwork<usize, DynIndex>,
) -> Result<SubDomainTreeTN<usize>, Box<dyn Error>> {
    let depth = patch.prefix[0].len();
    // The example binary owns its context and lends the engine to the builder.
    let context = tensor4all_tensorbackend::CpuExecutionContext::from_backend(
        tensor4all_tensorbackend::CpuBackend::with_threads(1)?,
    );
    let (mut tree, local_sites) = context.with_session(|exec| {
        interpolate_single_scale_nd(
            exec,
            |x: &[f64]| patch_polynomial(patch, basis, x),
            &patch.lower,
            &patch.upper,
            bits - depth,
            degree,
            &InterpolativeQttOptions::default().with_tolerance(1e-14),
        )
    })??;
    // Attach the prefix as one-hot sites so that every physical site exists.
    let mut first = tree.node_index(&0).ok_or("missing first QTT node")?;
    for level in (0..depth).rev() {
        let link = DynIndex::new_dyn(1);
        let unit = IdxTensor::ones(std::slice::from_ref(&link))?;
        let endpoint = tree.tensor(first).ok_or("missing QTT endpoint")?;
        tree.replace_tensor(first, outer_product(endpoint, &unit)?)?;
        let prefix = IdxTensor::onehot(&[
            (sites[2 * level].clone(), patch.prefix[0][level]),
            (sites[2 * level + 1].clone(), patch.prefix[1][level]),
            (link.clone(), 0),
        ])?;
        let node = tree.add_tensor(local_sites.len() + level, prefix)?;
        tree.connect(node, &link, first, &link)?;
        first = node;
    }
    // Unfuse each 4-dim digit (axis 0 fastest) into the interleaved pair.
    for (level, site) in local_sites.iter().enumerate() {
        let start = 2 * (depth + level);
        tree = tree.replace_site_index_with_indices(
            site,
            &sites[start..start + 2],
            LinearizationOrder::ColumnMajor,
        )?;
    }
    let tree = tree.split_to(target, &SplitOptions::default())?;
    let projector =
        Projector::from_pairs(patch.prefix.iter().enumerate().flat_map(|(axis, prefix)| {
            prefix
                .iter()
                .enumerate()
                .map(move |(level, &bit)| (sites[2 * level + axis].clone(), bit))
        }))?;
    Ok(SubDomainTreeTN::new(tree, projector)?)
}

struct Summary {
    patches: usize,
    parameters: usize,
    max_bond: usize,
    max_error: f64,
    rel_rms_error: f64,
}

fn summarize(
    partition: &PartitionedTreeTN<usize>,
    sites: &[DynIndex],
    bits: usize,
    beta: f64,
    rms_f: f64,
) -> Result<Summary, Box<dyn Error>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(7);
    let points = 20_000;
    let mut digits = Vec::with_capacity(points * 2 * bits);
    let mut exact = Vec::with_capacity(points);
    let scale = (1u64 << bits) as f64;
    for _ in 0..points {
        let ix: u64 = rng.random_range(0..1u64 << bits);
        let iy: u64 = rng.random_range(0..1u64 << bits);
        for level in 0..bits {
            let shift = bits - 1 - level;
            digits.push(((ix >> shift) & 1) as usize);
            digits.push(((iy >> shift) & 1) as usize);
        }
        exact.push(green(
            &[PI * ix as f64 / scale, PI * iy as f64 / scale],
            beta,
        ));
    }
    let values =
        partition.evaluate_batched(sites, ColMajorArrayRef::new(&digits, &[2 * bits, points])?)?;
    let mut max_error = 0.0_f64;
    let mut squares = 0.0;
    for (value, exact) in values.iter().zip(&exact) {
        let error = (value.real() - exact).abs();
        max_error = max_error.max(error);
        squares += error * error;
    }
    Ok(Summary {
        patches: partition.len(),
        parameters: partition.values().map(|p| parameters(p, sites)).sum(),
        max_bond: partition
            .values()
            .map(|p| p.max_bond_dim())
            .max()
            .unwrap_or(0),
        max_error,
        rel_rms_error: (squares / points as f64).sqrt() / rms_f,
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    let degree: usize = args.get(1).map_or(Ok(8), |s| s.parse())?;
    let rtol: f64 = args.get(2).map_or(Ok(1e-5), |s| s.parse())?;
    let parse_list = |index: usize, default: &str| -> Vec<f64> {
        args.get(index)
            .map_or(default, String::as_str)
            .split(',')
            .filter_map(|s| s.parse().ok())
            .collect()
    };
    let chis: Vec<usize> = parse_list(3, "16,32,64")
        .into_iter()
        .map(|c| c as usize)
        .collect();
    let betas = parse_list(4, "10,20,40,80,160");
    let bits: usize = args.get(5).map_or(Ok(16), |s| s.parse())?;

    let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(1)?);
    let basis = get_chebyshev_grid(degree)?;
    let sites: Vec<DynIndex> = (0..2 * bits).map(|_| DynIndex::new_dyn(2)).collect();
    let mut target = SiteIndexNetwork::new();
    for (node, site) in sites.iter().enumerate() {
        target.add_node(node, HashSet::from([site.clone()]))?;
        if node > 0 {
            target.add_edge(&(node - 1), &node)?;
        }
    }

    println!("# 2D Hubbard U=0 G(k, tau=beta/2), mu={MU}, [0,pi)^2 on 2^{bits} x 2^{bits}, degree={degree}, rtol={rtol:e} (relative L2)");
    println!("# stage chi patches params cheb_params max_bond max_abs_err rel_rms_err seconds");
    for &beta in &betas {
        // RMS of G over the grid, from a fine uniform sample (the budget's scale).
        let m = 2048;
        let mut sum = 0.0;
        for i in 0..m {
            for j in 0..m {
                let g = green(&[PI * i as f64 / m as f64, PI * j as f64 / m as f64], beta);
                sum += g * g;
            }
        }
        let rms_f = (sum / (m * m) as f64).sqrt();
        let tau = rtol * rms_f;

        let start = Instant::now();
        let chebyshev = interpolate_chebyshev_patches(
            &context,
            |points: ColMajorArrayRef<'_, f64>, _exec| {
                Ok::<_, Infallible>(
                    points
                        .data()
                        .chunks_exact(2)
                        .map(|p| green(p, beta))
                        .collect(),
                )
            },
            &[0.0, 0.0],
            &[PI, PI],
            degree,
            &ChebyshevPatchOptions {
                tolerance: CHEBYSHEV_SHARE * tau,
                max_patches: 1_000_000,
                ..Default::default()
            },
        )?;
        let mut subdomains = Vec::new();
        let mut bounds = Vec::new();
        for patch in chebyshev.patches() {
            let subdomain = embed(patch, &basis, degree, bits, &sites, &target)?;
            bounds.push(PatchErrorBound::new(
                subdomain.projector().clone(),
                patch.error_estimate,
            ));
            subdomains.push(subdomain);
        }
        let partition = PartitionedTreeTN::from_subdomains(subdomains)?;
        let built = start.elapsed().as_secs_f64();
        let cheb_params = chebyshev.patches().len() * (degree + 1).pow(2);
        println!("# beta={beta} rms(G)={rms_f:.3e} tau={tau:.3e}");
        let s = summarize(&partition, &sites, bits, beta, rms_f)?;
        println!(
            "beta={beta} unmerged - {} {} {cheb_params} {} {:.2e} {:.2e} {built:.2}",
            s.patches, s.parameters, s.max_bond, s.max_error, s.rel_rms_error
        );
        for &chi in &chis {
            let start = Instant::now();
            let merged = merge_sibling_patches(
                partition.clone(),
                bounds.clone(),
                &sites,
                tau,
                &0,
                &SiblingPatchMergeOptions::new(chi + 1).with_leaf_slack_share(LEAF_SHARE),
            )?;
            let elapsed = start.elapsed().as_secs_f64();
            let s = summarize(&merged.partition, &sites, bits, beta, rms_f)?;
            println!(
                "beta={beta} merged {chi} {} {} - {} {:.2e} {:.2e} {elapsed:.2} (merges {}, cap-rejected {})",
                s.patches, s.parameters, s.max_bond, s.max_error, s.rel_rms_error,
                merged.merge.merges.len(), merged.merge.rejected_by_cap
            );
        }
    }
    Ok(())
}
