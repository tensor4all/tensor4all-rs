//! Bounded input construction and independent dense reference calculations.
//! Fourier construction follows #1130's public reproducer and the #1127 probe.

use tensor4all_core::DynIndex;
use tensor4all_partitionedtreetn::{Projector, SubDomainTreeTN};
use tensor4all_quanticstransform::{quantics_fourier_operator, FourierOptions};
use tensor4all_treetn::{apply_linear_operator_to_indices, ApplyOptions};

use super::sum_benchmark_inputs::{signed, term};
use super::{Config, IdxTensor, Net, Result, C};

pub(super) struct Case {
    pub(super) terms: Vec<Net>,
    pub(super) oracle: IdxTensor,
    pub(super) mask: Vec<bool>,
}

pub(super) fn normalized_dft(input: &[C], nf: usize) -> Result<Vec<C>> {
    if !(1..=6).contains(&nf) {
        return Err("bounded DFT requires 1 <= n_F <= 6".into());
    }
    let points = 1usize << nf;
    if input.is_empty()
        || input.len() > 512
        || !input.len().is_multiple_of(points)
        || input
            .iter()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err("DFT requires a finite, bounded input with complete Fourier blocks".into());
    }
    let mut output = vec![C::new(0.0, 0.0); input.len()];
    for (source, value) in input.iter().enumerate() {
        let mut k_in = 0;
        for j in 0..nf {
            k_in = (k_in << 1) | ((source >> j) & 1);
        }
        for k in 0..points {
            let phase = -2.0 * std::f64::consts::PI * (k * k_in) as f64 / points as f64;
            output[((source >> nf) << nf) | k] +=
                value * C::from_polar(1.0 / (points as f64).sqrt(), phase);
        }
    }
    Ok(output)
}

fn random_chain(sites: &[DynIndex], state: &mut u64) -> Result<Net> {
    let n = sites.len();
    let bonds: Vec<_> = (0..n - 1)
        .map(|i| DynIndex::new_dyn(4.min(1 << (i + 1)).min(1 << (n - 1 - i))))
        .collect();
    let mut tensors = Vec::with_capacity(n);
    for (i, site) in sites.iter().enumerate() {
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
            .map(|_| C::new(signed(state), signed(state)))
            .collect();
        tensors.push(IdxTensor::from_dense(indices, data)?);
    }
    Ok(Net::from_tensors(tensors, (0..n).collect())?)
}

fn fourier_terms(config: &Config, sites: &[DynIndex]) -> Result<(Vec<Net>, IdxTensor)> {
    let nf = config.size;
    let mut seed = (0x5EED_0000 + nf as u64)
        .checked_add(config.input_seed)
        .ok_or("Fourier input seed offset overflows u64")?;
    let tree = random_chain(sites, &mut seed)?;
    let input = tree.to_dense()?.permute_indices(sites)?.to_vec::<C>()?;
    let oracle = IdxTensor::from_dense(sites.to_vec(), normalized_dft(&input, nf)?)?;
    let input_norm = input
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt();
    let parseval_error = (oracle.norm()? / input_norm - 1.0).abs();
    if !parseval_error.is_finite() || parseval_error > 1e-12 {
        return Err(format!("DFT Parseval self-check failed: {parseval_error:e}").into());
    }
    let operator = quantics_fourier_operator(nf, FourierOptions::default())?;
    let in_pairs = (0..nf)
        .map(|node| {
            let index = operator
                .get_input_mapping(&node)
                .ok_or("Fourier input mapping missing")?;
            Ok((index.true_index.clone(), sites[node].clone()))
        })
        .collect::<Result<Vec<_>>>()?;
    let out_pairs = (0..nf)
        .map(|node| {
            let index = operator
                .get_output_mapping(&node)
                .ok_or("Fourier output mapping missing")?;
            Ok((index.true_index.clone(), sites[node].clone()))
        })
        .collect::<Result<Vec<_>>>()?;
    let full = SubDomainTreeTN::from_treetn(tree)?;
    let terms = (0..config.terms)
        .map(|leaf| {
            let projector = Projector::from_pairs(
                (0..nf).map(|j| (sites[j].clone(), (leaf >> (nf - 1 - j)) & 1)),
            )?;
            let projected = full.project(&projector)?.ok_or("zero Fourier leaf")?;
            Ok(apply_linear_operator_to_indices(
                &operator,
                projected.data(),
                &in_pairs,
                &out_pairs,
                ApplyOptions::naive(),
            )?)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((terms, oracle))
}

pub(super) fn build_case(config: &Config) -> Result<Case> {
    let site_count = config.size + if config.family == "fourier" { 3 } else { 0 };
    let sites: Vec<_> = (0..site_count).map(|_| DynIndex::new_dyn(2)).collect();
    let (terms, fourier_oracle) = if config.family == "fourier" {
        let (terms, oracle) = fourier_terms(config, &sites)?;
        (terms, Some(oracle))
    } else {
        let terms = (0..config.terms)
            .map(|i| term(&sites, &config.family, i, config.input_seed))
            .collect::<Result<Vec<_>>>()?;
        (terms, None)
    };
    // Explicit dense reference: validated sizes cap this at 4096 entries.
    let mut sum: Option<IdxTensor> = None;
    let mut absolute_sum = vec![0.0; 1usize << site_count];
    for network in &terms {
        let dense = network.to_dense()?.permute_indices(&sites)?;
        for (total, value) in absolute_sum.iter_mut().zip(dense.to_vec::<C>()?) {
            *total += value.norm();
        }
        sum = Some(match sum {
            Some(acc) => acc.add(&dense)?,
            None => dense,
        });
    }
    let sum = sum.ok_or("fixture has no terms")?;
    let oracle = if let Some(oracle) = fourier_oracle {
        let error = sum.sub(&oracle)?.maxabs()? / oracle.maxabs()?;
        if !error.is_finite() || error > 1e-10 {
            return Err(
                format!("Fourier image sum differs from independent DFT: {error:e}").into(),
            );
        }
        oracle
    } else {
        sum
    };
    let values = oracle.to_vec::<C>()?;
    let mask = values
        .iter()
        .zip(&absolute_sum)
        .map(|(value, denominator)| *denominator > 0.0 && value.norm() / denominator <= 0.1)
        .collect();
    Ok(Case {
        terms,
        oracle,
        mask,
    })
}
