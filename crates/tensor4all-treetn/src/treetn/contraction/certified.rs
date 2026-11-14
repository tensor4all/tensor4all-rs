//! Truncating chain zip-up with a certified error bound (#1286).
//!
//! The sweep replaces each block `C_k` by `P_k C_k`, with `P_k = U_k U_k^H`
//! nested projectors, so the step errors are mutually orthogonal and
//!
//! ```text
//! ||A x - y||^2 = sum_k ||D_k L_k||_F^2,   D_k = (I - P_k) C_k,
//! ```
//!
//! where `L_k` is a factor of the Gram matrix of the exact unswept part of
//! `A x`. The factors are built exactly (untruncated QR) before the sweep; each
//! step keeps the smallest rank whose certified error fits its share of the
//! budget. See "Truncating application with a certified bound" in
//! `docs/design/orthogonal-target-reconstruction.md`.

use std::hash::Hash;

use anyhow::Context;
use tensor4all_core::{
    Canonical, FactorizeAlg, FactorizeOptions, IndexLike, SvdTruncationPolicy, TensorLike,
};

use crate::algorithm::CanonicalForm;
use crate::error::{CertifiedApplyError, TreeTNOperationError};

use super::super::TreeTN;

/// The default memory limit of a certified application: 4 GiB.
pub(crate) const DEFAULT_MEMORY_LIMIT: usize = 4 << 30;

/// The result of [`TreeTN::contract_certified_chain`].
#[derive(Debug)]
pub(crate) struct CertifiedChain<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    pub(crate) network: TreeTN<T, V>,
    pub(crate) step_bounds: Vec<f64>,
    pub(crate) bound: f64,
    pub(crate) estimated_bytes: usize,
    pub(crate) factor_bytes: usize,
}

/// Rounding constants of the certificate: the unit roundoff of the data and
/// `gamma_j = j u / (1 - j u)`.
#[derive(Debug, Clone, Copy)]
struct Rounding {
    unit: f64,
}

impl Rounding {
    fn gamma(self, count: usize) -> f64 {
        let ju = count as f64 * self.unit;
        if ju >= 1.0 {
            f64::INFINITY
        } else {
            ju / (1.0 - ju)
        }
    }

    /// An upper bound of the exact Frobenius norm of a tensor with `count`
    /// entries, from its computed norm.
    fn norm_upper(self, computed: f64, count: usize) -> f64 {
        computed * (1.0 + self.gamma(2 * count + 2))
    }
}

/// Memory estimate of building the factors for one sweep direction, in
/// elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Estimate {
    factor_elements: u128,
    peak_elements: u128,
}

/// Estimate the stored factors and the peak working set of building them,
/// along a chain with joint bond dimensions `rows[k]` (cut between sites `k`
/// and `k + 1`) and output dimensions `outputs[k]`.
///
/// Factor `L_k` has `rows[k]` rows and
/// `cols[k] = min(rows[k], outputs[k + 1] * cols[k + 1])` columns, with one
/// column beyond the last site. The peak adds the largest QR input and its
/// `Q`. Both are exact functions of the bond dimensions. The sweep's working
/// tensors are not included: their size depends on the ranks the sweep keeps,
/// so the sweep checks each block against the limit before forming it.
fn estimate_sweep(rows: &[usize], outputs: &[usize]) -> Estimate {
    let cuts = rows.len();
    let mut cols = vec![0_u128; cuts];
    let mut next_cols = 1_u128;
    for k in (0..cuts).rev() {
        let row = rows[k] as u128;
        cols[k] = row.min((outputs[k + 1] as u128).saturating_mul(next_cols));
        next_cols = cols[k];
    }
    let mut factor_elements = 0_u128;
    let mut transient = 0_u128;
    for k in 0..cuts {
        let row = rows[k] as u128;
        factor_elements = factor_elements.saturating_add(row.saturating_mul(cols[k]));
        let after = if k + 1 < cuts { cols[k + 1] } else { 1 };
        let qr_input = row
            .saturating_mul(outputs[k + 1] as u128)
            .saturating_mul(after);
        transient = transient.max(qr_input.saturating_mul(2));
    }
    Estimate {
        factor_elements,
        peak_elements: factor_elements.saturating_add(transient),
    }
}

/// `indices` without those in `excluded`.
fn all_but_list<I: Clone + PartialEq>(indices: &[I], excluded: &[I]) -> Vec<I> {
    indices
        .iter()
        .filter(|index| !excluded.contains(index))
        .cloned()
        .collect()
}

fn element_count<T: TensorLike>(tensor: &T) -> usize {
    tensor
        .external_indices()
        .iter()
        .map(IndexLike::dim)
        .product()
}

fn operation(source: impl Into<anyhow::Error>) -> CertifiedApplyError {
    CertifiedApplyError::Operation {
        source: TreeTNOperationError::from(source.into()),
    }
}

impl<T, V> TreeTN<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
    <T::Index as IndexLike>::Id: Clone + Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
{
    /// Contract `self` (a state) with `mpo` on a chain, truncating with a
    /// certified bound on `||self * mpo - result||_F`.
    ///
    /// `mpo` must have the state's nodes and edges; the two networks contract
    /// over their shared site indices. The result is canonical at `center`. The
    /// sweep direction is the one whose factors take less memory to build. A
    /// build estimate above `memory_limit` bytes fails before any factor is
    /// built, and a sweep block whose working set with the remaining factors
    /// would exceed it fails before that block is formed.
    pub(crate) fn contract_certified_chain(
        &self,
        mpo: &Self,
        center: &V,
        budget: f64,
        memory_limit: usize,
    ) -> Result<CertifiedChain<T, V>, CertifiedApplyError> {
        let unsupported = |reason: String| CertifiedApplyError::UnsupportedTopology { reason };
        let chain = self.chain_order(center).ok_or_else(|| {
            unsupported(
                "the state is not a chain; certified application supports chains only".into(),
            )
        })?;
        let node_count = chain.len();
        if mpo.node_count() != node_count {
            return Err(unsupported(format!(
                "the operator has {} nodes and the state {node_count}",
                mpo.node_count()
            )));
        }
        for pair in chain.windows(2) {
            if mpo.edge_between(&pair[0], &pair[1]).is_none() {
                return Err(unsupported(format!(
                    "the operator has no bond between the state's neighbours {:?} and {:?}",
                    pair[0], pair[1]
                )));
            }
        }

        // Scalar type of the computation: complex when either operand is,
        // single precision only when every tensor is.
        let tensors = self
            .node_indices()
            .into_iter()
            .filter_map(|node| self.tensor(node))
            .chain(
                mpo.node_indices()
                    .into_iter()
                    .filter_map(|node| mpo.tensor(node)),
            );
        let (mut complex, mut single) = (false, true);
        for tensor in tensors {
            complex |= tensor.is_complex();
            single &= tensor.is_single_precision();
        }
        let element_bytes = (if single { 4 } else { 8 }) * (if complex { 2 } else { 1 });
        let rounding = Rounding {
            unit: if single {
                f64::from(f32::EPSILON) / 2.0
            } else {
                f64::EPSILON / 2.0
            },
        };

        // Memory gate, from the input bond dimensions (upper bounds of the
        // canonicalized ones), before anything is allocated.
        let bond_dim = |network: &Self, a: &V, b: &V| {
            network
                .edge_between(a, b)
                .and_then(|edge| network.bond_index(edge))
                .map_or(1, IndexLike::dim)
        };
        let rows: Vec<usize> = chain
            .windows(2)
            .map(|pair| {
                bond_dim(self, &pair[0], &pair[1]).saturating_mul(bond_dim(mpo, &pair[0], &pair[1]))
            })
            .collect();
        let outputs: Vec<usize> = chain
            .iter()
            .map(|node| {
                let state_sites = self.site_space(node);
                let mpo_sites = mpo.site_space(node);
                let unshared =
                    |own: Option<&std::collections::HashSet<T::Index>>,
                     other: Option<&std::collections::HashSet<T::Index>>| {
                        own.into_iter()
                            .flatten()
                            .filter(|index| other.is_none_or(|other| !other.contains(*index)))
                            .map(IndexLike::dim)
                            .product::<usize>()
                    };
                unshared(mpo_sites, state_sites).saturating_mul(unshared(state_sites, mpo_sites))
            })
            .collect();
        let forward = estimate_sweep(&rows, &outputs);
        let reversed_rows: Vec<usize> = rows.iter().rev().copied().collect();
        let reversed_outputs: Vec<usize> = outputs.iter().rev().copied().collect();
        let backward = estimate_sweep(&reversed_rows, &reversed_outputs);
        let (chain, estimate) = if backward.peak_elements < forward.peak_elements {
            (chain.into_iter().rev().collect::<Vec<V>>(), backward)
        } else {
            (chain, forward)
        };
        let estimated_bytes =
            usize::try_from(estimate.peak_elements.saturating_mul(element_bytes as u128))
                .unwrap_or(usize::MAX);
        if estimated_bytes > memory_limit {
            return Err(CertifiedApplyError::MemoryLimit {
                estimated_bytes,
                limit_bytes: memory_limit,
            });
        }

        if node_count == 1 {
            let node = &chain[0];
            let tensor = |network: &Self| {
                network
                    .node_index(node)
                    .and_then(|index| network.tensor(index))
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("certified zip-up: tensor of {node:?} missing"))
            };
            let product = T::contract(&[
                &tensor(self).map_err(operation)?,
                &tensor(mpo).map_err(operation)?,
            ])
            .context("certified zip-up: single-site product")
            .map_err(operation)?;
            let mut network = TreeTN::new();
            network
                .add_tensor(node.clone(), product)
                .map_err(operation)?;
            network
                .set_canonical_region([node.clone()])
                .map_err(operation)?;
            return Ok(CertifiedChain {
                network,
                step_bounds: Vec::new(),
                bound: 0.0,
                estimated_bytes,
                factor_bytes: 0,
            });
        }

        self.certified_sweep(
            mpo,
            center,
            &chain,
            budget,
            rounding,
            element_bytes,
            estimated_bytes,
            memory_limit,
        )
        .map_err(|error| match error.downcast::<CertifiedApplyError>() {
            Ok(error) => error,
            Err(error) => operation(error),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn certified_sweep(
        &self,
        mpo: &Self,
        center: &V,
        chain: &[V],
        budget: f64,
        rounding: Rounding,
        element_bytes: usize,
        build_estimate: usize,
        memory_limit: usize,
    ) -> anyhow::Result<CertifiedChain<T, V>> {
        let mut estimated_bytes = build_estimate;
        let node_count = chain.len();
        let cuts = node_count - 1;

        // Exact QR form at the first sweep site, as in the plain zip-up, and
        // fresh bond indices so that the operands cannot share one.
        let mut state = self.clone();
        let mut operator = mpo.clone();
        state.canonicalize_impl(
            [chain[0].clone()],
            CanonicalForm::Unitary,
            "certified zip-up: state QR",
        )?;
        operator.canonicalize_impl(
            [chain[0].clone()],
            CanonicalForm::Unitary,
            "certified zip-up: operator QR",
        )?;
        let state = state.sim_internal_inds();
        let operator = operator.sim_internal_inds();

        let site = |network: &Self, node: &V| -> anyhow::Result<T> {
            network
                .node_index(node)
                .and_then(|index| network.tensor(index))
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("certified zip-up: tensor of {node:?} missing"))
        };
        let bond = |network: &Self, a: &V, b: &V| -> anyhow::Result<T::Index> {
            network
                .edge_between(a, b)
                .and_then(|edge| network.bond_index(edge))
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("certified zip-up: bond {a:?}-{b:?} missing"))
        };
        let mut states = Vec::with_capacity(node_count);
        let mut operators = Vec::with_capacity(node_count);
        for node in chain {
            states.push(site(&state, node)?);
            operators.push(site(&operator, node)?);
        }
        let mut joint_bonds = Vec::with_capacity(cuts);
        for pair in chain.windows(2) {
            joint_bonds.push([
                bond(&state, &pair[0], &pair[1])?,
                bond(&operator, &pair[0], &pair[1])?,
            ]);
        }
        let all_but = |tensor: &T, excluded: &[T::Index]| -> Vec<T::Index> {
            tensor
                .external_indices()
                .into_iter()
                .filter(|index| !excluded.contains(index))
                .collect()
        };

        // Factors of the unswept side, built exactly from the far end:
        // `M_k = L_k Q_k` with `L_k` indexed by the joint bond of cut `k` and
        // one new column index.
        let mut factors: Vec<Option<T>> = vec![None; cuts];
        let mut factor_sizes = vec![0_usize; cuts];
        let mut factor_bytes = 0_usize;
        for k in (0..cuts).rev() {
            let site_product = match &factors.get(k + 1).and_then(Option::as_ref) {
                Some(previous) => T::contract(&[previous, &states[k + 1], &operators[k + 1]]),
                None => T::contract(&[&states[k + 1], &operators[k + 1]]),
            }
            .context("certified zip-up: factor contraction")?;
            let columns = all_but(&site_product, &joint_bonds[k]);
            let qr = site_product
                .factorize_full_rank(&columns, FactorizeAlg::QR, Canonical::Left)
                .context("certified zip-up: factor QR")?;
            factor_sizes[k] = element_count(&qr.right).saturating_mul(element_bytes);
            factor_bytes = factor_bytes.saturating_add(factor_sizes[k]);
            factors[k] = Some(qr.right);
        }
        // Factors still stored at step k: those of cuts k and beyond.
        let mut stored_after = vec![0_usize; cuts + 1];
        for k in (0..cuts).rev() {
            stored_after[k] = stored_after[k + 1].saturating_add(factor_sizes[k]);
        }
        let dims = |indices: &[T::Index]| -> usize { indices.iter().map(IndexLike::dim).product() };

        // The bound is inflated once for the rounding of its own sum, so the
        // shares are taken from a budget deflated by the same factor.
        let total_inflation = 1.0 + rounding.gamma(cuts + 2);
        let effective_budget = budget / total_inflation;
        let mut spent_squared = 0.0_f64;
        let mut step_bounds = Vec::with_capacity(cuts);
        let mut result_tensors: Vec<T> = Vec::with_capacity(node_count);
        let mut remainder: Option<T> = None;
        for k in 0..cuts {
            let factor = factors[k]
                .take()
                .ok_or_else(|| anyhow::anyhow!("certified zip-up: factor {k} missing"))?;
            // Check the working set of this step from the actual dimensions:
            // the block, its projection and its discarded part, and the
            // weighted block, next to the factors still stored.
            let incoming = remainder.as_ref().map_or(1, |remainder| {
                dims(&all_but(remainder, &joint_bonds[k - 1]))
            });
            let mut site_legs: Vec<T::Index> = states[k].external_indices();
            for index in operators[k].external_indices() {
                if let Some(position) = site_legs.iter().position(|leg| leg == &index) {
                    site_legs.swap_remove(position);
                } else {
                    site_legs.push(index);
                }
            }
            let mut excluded: Vec<T::Index> = joint_bonds[k].to_vec();
            if k > 0 {
                excluded.extend(joint_bonds[k - 1].iter().cloned());
            }
            let outputs = dims(&all_but_list(&site_legs, &excluded));
            let rows = dims(&joint_bonds[k]);
            let columns = element_count(&factor) / rows.max(1);
            let block_elements = (incoming as u128)
                .saturating_mul(outputs as u128)
                .saturating_mul(rows as u128);
            let weighted_elements = (incoming as u128)
                .saturating_mul(outputs as u128)
                .saturating_mul(columns as u128);
            let working = block_elements
                .saturating_mul(3)
                .saturating_add(weighted_elements)
                .saturating_mul(element_bytes as u128);
            let step_bytes = usize::try_from(working)
                .unwrap_or(usize::MAX)
                .saturating_add(stored_after[k]);
            estimated_bytes = estimated_bytes.max(step_bytes);
            if step_bytes > memory_limit {
                return Err(CertifiedApplyError::MemoryLimit {
                    estimated_bytes: step_bytes,
                    limit_bytes: memory_limit,
                }
                .into());
            }
            let block = match &remainder {
                Some(remainder) => T::contract(&[remainder, &states[k], &operators[k]]),
                None => T::contract(&[&states[k], &operators[k]]),
            }
            .context("certified zip-up: block contraction")?;
            let left = all_but(&block, &joint_bonds[k]);
            let weighted =
                T::contract(&[&block, &factor]).context("certified zip-up: weighted block")?;
            let full = weighted
                .factorize_full_rank(&left, FactorizeAlg::SVD, Canonical::Left)
                .context("certified zip-up: weighted block SVD")?;
            let singular_values = full.singular_values.unwrap_or_default();
            let max_rank = singular_values.len().max(1);

            let remaining_steps = (cuts - k) as f64;
            let share = ((effective_budget * effective_budget - spent_squared).max(0.0)
                / remaining_steps)
                .sqrt();
            let inner = joint_bonds[k].iter().map(IndexLike::dim).product::<usize>();
            let product_entries = element_count(&weighted);
            let block_norm = rounding.norm_upper(block.norm()?, element_count(&block));
            let factor_norm = rounding.norm_upper(factor.norm()?, element_count(&factor));
            let margin_scale = rounding.gamma(2 * inner + 2) * factor_norm;

            // Smallest rank whose predicted certificate fits; the prediction
            // uses the exact-arithmetic tail and `||D_k|| <= ||C_k||`.
            let mut tails = vec![0.0_f64; max_rank + 1];
            for rank in (0..max_rank).rev() {
                let value = singular_values.get(rank).copied().unwrap_or(0.0);
                tails[rank] = (tails[rank + 1] * tails[rank + 1] + value * value).sqrt();
            }
            let mut rank = (1..=max_rank)
                .find(|&rank| {
                    tails[rank] * (1.0 + rounding.gamma(2 * product_entries + 2))
                        + margin_scale * block_norm
                        <= share
                })
                .unwrap_or(max_rank);

            let (kept, next, step_bound) = loop {
                let options = FactorizeOptions::svd()
                    .with_svd_policy(SvdTruncationPolicy::new(0.0))
                    .with_max_bond_dim(rank);
                let kept = weighted
                    .factorize(&left, &options)
                    .context("certified zip-up: truncated SVD")?
                    .left;
                let next =
                    T::contract(&[&kept.conj(), &block]).context("certified zip-up: remainder")?;
                let projected =
                    T::contract(&[&kept, &next]).context("certified zip-up: projection")?;
                let discarded = block.sub(&projected)?;
                let weighted_discarded = T::contract(&[&discarded, &factor])
                    .context("certified zip-up: weighted discarded block")?;
                let step_bound = weighted_discarded.norm()?
                    * (1.0 + rounding.gamma(2 * element_count(&weighted_discarded) + 2))
                    + margin_scale
                        * rounding.norm_upper(discarded.norm()?, element_count(&discarded));
                if step_bound <= share {
                    break (kept, next, step_bound);
                }
                let kept_rank = kept
                    .external_indices()
                    .iter()
                    .filter(|index| !left.contains(index))
                    .map(IndexLike::dim)
                    .product::<usize>();
                // A truncated SVD may keep fewer columns than requested when
                // the trailing singular values are exactly zero, so the next
                // request grows from the larger of the two.
                rank = rank.max(kept_rank) + 1;
                if rank > max_rank {
                    return Err(CertifiedApplyError::BudgetUnreachable {
                        budget,
                        step: k,
                        step_bound,
                        share,
                    }
                    .into());
                }
            };
            spent_squared += step_bound * step_bound;
            step_bounds.push(step_bound);
            result_tensors.push(kept);
            remainder = Some(next);
        }

        // Closing step: nothing is discarded here, so the bound stays valid.
        let last = match &remainder {
            Some(remainder) => T::contract(&[remainder, &states[cuts], &operators[cuts]]),
            None => T::contract(&[&states[cuts], &operators[cuts]]),
        }
        .context("certified zip-up: closing block")?;
        result_tensors.push(last);

        let bound = step_bounds
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt()
            * total_inflation;
        let mut network = TreeTN::from_tensors(result_tensors, chain.to_vec())?;
        // The kept factors are isometries towards the sweep end; one exact QR
        // pass records the canonical form at the requested centre.
        network.canonicalize_impl(
            [center.clone()],
            CanonicalForm::Unitary,
            "certified zip-up: canonical centre",
        )?;
        Ok(CertifiedChain {
            network,
            step_bounds,
            bound,
            estimated_bytes,
            factor_bytes,
        })
    }
}

#[cfg(test)]
mod tests;
