//! Exact sums of many TreeTNs with a selectable construction method.
//!
//! [`TreeTN::exact_sum`] builds the exact sum of `N` networks as a chain of
//! binary adds (the default, and the method the summing routines of this
//! workspace use), as a balanced tree of binary adds, or as one N-ary direct
//! sum, which is much cheaper in time and is chosen explicitly with
//! [`SumEngine::Nary`] or [`SumEngine::NaryWithin`].
//!
//! [`TreeTN::pairwise_sum`] is the same pairwise reduction with the truncation
//! of the caller's choice after every add; its per-add cutoff bounds the
//! end-to-end error only through the number of adds.

use std::hash::Hash;

use tensor4all_core::{IndexLike, TensorLike};

use super::TreeTN;
use crate::error::TreeTNOperationError;
use crate::TruncationOptions;

/// How the exact sum of `N` tensor networks is built.
///
/// Every engine gives the same network, node tensor by node tensor, up to the
/// bond index IDs; they differ in cost.
///
/// | engine | method | time | peak memory |
/// |---|---|---|---|
/// | [`Binary`](Self::Binary) (default) | `((t0 + t1) + t2) + ...` with [`TreeTN::add`] | grows as `N^3` for terms of equal bond dimension: every add copies the partial sum | the partial sum and the next one |
/// | [`Balanced`](Self::Balanced) | the same adds on a balanced binary tree over the terms, `ceil(log2 N)` levels | one `N`-independent reduction level at a time, about `N^2` | one level of partial sums, at most `N/2` of them |
/// | [`Nary`](Self::Nary) | one [`TreeTN::add_many`] | grows as `N^2`: every element is written once | the built sum |
/// | [`NaryWithin`](Self::NaryWithin) | `Nary` when the [estimated size](ExactSum::estimated_nary_bytes) of the built sum is at most `limit_bytes`, else `Binary` | | |
///
/// Measured in tensor4all/tensor4all-rs#1107 at `R = 12` sites, bond `D = 10`,
/// `N = 64` summands, one core, before the `direct_sum` kernel of #1113: the
/// sequential add chain took 8.9 to 9.2 s, the balanced tree 0.64 to 0.66 s
/// (14x less), and the N-ary sum 0.008 to 0.011 s.
///
/// The default is `Binary`, the method the summing routines of the
/// workspace use. `Nary` is opt-in: it allocates the whole summed network at
/// once, without a size check, so it is chosen explicitly. `NaryWithin`
/// bounds that allocation and reports in [`ExactSum::used_nary`] which method
/// ran, so a slow call can be traced to the fallback.
///
/// The estimate counts the built network only. A truncation sweep run on the
/// sum afterwards needs a transient on top of it, measured in
/// tensor4all/tensor4all-rs#1107 at 0.65 to 3.8 times the built sum; leave
/// that margin in `limit_bytes` when the sum is truncated next.
///
/// Measured in tensor4all/tensor4all-rs#1107 (64 complex chains of 12 sites,
/// bond 10, one core), the add chain took 2.5 to 2.7 s and the N-ary sum
/// 0.02 to 0.04 s.
///
/// # Examples
///
/// ```
/// use tensor4all_treetn::SumEngine;
///
/// assert_eq!(SumEngine::default(), SumEngine::Binary);
/// let capped = SumEngine::NaryWithin { limit_bytes: SumEngine::DEFAULT_NARY_LIMIT_BYTES };
/// assert_eq!(capped, SumEngine::NaryWithin { limit_bytes: 256 * 1024 * 1024 });
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum SumEngine {
    /// A chain of binary adds, `((t0 + t1) + t2) + ...`.
    #[default]
    Binary,
    /// The same binary adds on a balanced tree over the terms,
    /// `ceil(log2 N)` levels deep.
    ///
    /// Same sum and more peak memory than [`Binary`](Self::Binary) at large
    /// `N`, because a level holds several partial sums at once, but far less
    /// time: the depth, not `N`, bounds how often a partial sum is copied.
    Balanced,
    /// One N-ary direct sum ([`TreeTN::add_many`]), whatever its size.
    Nary,
    /// One N-ary direct sum when its estimated size is at most
    /// `limit_bytes`, otherwise a chain of binary adds.
    NaryWithin {
        /// Largest estimated size, in bytes, of the built sum for which the
        /// N-ary method runs. [`SumEngine::DEFAULT_NARY_LIMIT_BYTES`] is a
        /// conservative choice.
        limit_bytes: usize,
    },
}

impl SumEngine {
    /// A conservative `limit_bytes` for [`SumEngine::NaryWithin`]: 256 MiB.
    ///
    /// It is the amount a library call may allocate for the built sum
    /// without being asked; raise it explicitly for larger sums.
    pub const DEFAULT_NARY_LIMIT_BYTES: usize = 256 * 1024 * 1024;
}

/// The result of [`TreeTN::exact_sum`]: the sum and how it was built.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{DynIndex, IdxTensor};
/// use tensor4all_treetn::{SumEngine, TreeTN};
///
/// let s = DynIndex::new_dyn(2);
/// let term = TreeTN::<IdxTensor, usize>::from_tensors(
///     vec![IdxTensor::from_dense(vec![s], vec![1.0, 2.0]).unwrap()],
///     vec![0],
/// )
/// .unwrap();
/// let sum = TreeTN::exact_sum(&[&term, &term], SumEngine::Nary).unwrap();
/// assert!(sum.used_nary);
/// assert_eq!(sum.estimated_nary_bytes, 2 * 16);
/// ```
#[derive(Debug, Clone)]
pub struct ExactSum<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    /// The exact sum of the terms.
    pub network: TreeTN<T, V>,
    /// `true` when the N-ary method built the sum, `false` for the chain of
    /// binary adds.
    pub used_nary: bool,
    /// Estimated size in bytes of the N-ary sum, the quantity
    /// [`SumEngine::NaryWithin`] compares with its limit: for every node, the
    /// product of its site dimensions and of its summed bond dimensions, times
    /// 16 bytes (one `Complex64`, an upper bound for every supported scalar
    /// type). It saturates at `usize::MAX`.
    pub estimated_nary_bytes: usize,
}

impl<T, V> TreeTN<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
    T::Index: Eq + Hash,
    <T::Index as IndexLike>::Id: Clone + Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
{
    /// Sum `terms` exactly, building the sum with the method chosen by
    /// `engine`.
    ///
    /// # Arguments
    /// * `terms` - The networks to sum; at least one. All must share the
    ///   topology and the site index IDs of `terms[0]`, and one scalar type.
    /// * `engine` - How the sum is built; see [`SumEngine`] for the trade-off.
    ///   [`SumEngine::default`] is [`SumEngine::Binary`].
    ///
    /// # Returns
    /// The exact sum, which method built it, and the size estimate that
    /// method choice used. Nothing is truncated; a single term is returned as
    /// a clone.
    ///
    /// # Errors
    ///
    /// Returns [`TreeTNOperationError`] when `terms` is empty, when a term
    /// differs from `terms[0]` in topology, site space or scalar type, or when
    /// building a node tensor or the resulting network fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IdxTensor};
    /// use tensor4all_treetn::{SumEngine, TreeTN};
    ///
    /// let (s0, s1) = (DynIndex::new_dyn(2), DynIndex::new_dyn(2));
    /// let bond = DynIndex::new_dyn(1);
    /// let t0 = IdxTensor::from_dense(vec![s0.clone(), bond.clone()], vec![1.0, 2.0]).unwrap();
    /// let t1 = IdxTensor::from_dense(vec![bond, s1.clone()], vec![3.0, 4.0]).unwrap();
    /// let term = TreeTN::<IdxTensor, usize>::from_tensors(vec![t0, t1], vec![0, 1]).unwrap();
    /// let terms = [&term, &term, &term];
    ///
    /// // Built sum: 2 x 3 at each of the two nodes, 16 bytes each.
    /// let binary = TreeTN::exact_sum(&terms, SumEngine::Binary).unwrap();
    /// let nary = TreeTN::exact_sum(&terms, SumEngine::Nary).unwrap();
    /// let capped = TreeTN::exact_sum(&terms, SumEngine::NaryWithin { limit_bytes: 100 }).unwrap();
    /// assert_eq!(nary.estimated_nary_bytes, 2 * 6 * 16);
    /// assert!(!binary.used_nary && nary.used_nary && !capped.used_nary);
    ///
    /// let expected = IdxTensor::from_dense(vec![s0, s1], vec![9.0, 18.0, 12.0, 24.0]).unwrap();
    /// for sum in [binary, nary, capped] {
    ///     assert_eq!(sum.network.link_dims(), vec![3]);
    ///     assert!(sum.network.to_dense().unwrap().distance(&expected).unwrap() < 1e-12);
    /// }
    /// ```
    pub fn exact_sum(
        terms: &[&Self],
        engine: SumEngine,
    ) -> std::result::Result<ExactSum<T, V>, TreeTNOperationError> {
        let first = Self::ensure_summable(terms)?;
        let estimated_nary_bytes = estimate_nary_bytes(terms)?;
        let used_nary = match engine {
            SumEngine::Binary | SumEngine::Balanced => false,
            SumEngine::Nary => true,
            SumEngine::NaryWithin { limit_bytes } => estimated_nary_bytes <= limit_bytes,
        };
        let _ = first;
        let network = if used_nary {
            Self::add_many(terms)?
        } else {
            let order = match engine {
                SumEngine::Balanced => PairwiseSumOrder::Balanced,
                _ => PairwiseSumOrder::Sequential,
            };
            reduce_in_order(terms, order, |left, right| left.add(right))?
        };
        Ok(ExactSum {
            network,
            used_nary,
            estimated_nary_bytes,
        })
    }

    /// Sum `terms` pairwise in `order`, with an optional truncation after
    /// every add.
    ///
    /// One entry point covers both orders and both the exact and the truncating
    /// variants. With `per_add_cutoff = None` the result is exact and equals
    /// [`Self::exact_sum`] with [`SumEngine::Binary`] for
    /// [`PairwiseSumOrder::Sequential`], or with [`SumEngine::Balanced`] for
    /// [`PairwiseSumOrder::Balanced`].
    ///
    /// # Per-add cutoff, not an end-to-end bound
    ///
    /// `per_add_cutoff` is applied to **every** add, so its error accumulates
    /// over the reduction; it is not the end-to-end error the caller gets. With
    /// a relative squared discarded-weight cutoff `tau` and `R` sites, one
    /// truncation discards at most a relative error `2 (R - 1) sqrt(tau)` in the
    /// worst case (one two-site step per edge in each direction, each discarding
    /// at most `sqrt(tau)` of the current norm). Over the reduction, the
    /// relative error of the result is then bounded by
    /// `kappa ((1 + 2 (R - 1) sqrt(tau))^L - 1)`, first order
    /// `2 (R - 1) L kappa sqrt(tau)`, where `L` is the number of adds on the
    /// longest path (`N - 1` for [`PairwiseSumOrder::Sequential`],
    /// `ceil(log2 N)` for [`PairwiseSumOrder::Balanced`]) and `kappa >= 1` is the
    /// largest norm of a partial sum over the norm of the result. `kappa` is not
    /// known before the sum, so the bound is only quantitative once the caller
    /// bounds it.
    ///
    /// To reach an end-to-end relative error `eps` with the sequential chain,
    /// use `tau = eps / (2 (R - 1) (N - 1) kappa))^2`; for squared cutoffs of
    /// the same meaning as `eps` this is the `eps / ((N - 1)^2 kappa^2)` of
    /// tensor4all/tensor4all-rs#1107, first order. The balanced tree has the
    /// same bound with `L = ceil(log2 N)` in place of `N - 1`.
    ///
    /// # Arguments
    /// * `terms` - The networks to sum; at least one, as in
    ///   [`Self::exact_sum`]. `terms[0]` fixes the topology, the site indices
    ///   and the scalar type.
    /// * `order` - [`PairwiseSumOrder::Sequential`] or
    ///   [`PairwiseSumOrder::Balanced`].
    /// * `per_add_cutoff` - Truncation after every add, or `None` for the exact
    ///   sum.
    ///
    /// # Returns
    /// The sum built in `order`. Nothing is truncated without
    /// `per_add_cutoff`, and a single term is returned as a clone.
    ///
    /// # Errors
    ///
    /// Returns [`TreeTNOperationError`] when `terms` is empty, when a term
    /// differs from `terms[0]` in topology, site space or scalar type, when
    /// `per_add_cutoff` is invalid (a negative or non-finite cutoff, a zero
    /// bond cap, or a `center` that is not a node of `terms[0]`), or when an
    /// add or a truncation fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IdxTensor};
    /// use tensor4all_treetn::{PairwiseSumOrder, PerAddCutoff, TreeTN};
    ///
    /// let s = DynIndex::new_dyn(2);
    /// let term = TreeTN::<IdxTensor, usize>::from_tensors(
    ///     vec![IdxTensor::from_dense(vec![s], vec![1.0, 2.0]).unwrap()],
    ///     vec![0],
    /// )
    /// .unwrap();
    /// let terms = [&term, &term, &term];
    ///
    /// let chain = TreeTN::pairwise_sum(&terms, PairwiseSumOrder::Sequential, None).unwrap();
    /// let tree = TreeTN::pairwise_sum(&terms, PairwiseSumOrder::Balanced, None).unwrap();
    /// assert_eq!(chain.to_dense().unwrap().to_vec::<f64>().unwrap(), vec![3.0, 6.0]);
    /// assert!(tree.to_dense().unwrap().distance(&chain.to_dense().unwrap()).unwrap() < 1e-12);
    ///
    /// // Truncate after every add, with a per-add cutoff.
    /// let truncated = TreeTN::pairwise_sum(
    ///     &terms,
    ///     PairwiseSumOrder::Balanced,
    ///     Some(PerAddCutoff { center: &0, cutoff: 1e-12, max_bond_dim: None }),
    /// )
    /// .unwrap();
    /// assert!(truncated.to_dense().unwrap().distance(&chain.to_dense().unwrap()).unwrap() < 1e-10);
    /// ```
    pub fn pairwise_sum(
        terms: &[&Self],
        order: PairwiseSumOrder,
        per_add_cutoff: Option<PerAddCutoff<'_, V>>,
    ) -> std::result::Result<Self, TreeTNOperationError> {
        Self::ensure_summable(terms)?;
        let Some(per_add) = per_add_cutoff else {
            return reduce_in_order(terms, order, |left, right| left.add(right));
        };
        per_add.validate()?;
        let truncation = per_add.truncation_options();
        reduce_in_order(terms, order, |left, right| {
            let sum = left.add(right)?;
            sum.truncate([per_add.center.clone()], truncation)
        })
    }
}

/// Order in which [`TreeTN::pairwise_sum`] combines its terms.
///
/// Both orders build the same sum when nothing is truncated, node tensor by
/// node tensor, up to the bond index IDs.
///
/// # Examples
///
/// ```
/// use tensor4all_treetn::PairwiseSumOrder;
///
/// assert_eq!(PairwiseSumOrder::default(), PairwiseSumOrder::Sequential);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PairwiseSumOrder {
    /// `((t0 + t1) + t2) + ...`: one accumulator, `N - 1` adds, each add's
    /// result carried into the next one.
    #[default]
    Sequential,
    /// A balanced binary tree of adds, `ceil(log2 N)` levels, adjacent terms
    /// paired at each level and an odd term carried up unchanged.
    Balanced,
}

/// Truncation applied after every add of [`TreeTN::pairwise_sum`].
///
/// The cutoff is **per add**, not end-to-end; see
/// [`TreeTN::pairwise_sum`] for how the per-add bound accumulates over the
/// reduction and for the per-add value that reaches a given end-to-end error.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{DynIndex, IdxTensor};
/// use tensor4all_treetn::{PairwiseSumOrder, PerAddCutoff, TreeTN};
///
/// let s = DynIndex::new_dyn(2);
/// let term = TreeTN::<IdxTensor, usize>::from_tensors(
///     vec![IdxTensor::from_dense(vec![s], vec![1.0, 2.0]).unwrap()],
///     vec![0],
/// )
/// .unwrap();
/// let per_add = PerAddCutoff { center: &0, cutoff: 1e-12, max_bond_dim: Some(4) };
/// let sum = TreeTN::pairwise_sum(&[&term, &term], PairwiseSumOrder::Balanced, Some(per_add)).unwrap();
/// assert_eq!(sum.to_dense().unwrap().to_vec::<f64>().unwrap(), vec![2.0, 4.0]);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct PerAddCutoff<'a, V> {
    /// Node every truncation sweep canonicalizes towards. It is not stored in
    /// the result.
    pub center: &'a V,
    /// Relative squared discarded-weight cutoff of one truncation, the
    /// [`TruncationRule::DiscardedTailSum`](tensor4all_core::TruncationRule)
    /// policy of [`TreeTN::truncate`]. It bounds one add, not the sum.
    pub cutoff: f64,
    /// Optional bond cap applied after every add. A cap changes the retained
    /// rank and therefore the cost, not the cutoff bound.
    pub max_bond_dim: Option<usize>,
}

impl<V> PerAddCutoff<'_, V> {
    fn validate(&self) -> std::result::Result<(), TreeTNOperationError> {
        tensor4all_core::validate_svd_truncation_options(self.max_bond_dim, Some(self.policy()))
            .map_err(|error| anyhow::anyhow!("pairwise_sum: {error}"))
            .map_err(TreeTNOperationError::from)
    }

    /// Relative squared discarded-tail budget of one truncation, the policy
    /// the bound in [`TreeTN::pairwise_sum`] is stated for.
    fn policy(&self) -> tensor4all_core::SvdTruncationPolicy {
        tensor4all_core::SvdTruncationPolicy::new(self.cutoff)
            .with_relative()
            .with_squared_values()
            .with_discarded_tail_sum()
    }

    fn truncation_options(&self) -> TruncationOptions {
        let options = TruncationOptions::default().with_svd_policy(self.policy());
        match self.max_bond_dim {
            Some(cap) => options.with_max_bond_dim(cap),
            None => options,
        }
    }
}

/// Reduce `terms` pairwise in `order`, combining each pair with `combine`.
///
/// [`TreeTN::exact_sum`] and [`TreeTN::pairwise_sum`] share this reduction;
/// they differ only in `combine` (a plain add, or an add plus a truncation).
fn reduce_in_order<T, V>(
    terms: &[&TreeTN<T, V>],
    order: PairwiseSumOrder,
    combine: impl Fn(
        &TreeTN<T, V>,
        &TreeTN<T, V>,
    ) -> std::result::Result<TreeTN<T, V>, TreeTNOperationError>,
) -> std::result::Result<TreeTN<T, V>, TreeTNOperationError>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
    T::Index: Eq + Hash,
    <T::Index as IndexLike>::Id: Clone + Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
{
    let first = terms.first().ok_or_else(|| {
        TreeTNOperationError::invalid_configuration(anyhow::anyhow!(
            "pairwise_sum: at least one term is required"
        ))
    })?;
    match order {
        PairwiseSumOrder::Sequential => {
            let mut sum = (*first).clone();
            for term in &terms[1..] {
                sum = combine(&sum, term)?;
            }
            Ok(sum)
        }
        PairwiseSumOrder::Balanced => {
            let mut level: Vec<std::borrow::Cow<'_, TreeTN<T, V>>> = terms
                .iter()
                .map(|term| std::borrow::Cow::Borrowed(*term))
                .collect();
            while level.len() > 1 {
                let mut next = Vec::with_capacity(level.len().div_ceil(2));
                let mut pending = level.into_iter();
                while let Some(left) = pending.next() {
                    match pending.next() {
                        Some(right) => next.push(std::borrow::Cow::Owned(combine(&left, &right)?)),
                        None => next.push(left),
                    }
                }
                level = next;
            }
            level
                .pop()
                .map(std::borrow::Cow::into_owned)
                .ok_or_else(|| {
                    TreeTNOperationError::invalid_configuration(anyhow::anyhow!(
                        "pairwise_sum: at least one term is required"
                    ))
                })
        }
    }
}

/// Size in bytes, at 16 bytes per element, of the N-ary direct sum of
/// `terms`, laid out on `terms[0]`'s topology; saturating.
fn estimate_nary_bytes<T, V>(
    terms: &[&TreeTN<T, V>],
) -> std::result::Result<usize, TreeTNOperationError>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    const BYTES_PER_ELEMENT: usize = 16;
    let first = terms[0];
    let mut total = 0usize;
    for node in first.node_names() {
        let mut elements: usize = first
            .site_space(&node)
            .map(|sites| sites.iter().map(IndexLike::dim).product())
            .unwrap_or(1);
        for neighbor in first.site_index_network().neighbors(&node) {
            let mut bond = 0usize;
            for (term_position, term) in terms.iter().enumerate() {
                let dim = term
                    .edge_between(&node, &neighbor)
                    .and_then(|edge| term.bond_index(edge))
                    .map(IndexLike::dim)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "exact_sum: no bond between {:?} and {:?} in term {term_position}",
                            node,
                            neighbor
                        )
                    })?;
                bond = bond.saturating_add(dim);
            }
            elements = elements.saturating_mul(bond);
        }
        total = total.saturating_add(elements.saturating_mul(BYTES_PER_ELEMENT));
    }
    Ok(total)
}

#[cfg(test)]
mod tests;
