//! Exact sums of many TreeTNs with a selectable construction method.
//!
//! [`TreeTN::exact_sum`] builds the exact sum of `N` networks either as a
//! chain of binary adds (the default, and the method the summing routines of
//! this workspace use) or as one N-ary direct sum, which is much cheaper in
//! time and is chosen explicitly with [`SumEngine::Nary`] or
//! [`SumEngine::NaryWithin`].

use std::hash::Hash;

use tensor4all_core::{IndexLike, TensorLike};

use super::TreeTN;
use crate::error::TreeTNOperationError;

/// How the exact sum of `N` tensor networks is built.
///
/// Both methods give the same network, node tensor by node tensor, up to the
/// bond index IDs; they differ in cost.
///
/// | engine | method | time | peak memory |
/// |---|---|---|---|
/// | [`Binary`](Self::Binary) (default) | `((t0 + t1) + t2) + ...` with [`TreeTN::add`] | grows as `N^3` for terms of equal bond dimension: every add copies the partial sum | the partial sum and the next one |
/// | [`Nary`](Self::Nary) | one [`TreeTN::add_many`] | grows as `N^2`: every element is written once | the built sum |
/// | [`NaryWithin`](Self::NaryWithin) | `Nary` when the [estimated size](ExactSum::estimated_nary_bytes) of the built sum is at most `limit_bytes`, else `Binary` | | |
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
            SumEngine::Binary => false,
            SumEngine::Nary => true,
            SumEngine::NaryWithin { limit_bytes } => estimated_nary_bytes <= limit_bytes,
        };
        let network = if used_nary {
            Self::add_many(terms)?
        } else {
            let mut sum = (*first).clone();
            for term in &terms[1..] {
                sum = sum.add(term)?;
            }
            sum
        };
        Ok(ExactSum {
            network,
            used_nary,
            estimated_nary_bytes,
        })
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
