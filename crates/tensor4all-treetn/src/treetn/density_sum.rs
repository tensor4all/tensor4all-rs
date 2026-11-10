//! Density-matrix compression of a sum without constructing summed bonds.
//!
//! Conceptual reference: [density-matrix MPO/MPS compression](https://tensornetwork.org/mps/algorithms/denmat_mpo_mps/).
//! This implementation extends partial overlap environments and retained local
//! bases to sums on arbitrary trees.

use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

use anyhow::{ensure, Context};
use tensor4all_core::{
    AnyScalar, DynIndex, IdxTensor, IndexLike, LinearizationOrder, TensorContractionLike,
    TensorIndex,
};

use super::contraction::GlobalLeaves;
use super::fit::SumFitEnvironment;
use super::TreeTN;
use crate::{CanonicalForm, TreeTNOperationError};

/// A compressed sum and its squared error bound from discarded density weights.
///
/// Unlike [`super::ExactSum`], this result need not retain every bond direction.
///
/// # Examples
/// ```
/// use tensor4all_core::{DynIndex, IdxTensor};
/// use tensor4all_treetn::TreeTN;
/// let term = TreeTN::from_tensors(
///     vec![IdxTensor::from_dense(vec![DynIndex::new_dyn(2)], vec![1.0, 2.0]).unwrap()],
///     vec![0],
/// ).unwrap();
/// let sum = TreeTN::density_matrix_sum(&[&term, &term], &0, 0.0, None).unwrap();
/// assert_eq!(sum.network.to_dense().unwrap().to_vec::<f64>().unwrap(), vec![2.0, 4.0]);
/// assert_eq!(sum.discarded_weight, 0.0);
/// ```
#[derive(Debug, Clone)]
pub struct DensityMatrixSumResult<V>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    /// The compressed sum, unitary toward the requested center.
    pub network: TreeTN<IdxTensor, V>,
    /// Sum of absolute discarded density-eigenvalue tails.
    ///
    /// This bounds the squared Frobenius error against the exact sum in exact
    /// arithmetic, including any extra loss forced by the bond cap. Floating
    /// point contraction and eigensolver errors are not included.
    pub discarded_weight: f64,
}

impl<V> TreeTN<IdxTensor, V>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    /// Compress a sum using density matrices without constructing summed bonds.
    ///
    /// Every noncentral node is processed once, in the canonicalization
    /// leaves-to-center order. Exact pairwise overlap environments of the
    /// original terms supply the exterior metric, including interference
    /// between different terms. The retained eigenvectors directly form an
    /// isometric output node, so no additional canonicalization sweep is needed.
    ///
    /// # Arguments
    /// * `terms` - Nonempty networks with one shared named tree, identical full
    ///   site indices at every node, and one scalar kind throughout. Any number
    ///   of site indices per node is supported, including site-free nodes.
    /// * `center` - The output orthogonality center; must exist in every term.
    /// * `cutoff` - Finite nonnegative relative discarded density weight per
    ///   edge. Retain the smallest positive rank whose discarded eigenvalue
    ///   tail is at most `cutoff * trace(density)`. Use `0.0` for no
    ///   tolerance-based loss; a small value such as `1e-12` is typical.
    /// * `max_bond_dim` - Optional positive hard cap, applied after the cutoff.
    ///   `None` leaves the retained rank uncapped; a binding cap can exceed the
    ///   cutoff budget, and its full discarded tail is still reported.
    ///
    /// # Returns
    /// The compressed sum and the sum of absolute discarded eigenvalue tails.
    /// In exact arithmetic, `||sum - output||² <= discarded_weight`. Without
    /// loss forced by the cap, a tree with `R` nodes satisfies
    /// `||sum - output|| <= sqrt((R - 1) * cutoff) * ||sum||`.
    /// These bounds exclude floating point errors; density matrices square
    /// singular-value conditioning and cancellation can limit relative accuracy.
    ///
    /// Pairwise environments require quadratic work/storage in the number of
    /// terms. Dense intermediates involve only original pairwise bond spaces
    /// and one node's site and retained-child spaces, never a full physical
    /// tensor or an exact network with summed bond dimensions. Backend tensor
    /// operations preserve AD metadata where supported; eigenvalues are read
    /// as detached primal values for nondifferentiable rank selection.
    /// For an `R`-site chain with `N` terms, input bond bound `D`, site size
    /// `d`, retained bond bound `k`, and `m = d * k`, overlap work/storage is
    /// `O(R * N² * d * D³)` / `O(R * N² * D²)`. Each node adds
    /// `O(N² * (m * D² + m² * D) + m³)` local work. On a branch, `m` is the
    /// product of that node's site dimensions and retained child dimensions.
    ///
    /// # Errors
    /// Returns [`TreeTNOperationError`] for empty or incompatible networks,
    /// mixed scalar kinds, invalid options or center, nonfinite or materially
    /// negative density eigenvalues, or failed contractions/eigendecompositions.
    /// Inputs are left unchanged on every path.
    ///
    /// # Examples
    /// ```
    /// use tensor4all_core::{DynIndex, IdxTensor};
    /// use tensor4all_treetn::TreeTN;
    /// let (s0, s1, bond) = (
    ///     DynIndex::new_dyn(2), DynIndex::new_dyn(2), DynIndex::new_dyn(1),
    /// );
    /// let term = TreeTN::from_tensors(vec![
    ///     IdxTensor::from_dense(vec![s0, bond.clone()], vec![1.0, 2.0]).unwrap(),
    ///     IdxTensor::from_dense(vec![bond, s1], vec![3.0, 4.0]).unwrap(),
    /// ], vec![0, 1]).unwrap();
    /// let result = TreeTN::density_matrix_sum(&[&term, &term], &0, 0.0, Some(1)).unwrap();
    /// let expected = term.to_dense().unwrap().scale(2.0.into()).unwrap();
    /// assert!(result.network.to_dense().unwrap().distance(&expected).unwrap() < 1e-12);
    /// ```
    pub fn density_matrix_sum(
        terms: &[&Self],
        center: &V,
        cutoff: f64,
        max_bond_dim: Option<usize>,
    ) -> Result<DensityMatrixSumResult<V>, TreeTNOperationError> {
        validate_density_sum(terms, center, cutoff, max_bond_dim)?;
        density_matrix_sum_impl(terms, center, cutoff, max_bond_dim)
            .context("density_matrix_sum failed")
            .map_err(TreeTNOperationError::from)
    }
}

fn validate_density_sum<V>(
    terms: &[&TreeTN<IdxTensor, V>],
    center: &V,
    cutoff: f64,
    max_bond_dim: Option<usize>,
) -> Result<(), TreeTNOperationError>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let invalid_configuration =
        |message: &str| TreeTNOperationError::invalid_configuration(anyhow::anyhow!("{message}"));
    let invalid_graph =
        |message: &str| TreeTNOperationError::invalid_graph(anyhow::anyhow!("{message}"));
    if !cutoff.is_finite() || cutoff < 0.0 {
        return Err(invalid_configuration(
            "density_matrix_sum: cutoff must be finite and nonnegative",
        ));
    }
    if max_bond_dim == Some(0) {
        return Err(invalid_configuration(
            "density_matrix_sum: max_bond_dim must be positive",
        ));
    }
    let first = terms
        .first()
        .ok_or_else(|| invalid_configuration("density_matrix_sum requires at least one term"))?;
    if first.node_count() == 0 || first.node_index(center).is_none() {
        return Err(invalid_graph(
            "density_matrix_sum: center is not a node of the terms",
        ));
    }
    TreeTN::ensure_summable(terms).map_err(TreeTNOperationError::invalid_graph)?;
    let scalar_kind = |tensor: &IdxTensor| {
        [
            tensor.is_f32(),
            tensor.is_f64(),
            tensor.is_c32(),
            tensor.is_c64(),
        ]
    };
    let kind = scalar_kind(tree_tensor(first, center)?);
    if !kind.iter().any(|&supported| supported) {
        return Err(invalid_configuration(
            "density_matrix_sum: unsupported scalar kind",
        ));
    }
    for term in terms {
        term.validate_tree()?;
        for node in term.node_names() {
            if scalar_kind(tree_tensor(term, &node)?) != kind {
                return Err(invalid_configuration(
                    "density_matrix_sum: terms must share one scalar kind at every node",
                ));
            }
        }
    }
    Ok(())
}

fn density_matrix_sum_impl<V>(
    terms: &[&TreeTN<IdxTensor, V>],
    center: &V,
    cutoff: f64,
    max_bond_dim: Option<usize>,
) -> anyhow::Result<DensityMatrixSumResult<V>>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let first = terms.first().context("validated terms are empty")?;
    let edge_order = first
        .site_index_network()
        .topology()
        .edges_to_canonicalize_by_names(center)
        .context("failed to build canonicalization edge order")?;
    // Bra links must be fresh even for i == j; shared physical indices alone
    // are contracted inside overlap environments.
    let ket: Vec<_> = terms.iter().map(|term| term.sim_internal_inds()).collect();
    let bra: Vec<_> = ket.iter().map(|term| term.sim_internal_inds()).collect();
    // INVARIANT: cross terms define the density of the sum; this operation-owned
    // cache has the quadratic term-count cost documented by the public API.
    let mut environments: Vec<Vec<_>> = (0..terms.len())
        .map(|_| (0..terms.len()).map(|_| SumFitEnvironment::new()).collect())
        .collect();
    let mut working: Vec<HashMap<V, IdxTensor>> = ket
        .iter()
        .map(|term| {
            term.node_names()
                .into_iter()
                .map(|node| {
                    let tensor = tree_tensor(term, &node)?.clone();
                    Ok((node, tensor))
                })
                .collect::<anyhow::Result<_>>()
        })
        .collect::<anyhow::Result<_>>()?;
    let mut output = Vec::with_capacity(first.node_count());
    let mut names = Vec::with_capacity(first.node_count());
    let mut discarded_weight = 0.0;

    for (child, parent) in &edge_order {
        let locals = working
            .iter_mut()
            .map(|nodes| nodes.remove(child).context("missing reduced child tensor"))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let ket_bonds = ket
            .iter()
            .map(|term| tree_bond(term, child, parent))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let bra_bonds = bra
            .iter()
            .map(|term| tree_bond(term, child, parent))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let mut q: Vec<_> = locals[0]
            .external_indices()
            .into_iter()
            .filter(|index| index != &ket_bonds[0])
            .collect();
        // A site-free leaf still has a one-dimensional local Hilbert space.
        let dummy = q.is_empty().then(|| DynIndex::new_dyn(1));
        let density_locals = if let Some(index) = &dummy {
            q.push(index.clone());
            locals
                .iter()
                .map(|tensor| IdxTensor::stack_along_new_index(&[tensor], index.clone(), -1))
                .collect::<Result<Vec<_>, _>>()?
        } else {
            locals.clone()
        };
        let left: Vec<_> = q
            .iter()
            .map(|index| DynIndex::new_dyn(index.dim()))
            .collect();
        let right: Vec<_> = q
            .iter()
            .map(|index| DynIndex::new_dyn(index.dim()))
            .collect();
        let mut density: Option<IdxTensor> = None;
        for i in 0..terms.len() {
            let local_ket = density_locals[i].replace_indices(&q, &left)?;
            for j in 0..terms.len() {
                let exterior = environments[i][j].get_or_compute(
                    parent,
                    child,
                    &ket[i],
                    &bra[j],
                    &GlobalLeaves,
                )?;
                let local_bra = density_locals[j]
                    .conj()
                    .replaceind(&ket_bonds[j], &bra_bonds[j])?
                    .replace_indices(&q, &right)?;
                let contribution = IdxTensor::contract(&[&local_ket, &exterior, &local_bra])?;
                density = Some(match density {
                    Some(accumulated) => accumulated.add(&contribution)?,
                    None => contribution,
                });
            }
        }
        let (mut basis, tail) = density_basis(
            &density.context("no density contributions")?,
            &q,
            &left,
            &right,
            cutoff,
            max_bond_dim,
        )?;
        discarded_weight += tail;
        ensure!(discarded_weight.is_finite(), "discarded weight overflowed");
        if let Some(index) = dummy {
            basis = basis.select_indices(&[index], &[0])?;
        }
        for (i, local) in locals.iter().enumerate() {
            let coefficients = super::contract_or_outer_product(&basis.conj(), local)?;
            let old_parent = working[i]
                .remove(parent)
                .context("missing reduced parent tensor")?;
            working[i].insert(
                parent.clone(),
                IdxTensor::contract(&[&old_parent, &coefficients])?,
            );
        }
        output.push(basis);
        names.push(child.clone());
    }
    let mut root: Option<IdxTensor> = None;
    for nodes in &mut working {
        let tensor = nodes.remove(center).context("missing root tensor")?;
        root = Some(match root {
            Some(accumulated) => accumulated.add(&tensor)?,
            None => tensor,
        });
    }
    output.push(root.context("no root contributions")?);
    names.push(center.clone());
    let mut network = TreeTN::from_tensors(output, names)?;
    for (child, parent) in edge_order {
        let edge = network
            .edge_between(&child, &parent)
            .context("missing output edge")?;
        network.set_edge_ortho_towards(edge, Some(parent))?;
    }
    network.set_canonical_region([center.clone()])?;
    network.set_canonical_form(Some(CanonicalForm::Unitary));
    Ok(DensityMatrixSumResult {
        network,
        discarded_weight,
    })
}

fn tree_tensor<'a, V>(tree: &'a TreeTN<IdxTensor, V>, node: &V) -> anyhow::Result<&'a IdxTensor>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    tree.node_index(node)
        .and_then(|index| tree.tensor(index))
        .context("missing node tensor")
}

fn tree_bond<V>(tree: &TreeTN<IdxTensor, V>, child: &V, parent: &V) -> anyhow::Result<DynIndex>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    tree.edge_between(child, parent)
        .and_then(|edge| tree.bond_index(edge))
        .cloned()
        .context("missing edge bond")
}

fn density_basis(
    density: &IdxTensor,
    q: &[DynIndex],
    left: &[DynIndex],
    right: &[DynIndex],
    cutoff: f64,
    max_bond_dim: Option<usize>,
) -> anyhow::Result<(IdxTensor, f64)> {
    let dim = q
        .iter()
        .try_fold(1usize, |size, index| size.checked_mul(index.dim()))
        .context("local density dimension overflowed")?;
    let row = DynIndex::new_dyn(dim);
    let column = DynIndex::new_dyn(dim);
    let matrix = density
        .fuse_indices(left, row.clone(), LinearizationOrder::ColumnMajor)?
        .fuse_indices(right, column.clone(), LinearizationOrder::ColumnMajor)?
        .permuteinds(&[row.clone(), column.clone()])?;
    let adjoint = matrix
        .conj()
        .replace_indices(&[row.clone(), column.clone()], &[column, row.clone()])?;
    let matrix = matrix.axpby(AnyScalar::new_real(0.5), &adjoint, AnyScalar::new_real(0.5))?;
    let decomposition = matrix.hermitian_eigendecomposition(1e-12)?;
    let mut weights = decomposition.eigenvalues;
    ensure!(
        weights.iter().all(|value| value.is_finite()),
        "density has nonfinite eigenvalues"
    );
    let scale = weights
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    let epsilon = if matrix.is_f32() || matrix.is_c32() {
        f32::EPSILON as f64
    } else {
        f64::EPSILON
    };
    let roundoff = 64.0 * epsilon * dim as f64 * scale;
    ensure!(
        weights.iter().all(|&value| value >= -roundoff),
        "density has materially negative eigenvalues"
    );
    for value in &mut weights {
        *value = value.max(0.0);
    }
    let mut order: Vec<_> = (0..dim).collect();
    order.sort_by(|&a, &b| weights[b].total_cmp(&weights[a]));
    let trace: f64 = weights.iter().sum();
    ensure!(trace.is_finite(), "density trace overflowed");
    let mut rank = dim;
    let mut tail = 0.0;
    while rank > 1 && tail + weights[order[rank - 1]] <= cutoff * trace {
        rank -= 1;
        tail += weights[order[rank]];
    }
    if let Some(cap) = max_bond_dim {
        while rank > cap {
            rank -= 1;
            tail += weights[order[rank]];
        }
    }
    let vectors = order[..rank]
        .iter()
        .map(|&position| {
            decomposition
                .eigenvectors
                .select_indices(
                    std::slice::from_ref(&decomposition.eigenvector_index),
                    &[position],
                )?
                .unfuse_index(&row, q, LinearizationOrder::ColumnMajor)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let bond = DynIndex::new_bond(rank).map_err(|error| anyhow::anyhow!("{error:?}"))?;
    let refs: Vec<_> = vectors.iter().collect();
    Ok((IdxTensor::stack_along_new_index(&refs, bond, -1)?, tail))
}

#[cfg(test)]
mod tests;
