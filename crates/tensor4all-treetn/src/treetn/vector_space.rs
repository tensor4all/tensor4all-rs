//! Scalable vector-space algebra for TreeTN-backed Krylov vectors.

use std::hash::Hash;

use tensor4all_core::{AnyScalar, IndexLike, TensorLike, TensorVectorSpace};

use super::TreeTN;
use crate::TreeTNOperationError;

/// Vector-space operations on nonempty tree tensor networks.
///
/// Linear combinations are exact direct sums; compression belongs in the
/// Krylov solver's truncation callback. Both operands must have the same
/// topology and full site indices. If an operator changes the site indices,
/// explicitly call [`TreeTN::reindex_site_space_like`] on its output when
/// node-by-node matching describes the intended vector space.
///
/// Trait norms canonicalize a copy and leave the input's gauge unchanged.
/// Use `TreeTN::norm(&mut tree)` to request in-place canonicalization instead.
/// Scaling preserves a single canonical center through [`TreeTN::scale`].
/// Existing core-factorization limitations still apply: `IdxTensor` norms
/// fail for f32/Complex32 cores when unitary QR is required.
///
/// Elementwise extrema are unsupported: they would require materializing the
/// full tensor. `is_all_finite` checks stored core components; finite cores do
/// not guarantee that every contracted value is representable.
/// The existing direct-sum host-copy path is not differentiable; this
/// implementation does not promise differentiation through a Krylov solve.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{AnyScalar, DynIndex, IdxTensor, TensorVectorSpace};
/// use tensor4all_core::krylov::{gmres_with_truncation, GmresOptions};
/// use tensor4all_treetn::TreeTN;
///
/// # fn main() -> anyhow::Result<()> {
/// let site = DynIndex::new_dyn(2);
/// let tensor = IdxTensor::from_dense(vec![site], vec![6.0_f64, 8.0])?;
/// let rhs = TreeTN::<_, usize>::from_tensors(vec![tensor], vec![0])?;
/// let initial = rhs.scale(AnyScalar::new_real(0.0))?;
/// let result = gmres_with_truncation(
///     |x: &TreeTN<IdxTensor, usize>| Ok(x.scale(AnyScalar::new_real(2.0))?),
///     &rhs, &initial, &GmresOptions::default(), |_x| Ok(()),
/// )?;
/// let expected = rhs.scale(AnyScalar::new_real(0.5))?;
/// let residual = TensorVectorSpace::sub(&result.solution, &expected)?;
/// assert!(result.converged);
/// assert!(TensorVectorSpace::norm(&residual)? < 1e-12);
/// # Ok(())
/// # }
/// ```
impl<T, V> TensorVectorSpace for TreeTN<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
    <T::Index as IndexLike>::Id: Clone + Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
{
    fn axpby(&self, a: AnyScalar, other: &Self, b: AnyScalar) -> Result<Self, Self::Error> {
        TreeTN::axpby(self, a, other, b)
    }

    fn scale(&self, scalar: AnyScalar) -> Result<Self, Self::Error> {
        TreeTN::scale(self, scalar)
    }

    fn inner_product(&self, other: &Self) -> Result<AnyScalar, Self::Error> {
        self.inner(other)
    }

    fn norm(&self) -> Result<f64, Self::Error> {
        TreeTN::norm(&mut self.clone())
    }

    fn norm_squared(&self) -> Result<f64, Self::Error> {
        TreeTN::norm_squared(&mut self.clone())
    }

    fn is_single_precision(&self) -> bool {
        self.graph
            .graph()
            .node_weights()
            .any(TensorVectorSpace::is_single_precision)
    }

    fn is_complex(&self) -> bool {
        self.graph
            .graph()
            .node_weights()
            .any(TensorVectorSpace::is_complex)
    }

    fn maxabs(&self) -> Result<f64, Self::Error> {
        Err(TreeTNOperationError::invalid_configuration(anyhow::anyhow!(
            "TreeTN::maxabs requires explicit dense materialization; use to_dense()?.maxabs() only for small reference checks, or use a norm-based residual"
        )))
    }

    fn min_nonzero_abs(&self) -> Result<Option<f64>, Self::Error> {
        Err(TreeTNOperationError::invalid_configuration(anyhow::anyhow!(
            "TreeTN::min_nonzero_abs requires explicit dense materialization; use to_dense()?.min_nonzero_abs() only for small reference checks"
        )))
    }

    fn is_all_finite(&self) -> Result<bool, Self::Error> {
        for tensor in self.graph.graph().node_weights() {
            if !tensor.is_all_finite().map_err(anyhow::Error::new)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn validate(&self) -> Result<(), Self::Error> {
        if self.node_count() == 0 {
            return Err(TreeTNOperationError::invalid_graph(anyhow::anyhow!(
                "TreeTN vector-space operations require at least one node; construct a nonempty tree with the intended site indices"
            )));
        }
        self.verify_internal_consistency()?;
        self.validate_ortho_consistency()?;
        for tensor in self.graph.graph().node_weights() {
            TensorVectorSpace::validate(tensor).map_err(anyhow::Error::new)?;
        }
        Ok(())
    }
}
