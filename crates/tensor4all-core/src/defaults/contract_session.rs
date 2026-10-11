use std::sync::Arc;

use crate::defaults::contract::PairwiseContractionOptions;
use crate::defaults::idx_tensor::{IdxTensor, IdxTensorError, SessionDenseOperand};
use crate::index_ops::prepare_contraction;

/// Operation name the refusals and errors of this entry point carry.
///
/// Exceptions are the errors that carry a more specific origin: a deferred storage failure keeps
/// [`IdxTensorError::Storage`], and a planning or publication failure keeps its own operation name.
const OP: &str = "contract_pair_on_session";

/// Contract two dense host tensors on a caller-held explicit concrete session.
///
/// The planning is the compatibility route's own (`operand_indices_for_contraction` plus
/// `prepare_contraction`), so the result's indices - id, prime level and tags - are the ones
/// [`contract_pair_with_operand_options`](crate::contract_pair_with_operand_options) produces. The
/// execution is the caller's session ([`Session::contraction`](tensor4all_tensorbackend::explicit::Session::contraction),
/// which also promotes mixed `f64`/`c64` operands on that session), and the result is published as dense
/// host storage, so neither the process-global session nor an ambient eager runtime is entered and no
/// graph is compiled.
///
/// Operands are adapted by representation rather than by materialization: dense host storage goes through
/// the existing host/native bridge, and an untracked eager host value through its borrowed read
/// representation, which the session consumes itself.
///
/// # Errors
///
/// Returns a typed [`IdxTensorError`] for an operand whose representation or dtype this slice does not
/// serve - a tracked value, a device-resident operand, a diagonal, structured or compact payload, an
/// integer or `f32`/`c32` operand - for an operand whose index list repeats an index, for incompatible
/// index spaces (the planner's shape and dimension errors), when the session rejects the contraction, or
/// when the result cannot be published. A deferred storage error is returned as
/// [`IdxTensorError::Storage`] with its original source.
///
/// # Examples
///
/// ```
/// use tensor4all_core::index::{DefaultIndex as Index, DynId};
/// use tensor4all_core::{contract_pair_on_session, IdxTensor, PairwiseContractionOptions};
/// use tensor4all_tensorbackend::CpuExecutionContext;
/// use tenferro_cpu::CpuBackend;
///
/// let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(1)?);
/// let i = Index::new_dyn(2);
/// let j = Index::new_dyn(3);
/// let k = Index::new_dyn(2);
/// let lhs: IdxTensor =
///     IdxTensor::from_dense(vec![i.clone(), k.clone()], vec![1.0_f64, 2.0, 3.0, 4.0])?;
/// let rhs: IdxTensor = IdxTensor::from_dense(
///     vec![k.clone(), j.clone()],
///     vec![1.0_f64, 1.0, 1.0, 1.0, 1.0, 1.0],
/// )?;
///
/// let product = context.with_concrete_session(|session| {
///     contract_pair_on_session(session, &lhs, &rhs, PairwiseContractionOptions::new())
/// })??;
/// assert_eq!(product.indices().len(), 2);
/// // Column-major over `[i, j]`: each `j` column sums the two `k` entries of that `i` row.
/// assert_eq!(product.to_vec::<f64>()?, vec![4.0, 6.0, 4.0, 6.0, 4.0, 6.0]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn contract_pair_on_session(
    session: &mut tensor4all_tensorbackend::explicit::Session<'_>,
    lhs: &IdxTensor,
    rhs: &IdxTensor,
    options: PairwiseContractionOptions,
) -> Result<IdxTensor, IdxTensorError> {
    // Malformed index lists are refused before planning: a repeated index makes the planner's
    // result-length arithmetic underflow, and `replace_indices` can produce one through public API.
    if !lhs.has_unique_indices() || !rhs.has_unique_indices() {
        return Err(IdxTensorError::Operation {
            operation: OP,
            source: Arc::from(
                anyhow::anyhow!(
                    "every index of an operand must be distinct; a repeated index cannot be contracted"
                )
                .into_boxed_dyn_error(),
            ),
        });
    }

    // Planning: the compatibility route's own preparation, with the same conjugated operand index lists.
    let lhs_indices = lhs.operand_indices_for_contraction(options.lhs_conj);
    let rhs_indices = rhs.operand_indices_for_contraction(options.rhs_conj);
    let lhs_dims = lhs.dims();
    let rhs_dims = rhs.dims();
    let spec = prepare_contraction(&lhs_indices, &lhs_dims, &rhs_indices, &rhs_dims)
        .map_err(|error| IdxTensorError::operation(OP, anyhow::Error::new(error)))?;

    // Adaptation: refuse every representation this slice does not serve before touching a session.
    let lhs_operand = lhs.session_dense_operand(OP)?;
    let rhs_operand = rhs.session_dense_operand(OP)?;

    let mut lhs_value = materialize_on_session(session, lhs_operand)?;
    let mut rhs_value = materialize_on_session(session, rhs_operand)?;
    if options.lhs_conj {
        lhs_value = session.conj(&lhs_value).map_err(session_error)?;
    }
    if options.rhs_conj {
        rhs_value = session.conj(&rhs_value).map_err(session_error)?;
    }

    let value = session
        .contraction(&lhs_value, &spec.axes_a, &rhs_value, &spec.axes_b)
        .map_err(session_error)?;

    IdxTensor::publish_dense_session_result(spec.result_indices.into_vec(), &value)
}

/// Compute the inner product of two dense host tensors on a caller-held explicit session.
///
/// This is the reduction companion of [`contract_pair_on_session`] and the operation #510's TT inner
/// product runs hottest. It mirrors [`IdxTensor::inner_product`]: when both operands carry the same index
/// set the right operand is aligned to the left one's order on the session and contracted with the left
/// operand conjugated, and otherwise the two are contracted over their shared indices and the remaining
/// axes are reduced on the session. The result is a plain scalar.
///
/// # Errors
///
/// Returns a typed [`IdxTensorError`] under exactly the conditions [`contract_pair_on_session`] refuses an
/// operand - a tracked value, a device-resident operand, a diagonal, structured or compact payload, an
/// unsupported dtype, a repeated index, an incompatible index space - when the session rejects the
/// contraction or the reduction, or when the result cannot be published.
///
/// # Examples
///
/// ```
/// use tensor4all_core::index::{DefaultIndex as Index, DynId};
/// use tensor4all_core::{inner_product_on_session, IdxTensor};
/// use tensor4all_tensorbackend::CpuExecutionContext;
/// use tenferro_cpu::CpuBackend;
///
/// let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(1)?);
/// let i = Index::new_dyn(2);
/// let j = Index::new_dyn(2);
/// let lhs: IdxTensor =
///     IdxTensor::from_dense(vec![i.clone(), j.clone()], vec![1.0_f64, 2.0, 3.0, 4.0])?;
/// let rhs: IdxTensor =
///     IdxTensor::from_dense(vec![i.clone(), j.clone()], vec![1.0_f64, 1.0, 1.0, 1.0])?;
///
/// let value = context.with_concrete_session(|session| {
///     inner_product_on_session(session, &lhs, &rhs)
/// })??;
/// // conj(lhs) contracted with rhs over both indices: 1 + 2 + 3 + 4.
/// assert_eq!(value.real(), 10.0);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn inner_product_on_session(
    session: &mut tensor4all_tensorbackend::explicit::Session<'_>,
    lhs: &IdxTensor,
    rhs: &IdxTensor,
) -> Result<crate::AnyScalar, IdxTensorError> {
    // Malformed index lists are refused before planning, as in the contraction entry point.
    if !lhs.has_unique_indices() || !rhs.has_unique_indices() {
        return Err(IdxTensorError::Operation {
            operation: OP,
            source: Arc::from(
                anyhow::anyhow!(
                    "every index of an operand must be distinct; an inner product needs a valid index space"
                )
                .into_boxed_dyn_error(),
            ),
        });
    }

    let lhs_dims = lhs.dims();
    let rhs_dims = rhs.dims();
    // Both operands are adapted - and therefore refused - before either is materialized, so a refusal on
    // the right cannot follow session work on the left.
    let lhs_operand = lhs.session_dense_operand(OP)?;
    let rhs_operand = rhs.session_dense_operand(OP)?;
    // Planning is metadata too, so it happens before any numerical session work.
    let aligned = alignment_permutation(lhs.indices(), rhs.indices());
    let spec = match aligned {
        Some(_) => None,
        None => {
            let lhs_indices = lhs.operand_indices_for_contraction(true);
            let rhs_indices = rhs.operand_indices_for_contraction(false);
            Some(
                prepare_contraction(&lhs_indices, &lhs_dims, &rhs_indices, &rhs_dims)
                    .map_err(|error| IdxTensorError::operation(OP, anyhow::Error::new(error)))?,
            )
        }
    };

    let mut lhs_value = materialize_on_session(session, lhs_operand)?;
    let mut rhs_value = materialize_on_session(session, rhs_operand)?;
    // The left operand is conjugated, exactly as the compatibility route conjugates it.
    lhs_value = session.conj(&lhs_value).map_err(session_error)?;

    let product = match (aligned, spec) {
        // Same index set: align the right operand to the left one's order and contract every axis.
        (Some(perm), None) => {
            let axes: Vec<usize> = (0..rhs_dims.len()).collect();
            if perm != axes {
                rhs_value = session.permute(&rhs_value, &perm).map_err(session_error)?;
            }
            session
                .contraction(&lhs_value, &axes, &rhs_value, &axes)
                .map_err(session_error)?
        }
        // Different index sets: contract the shared axes and reduce the rest.
        (None, Some(spec)) => {
            let contracted = session
                .contraction(&lhs_value, &spec.axes_a, &rhs_value, &spec.axes_b)
                .map_err(session_error)?;
            session.sum(&contracted).map_err(session_error)?
        }
        _ => unreachable!("alignment and planning are decided together"),
    };

    // The scalar is taken from the concrete value, not through the tensor's eager representation, which
    // would adopt it into the process-global eager runtime.
    let published = IdxTensor::publish_dense_session_result(Vec::new(), &product)?;
    let scalar = match product.dtype() {
        tenferro::DType::F64 => {
            let values = product
                .as_slice::<f64>()
                .map_err(|error| IdxTensorError::operation(OP, anyhow::Error::new(error)))?;
            crate::AnyScalar::from_published_f64(values[0], published)
        }
        tenferro::DType::C64 => {
            let values = product
                .as_slice::<num_complex::Complex64>()
                .map_err(|error| IdxTensorError::operation(OP, anyhow::Error::new(error)))?;
            crate::AnyScalar::from_published_c64(values[0], published)
        }
        other => {
            return Err(IdxTensorError::operation(
                OP,
                anyhow::anyhow!(
                    "the explicit-session route returns f64 and c64 scalars, not {other:?}"
                ),
            ));
        }
    };
    Ok(scalar)
}

/// The permutation that reorders `rhs`'s axes into `lhs`'s index order, when both carry the same set.
///
/// Returns `None` when an index of `lhs` does not appear in `rhs`, which is the case the caller handles by
/// contracting the shared axes and reducing the rest.
fn alignment_permutation(
    lhs_indices: &[crate::defaults::DynIndex],
    rhs_indices: &[crate::defaults::DynIndex],
) -> Option<Vec<usize>> {
    if lhs_indices.len() != rhs_indices.len() {
        return None;
    }
    let mut perm = Vec::with_capacity(lhs_indices.len());
    for index in lhs_indices {
        let position = rhs_indices
            .iter()
            .position(|candidate| candidate == index)?;
        perm.push(position);
    }
    Some(perm)
}

/// Materialize one operand as an owned concrete host value of the caller's session.
///
/// Host storage is converted by the existing bridge, which does not enter a runtime; an untracked eager
/// host value is read through its borrowed representation and copied by the session itself.
fn materialize_on_session(
    session: &mut tensor4all_tensorbackend::explicit::Session<'_>,
    operand: SessionDenseOperand<'_>,
) -> Result<tenferro::Tensor, IdxTensorError> {
    match operand {
        SessionDenseOperand::Storage(storage, dims) => {
            tensor4all_tensorbackend::storage_to_native_tensor(&storage, &dims)
                .map_err(|error| anyhow_error(anyhow::Error::new(error)))
        }
        SessionDenseOperand::Read(read, rank) => {
            let ids: Vec<usize> = (0..rank).collect();
            session
                .einsum_reads(&[*read], &[&ids], &ids)
                .map_err(session_error)
        }
    }
}

/// Map a session-side failure into this crate's error.
fn session_error<E: Into<Box<dyn std::error::Error + Send + Sync>>>(error: E) -> IdxTensorError {
    IdxTensorError::Operation {
        operation: OP,
        source: Arc::from(error.into()),
    }
}

/// Map a planning-side failure into this crate's error.
fn anyhow_error(error: anyhow::Error) -> IdxTensorError {
    IdxTensorError::Operation {
        operation: OP,
        source: Arc::from(error.into_boxed_dyn_error()),
    }
}
