use std::sync::Arc;

use tenferro::{Tensor as NativeTensor, TensorScalar};

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
/// serve - a tracked value, a device-resident operand, a diagonal, structured or compact payload, or an
/// `f32`/`c32` or non-floating dtype, because this route's scalar carrier is a rank-zero `IdxTensor` whose
/// compact storage holds only `f64` and `c64` - for an operand whose index list repeats an index, for incompatible
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

    let mut lhs_value = materialize_on_session(OP, session, lhs_operand)?;
    let mut rhs_value = materialize_on_session(OP, session, rhs_operand)?;
    if options.lhs_conj {
        lhs_value = session
            .conj(&lhs_value)
            .map_err(|error| session_error(OP, error))?;
    }
    if options.rhs_conj {
        rhs_value = session
            .conj(&rhs_value)
            .map_err(|error| session_error(OP, error))?;
    }

    let value = session
        .contraction(&lhs_value, &spec.axes_a, &rhs_value, &spec.axes_b)
        .map_err(|error| session_error(OP, error))?;

    IdxTensor::publish_dense_session_result(spec.result_indices, &value)
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
    // Planning is metadata too, so it happens before any numerical session work, and the route is decided
    // once: an aligned pair needs only the permutation, an unaligned one needs the planner's axes.
    let route = match alignment_permutation(lhs.indices(), rhs.indices()) {
        Some(perm) => Route::Aligned(perm),
        None => {
            let lhs_indices = lhs.operand_indices_for_contraction(true);
            let rhs_indices = rhs.operand_indices_for_contraction(false);
            Route::Planned(Box::new(
                prepare_contraction(&lhs_indices, &lhs_dims, &rhs_indices, &rhs_dims)
                    .map_err(|error| IdxTensorError::operation(OP, anyhow::Error::new(error)))?,
            ))
        }
    };

    let mut lhs_value = materialize_on_session(OP, session, lhs_operand)?;
    let mut rhs_value = materialize_on_session(OP, session, rhs_operand)?;
    // The left operand is conjugated, exactly as the compatibility route conjugates it.
    lhs_value = session
        .conj(&lhs_value)
        .map_err(|error| session_error(OP, error))?;

    let product = match route {
        // Same index set: align the right operand to the left one's order and contract every axis.
        Route::Aligned(perm) => {
            let axes: Vec<usize> = (0..rhs_dims.len()).collect();
            if perm != axes {
                rhs_value = session
                    .permute(&rhs_value, &perm)
                    .map_err(|error| session_error(OP, error))?;
            }
            session
                .contraction(&lhs_value, &axes, &rhs_value, &axes)
                .map_err(|error| session_error(OP, error))?
        }
        // Different index sets: contract the shared axes and reduce the rest.
        Route::Planned(spec) => {
            let contracted = session
                .contraction(&lhs_value, &spec.axes_a, &rhs_value, &spec.axes_b)
                .map_err(|error| session_error(OP, error))?;
            session
                .sum(&contracted)
                .map_err(|error| session_error(OP, error))?
        }
    };

    // The scalar is taken from the concrete value, not through the tensor's eager representation, which
    // would adopt it into the process-global eager runtime.
    let published = IdxTensor::publish_dense_session_result(Vec::new(), &product)?;
    let scalar = match product.dtype() {
        tenferro::DType::F64 => {
            let value = first_scalar::<f64>(&product)?;
            crate::AnyScalar::from_published(value, published)
        }
        tenferro::DType::C64 => {
            let value = first_scalar::<num_complex::Complex64>(&product)?;
            crate::AnyScalar::from_published(value, published)
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

/// How the session should combine the two operands, decided before any session work.
enum Route {
    /// Both operands carry the same index set: permute the right one into the left one's order.
    Aligned(Vec<usize>),
    /// The index sets differ: contract the planned axes and reduce what is left.
    ///
    /// Boxed because the plan is much larger than the permutation and this enum is matched by value.
    Planned(Box<crate::index_ops::ContractionSpec<crate::defaults::DynIndex>>),
}

/// Read the single element of a rank-zero result.
fn first_scalar<T: TensorScalar>(value: &NativeTensor) -> Result<T, IdxTensorError> {
    value
        .as_slice::<T>()
        .map_err(|error| IdxTensorError::operation(OP, anyhow::Error::new(error)))?
        .first()
        .copied()
        .ok_or_else(|| {
            IdxTensorError::operation(OP, anyhow::anyhow!("the reduced result is empty"))
        })
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

/// Operation name the sum entry point reports; it shares the seam's adaptation, not its identity.
const SUM_OP: &str = "sum_on_session";

/// Sum every element of a dense host tensor on a caller-held explicit session.
///
/// The reduction companion of the F1 seam for the dense host representations this slice serves, so an app
/// can reduce a value inside a region it holds open. The result is a plain scalar, published from the
/// concrete value rather than adopted into any eager runtime.
///
/// **A zero-length axis is not parity with [`IdxTensor::sum`] on purpose** in this slice: the compatibility
/// route accumulates from zero over the payload and answers `0`, while the session's reduction rejects a
/// zero-length axis. The session keeps its backend's refusal rather than inventing a special case, and the
/// difference is covered by a regression test. Everything else in the served domain matches.
///
/// # Errors
///
/// Returns a typed [`IdxTensorError`] under the same conditions as [`contract_pair_on_session`] refuses an
/// operand - a tracked value, a device-resident operand, a diagonal, structured or compact payload, a
/// deferred storage error, an unsupported dtype - when the session rejects the reduction, or when the
/// result cannot be published.
///
/// # Examples
///
/// ```
/// use tensor4all_core::index::{DefaultIndex as Index, DynId};
/// use tensor4all_core::{sum_on_session, IdxTensor};
/// use tensor4all_tensorbackend::CpuExecutionContext;
/// use tenferro_cpu::CpuBackend;
///
/// let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(1)?);
/// let i = Index::new_dyn(2);
/// let k = Index::new_dyn(3);
/// let tensor: IdxTensor =
///     IdxTensor::from_dense(vec![i.clone(), k.clone()], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])?;
///
/// let total = context.with_concrete_session(|session| sum_on_session(session, &tensor))??;
/// assert_eq!(total.real(), 21.0);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn sum_on_session(
    session: &mut tensor4all_tensorbackend::explicit::Session<'_>,
    tensor: &IdxTensor,
) -> Result<crate::AnyScalar, IdxTensorError> {
    // The operand is adapted - and therefore refused - before any session work, as in the other slices. The
    // scalar's carrier here is a rank-zero `IdxTensor`, whose compact storage holds `f64` and `c64`, so this
    // route keeps the narrow policy even though its result is a scalar.
    let operand = tensor.session_dense_operand(SUM_OP)?;
    let value = materialize_on_session(SUM_OP, session, operand)?;
    // `Session::sum` reduces every axis and duplicates an already rank-zero value, so one call covers both.
    let reduced = session
        .sum(&value)
        .map_err(|error| session_error(SUM_OP, error))?;
    let published = IdxTensor::publish_dense_session_result(Vec::new(), &reduced)?;
    let scalar = match reduced.dtype() {
        tenferro::DType::F64 => {
            crate::AnyScalar::from_published(first_scalar::<f64>(&reduced)?, published)
        }
        tenferro::DType::C64 => crate::AnyScalar::from_published(
            first_scalar::<num_complex::Complex64>(&reduced)?,
            published,
        ),
        other => {
            return Err(IdxTensorError::operation(
                SUM_OP,
                anyhow::anyhow!(
                    "the explicit-session route reduces the dtypes this seam serves, not {other:?}"
                ),
            ));
        }
    };
    Ok(scalar)
}

/// Permute a dense host tensor's axes on a caller-held explicit session.
///
/// The numerical counterpart of [`IdxTensor::permute`]: the compatibility entry point materializes an eager
/// transpose, which is numerical work rather than metadata, so an app that holds a region open needs this
/// route too. The permutation convention is the compatibility one - `perm[new_axis] = old_axis` - and the
/// result is the same materialized dense host value with the same permuted index list. An identity
/// permutation is returned unchanged, exactly as the compatibility route returns a clone, and performs no
/// session work.
///
/// # Errors
///
/// Returns a typed [`IdxTensorError`] for a permutation whose length does not match the rank, an axis out of
/// range or repeated - with the shared validator's messages - and, **for a nonidentity permutation**, under
/// the same conditions as [`contract_pair_on_session`] refuses an operand: a tracked value, a
/// device-resident operand, a diagonal, structured or compact payload, a deferred storage error, or an
/// `f32`/`c32` or non-floating dtype. It also returns one when the session rejects the permutation or the
/// result cannot be published.
///
/// **An identity permutation never reaches those refusals**: like the compatibility route it returns the
/// tensor unchanged, whatever its representation or dtype, because it is metadata only. Execution - and
/// therefore the `f64`/`c64` domain of this seam - applies to the nonidentity case.
///
/// # Examples
///
/// ```
/// use tensor4all_core::index::{DefaultIndex as Index, DynId};
/// use tensor4all_core::{permute_on_session, IdxTensor};
/// use tensor4all_tensorbackend::CpuExecutionContext;
/// use tenferro_cpu::CpuBackend;
///
/// let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(1)?);
/// let i = Index::new_dyn(2);
/// let k = Index::new_dyn(3);
/// let tensor: IdxTensor =
///     IdxTensor::from_dense(vec![i.clone(), k.clone()], (1..=6).map(|v| v as f64).collect())?;
///
/// let transposed = context.with_concrete_session(|session| {
///     permute_on_session(session, &tensor, &[1, 0])
/// })??;
/// assert_eq!(transposed.indices()[0], k);
/// assert_eq!(transposed.indices()[1], i);
/// // Column-major with the axes swapped: row major in the old order.
/// assert_eq!(transposed.to_vec::<f64>()?, vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn permute_on_session(
    session: &mut tensor4all_tensorbackend::explicit::Session<'_>,
    tensor: &IdxTensor,
    perm: &[usize],
) -> Result<IdxTensor, IdxTensorError> {
    const PERMUTE_OP: &str = "permute_on_session";
    let rank = tensor.indices().len();
    // The shared validator, before any session work: one definition of the contract for both entry points.
    if let Err(message) = crate::defaults::idx_tensor::validate_permutation(rank, perm) {
        return Err(IdxTensorError::operation(
            PERMUTE_OP,
            anyhow::anyhow!(message),
        ));
    }
    if perm.iter().copied().eq(0..rank) {
        // Identity is metadata only: no session work, exactly as the compatibility route clones.
        return Ok(tensor.clone());
    }

    let operand = tensor.session_dense_operand(PERMUTE_OP)?;
    let value = materialize_on_session(PERMUTE_OP, session, operand)?;
    let input_ids: Vec<usize> = (0..rank).collect();
    let read = tenferro::TensorRead::from_tensor(&value);
    // Output labels in the requested order: new axis `i` takes old axis `perm[i]`.
    let permuted = session
        .einsum_reads(&[read], &[&input_ids], perm)
        .map_err(|error| session_error(PERMUTE_OP, error))?;

    let new_indices: Vec<crate::defaults::DynIndex> = perm
        .iter()
        .map(|&axis| tensor.indices()[axis].clone())
        .collect();
    IdxTensor::publish_dense_session_result(new_indices, &permuted)
}

/// Align a dense host tensor to a caller-supplied index order on a caller-held explicit session.
///
/// The numerical counterpart of [`IdxTensor::permute_indices`], which the app crates use to align operands
/// before a contraction. The index list must be a permutation of the tensor's own indices by full index
/// equality, exactly as the compatibility route requires; the permutation is derived with the same helper,
/// so the two routes refuse the same inputs with the same messages. Execution is
/// [`permute_on_session`]'s, so this route serves the same dense `f64`/`c64` domain, and an identity
/// permutation is metadata only - it returns the same storage under the supplied index list, for any
/// representation, without touching the session.
///
/// # Errors
///
/// Returns a typed [`IdxTensorError`] when the index list is not a permutation of the tensor's indices (the
/// compatibility helper's reason, reported under this entry point's operation name), when a supplied index
/// carries a different dimension than the tensor's matching index (this route's own message), under the same
/// conditions as a nonidentity [`permute_on_session`] refuses an operand - whose errors keep that route's
/// operation name, because the execution is delegated - and when the session rejects the permutation or the
/// result cannot be published. A delegated storage error keeps its original source rather than being
/// re-labelled.
///
/// # Examples
///
/// ```
/// use tensor4all_core::index::{DefaultIndex as Index, DynId};
/// use tensor4all_core::{permute_indices_on_session, IdxTensor};
/// use tensor4all_tensorbackend::CpuExecutionContext;
/// use tenferro_cpu::CpuBackend;
///
/// let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(1)?);
/// let i = Index::new_dyn(2);
/// let k = Index::new_dyn(3);
/// let tensor: IdxTensor =
///     IdxTensor::from_dense(vec![i.clone(), k.clone()], (1..=6).map(|v| v as f64).collect())?;
///
/// let aligned = context.with_concrete_session(|session| {
///     permute_indices_on_session(session, &tensor, &[k.clone(), i.clone()])
/// })??;
/// assert_eq!(aligned.indices()[0], k);
/// assert_eq!(aligned.to_vec::<f64>()?, vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn permute_indices_on_session(
    session: &mut tensor4all_tensorbackend::explicit::Session<'_>,
    tensor: &IdxTensor,
    new_indices: &[crate::defaults::DynIndex],
) -> Result<IdxTensor, IdxTensorError> {
    const OP: &str = "permute_indices_on_session";
    // The compatibility route's own permutation derivation, so invalid index lists are refused identically.
    let perm = crate::defaults::idx_tensor::compute_permutation_from_indices(
        tensor.indices(),
        new_indices,
    )
    .map_err(|error| IdxTensorError::operation(OP, anyhow::Error::new(error)))?;

    // Full index equality ignores the dimension, so a supplied list can carry a different shape than the
    // tensor's matching index. The compatibility route rejects that for a nonidentity order when it builds
    // the result, and this route refuses it before either branch rather than silently keeping the tensor's
    // own dimensions. For an identity order that is deliberately stricter than compatibility, which accepts
    // malformed dimension metadata there.
    for (position, supplied) in new_indices.iter().enumerate() {
        let expected = &tensor.indices()[perm[position]];
        if supplied.dim != expected.dim {
            return Err(IdxTensorError::operation(
                OP,
                anyhow::anyhow!(
                    "index {position} is supplied with dimension {} but the tensor's matching index has dimension {}",
                    supplied.dim,
                    expected.dim
                ),
            ));
        }
    }

    if perm.iter().copied().eq(0..perm.len()) {
        // Identity is metadata only: the same storage and eager cache under the supplied index list, for
        // any representation.
        return Ok(tensor.aligned_identity_metadata(new_indices));
    }

    permute_on_session(session, tensor, &perm)
}

/// Compute the squared Euclidean norm of a dense host tensor on a caller-held explicit session.
///
/// The compatibility entry point accumulates the norm over the tensor's payload through a scaled
/// accumulator), so it computes the same value as the compatibility route rather than an algebraic
/// shortcut. The session's part is the materialization: the operand is adapted and materialized on the
/// caller's session instead of through the compatibility route.
///
/// Note what that accumulator does and does not change here: this API squares the accumulated norm, so for a
/// tensor whose norm is representable but whose square is not - the `1e-200` scale - the returned value is
/// `0` in the compatibility route as well, and [`norm_on_session`] squares-and-roots the same value. The
/// accumulator's benefit is that individually underflowing squares can still add up to a representable
/// result; it is not a promise about any single fixture.
///
/// # Errors
///
/// Returns [`IdxTensorError::NaNInput`] naming `norm_squared_on_session` when an element or the accumulated
/// result is NaN. Otherwise it returns a typed [`IdxTensorError`] for the representation and dtype refusals
/// the shared admission applies - a tracked value, a device-resident operand, a diagonal, structured or
/// compact payload, a deferred storage error, or a non-floating dtype; note that this is the admission's
/// refusal, not the compatibility route's `ScalarTypeMismatch`, and that the contraction route's
/// repeated-index and planning refusals do not apply here - and for a materialization, session-copy or
/// host-read failure while the operand is brought onto the caller's session.
///
/// **`f32` and `c32` operands are served here**, because this route returns a scalar: they are promoted into
/// the same `f64` accumulation the compatibility route uses for them. A route whose result is a dense tensor
/// cannot serve them, because this crate's compact storage holds only `f64` and `c64`.
///
/// # Examples
///
/// ```
/// use tensor4all_core::index::{DefaultIndex as Index, DynId};
/// use tensor4all_core::{norm_squared_on_session, IdxTensor};
/// use tensor4all_tensorbackend::CpuExecutionContext;
/// use tenferro_cpu::CpuBackend;
///
/// let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(1)?);
/// let i = Index::new_dyn(2);
/// let tensor: IdxTensor = IdxTensor::from_dense(vec![i.clone()], vec![3.0_f64, 4.0])?;
///
/// let squared = context.with_concrete_session(|session| {
///     norm_squared_on_session(session, &tensor)
/// })??;
/// assert!((squared - 25.0).abs() < 1e-12);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn norm_squared_on_session(
    session: &mut tensor4all_tensorbackend::explicit::Session<'_>,
    tensor: &IdxTensor,
) -> Result<f64, IdxTensorError> {
    const OP: &str = "norm_squared_on_session";
    let operand = tensor.session_dense_operand(OP)?;
    let value = materialize_on_session(OP, session, operand)?;
    let squared =
        match value.dtype() {
            // The compatibility route accumulates every accepted floating dtype in `f64`, so promoting here is
            // its behaviour rather than a widening of this route's own.
            tenferro::DType::F32 => {
                let values = value
                    .as_slice::<f32>()
                    .map_err(|error| IdxTensorError::operation(OP, anyhow::Error::new(error)))?;
                crate::defaults::idx_tensor::lassq_norm_squared(
                    values
                        .iter()
                        .map(|value| num_complex::Complex64::new(f64::from(*value), 0.0)),
                )
            }
            tenferro::DType::C32 => {
                let values = value
                    .as_slice::<num_complex::Complex32>()
                    .map_err(|error| IdxTensorError::operation(OP, anyhow::Error::new(error)))?;
                crate::defaults::idx_tensor::lassq_norm_squared(values.iter().map(|value| {
                    num_complex::Complex64::new(f64::from(value.re), f64::from(value.im))
                }))
            }
            tenferro::DType::F64 => {
                let values = value
                    .as_slice::<f64>()
                    .map_err(|error| IdxTensorError::operation(OP, anyhow::Error::new(error)))?;
                crate::defaults::idx_tensor::lassq_norm_squared(
                    values
                        .iter()
                        .map(|value| num_complex::Complex64::new(*value, 0.0)),
                )
            }
            tenferro::DType::C64 => {
                let values = value
                    .as_slice::<num_complex::Complex64>()
                    .map_err(|error| IdxTensorError::operation(OP, anyhow::Error::new(error)))?;
                crate::defaults::idx_tensor::lassq_norm_squared(values.iter().copied())
            }
            other => {
                return Err(IdxTensorError::operation(
                    OP,
                    anyhow::anyhow!(
                    "the explicit-session route serves the dtypes this seam serves, not {other:?}"
                ),
                ));
            }
        };
    squared.ok_or(IdxTensorError::NaNInput { operation: OP })
}

/// Compute the Euclidean norm of a dense host tensor on a caller-held explicit session.
///
/// The square root of [`norm_squared_on_session`], exactly as the compatibility route defines
/// [`IdxTensor::norm`] as the square root of [`IdxTensor::norm_squared`].
///
/// # Errors
///
/// Returns [`IdxTensorError::NaNInput`] when an element or the accumulated result is NaN, and
/// [`IdxTensorError::Operation`] for the representation and dtype refusals the shared admission applies (a
/// tracked value, a device-resident operand, a diagonal, structured or compact payload, a deferred storage
/// error, or a non-floating dtype) and for a materialization, session-copy or host-read failure. `f32` and
/// `c32` operands are **served** by this route, promoted into the same `f64` accumulation the compatibility
/// route uses.
/// Both carry `norm_squared_on_session` as their operation name, because the computation is delegated - the
/// same shape as slice 5's delegation and as the compatibility route's own `norm` delegating to
/// `norm_squared`.
///
/// # Examples
///
/// ```
/// use tensor4all_core::index::{DefaultIndex as Index, DynId};
/// use tensor4all_core::{norm_on_session, IdxTensor};
/// use tensor4all_tensorbackend::CpuExecutionContext;
/// use tenferro_cpu::CpuBackend;
///
/// let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(1)?);
/// let i = Index::new_dyn(2);
/// let tensor: IdxTensor = IdxTensor::from_dense(vec![i.clone()], vec![3.0_f64, 4.0])?;
///
/// let norm = context.with_concrete_session(|session| norm_on_session(session, &tensor))??;
/// assert!((norm - 5.0).abs() < 1e-12);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn norm_on_session(
    session: &mut tensor4all_tensorbackend::explicit::Session<'_>,
    tensor: &IdxTensor,
) -> Result<f64, IdxTensorError> {
    Ok(norm_squared_on_session(session, tensor)?.sqrt())
}

/// Materialize one operand as an owned concrete host value of the caller's session.
///
/// Host storage is converted by the existing bridge, which does not enter a runtime; an untracked eager
/// host value is read through its borrowed representation and copied by the session itself.
fn materialize_on_session(
    op: &'static str,
    session: &mut tensor4all_tensorbackend::explicit::Session<'_>,
    operand: SessionDenseOperand<'_>,
) -> Result<tenferro::Tensor, IdxTensorError> {
    match operand {
        SessionDenseOperand::Storage(storage, dims) => {
            tensor4all_tensorbackend::storage_to_native_tensor(&storage, &dims)
                .map_err(|error| anyhow_error(op, anyhow::Error::new(error)))
        }
        SessionDenseOperand::Read(read, rank) => {
            let ids: Vec<usize> = (0..rank).collect();
            session
                .einsum_reads(&[*read], &[&ids], &ids)
                .map_err(|error| session_error(op, error))
        }
    }
}

/// Map a session-side failure into this crate's error, named by its own operation.
fn session_error<E: Into<Box<dyn std::error::Error + Send + Sync>>>(
    op: &'static str,
    error: E,
) -> IdxTensorError {
    IdxTensorError::Operation {
        operation: op,
        source: Arc::from(error.into()),
    }
}

/// Map a planning-side failure into this crate's error, named by its own operation.
fn anyhow_error(op: &'static str, error: anyhow::Error) -> IdxTensorError {
    IdxTensorError::Operation {
        operation: op,
        source: Arc::from(error.into_boxed_dyn_error()),
    }
}
