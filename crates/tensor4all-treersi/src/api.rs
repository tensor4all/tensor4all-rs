//! Public product entry points; the execution context and RNG have explicit owners.
use crate::{engine, Result, TreeRsiNode, TreeRsiOptions, TreeRsiResult, TreeRsiScalar};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use tensor4all_core::IdxTensor;
use tensor4all_tensorbackend::{explicit::Session, ExecutionContext};
use tensor4all_treetn::TreeTN;

/// Approximates the simultaneous Hadamard product using the default CPU context.
///
/// `inputs` must be nonempty and share labeled topology and physical indices.
/// `options` controls rank and local pivot selection, not global error. Returns
/// a tree with the original physical indices and local diagnostics. Use
/// [`hadamard_many_in`] for caller-owned contexts. Enabled by `global-defaults`.
///
/// # Errors
/// Returns [`crate::TreeRsiError`] for invalid topology, indices, options,
/// probes or scalar kind, resource limits, unsupported contexts, lost numerical
/// range or a backend failure. All inputs are validated before contraction.
///
/// # Examples
/// ```
/// use tensor4all_core::{DynIndex,IdxTensor};
/// use tensor4all_treetn::TreeTN;
/// use tensor4all_treersi::{hadamard_many,TreeRsiOptions};
/// let t=TreeTN::from_tensors(vec![IdxTensor::from_dense(vec![DynIndex::new_dyn(2)],
///     vec![2.0_f64,3.0])?],vec![0usize])?;
/// let options=TreeRsiOptions { max_bond_dim:Some(2),..Default::default() };
/// let out=hadamard_many::<f64,_>(&[t.clone(),t],&options)?;
/// assert_eq!(out.tree.to_dense()?.to_vec::<f64>()?,vec![4.0,9.0]);
/// # Ok::<(),Box<dyn std::error::Error>>(())
/// ```
#[cfg(feature = "global-defaults")]
pub fn hadamard_many<T: TreeRsiScalar, V: TreeRsiNode>(
    inputs: &[TreeTN<IdxTensor, V>],
    options: &TreeRsiOptions<V>,
) -> Result<TreeRsiResult<V>> {
    let context = ExecutionContext::Cpu(tensor4all_tensorbackend::default_cpu_execution_context());
    hadamard_many_in::<T, V>(inputs, options, &context)
}

/// Approximates the Hadamard product in a caller-owned execution context.
///
/// All `inputs` must belong to `context`. `options.seed` initializes a named
/// ChaCha8 RNG; [`hadamard_many_with_rng_in`] instead consumes a caller's stream.
/// The returned tensor cores also belong to `context`. Tolerances control
/// local pivot selection only; validate actual product accuracy separately.
///
/// A CPU context is entered once for the whole product, so a caller that
/// already holds a session of `context` gets the context's re-entry error;
/// such a caller should use [`hadamard_many_with_rng_on_session`] instead.
///
/// With the `tenferro-cuda` feature a CUDA context is accepted: sketch blocks
/// stay in host buffers, so every contraction uploads its operands and
/// downloads its result. Expect the transfer cost of that naive route. A CUDA
/// run also keeps two host dependencies: each input core is downloaded and
/// rebuilt in the process-global CPU eager context, whose session `to_vec`
/// enters, and the local rank decision uses the process-global CPU context. A
/// caller that already holds a session of that same process-global context
/// therefore fails at the readback instead of nesting into it.
///
/// # Errors
/// Returns [`crate::TreeRsiError`] for incompatible inputs/context, invalid
/// controls/probes, scalar mismatch, allocation bounds, lost floating-point
/// range or a backend failure.
///
/// # Examples
/// ```
/// use tensor4all_core::{DynIndex,IdxTensor};
/// use tensor4all_tensorbackend::{CpuBackend,CpuExecutionContext,ExecutionContext};
/// use tensor4all_treetn::TreeTN;
/// use tensor4all_treersi::{hadamard_many_in,TreeRsiOptions};
/// let context=ExecutionContext::Cpu(CpuExecutionContext::from_backend(CpuBackend::new()).into());
/// let t=TreeTN::from_tensors(vec![IdxTensor::from_dense_in(&context,
///     vec![DynIndex::new_dyn(2)],vec![2.0_f64,3.0])?],vec![0usize])?;
/// let options=TreeRsiOptions { max_bond_dim:Some(2),..Default::default() };
/// let out=hadamard_many_in::<f64,_>(&[t.clone(),t],&options,&context)?;
/// assert_eq!(out.tree.to_dense()?.to_vec::<f64>()?,vec![4.0,9.0]);
/// # Ok::<(),Box<dyn std::error::Error>>(())
/// ```
pub fn hadamard_many_in<T: TreeRsiScalar, V: TreeRsiNode>(
    inputs: &[TreeTN<IdxTensor, V>],
    options: &TreeRsiOptions<V>,
    context: &ExecutionContext,
) -> Result<TreeRsiResult<V>> {
    let mut rng = ChaCha8Rng::seed_from_u64(options.seed);
    hadamard_many_with_rng_in::<T, V, _>(inputs, options, &mut rng, context)
}

/// Approximates the Hadamard product using a caller-owned RNG and context.
///
/// Consumes `rng` directly: input order, sorted node order, then column-major
/// standard-normal probe entries. Explicit probes and dimension-one physical
/// groups consume no draws. `options.seed` is ignored. Inputs and returned
/// cores belong to `context`; consecutive calls consume successive draws. A
/// CUDA context pays one transfer per operand and one per result and keeps the two host
/// dependencies [`hadamard_many_in`] describes.
///
/// # Errors
/// Returns [`crate::TreeRsiError`] for incompatible inputs/context, invalid
/// controls/probes, scalar mismatch, allocation bounds, lost floating-point
/// range or a backend failure. Invalid input cores and required probe matrices
/// are rejected before the random stream advances.
///
/// # Examples
/// ```
/// use rand::SeedableRng;
/// use rand_chacha::ChaCha8Rng;
/// use tensor4all_core::{DynIndex,IdxTensor};
/// use tensor4all_tensorbackend::{default_cpu_execution_context,ExecutionContext};
/// use tensor4all_treetn::TreeTN;
/// use tensor4all_treersi::{hadamard_many_with_rng_in,TreeRsiOptions};
/// let context=ExecutionContext::Cpu(default_cpu_execution_context());
/// let t=TreeTN::from_tensors(vec![IdxTensor::from_dense_in(&context,
///     vec![DynIndex::new_dyn(2)],vec![2.0_f64,3.0])?],vec![0usize])?;
/// let mut rng=ChaCha8Rng::seed_from_u64(7);
/// let options=TreeRsiOptions { max_bond_dim:Some(2),..Default::default() };
/// let out=hadamard_many_with_rng_in::<f64,_,_>(&[t.clone(),t],&options,&mut rng,&context)?;
/// assert_eq!(out.tree.to_dense()?.to_vec::<f64>()?,vec![4.0,9.0]);
/// # Ok::<(),Box<dyn std::error::Error>>(())
/// ```
pub fn hadamard_many_with_rng_in<T: TreeRsiScalar, V: TreeRsiNode, R: Rng + ?Sized>(
    inputs: &[TreeTN<IdxTensor, V>],
    options: &TreeRsiOptions<V>,
    rng: &mut R,
    context: &ExecutionContext,
) -> Result<TreeRsiResult<V>> {
    engine::run_in::<T, V, R>(inputs, options, rng, context)
}

/// Approximates the Hadamard product on a caller-held explicit session.
///
/// Every numerical step -- input readback, sketch contractions and the local
/// rank decision -- runs on `session`; neither the process-global context nor
/// any other context is entered, so this is the entry for code that already
/// runs inside a session (a worker lane, a patch driver stage, an MPI rank).
/// It is CPU-only by construction: there is no CUDA route and no fallback to
/// the default CPU context.
///
/// `inputs` are host tensors: compact host storage (for example published by
/// another session) or untracked eager host values. A tracked, device-resident
/// or structured core is refused. The returned cores are published as compact
/// host storage, so the result stays usable after `session` is released.
/// `rng` is consumed exactly as [`hadamard_many_with_rng_in`] consumes it, and
/// for the same inputs, options, RNG stream and backend configuration both
/// entries return the same tree; [`hadamard_many_with_rng_in`] on a CPU context
/// is this implementation run on that context's session.
///
/// # Errors
/// Returns [`crate::TreeRsiError`] for invalid topology, indices, options or
/// probes, scalar mismatch, allocation bounds, lost floating-point range, a
/// refused input representation or a session failure. Invalid input cores and
/// required probe matrices are rejected before the random stream advances.
///
/// # Examples
/// ```
/// use rand::SeedableRng;
/// use rand_chacha::ChaCha8Rng;
/// use tensor4all_core::{DynIndex, IdxTensor};
/// use tensor4all_tensorbackend::{CpuBackend, CpuExecutionContext};
/// use tensor4all_treetn::TreeTN;
/// use tensor4all_treersi::{hadamard_many_with_rng_on_session, TreeRsiOptions};
/// let context = CpuExecutionContext::from_backend(CpuBackend::new());
/// let t = TreeTN::from_tensors(vec![IdxTensor::from_dense_compact(
///     vec![DynIndex::new_dyn(2)], vec![2.0_f64, 3.0])?], vec![0usize])?;
/// let mut rng = ChaCha8Rng::seed_from_u64(7);
/// let options = TreeRsiOptions { max_bond_dim: Some(2), ..Default::default() };
/// let out = context.with_concrete_session(|session| {
///     hadamard_many_with_rng_on_session::<f64, _, _>(&[t.clone(), t], &options, &mut rng, session)
/// })??;
/// // The session is released; the compact result is still readable.
/// assert_eq!(out.tree.to_dense()?.dense_host_values::<f64>()?, vec![4.0, 9.0]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn hadamard_many_with_rng_on_session<T: TreeRsiScalar, V: TreeRsiNode, R: Rng + ?Sized>(
    inputs: &[TreeTN<IdxTensor, V>],
    options: &TreeRsiOptions<V>,
    rng: &mut R,
    session: &mut Session<'_>,
) -> Result<TreeRsiResult<V>> {
    engine::run_on_session::<T, V, R>(inputs, options, rng, session)
}
