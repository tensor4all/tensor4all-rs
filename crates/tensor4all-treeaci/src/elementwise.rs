//! Public native TreeTN elementwise ACI entry points.

use rand::SeedableRng;
use tensor4all_core::IdxTensor;
use tensor4all_treetn::TreeTN;

use crate::exec::Exec;
use crate::{
    schedule::{run_directional_pass, run_local_sweeps, PassDirection},
    single_site::evaluate_single_site,
    state::TreeAciState,
    Result, TreeAciDiagnostics, TreeAciNode, TreeAciOptions, TreeAciResult, TreeAciScalar,
    TreeElementwiseBatch,
};

/// Approximates a batched pointwise operator directly as a tree tensor network.
///
/// All inputs must have the same labeled tree topology and identical full
/// physical indices at corresponding nodes. A node may own zero, one, or many
/// physical indices; no quantization is required. The callback receives an
/// `n_inputs × n_points` column-major batch.
/// Scalar support includes f32, f64, Complex32, and Complex64. Sampled callback
/// outputs and guard residuals must be finite; non-finite values are rejected
/// with [`crate::TreeAciError::NonFiniteValue`] (or a lower-layer numerical
/// error during local factorization).
///
/// # Arguments
///
/// * `operator` - Fallible batched pointwise operation.
/// * `inputs` - Nonempty, topology-compatible native TreeTNs.
/// * `options` - Sweep, guard, rank, traversal, and allocation controls.
///
/// # Returns
///
/// The interpolated TreeTN, pass histories, termination reason, and diagnostics.
///
/// # Errors
///
/// Returns [`crate::TreeAciError::NoInputs`] or [`crate::TreeAciError::EmptyTree`]
/// for an empty input list or tree, [`crate::TreeAciError::TopologyMismatch`] or
/// [`crate::TreeAciError::PhysicalIndexMismatch`] for incompatible inputs,
/// [`crate::TreeAciError::InvalidOption`] or
/// [`crate::TreeAciError::InvalidInitialGuess`] for a bad option or initial
/// guess, [`crate::TreeAciError::Callback`] when the operator returns an error,
/// [`crate::TreeAciError::NonFiniteValue`] or [`crate::TreeAciError::Numerical`]
/// on a non-finite value or failed factorization, and the resource, size,
/// scalar, batch, and [`crate::TreeAciError::TreeTN`] errors of preparation.
///
/// # Panics
///
/// This function does not intentionally panic. A callback panic is not caught.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{DynIndex, IdxTensor};
/// use tensor4all_treeaci::{tree_elementwise_batched, TreeAciOptions};
/// use tensor4all_treetn::TreeTN;
///
/// let site = DynIndex::new_dyn(2);
/// let tree = TreeTN::from_tensors(
///     vec![IdxTensor::from_dense(vec![site], vec![2.0_f64, 3.0])?],
///     vec![0usize],
/// )?;
/// let result = tree_elementwise_batched::<f64, _, _>(
///     |batch, output| {
///         for (point, value) in output.iter_mut().enumerate() {
///             *value = batch.get(0, point)?.powi(2);
///         }
///         Ok(())
///     },
///     &[tree],
///     &TreeAciOptions::default(),
/// )?;
/// assert_eq!(result.tree.to_dense()?.to_vec::<f64>()?, vec![4.0, 9.0]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn tree_elementwise_batched<T, V, F>(
    operator: F,
    inputs: &[TreeTN<IdxTensor, V>],
    options: &TreeAciOptions<V>,
) -> Result<TreeAciResult<V>>
where
    T: TreeAciScalar,
    V: TreeAciNode,
    F: for<'batch> FnMut(TreeElementwiseBatch<'batch, T>, &mut [T]) -> Result<()>,
{
    // The seeded high-level path uses an explicitly named RNG and delegates;
    // one stream serves the random initial output and every guard search.
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(options.rng_seed);
    tree_elementwise_batched_with_rng(operator, inputs, options, &mut rng)
}

/// Tree elementwise ACI whose run-time execution runs in a caller-owned CPU context.
///
/// The same algorithm, options, stopping rule and diagnostics as
/// [`tree_elementwise_batched_with_rng`]. It enters one session of `context`
/// and runs [`tree_elementwise_batched_with_rng_on_session`] on it, so every
/// numerical step of the run executes in `context`. The operator is called
/// while that session is held: an operator must not enter `context` itself.
///
/// # Errors
///
/// Returns the same errors as [`tree_elementwise_batched_with_rng`], plus
/// [`crate::TreeAciError::Numerical`] when `context` cannot open a session or
/// refuses a numerical step.
pub fn tree_elementwise_batched_with_rng_in<T, V, F, R>(
    operator: F,
    inputs: &[TreeTN<IdxTensor, V>],
    options: &TreeAciOptions<V>,
    rng: &mut R,
    context: &tensor4all_tensorbackend::CpuExecutionContext,
) -> Result<TreeAciResult<V>>
where
    T: TreeAciScalar,
    V: TreeAciNode,
    F: for<'batch> FnMut(TreeElementwiseBatch<'batch, T>, &mut [T]) -> Result<()>,
    R: rand::Rng + ?Sized,
{
    context
        .with_concrete_session(|session| {
            tree_elementwise_batched_with_rng_on_session(operator, inputs, options, rng, session)
        })
        .map_err(|error| crate::TreeAciError::Numerical {
            message: error.to_string(),
        })?
}

/// Tree elementwise ACI on a caller's explicit session.
///
/// The same algorithm, options, stopping rule, diagnostics, and random stream
/// as [`tree_elementwise_batched_with_rng`], with every numerical step - the
/// frame contractions, the candidate-matrix products, the pivoted LU, the
/// guard evaluations, and the final canonicalization - running on `session`.
/// The result's tensors are published as compact host storage, so the result
/// outlives the session, and the run enters no process-global or thread-local
/// runtime. This is the entry for a caller that already holds a session, such
/// as a worker lane of a patch driver.
///
/// The inputs are read through their host payloads without entering a
/// runtime: they must be compact host storage (as session routes publish) or
/// host-contiguous dense values.
///
/// # Errors
///
/// Returns the same errors as [`tree_elementwise_batched_with_rng`], plus
/// [`crate::TreeAciError::Numerical`] or [`crate::TreeAciError::TreeTN`] when
/// `session` refuses a numerical step or an input cannot be read without a
/// runtime.
///
/// # Examples
///
/// ```
/// use rand::SeedableRng;
/// use tensor4all_core::{DynIndex, IdxTensor};
/// use tensor4all_tensorbackend::{explicit::Session, CpuBackend};
/// use tensor4all_treeaci::{tree_elementwise_batched_with_rng_on_session, TreeAciOptions};
/// use tensor4all_treetn::TreeTN;
///
/// let (a, b) = (DynIndex::new_dyn(2), DynIndex::new_dyn(3));
/// let bond = DynIndex::new_dyn(1);
/// let tree = TreeTN::from_tensors(
///     vec![
///         IdxTensor::from_dense(vec![a.clone(), bond.clone()], vec![1.0_f64, 2.0])?,
///         IdxTensor::from_dense(vec![bond, b.clone()], vec![1.0, 2.0, 3.0])?,
///     ],
///     vec![0usize, 1],
/// )?;
/// let mut backend = CpuBackend::with_threads(1)?;
/// let mut session = Session::new(backend.engine_mut());
/// let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
/// // Square the product network (a + 1)(b + 1) pointwise.
/// let result = tree_elementwise_batched_with_rng_on_session::<f64, _, _, _>(
///     |batch, output| {
///         for (point, value) in output.iter_mut().enumerate() {
///             *value = batch.get(0, point)?.powi(2);
///         }
///         Ok(())
///     },
///     &[tree],
///     &TreeAciOptions::default(),
///     &mut rng,
///     &mut session,
/// )?;
/// let dense = result.tree.to_dense()?;
/// let expected = IdxTensor::from_dense(
///     vec![a, b],
///     vec![1.0, 4.0, 4.0, 16.0, 9.0, 36.0],
/// )?;
/// assert!(dense.sub(&expected)?.maxabs()? < 1e-12);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn tree_elementwise_batched_with_rng_on_session<T, V, F, R>(
    operator: F,
    inputs: &[TreeTN<IdxTensor, V>],
    options: &TreeAciOptions<V>,
    rng: &mut R,
    session: &mut tensor4all_tensorbackend::explicit::Session<'_>,
) -> Result<TreeAciResult<V>>
where
    T: TreeAciScalar,
    V: TreeAciNode,
    F: for<'batch> FnMut(TreeElementwiseBatch<'batch, T>, &mut [T]) -> Result<()>,
    R: rand::Rng + ?Sized,
{
    let mut stream: &mut R = rng;
    tree_elementwise_batched_erased(
        operator,
        inputs,
        options,
        &mut stream,
        Exec::Session(session),
    )
}

/// Tree elementwise ACI on a caller-owned random stream.
///
/// Same as [`tree_elementwise_batched`], but consumes `rng` and ignores
/// [`TreeAciOptions::rng_seed`] for the random
/// initial output and for every global guard search instead of deriving a seed
/// per pass, so the caller can reproduce or advance the whole run.
///
/// # Errors
///
/// Returns [`crate::TreeAciError::NoInputs`] or [`crate::TreeAciError::EmptyTree`]
/// for an empty input list or tree, [`crate::TreeAciError::TopologyMismatch`] or
/// [`crate::TreeAciError::PhysicalIndexMismatch`] for incompatible inputs,
/// [`crate::TreeAciError::InvalidOption`] or
/// [`crate::TreeAciError::InvalidInitialGuess`] for a bad option or initial
/// guess, [`crate::TreeAciError::Callback`] when the operator returns an error,
/// [`crate::TreeAciError::NonFiniteValue`] or [`crate::TreeAciError::Numerical`]
/// on a non-finite value or failed factorization, and the resource, size,
/// scalar, batch, and [`crate::TreeAciError::TreeTN`] errors of preparation.
pub fn tree_elementwise_batched_with_rng<T, V, F, R>(
    operator: F,
    inputs: &[TreeTN<IdxTensor, V>],
    options: &TreeAciOptions<V>,
    rng: &mut R,
) -> Result<TreeAciResult<V>>
where
    T: TreeAciScalar,
    V: TreeAciNode,
    F: for<'batch> FnMut(TreeElementwiseBatch<'batch, T>, &mut [T]) -> Result<()>,
    R: rand::Rng + ?Sized,
{
    // Erase the caller's RNG type once so the run below is instantiated once
    // per scalar type instead of once per (scalar, RNG) pair.
    let mut stream: &mut R = rng;
    tree_elementwise_batched_erased(operator, inputs, options, &mut stream, Exec::Global)
}

fn tree_elementwise_batched_erased<T, V, F>(
    mut operator: F,
    inputs: &[TreeTN<IdxTensor, V>],
    options: &TreeAciOptions<V>,
    rng: &mut dyn rand::RngCore,
    mut exec: Exec<'_, '_>,
) -> Result<TreeAciResult<V>>
where
    T: TreeAciScalar,
    V: TreeAciNode,
    F: for<'batch> FnMut(TreeElementwiseBatch<'batch, T>, &mut [T]) -> Result<()>,
{
    // The exact one-node path needs neither bootstrap samples nor frame/state
    // caches. Branch before `TreeAciState::initialize`; `evaluate_single_site`
    // performs the complete public-input and initial-guess validation itself.
    if inputs.first().is_some_and(|input| input.node_count() == 1) {
        return evaluate_single_site(inputs, options, &mut operator, &mut exec);
    }
    let mut state = TreeAciState::<T, V>::initialize_with_rng(inputs, options, rng, &mut exec)?;
    let history = run_local_sweeps(&mut state, options, &mut operator, rng, &mut exec)?;
    let mut evaluated_points = history.evaluated_points;
    if history.needs_cleanup {
        let direction = if history.max_ranks.len() % 2 == 0 {
            PassDirection::Forward
        } else {
            PassDirection::Reverse
        };
        let cleanup =
            run_directional_pass(&mut state, options, direction, &mut operator, &mut exec)?;
        evaluated_points = evaluated_points
            .checked_add(cleanup.evaluated_points)
            .ok_or(crate::TreeAciError::SizeOverflow {
                context: "final evaluated point count",
            })?;
    }
    state.output.verify_internal_consistency()?;
    let edge_ranks = state
        .edge_ranks
        .iter()
        .copied()
        .enumerate()
        .map(|(edge, rank)| {
            let prepared = &state.problem.directed_edges[2 * edge];
            (prepared.from.clone(), prepared.to.clone(), rank)
        })
        .collect::<Vec<_>>();
    let saturated_edges = edge_ranks
        .iter()
        .zip(&state.algebraic_edge_bounds)
        .filter(|((_, _, rank), algebraic)| {
            let limit = options.max_bond_dim.unwrap_or(usize::MAX).min(**algebraic);
            *rank >= limit
        })
        .map(|((from, to, _), _)| (from.clone(), to.clone()))
        .collect();
    Ok(TreeAciResult {
        tree: state.output,
        max_ranks: history.max_ranks,
        max_errors: history.max_errors,
        max_sample_magnitude: state.max_sample_magnitude,
        global_pivots_found: history.global_pivots_found,
        termination: history.termination,
        diagnostics: TreeAciDiagnostics {
            edge_ranks,
            saturated_edges,
            evaluated_points,
            sample_arena_records: state.sample_arena.record_count(),
            sample_arena_retained_bytes: state.sample_arena.retained_bytes(),
            frame_records: state.input_frames.records(),
            frame_retained_bytes: state.input_frames.retained_bytes(),
            candidate_set_sizes: state
                .problem
                .directed_edges
                .iter()
                .map(|edge| {
                    (
                        edge.from.clone(),
                        edge.to.clone(),
                        state.candidates.ids[edge.id].len(),
                    )
                })
                .collect(),
        },
    })
}

/// Approximates a scalar pointwise operator directly as a tree tensor network.
///
/// This convenience wrapper has the same topology, index, convergence, and
/// quantization rules as [`tree_elementwise_batched`].
///
/// # Arguments
///
/// * `operator` - Pointwise function receiving one scalar from every input.
/// * `inputs` - Nonempty, topology-compatible native TreeTNs.
/// * `options` - Sweep, guard, rank, traversal, and allocation controls.
///
/// # Returns
///
/// The same result structure as [`tree_elementwise_batched`].
///
/// # Errors
///
/// Returns [`crate::TreeAciError`] under the same conditions as the batched API.
///
/// # Panics
///
/// This function does not intentionally panic. A callback panic is not caught.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{DynIndex, IdxTensor};
/// use tensor4all_treeaci::{tree_elementwise, TreeAciOptions};
/// use tensor4all_treetn::TreeTN;
///
/// let site = DynIndex::new_dyn(2);
/// let tree = TreeTN::from_tensors(
///     vec![IdxTensor::from_dense(vec![site], vec![2.0_f64, 3.0])?],
///     vec![0usize],
/// )?;
/// let result = tree_elementwise::<f64, _, _>(
///     |values| values[0] + 1.0,
///     &[tree],
///     &TreeAciOptions::default(),
/// )?;
/// assert_eq!(result.tree.to_dense()?.to_vec::<f64>()?, vec![3.0, 4.0]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn tree_elementwise<T, V, F>(
    operator: F,
    inputs: &[TreeTN<IdxTensor, V>],
    options: &TreeAciOptions<V>,
) -> Result<TreeAciResult<V>>
where
    T: TreeAciScalar,
    V: TreeAciNode,
    F: FnMut(&[T]) -> T,
{
    // The seeded high-level path builds one explicitly named stream and
    // delegates, so initialization and the guards share it.
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(options.rng_seed);
    tree_elementwise_with_rng(operator, inputs, options, &mut rng)
}

/// Pointwise tree ACI on a caller-owned random stream.
///
/// Same as [`tree_elementwise`] with the randomness of
/// [`tree_elementwise_batched_with_rng`]: the supplied stream drives the random
/// initial output and every global guard search, and the `rng_seed` option is
/// ignored.
///
/// # Errors
/// Returns [`crate::TreeAciError`] under the same conditions as the batched API.
pub fn tree_elementwise_with_rng<T, V, F, R>(
    mut operator: F,
    inputs: &[TreeTN<IdxTensor, V>],
    options: &TreeAciOptions<V>,
    rng: &mut R,
) -> Result<TreeAciResult<V>>
where
    T: TreeAciScalar,
    V: TreeAciNode,
    F: FnMut(&[T]) -> T,
    R: rand::Rng + ?Sized,
{
    let mut stream: &mut R = rng;
    tree_elementwise_batched_erased(
        |batch, output| {
            for (value, point_inputs) in output
                .iter_mut()
                .zip(batch.as_col_major_slice().chunks_exact(batch.n_inputs()))
            {
                *value = operator(point_inputs);
            }
            Ok(())
        },
        inputs,
        options,
        &mut stream,
        Exec::Global,
    )
}

#[cfg(test)]
mod tests;
