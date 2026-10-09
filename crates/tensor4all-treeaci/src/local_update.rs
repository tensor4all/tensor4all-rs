//! Budgeted dense edge-local matrices and LUCI factors.

use std::mem::size_of;

use tensor4all_core::IdxTensor;
use tensor4all_core::{
    matrix_luci_factors_from_matrix_owned, matrix_luci_factors_from_pivots, RrLUOptions,
};
#[cfg(test)]
use tensor4all_tensorbackend::mat_mul;
#[cfg(not(test))]
use tensor4all_tensorbackend::mat_mul_owned;
use tensor4all_tensorbackend::Matrix;
use tensor4all_treetn::TreeTN;

use crate::{
    frames::InputFrameStore,
    problem::{enforce_limit, DirectedEdgeId, PreparedTreeProblem},
    samples::{CandidateSets, ComponentSample, SampleArena, SampleId},
    Result, TreeAciError, TreeAciNode, TreeAciOptions, TreeAciScalar, TreeElementwiseBatch,
};

pub(crate) type PreviousPivots<'a> = (&'a SampleArena, &'a [(SampleId, SampleId)]);

#[derive(Clone, Debug)]
pub(crate) struct LocalUpdateResult<T> {
    pub(crate) row_samples: Vec<ComponentSample>,
    pub(crate) col_samples: Vec<ComponentSample>,
    pub(crate) left: Matrix<T>,
    pub(crate) right: Matrix<T>,
    pub(crate) pivot_errors: Vec<f64>,
    pub(crate) sampled_scale: f64,
    pub(crate) row_count: usize,
    pub(crate) col_count: usize,
    #[cfg(test)]
    pub(crate) local_values: Vec<T>,
}

/// Converts a local matrix between sampled-output units and the units used by
/// the selected tolerance mode.
///
/// Relative truncation factors a unit-scaled matrix, so the RRLU pivot floor is
/// independent of the operator's overall magnitude. Absolute truncation keeps
/// the raw matrix so both its configured threshold and pivot floor stay in raw
/// units. The magnitude-bearing factor and pivot errors are restored after a
/// relative-mode factorization.
#[derive(Clone, Copy, Debug)]
struct LocalMatrixScale {
    normalizer: f64,
}

impl LocalMatrixScale {
    fn new(normalizer: f64) -> Self {
        Self { normalizer }
    }

    fn normalize<T: TreeAciScalar>(self, matrix: &mut Matrix<T>) {
        if self.normalizer == 1.0 {
            return;
        }
        for value in matrix.as_col_major_mut_slice() {
            *value = value.div_real(self.normalizer);
        }
    }

    fn restore<T: TreeAciScalar>(self, magnitude_factor: &mut Matrix<T>, pivot_errors: &mut [f64]) {
        if self.normalizer == 1.0 {
            return;
        }
        let multiplier = <T as tensor4all_core::Scalar>::from_f64(self.normalizer);
        for value in magnitude_factor.as_col_major_mut_slice() {
            *value = *value * multiplier;
        }
        for error in pivot_errors {
            *error *= self.normalizer;
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn materialize_and_factor_edge<T, V, F>(
    inputs: &[TreeTN<IdxTensor, V>],
    problem: &PreparedTreeProblem<V>,
    candidates: &CandidateSets,
    frames: &InputFrameStore<T>,
    forward: DirectedEdgeId,
    options: &TreeAciOptions<V>,
    left_orthogonal: bool,
    previous: Option<PreviousPivots<'_>>,
    operator: &mut F,
) -> Result<LocalUpdateResult<T>>
where
    T: TreeAciScalar,
    V: TreeAciNode,
    F: for<'batch> FnMut(TreeElementwiseBatch<'batch, T>, &mut [T]) -> Result<()>,
{
    #[cfg(test)]
    let preparation_started = std::time::Instant::now();
    if inputs.is_empty() {
        return Err(TreeAciError::NoInputs);
    }
    let edge = problem
        .directed_edges
        .get(forward)
        .ok_or(TreeAciError::InternalInvariant {
            message: "local update references an unknown directed edge",
        })?;
    let reverse = edge.reverse;
    // Plan both Cartesian sides and charge their records/pairs before either
    // Vec is allocated. Element limits alone do not bound candidate metadata.
    let row_layout = candidate_layout(
        problem,
        candidates,
        forward,
        "candidate rows",
        options.max_candidate_rows,
    )?;
    let col_layout = candidate_layout(
        problem,
        candidates,
        reverse,
        "candidate columns",
        options.max_candidate_cols,
    )?;
    let metadata_bytes =
        row_layout
            .bytes
            .checked_add(col_layout.bytes)
            .ok_or(TreeAciError::SizeOverflow {
                context: "local candidate metadata bytes",
            })?;
    enforce_limit("working bytes", metadata_bytes, options.max_working_bytes)?;
    let row_count = row_layout.count;
    let col_count = col_layout.count;
    let point_count = row_count
        .checked_mul(col_count)
        .ok_or(TreeAciError::SizeOverflow {
            context: "local matrix elements",
        })?;
    enforce_limit(
        "local matrix elements",
        point_count,
        problem.max_local_matrix_elements,
    )?;
    let input_value_elements =
        inputs
            .len()
            .checked_mul(point_count)
            .ok_or(TreeAciError::SizeOverflow {
                context: "local input value elements",
            })?;
    let max_cut_rank = (0..inputs.len()).try_fold(0usize, |max_rank, input| {
        Ok::<usize, TreeAciError>(max_rank.max(frames.bond_dim(input, forward)?))
    })?;
    // Everything charged here stays live while candidate frames are built, so
    // it is what the candidate-frame kernels may *not* spend.
    let reserved_bytes = reserved_working_bytes::<T>(
        input_value_elements,
        point_count,
        row_count,
        col_count,
        max_cut_rank,
    )?
    .checked_add(metadata_bytes)
    .ok_or(TreeAciError::SizeOverflow {
        context: "local frame reservation bytes",
    })?;
    let candidate_frame_scratch =
        inputs
            .iter()
            .enumerate()
            .try_fold(0usize, |peak, (input, _)| {
                let row_scratch = frames.enumerated_candidate_frame_scratch_elements(
                    problem,
                    input,
                    forward,
                    candidates,
                    reserved_bytes,
                )?;
                let col_scratch = frames.enumerated_candidate_frame_scratch_elements(
                    problem,
                    input,
                    reverse,
                    candidates,
                    reserved_bytes,
                )?;
                Ok::<usize, TreeAciError>(peak.max(row_scratch).max(col_scratch))
            })?;
    let working_bytes = candidate_frame_scratch
        .checked_mul(size_of::<T>())
        .and_then(|bytes| bytes.checked_add(reserved_bytes))
        .ok_or(TreeAciError::SizeOverflow {
            context: "local matrix working bytes",
        })?;
    enforce_limit("working bytes", working_bytes, options.max_working_bytes)?;
    let factor_rank_bound = options
        .max_bond_dim
        .unwrap_or(usize::MAX)
        .min(row_count)
        .min(col_count)
        .max(1);
    // A previous cross is reusable when at least one complete projection
    // remains in this Cartesian candidate space. No candidate union or rank
    // floor is introduced. Probe membership without allocation before charging
    // the retained matrix needed by the full residual check.
    let mut previous = match previous {
        Some((arena, pairs)) if !pairs.is_empty() && pairs.len() <= factor_rank_bound => {
            let mut rows_available = true;
            let mut cols_available = true;
            for &(a, b) in pairs {
                let (row, col) = if forward.is_multiple_of(2) {
                    (a, b)
                } else {
                    (b, a)
                };
                rows_available &=
                    candidate_index(problem, candidates, forward, arena.record(forward, row)?)?
                        .is_some();
                cols_available &=
                    candidate_index(problem, candidates, reverse, arena.record(reverse, col)?)?
                        .is_some();
            }
            if rows_available || cols_available {
                Some((arena, pairs))
            } else {
                None
            }
        }
        _ => None,
    };
    let left_elements =
        row_count
            .checked_mul(factor_rank_bound)
            .ok_or(TreeAciError::SizeOverflow {
                context: "left local factor elements",
            })?;
    let right_elements =
        col_count
            .checked_mul(factor_rank_bound)
            .ok_or(TreeAciError::SizeOverflow {
                context: "right local factor elements",
            })?;
    enforce_limit("core elements", left_elements, problem.max_core_elements)?;
    enforce_limit("core elements", right_elements, problem.max_core_elements)?;
    let luci_bytes = tensor4all_core::matrix_luci_factors_working_bytes::<T>(
        row_count,
        col_count,
        factor_rank_bound,
    )
    .ok_or(TreeAciError::SizeOverflow {
        context: "local LUCI working bytes",
    })?;
    let factor_bytes = input_value_elements
        .checked_mul(size_of::<T>())
        .and_then(|bytes| bytes.checked_add(luci_bytes))
        // Selected pivot samples coexist with both full candidate lists.
        .and_then(|bytes| {
            metadata_bytes
                .checked_mul(2)
                .and_then(|meta| bytes.checked_add(meta))
        })
        .ok_or(TreeAciError::SizeOverflow {
            context: "local factor working bytes",
        })?;
    enforce_limit("working bytes", factor_bytes, options.max_working_bytes)?;
    if previous.is_some() {
        // Frame retention is optional. The fresh factors coexist with the
        // retained original matrix and Core's reconstruction/completion
        // buffers. Charge another full conservative LUCI estimate rather
        // than relying on unused scratch within the ordinary estimate.
        // A tight budget keeps the owned-LUCI path without a new run failure.
        let extra = point_count
            .checked_mul(size_of::<T>())
            .and_then(|bytes| bytes.checked_add(luci_bytes))
            .and_then(|bytes| factor_bytes.checked_add(bytes));
        if !matches!(extra, Some(bytes) if bytes <= options.max_working_bytes) {
            previous = None;
        }
    }

    let row_candidates = materialize_candidates(problem, candidates, forward, row_layout.count);
    let col_candidates = materialize_candidates(problem, candidates, reverse, col_layout.count);

    let mut input_values = vec![T::default(); input_value_elements];
    #[cfg(test)]
    crate::state::profile_debug_stats::record(|stats| {
        stats.local_preparation += preparation_started.elapsed();
    });
    #[cfg(test)]
    let input_frames_started = std::time::Instant::now();
    // Per input, `input_values[.., point] = row_frames[row] . col_frames[col]`
    // is one (row_count x chi) times (chi x col_count) matrix product, not a
    // per-point scalar dot product: pack each side's candidate frame vectors
    // into packed dense matrices and let BLAS do the O(row*col*chi)
    // contraction in one `mat_mul_owned` call. The candidate-frame cache and
    // frame batching no longer create a Vec<Vec<T>> round trip; only the
    // row-side flat layout conversion and O(row*col) scatter remain plain
    // loops.
    if point_count > 0 {
        for input in 0..inputs.len() {
            #[cfg(test)]
            let row_frames_started = std::time::Instant::now();
            let row_input_frames = frames.candidate_frames_for_edge_rows(
                inputs,
                problem,
                input,
                forward,
                &row_candidates,
                reserved_bytes,
            )?;
            #[cfg(test)]
            crate::state::profile_debug_stats::record(|stats| {
                stats.local_row_frames += row_frames_started.elapsed();
            });
            #[cfg(test)]
            let col_frames_started = std::time::Instant::now();
            let col_input_frames = frames.candidate_frames_for_edge(
                inputs,
                problem,
                input,
                reverse,
                &col_candidates,
                reserved_bytes,
            )?;
            #[cfg(test)]
            crate::state::profile_debug_stats::record(|stats| {
                stats.local_col_frames += col_frames_started.elapsed();
            });
            #[cfg(test)]
            let pack_started = std::time::Instant::now();
            let bond_dim = row_input_frames.bond_dim();
            if col_input_frames.bond_dim() != bond_dim
                || row_input_frames.candidate_count() != row_count
                || col_input_frames.candidate_count() != col_count
            {
                return Err(TreeAciError::InternalInvariant {
                    message: "packed input frames have inconsistent candidate dimensions",
                });
            }
            if bond_dim == 0 {
                continue;
            }
            #[cfg(test)]
            let use_legacy_frame_pack =
                std::env::var("T4A_TREEACI_USE_LEGACY_LOCAL_FRAME_PACK").as_deref() == Ok("1");
            #[cfg(test)]
            let (row_candidate_matrix, col_bond_matrix) = if use_legacy_frame_pack {
                // [AI Supplied] Diagnostic-only pre-#714 path. It exists only
                // for the paired release measurement and reproduces the old
                // per-candidate extraction plus two flat repacks.
                crate::state::profile_debug_stats::record(|stats| {
                    stats.local_legacy_frame_vectors += row_count + col_count;
                    stats.local_legacy_frame_values += (row_count + col_count) * bond_dim;
                });
                let row_frames = row_input_frames.to_candidate_vecs();
                let col_frames = col_input_frames.to_candidate_vecs();
                let mut row_flat = Vec::with_capacity(row_count * bond_dim);
                for bond in 0..bond_dim {
                    for frame in &row_frames {
                        row_flat.push(frame[bond]);
                    }
                }
                let col_flat = col_frames
                    .iter()
                    .flat_map(|frame| frame.iter().copied())
                    .collect::<Vec<_>>();
                (
                    Matrix::from_col_major_vec(row_count, bond_dim, row_flat),
                    Matrix::from_col_major_vec(bond_dim, col_count, col_flat),
                )
            } else {
                crate::state::profile_debug_stats::record(|stats| {
                    stats.local_packed_frame_batches += 2;
                    stats.local_packed_frame_values += (row_count + col_count) * bond_dim;
                });
                (
                    row_input_frames.into_candidate_by_bond_matrix(),
                    col_input_frames.into_bond_by_candidate_matrix(),
                )
            };
            #[cfg(not(test))]
            let row_candidate_matrix = row_input_frames.into_candidate_by_bond_matrix();
            #[cfg(not(test))]
            let col_bond_matrix = col_input_frames.into_bond_by_candidate_matrix();
            #[cfg(test)]
            crate::state::profile_debug_stats::record(|stats| {
                stats.local_frame_pack += pack_started.elapsed();
            });
            #[cfg(test)]
            let matmul_started = std::time::Instant::now();
            // [AI Supplied] Keep the diagnostic A/B switch in tests while
            // production always consumes these short-lived matrices.
            #[cfg(test)]
            let product_result =
                if std::env::var("T4A_TREEACI_USE_OWNED_LOCAL_MATMUL").as_deref() == Ok("1") {
                    tensor4all_tensorbackend::mat_mul_owned(row_candidate_matrix, col_bond_matrix)
                } else {
                    mat_mul(&row_candidate_matrix, &col_bond_matrix)
                };
            #[cfg(not(test))]
            let product_result = mat_mul_owned(row_candidate_matrix, col_bond_matrix);
            let product = product_result.map_err(|error| TreeAciError::Numerical {
                message: error.to_string(),
            })?;
            #[cfg(test)]
            crate::state::profile_debug_stats::record(|stats| {
                stats.local_frame_matmul += matmul_started.elapsed();
            });
            #[cfg(test)]
            let scatter_started = std::time::Instant::now();
            for col in 0..col_count {
                for row in 0..row_count {
                    let point = row + row_count * col;
                    input_values[input + inputs.len() * point] = product[[row, col]];
                }
            }
            #[cfg(test)]
            crate::state::profile_debug_stats::record(|stats| {
                stats.local_frame_scatter += scatter_started.elapsed();
            });
        }
    }
    #[cfg(test)]
    crate::state::profile_debug_stats::record(|stats| {
        stats.local_input_frames += input_frames_started.elapsed();
    });
    let batch = TreeElementwiseBatch::new(&input_values, inputs.len(), point_count)?;
    let mut local_values = vec![T::default(); point_count];
    #[cfg(test)]
    let operator_started = std::time::Instant::now();
    operator(batch, &mut local_values)?;
    #[cfg(test)]
    crate::state::profile_debug_stats::record(|stats| {
        stats.operator += operator_started.elapsed();
    });
    let sampled_scale = local_values.iter().copied().fold(0.0_f64, |scale, value| {
        scale.max(tensor4all_core::Scalar::abs_val(value))
    });
    #[cfg(test)]
    let retained_local_values = local_values.clone();
    // Use the same numeric tolerance in RRLU as the selected matrix units:
    // relative mode normalizes the matrix, while absolute mode leaves it raw.
    let tolerance = options.tolerance_policy();
    let local_scale = LocalMatrixScale::new(tolerance.local_normalizer(sampled_scale));
    let mut matrix = Matrix::from_col_major_vec(row_count, col_count, local_values);
    local_scale.normalize(&mut matrix);
    let previous_matrix = previous.map(|_| matrix.clone());
    #[cfg(test)]
    let luci_started = std::time::Instant::now();
    let mut factors = matrix_luci_factors_from_matrix_owned(
        matrix,
        Some(RrLUOptions {
            max_bond_dim: options.max_bond_dim.unwrap_or(usize::MAX),
            rel_tol: 0.0,
            abs_tol: tolerance.local_threshold(),
            left_orthogonal,
        }),
    )
    .map_err(|error| TreeAciError::Numerical {
        message: error.to_string(),
    })?;
    if let (Some((arena, pairs)), Some(matrix)) = (previous, previous_matrix.as_ref()) {
        // Preserve an already sufficient nested frame instead of replacing it
        // with a different near-threshold basis. Fresh LUCI still determines
        // the rank ceiling here, so every rank decrease remains possible.
        if pairs.len() <= factors.rank {
            let mut rows = Vec::with_capacity(pairs.len());
            let mut cols = Vec::with_capacity(pairs.len());
            for &(a, b) in pairs {
                let (row, col) = if forward.is_multiple_of(2) {
                    (a, b)
                } else {
                    (b, a)
                };
                rows.push(candidate_index(
                    problem,
                    candidates,
                    forward,
                    arena.record(forward, row)?,
                )?);
                cols.push(candidate_index(
                    problem,
                    candidates,
                    reverse,
                    arena.record(reverse, col)?,
                )?);
            }
            let rows = rows.into_iter().collect::<Option<Vec<_>>>();
            let cols = cols.into_iter().collect::<Option<Vec<_>>>();
            // An adjacent update can invalidate one projection while the
            // other still defines a sufficient nested frame. Complete only
            // the missing axis, then validate the entire unchanged matrix.
            let (rows, cols) = match (rows, cols) {
                (Some(rows), Some(cols)) => (rows, cols),
                (Some(rows), None) => (rows, Vec::new()),
                (None, Some(cols)) => (Vec::new(), cols),
                (None, None) => {
                    return Err(TreeAciError::InternalInvariant {
                        message: "previous cross lost both axes after preflight",
                    })
                }
            };
            match matrix_luci_factors_from_pivots(matrix, &rows, &cols, left_orthogonal) {
                Ok(previous) if previous.pivot_errors[0] <= tolerance.local_threshold() => {
                    factors = previous
                }
                Ok(_) | Err(tensor4all_core::MatrixCIError::SingularMatrix) => {}
                Err(error) => {
                    return Err(TreeAciError::Numerical {
                        message: error.to_string(),
                    })
                }
            }
        }
    }
    drop(previous_matrix);
    #[cfg(test)]
    crate::state::profile_debug_stats::record(|stats| {
        stats.luci += luci_started.elapsed();
    });
    // Only the factor holding raw pivot rows/columns carries the magnitude;
    // the interpolative factor `A[:, J] A[I, J]^-1` (or its transpose) is
    // invariant under relative-mode normalization.
    let magnitude_factor = if left_orthogonal {
        &mut factors.right
    } else {
        &mut factors.left
    };
    local_scale.restore(magnitude_factor, &mut factors.pivot_errors);
    let (left, right, row_indices, col_indices) = if factors.rank == 0 {
        let (left, right) = zero_rank_one_skeleton(row_count, col_count, left_orthogonal)?;
        (left, right, vec![0], vec![0])
    } else {
        (
            factors.left,
            factors.right,
            factors.row_indices,
            factors.col_indices,
        )
    };
    let row_samples = select_pivot_samples(row_indices, &row_candidates)?;
    let col_samples = select_pivot_samples(col_indices, &col_candidates)?;
    Ok(LocalUpdateResult {
        row_samples,
        col_samples,
        left,
        right,
        pivot_errors: factors.pivot_errors,
        sampled_scale,
        row_count,
        col_count,
        #[cfg(test)]
        local_values: retained_local_values,
    })
}

/// Inverse of `materialize_candidates` for a retained component sample.
/// Missing incoming IDs mean that the previous sample is outside the current
/// Cartesian space; they never justify padding or reordering output axes.
fn candidate_index<V: TreeAciNode>(
    problem: &PreparedTreeProblem<V>,
    candidates: &CandidateSets,
    forward: DirectedEdgeId,
    sample: &ComponentSample,
) -> Result<Option<usize>> {
    let edge = &problem.directed_edges[forward];
    let local_dim = problem.physical[problem.node_positions[&edge.from]].local_dim;
    if sample.local_coordinate >= local_dim || sample.incoming.len() != edge.incoming_to_from.len()
    {
        return Ok(None);
    }
    let mut index = sample.local_coordinate;
    let mut stride = local_dim;
    for (&incoming, &(sample_edge, sample_id)) in edge.incoming_to_from.iter().zip(&sample.incoming)
    {
        if sample_edge != incoming {
            return Ok(None);
        }
        let ids = &candidates.ids[incoming];
        let Some(position) = ids.iter().position(|&id| id == sample_id) else {
            return Ok(None);
        };
        index = position
            .checked_mul(stride)
            .and_then(|offset| index.checked_add(offset))
            .ok_or(TreeAciError::SizeOverflow {
                context: "previous-cross candidate index",
            })?;
        stride = stride
            .checked_mul(ids.len())
            .ok_or(TreeAciError::SizeOverflow {
                context: "previous-cross candidate stride",
            })?;
    }
    Ok(Some(index))
}

/// Bytes of `max_working_bytes` that stay live for the whole candidate-frame
/// phase of one edge update.
///
/// This is the reservation the candidate-frame kernels are told about, so that
/// an arbitrary-degree edge whose batched cross fits the budget on its own,
/// but not once these buffers are counted, degrades to the scalar route on
/// both sides of the contract instead of being refused by the pre-flight and
/// then batched anyway by the kernel (#726). It is also the constant part of
/// the pre-flight charge, so both uses read the same rule from one place.
///
/// # Arguments
///
/// * `input_value_elements` - one per input per point, all live at once.
/// * `point_count` - the one point-sized output/product buffer that coexists
///   with the input values.
/// * `row_count`, `col_count`, `max_cut_rank` - the packed candidate frames of
///   both sides, counted twice because a side's candidate vectors and their
///   packed BLAS matrix coexist for one input at a time. Other inputs are
///   streamed and dropped.
///
/// # Errors
///
/// Returns [`TreeAciError::SizeOverflow`] when the charge does not fit `usize`.
pub(crate) fn reserved_working_bytes<T: TreeAciScalar>(
    input_value_elements: usize,
    point_count: usize,
    row_count: usize,
    col_count: usize,
    max_cut_rank: usize,
) -> Result<usize> {
    let candidate_frame_elements = row_count
        .checked_add(col_count)
        .and_then(|count| count.checked_mul(max_cut_rank))
        .and_then(|count| count.checked_mul(2))
        .ok_or(TreeAciError::SizeOverflow {
            context: "candidate frame working elements",
        })?;
    input_value_elements
        .checked_add(point_count)
        .and_then(|count| count.checked_add(candidate_frame_elements))
        .and_then(|count| count.checked_mul(size_of::<T>()))
        .ok_or(TreeAciError::SizeOverflow {
            context: "local update working elements",
        })
}

/// Rank-one factors that approximate a negligible local matrix by zero while
/// keeping the interpolative form of the nonzero-rank factors.
///
/// A LUCI factorization that selects no pivot (every sampled entry is zero, or
/// below the factorization's pivot floor) still has to commit a rank-one bond
/// with the pivot pair `(0, 0)`. Rank-`r` LUCI factors are interpolative: with
/// `left_orthogonal` the left factor is `A[:, J] A[I, J]^-1`, whose pivot rows
/// form the identity, and otherwise the right factor carries the identity on
/// the pivot columns. The zero approximation keeps that identity on its single
/// pivot and puts the zero in the other factor. An all-zero interpolating
/// factor would instead leave a core that no CI factorization can represent
/// with a nonzero bond, so the deferred CI canonicalization at the end of the
/// pass could not accept it.
///
/// # Arguments
///
/// * `row_count`, `col_count` - local matrix shape; both must be nonzero.
/// * `left_orthogonal` - which factor carries the pivot identity, as in
///   [`RrLUOptions::left_orthogonal`].
///
/// # Errors
///
/// Returns [`TreeAciError::InternalInvariant`] for an empty local matrix,
/// which has no pivot `(0, 0)` to select.
fn zero_rank_one_skeleton<T: TreeAciScalar>(
    row_count: usize,
    col_count: usize,
    left_orthogonal: bool,
) -> Result<(Matrix<T>, Matrix<T>)> {
    if row_count == 0 || col_count == 0 {
        return Err(TreeAciError::InternalInvariant {
            message: "an empty local matrix has no pivot to keep",
        });
    }
    let mut left = Matrix::zeros(row_count, 1);
    let mut right = Matrix::zeros(1, col_count);
    if left_orthogonal {
        left[[0, 0]] = T::one();
    } else {
        right[[0, 0]] = T::one();
    }
    Ok((left, right))
}

fn select_pivot_samples(
    indices: Vec<usize>,
    candidates: &[ComponentSample],
) -> Result<Vec<ComponentSample>> {
    indices
        .into_iter()
        .map(|index| {
            candidates
                .get(index)
                .cloned()
                .ok_or(TreeAciError::InternalInvariant {
                    message: "LUCI returned a pivot index outside the candidate matrix",
                })
        })
        .collect()
}

struct CandidateLayout {
    count: usize,
    bytes: usize,
}

fn candidate_layout<V: TreeAciNode>(
    problem: &PreparedTreeProblem<V>,
    candidate_sets: &CandidateSets,
    edge: DirectedEdgeId,
    resource: &'static str,
    limit: usize,
) -> Result<CandidateLayout> {
    let directed = &problem.directed_edges[edge];
    let node =
        *problem
            .node_positions
            .get(&directed.from)
            .ok_or(TreeAciError::InternalInvariant {
                message: "candidate source has no prepared node position",
            })?;
    let mut count = problem.physical[node].local_dim;
    for incoming in &directed.incoming_to_from {
        let ids = candidate_sets
            .ids
            .get(*incoming)
            .ok_or(TreeAciError::InternalInvariant {
                message: "candidate incoming edge has no candidate set",
            })?;
        if ids.is_empty() {
            return Err(TreeAciError::InternalInvariant {
                message: "candidate incoming edge has an empty candidate set",
            });
        }
        count = count
            .checked_mul(ids.len())
            .ok_or(TreeAciError::SizeOverflow {
                context: "candidate count",
            })?;
    }
    enforce_limit(resource, count, limit)?;
    let bytes = directed
        .incoming_to_from
        .len()
        .checked_mul(size_of::<(DirectedEdgeId, crate::samples::SampleId)>())
        .and_then(|pairs| pairs.checked_add(size_of::<ComponentSample>()))
        .and_then(|record| record.checked_mul(count))
        .ok_or(TreeAciError::SizeOverflow {
            context: "candidate metadata bytes",
        })?;
    Ok(CandidateLayout { count, bytes })
}

#[cfg(test)]
fn enumerate_candidates<V: TreeAciNode>(
    problem: &PreparedTreeProblem<V>,
    candidate_sets: &CandidateSets,
    edge: DirectedEdgeId,
    resource: &'static str,
    limit: usize,
) -> Result<Vec<ComponentSample>> {
    let layout = candidate_layout(problem, candidate_sets, edge, resource, limit)?;
    Ok(materialize_candidates(
        problem,
        candidate_sets,
        edge,
        layout.count,
    ))
}

fn materialize_candidates<V: TreeAciNode>(
    problem: &PreparedTreeProblem<V>,
    candidate_sets: &CandidateSets,
    edge: DirectedEdgeId,
    count: usize,
) -> Vec<ComponentSample> {
    let directed = &problem.directed_edges[edge];
    // The layout validated this source and every incoming candidate set.
    let node = problem.node_positions[&directed.from];
    let mut candidates = Vec::with_capacity(count);
    for encoded in 0..count {
        let mut quotient = encoded;
        let local_coordinate = quotient % problem.physical[node].local_dim;
        quotient /= problem.physical[node].local_dim;
        let mut incoming_samples = Vec::with_capacity(directed.incoming_to_from.len());
        for incoming in &directed.incoming_to_from {
            let ids = &candidate_sets.ids[*incoming];
            incoming_samples.push((*incoming, ids[quotient % ids.len()]));
            quotient /= ids.len();
        }
        candidates.push(ComponentSample {
            local_coordinate,
            incoming: incoming_samples,
        });
    }
    candidates
}

#[cfg(test)]
mod tests;
