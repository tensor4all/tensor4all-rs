//! Re-measurement of per-edge residuals that later commits made stale (#1451).
//!
//! A commit records its edge's residual: the output's error over that edge's
//! local matrix, as the commit leaves the output. Every later commit rewrites
//! two cores, and every output value contracts every core, so the recorded
//! residual no longer describes the current output. The continuous walk makes
//! each commit start where the previous one ended, so at the end of a pass
//! every edge except the last one committed is stale in this sense.
//!
//! The refresh evaluates the operator and the current output on the current
//! local matrix of each stale edge and records `max |f - output|` there. It
//! touches only those edges and only their current local matrices: no
//! retained pivot, Guard point or earlier local matrix is revisited (#1372).

use crate::{
    frames::InputFrameStore,
    local_update::{evaluate_local_matrix, local_input_values, plan_local_matrix},
    state::TreeAciState,
    Result, TreeAciError, TreeAciNode, TreeAciOptions, TreeAciScalar, TreeElementwiseBatch,
};

/// What one [`refresh_stale_residuals`] call did.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct ResidualRefreshReport {
    /// Edges whose residual was re-measured.
    pub(crate) refreshed_edges: usize,
    /// Operator points evaluated, one per local-matrix entry of those edges.
    pub(crate) evaluated_points: u64,
}

/// Re-measures the residual of every edge whose recorded residual is stale
/// (see [`TreeAciState::residual_is_stale`]) on the current output.
///
/// For each such edge the operator is evaluated on the edge's current local
/// matrix (the same points its next commit would read), and the output on the
/// same points through a frame store of the output. A commit across a vertex
/// of local dimension 1 also reads the candidates its unit-vertex expansion
/// adds for that one commit (`crate::unit_vertex`); the refresh does not: it
/// measures the plain local matrix of the current candidate sets, which no
/// expansion outlives. `edge_errors` receives
/// `max |f - output|` over those points and `edge_scales` the largest `|f|`
/// there, in the raw units a commit records. Fresh edges are untouched. The
/// state is updated only after every stale edge was measured, so a failure
/// leaves it unchanged.
///
/// Cost: one operator evaluation per local-matrix entry of each stale edge,
/// which is what re-committing those edges would cost, plus one frame store of
/// the output over the retained samples.
///
/// # Errors
///
/// Propagates the local-matrix budget refusals of a commit
/// ([`TreeAciError::ResourceLimit`]), frame and operator failures, and returns
/// [`TreeAciError::NonFiniteValue`] for a non-finite residual.
pub(crate) fn refresh_stale_residuals<T, V, F>(
    state: &mut TreeAciState<'_, T, V>,
    options: &TreeAciOptions<V>,
    operator: &mut F,
    exec: &mut crate::exec::Exec<'_, '_>,
) -> Result<ResidualRefreshReport>
where
    T: TreeAciScalar,
    V: TreeAciNode,
    F: for<'batch> FnMut(TreeElementwiseBatch<'batch, T>, &mut [T]) -> Result<()>,
{
    let edge_count = state.residual_revisions.len();
    if state.edge_errors.len() != edge_count || state.edge_scales.len() != edge_count {
        return Err(TreeAciError::InternalInvariant {
            message: "residual refresh state metadata is incomplete",
        });
    }
    let mut stale = Vec::new();
    for edge_number in 0..edge_count {
        if state.residual_is_stale(edge_number)? {
            stale.push(edge_number);
        }
    }
    if stale.is_empty() {
        return Ok(ResidualRefreshReport::default());
    }
    let output = std::slice::from_ref(&state.output);
    let output_frames =
        InputFrameStore::<T>::from_samples(output, &state.problem, &state.sample_arena, exec)?;
    let mut measured = Vec::with_capacity(stale.len());
    let mut evaluated_points = 0u64;
    for &edge_number in &stale {
        let forward = edge_number
            .checked_mul(2)
            .ok_or(TreeAciError::SizeOverflow {
                context: "directed edge identifier",
            })?;
        let plan = plan_local_matrix(
            state.inputs,
            &state.problem,
            &state.candidates,
            &state.input_frames,
            forward,
            options,
            0,
        )?;
        let (rows, cols, targets) = evaluate_local_matrix(
            state.inputs,
            &state.problem,
            &state.candidates,
            &state.input_frames,
            &plan,
            operator,
            exec,
        )?;
        // The operator values stay live while the output is evaluated.
        let held_bytes = targets.len().checked_mul(std::mem::size_of::<T>()).ok_or(
            TreeAciError::SizeOverflow {
                context: "residual refresh target bytes",
            },
        )?;
        let output_plan = plan_local_matrix(
            output,
            &state.problem,
            &state.candidates,
            &output_frames,
            forward,
            options,
            held_bytes,
        )?;
        let approximations = local_input_values(
            output,
            &state.problem,
            &output_frames,
            &output_plan,
            &rows,
            &cols,
            exec,
        )?;
        if approximations.len() != targets.len() {
            return Err(TreeAciError::InternalInvariant {
                message: "refreshed output and operator values disagree on the local matrix",
            });
        }
        let mut error = 0.0_f64;
        let mut scale = 0.0_f64;
        for (&target, &approximation) in targets.iter().zip(&approximations) {
            let residual = tensor4all_core::Scalar::abs_val(target - approximation);
            if !residual.is_finite() {
                return Err(TreeAciError::NonFiniteValue {
                    context: "refreshed edge residual",
                });
            }
            error = error.max(residual);
            scale = scale.max(tensor4all_core::Scalar::abs_val(target));
        }
        evaluated_points = evaluated_points
            .checked_add(u64::try_from(plan.point_count).map_err(|_| {
                TreeAciError::SizeOverflow {
                    context: "residual refresh point count",
                }
            })?)
            .ok_or(TreeAciError::SizeOverflow {
                context: "residual refresh point count",
            })?;
        measured.push((edge_number, error, scale));
    }
    let revision = Some(state.output_revision);
    for &(edge_number, error, scale) in &measured {
        state.edge_errors[edge_number] = error;
        state.edge_scales[edge_number] = scale;
        state.residual_revisions[edge_number] = revision;
    }
    Ok(ResidualRefreshReport {
        refreshed_edges: measured.len(),
        evaluated_points,
    })
}

#[cfg(test)]
mod tests;
