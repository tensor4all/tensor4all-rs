//! Query candidate membership is independent of the selected output basis.
use super::{CandidateSets, SampleId};
use crate::{
    problem::{enforce_limit, PreparedTreeProblem},
    Result, TreeAciError, TreeAciNode,
};

#[derive(Clone, Copy)]
pub(crate) struct CandidateCapacity<'a> {
    /// Maximum distinct useful samples per direction, independent of the output basis.
    pub(crate) limits: &'a [usize],
    /// Component directions whose child membership must be preserved.
    pub(crate) required_closure: &'a [bool],
}

impl CandidateCapacity<'_> {
    pub(super) fn requested<V: TreeAciNode>(
        &self,
        problem: &PreparedTreeProblem<V>,
        candidates: &CandidateSets,
    ) -> Result<Vec<bool>> {
        let n = problem.directed_edges.len();
        if self.limits.len().checked_mul(2) != Some(n) || self.required_closure.len() != n {
            return Err(TreeAciError::InternalInvariant {
                message: "candidate capacity differs from prepared topology",
            });
        }
        let bytes = n
            .checked_mul(
                4 * std::mem::size_of::<usize>() + 2 * std::mem::size_of::<Option<usize>>() + 5,
            )
            .ok_or(TreeAciError::SizeOverflow {
                context: "candidate activation planning bytes",
            })?;
        enforce_limit("working bytes", bytes, problem.max_working_bytes)?;
        candidates
            .ids
            .iter()
            .enumerate()
            .map(|(edge, ids)| {
                let limit = self.limits[edge / 2];
                if ids.len() > limit {
                    return Err(TreeAciError::InternalInvariant {
                        message: "useful candidate count exceeds its pool limit",
                    });
                }
                Ok(ids.len() < limit)
            })
            .collect()
    }
}

pub(super) struct CandidateActivation {
    pub(super) additions: Vec<Option<SampleId>>,
}

impl CandidateActivation {
    pub(super) fn plan<V: TreeAciNode>(
        problem: &PreparedTreeProblem<V>,
        candidates: &CandidateSets,
        projected: &[Option<SampleId>],
        budget: CandidateCapacity<'_>,
    ) -> Self {
        let n = projected.len();
        let present: Vec<_> = projected
            .iter()
            .enumerate()
            .map(|(e, id)| id.is_some_and(|id| candidates.ids[e].contains(&id)))
            .collect();
        let mut additions: Vec<_> = projected
            .iter()
            .enumerate()
            .map(|(e, &id)| {
                if !present[e] && candidates.ids[e].len() < budget.limits[e / 2] {
                    id
                } else {
                    None
                }
            })
            .collect();
        let mut available: Vec<_> = present
            .iter()
            .zip(&additions)
            .map(|(&p, id)| p || id.is_some())
            .collect();
        let mut queued = vec![false; n];
        let mut queue = Vec::with_capacity(n);
        for edge in 0..n {
            if budget.required_closure[edge]
                && additions[edge].is_some()
                && problem.directed_edges[edge]
                    .incoming_to_from
                    .iter()
                    .any(|&child| !available[child])
            {
                queued[edge] = true;
                queue.push(edge);
            }
        }
        // INVARIANT: each addition is withdrawn at most once. Prepared reverse
        // dependencies supply consumers, so no whole-tree rescans are needed.
        while let Some(edge) = queue.pop() {
            if additions[edge].take().is_none() {
                continue;
            }
            available[edge] = false;
            for &incoming in
                &problem.directed_edges[problem.directed_edges[edge].reverse].incoming_to_from
            {
                let consumer = problem.directed_edges[incoming].reverse;
                if budget.required_closure[consumer]
                    && additions[consumer].is_some()
                    && !queued[consumer]
                {
                    queued[consumer] = true;
                    queue.push(consumer);
                }
            }
        }
        Self { additions }
    }
}

#[cfg(test)]
mod tests;
