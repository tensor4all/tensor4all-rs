//! Completion policies for the existing zip-up schedule.
//!
//! Rank-triggered cancellation follows the observation in Grosso et al.,
//! [arXiv:2602.22372v1, Sec. 4.3](https://arxiv.org/html/2602.22372v1#S4.SS3).
//! The tree-specific justification is independently derived in
//! `docs/design/treetn-contraction-outcome.md`: a retained link is never
//! resized again on the general-tree path. This is a stored-dimension
//! predicate, not an accuracy certificate. The chain's return compression
//! prevents using this hook there. No external source code is translated.

use std::{fmt::Debug, hash::Hash, num::NonZeroUsize};

use anyhow::Result;
use tensor4all_core::TensorLike;

use super::{outcome, ContractionOutcome, EdgeRank, TreeTN};

/// Static policies let ordinary contraction keep its by-value return type
/// without allocating a report or creating an unreachable abort branch.
pub(super) trait ZipupControl<T: TensorLike, V: Clone + Hash + Eq + Send + Sync + Debug> {
    type Output;

    fn edge(&self, child: &V, parent: &V, rank: usize) -> Option<Self::Output>;
    fn finish(self, network: TreeTN<T, V>) -> Result<Self::Output>;
}

pub(super) struct Complete;

impl<T: TensorLike, V: Clone + Hash + Eq + Send + Sync + Debug> ZipupControl<T, V> for Complete {
    type Output = TreeTN<T, V>;

    fn edge(&self, _child: &V, _parent: &V, _rank: usize) -> Option<Self::Output> {
        None
    }

    fn finish(self, network: TreeTN<T, V>) -> Result<Self::Output> {
        Ok(network)
    }
}

pub(super) struct AbortAtRank<'a, V> {
    pub center: &'a V,
    pub threshold: NonZeroUsize,
}

impl<T: TensorLike, V: Clone + Hash + Eq + Ord + Send + Sync + Debug> ZipupControl<T, V>
    for AbortAtRank<'_, V>
{
    type Output = ContractionOutcome<T, V>;

    fn edge(&self, child: &V, parent: &V, rank: usize) -> Option<Self::Output> {
        (rank >= self.threshold.get()).then(|| ContractionOutcome::Saturated {
            edge: EdgeRank {
                child: child.clone(),
                parent: parent.clone(),
                rank,
            },
        })
    }

    fn finish(self, network: TreeTN<T, V>) -> Result<Self::Output> {
        outcome::finish(network, self.center, Some(self.threshold)).map_err(Into::into)
    }
}

#[cfg(test)]
pub(super) mod tests;
