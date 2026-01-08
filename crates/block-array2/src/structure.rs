//! Block structure metadata used for planning and cost estimation.
//!
//! Note: This is intentionally a lightweight representation for *planning*.
//! It does not attempt to track the exact nonzero block pattern of intermediate tensors.

use crate::partition::BlockPartition;

/// Block structure metadata (partitions + an estimate of sparsity).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockStructure {
    partitions: Vec<BlockPartition>,
    /// Estimated number of nonzero blocks.
    ///
    /// For input tensors, callers should set this to the true number of nonzero blocks if known.
    /// For intermediate tensors, this is updated heuristically by the planner.
    nnz_blocks_estimate: usize,
}

impl BlockStructure {
    /// Create a new `BlockStructure`.
    pub fn new(partitions: Vec<BlockPartition>, nnz_blocks_estimate: usize) -> Self {
        Self {
            partitions,
            nnz_blocks_estimate,
        }
    }

    /// Rank (number of dimensions).
    pub fn rank(&self) -> usize {
        self.partitions.len()
    }

    /// Partitions.
    pub fn partitions(&self) -> &[BlockPartition] {
        &self.partitions
    }

    /// Total dimension along each axis.
    pub fn shape(&self) -> Vec<usize> {
        self.partitions.iter().map(|p| p.total_dim()).collect()
    }

    /// Number of blocks along each axis.
    pub fn num_blocks(&self) -> Vec<usize> {
        self.partitions.iter().map(|p| p.num_blocks()).collect()
    }

    /// Estimated number of nonzero blocks.
    pub fn nnz_blocks_estimate(&self) -> usize {
        self.nnz_blocks_estimate
    }

    /// Total number of blocks (dense block grid size).
    pub fn total_blocks(&self) -> usize {
        self.num_blocks().into_iter().product::<usize>().max(1)
    }

    /// Heuristic cost estimate for contracting `self` and `other`.
    ///
    /// Dense FLOP estimate (multiply + add), assuming dense blocks.
    pub fn estimate_tensordot_cost(
        &self,
        other: &Self,
        axes_self: &[usize],
        axes_other: &[usize],
    ) -> u64 {
        assert_eq!(axes_self.len(), axes_other.len());

        for (&a, &b) in axes_self.iter().zip(axes_other.iter()) {
            assert_eq!(self.partitions[a], other.partitions[b]);
        }

        let rank_self = self.rank();
        let rank_other = other.rank();

        let free_self: Vec<usize> = (0..rank_self).filter(|a| !axes_self.contains(a)).collect();
        let free_other: Vec<usize> = (0..rank_other)
            .filter(|a| !axes_other.contains(a))
            .collect();

        let contracted_dim: u128 = axes_self
            .iter()
            .map(|&a| self.partitions[a].total_dim() as u128)
            .product::<u128>()
            .max(1);
        let free_self_dim: u128 = free_self
            .iter()
            .map(|&a| self.partitions[a].total_dim() as u128)
            .product::<u128>()
            .max(1);
        let free_other_dim: u128 = free_other
            .iter()
            .map(|&a| other.partitions[a].total_dim() as u128)
            .product::<u128>()
            .max(1);

        // Dense FLOP estimate for tensordot: ~ 2 * free_self * contracted * free_other
        let dense_cost = 2u128 * free_self_dim * contracted_dim * free_other_dim;
        dense_cost.min(u64::MAX as u128) as u64
    }

    /// Heuristic update for intermediate `nnz_blocks_estimate`.
    pub fn estimate_nnz_after_contraction(
        lhs: &Self,
        rhs: &Self,
        contracted_labels: usize,
    ) -> usize {
        // Very conservative: assume multiplicative growth, but dampen by contracted count.
        let denom = contracted_labels.max(1);
        let raw =
            (lhs.nnz_blocks_estimate().max(1) as u128) * (rhs.nnz_blocks_estimate().max(1) as u128);
        ((raw / (denom as u128)).min(usize::MAX as u128)) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::partition::BlockPartition;

    #[test]
    fn test_structure_basics() {
        let s = BlockStructure::new(
            vec![BlockPartition::trivial(2), BlockPartition::uniform(3, 2)],
            5,
        );
        assert_eq!(s.rank(), 2);
        assert_eq!(s.shape(), vec![2, 6]);
        assert_eq!(s.num_blocks(), vec![1, 2]);
        assert_eq!(s.total_blocks(), 2);
        assert_eq!(s.nnz_blocks_estimate(), 5);
    }
}
