//! Error types for block-array2 planning.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum BlockArray2Error {
    #[error("rank mismatch: tensor rank {rank} != labels length {labels_len}")]
    RankMismatch { rank: usize, labels_len: usize },

    #[error("duplicate index label within a single tensor: '{label}'")]
    DuplicateIndexLabelWithinTensor { label: String },

    #[error(
        "index label '{label}' appears on 3 or more tensors (hyperedge), which is not supported"
    )]
    HyperedgeNotSupported { label: String },

    #[error("index label '{label}' partition mismatch between tensors")]
    PartitionMismatch { label: String },

    #[error("cannot fully contract the network: remaining tensors are disconnected (no shared internal indices)")]
    DisconnectedNetwork,
}

pub type Result<T> = std::result::Result<T, BlockArray2Error>;
