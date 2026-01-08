//! Block tensor planning utilities.
//!
//! This crate intentionally does **not** implement numeric tensor contraction.
//! It only provides:
//! - Block metadata (`BlockPartition`, `BlockStructure`)
//! - Contraction order optimization (`TensorNetworkPlanner`)
//! - Per-step contraction plan generation (`PairContractionPlan`)

mod error;
mod partition;
mod plan;
mod structure;

pub use error::{BlockArray2Error, Result};
pub use partition::{block_linear_index, block_multi_index, BlockIndex, BlockPartition};
pub use plan::{
    ContractionPathPlan, IndexLabel, PairContractionPlan, TensorLabel, TensorLabelAtom,
    TensorNetworkPlanner,
};
pub use structure::BlockStructure;
