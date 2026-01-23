//! Index mapping between true site indices and internal MPO indices.
//!
//! In the equation `A * x = b`:
//! - The state `x` has site indices with certain IDs
//! - The MPO `A` internally uses different IDs (`s_in_tmp`, `s_out_tmp`)
//! - This mapping defines the correspondence

use tensor4all_core::IndexLike;

/// Mapping between true site indices and internal MPO indices.
///
/// In the equation `A * x = b`:
/// - The state `x` has site indices with certain IDs
/// - The MPO `A` internally uses different IDs (`s_in_tmp`, `s_out_tmp`)
/// - This mapping defines the correspondence
#[derive(Debug, Clone)]
pub struct IndexMapping<I>
where
    I: IndexLike,
{
    /// True site index (from state x or b)
    pub true_index: I,
    /// Internal MPO index (s_in_tmp or s_out_tmp)
    pub internal_index: I,
}
