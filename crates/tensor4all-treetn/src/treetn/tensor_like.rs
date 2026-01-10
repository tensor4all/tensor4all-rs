//! TensorLike is NOT implemented for TreeTN.
//!
//! **Design Decision**: TreeTN intentionally does NOT implement TensorLike.
//!
//! ## Reasons
//!
//! 1. **Unclear semantics**: What would `tensordot` between two TreeTNs mean?
//! 2. **Hidden costs**: `to_tensor()` has exponential cost that shouldn't be hidden
//! 3. **Separation of concerns**: Dense tensors and TNs are fundamentally different
//!
//! ## Alternative API
//!
//! TreeTN provides its own methods instead:
//! - `site_indices()`: Returns physical indices (not bonds)
//! - `contract_to_tensor()`: Explicit method for full contraction (exponential cost)
//! - `contract_nodes()`: Graph operations for node contraction
