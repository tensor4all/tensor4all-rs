//! Torch/libtorch backend for tensor operations.
//!
//! This module provides PyTorch tensor storage and operations via the tch-rs crate.
//! It enables autograd (automatic differentiation) for tensor operations.
//!
//! ## Usage
//!
//! Enable the `backend-libtorch` feature to use this module:
//!
//! ```toml
//! [dependencies]
//! tensor4all-tensorbackend = { version = "...", features = ["backend-libtorch"] }
//! ```
//!
//! ## Environment Setup
//!
//! Requires libtorch to be installed. Set environment variables:
//!
//! ```bash
//! export LIBTORCH=/path/to/libtorch
//! export DYLD_LIBRARY_PATH="$LIBTORCH/lib:$DYLD_LIBRARY_PATH"  # macOS
//! export LD_LIBRARY_PATH="$LIBTORCH/lib:$LD_LIBRARY_PATH"      # Linux
//! ```

mod storage;

pub use storage::TorchStorage;

// Re-export tch for downstream convenience
pub use tch;
