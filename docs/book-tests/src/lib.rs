//! Test harness for mdBook guide code examples.
//!
//! Each module embeds a markdown file via `include_str!`. Running
//! `cargo test --doc -p book-tests` executes every fenced Rust code block
//! in these files as a doctest.
//!
//! `scripts/test-mdbook.sh` reuses this crate's resolved `--extern` flags to
//! run the same snippets through `mdbook test docs/book`, avoiding ambiguous
//! dependency resolution from the workspace-wide `target/release/deps` cache.
//!
//! The repository root `README.md` is included here as well so its short
//! runnable examples stay covered by CI.

#[cfg(test)]
const BOOK_INTRODUCTION: &str = include_str!("../../book/src/README.md");

#[doc = include_str!("../../../README.md")]
mod root_readme {}

#[doc = include_str!("../../book/src/getting-started.md")]
mod getting_started {}

#[doc = include_str!("../../book/src/guides/tensor-basics.md")]
mod tensor_basics {}

#[doc = concat!("```rust\n", include_str!("../../../crates/tensor4all-treetn/examples/randomized_factorization.rs"), "\n```")]
mod randomized_factorization {}

#[doc = include_str!("../../book/src/guides/tensor-train.md")]
mod tensor_train {}

#[doc = include_str!("../../book/src/guides/compress.md")]
mod compress {}

#[doc = include_str!("../../book/src/guides/tci.md")]
mod tci {}

#[doc = include_str!("../../book/src/guides/tci-advanced.md")]
mod tci_advanced {}

#[doc = include_str!("../../book/src/guides/tree-tn.md")]
mod tree_tn {}

#[doc = include_str!("../../book/src/guides/hdf5-serialization.md")]
mod hdf5_serialization {}

#[doc = include_str!("../../book/src/guides/quantics.md")]
mod quantics {}

#[doc = include_str!("../../book/src/guides/qft.md")]
mod qft {}

#[cfg(test)]
mod tests;

// The RSI guide includes this exact source via mdBook preprocessing. Embed
// the source here so workspace doctests exercise it without that preprocessor.
#[doc = concat!("```rust\n", include_str!("../../../crates/tensor4all-treersi/examples/validated_product.rs"), "\n``` ")]
mod tree_rsi {}

// Execute the patched-QFT example from the same source linked by the QFT guide.
#[doc = concat!("```rust\n", include_str!("../../../crates/tensor4all-quanticstransform/examples/patched_fourier.rs"), "\n``` ")]
mod patched_fourier {}
