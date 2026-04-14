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

#[doc = include_str!("../../../README.md")]
mod root_readme {}

#[doc = include_str!("../../book/src/getting-started.md")]
mod getting_started {}

#[doc = include_str!("../../book/src/guides/tensor-basics.md")]
mod tensor_basics {}

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

#[doc = include_str!("../../book/src/guides/quantics.md")]
mod quantics {}

#[doc = include_str!("../../book/src/guides/qft.md")]
mod qft {}
