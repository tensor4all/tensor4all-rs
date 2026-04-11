//! Test harness for mdBook guide code examples.
//!
//! Each module embeds a markdown guide file via `include_str!`. Running
//! `cargo test --doc -p book-tests` executes every fenced Rust code block
//! in these files as a doctest.

#[doc = include_str!("../../book/src/getting-started.md")]
mod getting_started {}

#[doc = include_str!("../../book/src/guides/tensor-basics.md")]
mod tensor_basics {}

#[doc = include_str!("../../book/src/guides/tensor-train.md")]
mod tensor_train {}

#[doc = include_str!("../../book/src/guides/compress.md")]
mod compress {}
