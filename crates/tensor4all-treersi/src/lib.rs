//! Experimental one-pass recursive sketched interpolation of Hadamard products
//! on tree tensor networks, following Meng et al., [arXiv:2602.17974v1](https://arxiv.org/abs/2602.17974v1).
//!
//! The tree extension selects rows of products of sketched input frames, then
//! re-interpolates each original input at those rows. It does not materialize
//! the product tensor or multiply all input bond dimensions. Exact and sketch
//! messages carry binary scales throughout contraction.
//!
//! A small local pivot diagnostic is **not a global error estimate**. Even a
//! low-rank true product may be poorly approximated by finite sketches. Check
//! actual pointwise products on independent samples or a small dense oracle.
//! Compared with `tensor4all-treeaci`, this method makes one pass and chooses
//! pivots from sketches rather than adaptive samples of the true function.
//!
//! Only Hadamard products are supported. Applying arbitrary nonlinear maps to
//! sketch values is not a justified general-purpose function approximation.
//! The implementation uses host buffers and discrete pivot selection; with the
//! `tenferro-cuda` feature a caller-supplied CUDA context runs the contractions
//! on the device and transfers each operand and result explicitly. Two host
//! dependencies remain for a CUDA run: the input-core readback enters the
//! process-global CPU eager context, and the rank decision uses the
//! process-global CPU context. Automatic differentiation through this operation
//! is not supported.
//!
//! [`hadamard_many_with_rng_on_session`] runs the same implementation on a
//! caller-held explicit session (CPU only): every step stays on that session,
//! and the result is compact host storage that outlives it. A CPU context entry
//! is that implementation run on the context's own session.
//!
//! # Examples
//! ```
//! use tensor4all_core::{DynIndex, IdxTensor};
//! use tensor4all_treetn::TreeTN;
//! use tensor4all_treersi::{hadamard_many, TreeRsiOptions};
//! let t = TreeTN::from_tensors(vec![IdxTensor::from_dense(
//!     vec![DynIndex::new_dyn(2)], vec![2.0_f64, -3.0])?], vec![0usize])?;
//! let options = TreeRsiOptions { max_bond_dim: Some(4), ..Default::default() };
//! let squared = hadamard_many::<f64,_>(&[t.clone(),t], &options)?;
//! assert_eq!(squared.tree.to_dense()?.to_vec::<f64>()?, vec![4.0,9.0]);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod api;
mod dense;
mod engine;
mod error;
mod interpolator;
mod options;
mod plan;
mod result;
mod scalar;
mod scaled;

#[cfg(feature = "global-defaults")]
pub use api::hadamard_many;
pub use api::{hadamard_many_in, hadamard_many_with_rng_in, hadamard_many_with_rng_on_session};
pub use error::{Result, TreeRsiError};
pub use interpolator::TreeRsiInterpolator;
pub use options::{TreeRsiOptions, TreeRsiProbes};
pub use result::{TreeRsiDiagnostics, TreeRsiEdgeReport, TreeRsiResult};
pub use scalar::{TreeRsiNode, TreeRsiScalar};
