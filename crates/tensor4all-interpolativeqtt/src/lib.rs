#![warn(missing_docs)]
//! Interpolative Quantics Tensor Train construction.
//!
//! This crate ports the tested construction surface of
//! `InterpolativeQTT.jl` to Rust. The QTT constructors ([`interpolate_single_scale`],
//! [`interpolate_multi_scale`], [`interpolate_adaptive`] and
//! [`interpolate_patched_nd`]) return [`InterpolativeQtt`] values for evaluation
//! through TreeTN; [`interpolate_chebyshev_patches`] instead returns a piecewise
//! [`ChebyshevPatches`] interpolant that is evaluated directly.
//! Every constructor takes real (`f64`) or complex (`Complex64`) function
//! values; see [`InterpolativeScalar`].
//!
//! # Examples
//!
//! ```
//! use tensor4all_interpolativeqtt::{
//!     interpolate_single_scale, InterpolativeQttOptions,
//! };
//!
//! let context = tensor4all_tensorbackend::CpuExecutionContext::from_backend(
//!     tensor4all_tensorbackend::CpuBackend::with_threads(1).unwrap(),
//! );
//! let (tree, sites) = context
//!     .with_session(|exec| interpolate_single_scale(exec,
//!     |x| (-x * x).exp(),
//!     -1.0,
//!     1.0,
//!     4,
//!     10,
//!     &InterpolativeQttOptions::default(),
//! )).unwrap().unwrap();
//!
//! assert_eq!(tree.node_count(), 4);
//! let value = tree.evaluate_point(&sites, &[0, 0, 0, 0]).unwrap().real();
//! assert!((value - (-1.0_f64).exp()).abs() < 1e-10);
//! ```

#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;

pub mod prelude;

mod basis;
mod chebyshev_patches;
mod core_array;
mod error;
mod interpolation;
mod interval;
mod options;
mod patched;

pub use basis::{
    direct_product_core_tensors, get_chebyshev_grid, interpolation_tensor, LagrangePolynomials,
};
#[cfg(feature = "parallel")]
pub use chebyshev_patches::interpolate_chebyshev_patches_on_lanes;
pub use chebyshev_patches::{
    interpolate_chebyshev_patches, ChebyshevPatch, ChebyshevPatchOptions, ChebyshevPatches,
    ExtraValidation,
};
pub use error::{InterpolativeQttError, Result};
pub use interpolation::{
    estimate_interpolation_error, estimate_interpolation_error_nd, interpolate_adaptive,
    interpolate_adaptive_nd, interpolate_multi_scale, interpolate_multi_scale_nd,
    interpolate_single_scale, interpolate_single_scale_nd, interpolate_single_scale_sparse,
    interpolate_single_scale_sparse_nd, invert_qtt, InterpolativeScalar,
};
pub use options::InterpolativeQttOptions;
#[cfg(feature = "parallel")]
pub use patched::interpolate_patched_nd_on_lanes;
pub use patched::{interpolate_patched_nd, InterpolativePatch, PatchedInterpolativeOptions};
/// An interpolated TreeTN and its ordered physical indices.
/// Node names are bit levels `0..K`, most significant first. Within each
/// site, the fused digit stores axis zero in the least significant bit.
/// Pass the indices in this order to TreeTN evaluators and [`invert_qtt`].
///
/// # Examples
///
/// ```
/// use tensor4all_interpolativeqtt::{interpolate_single_scale, InterpolativeQtt};
/// use tensor4all_tensorbackend::{CpuBackend, CpuExecutionContext};
/// let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(1)?);
/// let (tree, sites): InterpolativeQtt = context.with_session(|exec| {
///     interpolate_single_scale(exec, |x| 1.0 + x, 0.0, 1.0, 3, 2, &Default::default())
/// })??;
/// let value = tree.evaluate_point(&sites, &[1, 0, 1])?;
/// assert!((value.real() - 1.625).abs() < 1e-12);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub type InterpolativeQtt = (
    tensor4all_treetn::TreeTN<tensor4all_core::IdxTensor, usize>,
    Vec<tensor4all_core::DynIndex>,
);

#[cfg(test)]
mod tests;
