//! Quantics transformation operators for tensor train methods.
//!
//! This crate provides LinearOperator constructors for various transformations
//! in Quantics representation. It is a Rust port of transformation functionality
//! from [Quantics.jl](https://github.com/tensor4all/Quantics.jl).
//!
//! # Overview
//!
//! All transformations return `LinearOperator` from tensor4all-treetn for
//! consistent operator application to tensor train states.
//!
//! ## Available Transformations
//!
//! - **Flip**: f(x) = g(2^R - x)
//! - **Shift**: f(x) = g(x + offset) mod 2^R
//! - **Phase Rotation**: f(x) = exp(i*θ*x) * g(x)
//! - **Cumulative Sum**: y_i = Σ_{j<i} x_j
//! - **Fourier Transform**: Quantics Fourier Transform (QFT)
//! - **Binary Operation**: f(x, y) where first variable is a*x + b*y
//! - **Affine Transform**: y = A*x + b with rational coefficients
//!
//! # Example
//!
//! ```ignore
//! use tensor4all_quantics_transform::{flip_operator, BoundaryCondition};
//!
//! // Create a flip operator for 8-bit quantics representation
//! let op = flip_operator(8, BoundaryCondition::Periodic).unwrap();
//! ```

mod affine;
mod binaryop;
mod common;
mod cumsum;
mod flip;
mod fourier;
mod phase_rotation;
mod shift;

pub use affine::{affine_operator, AffineParams};
pub use binaryop::{binaryop_operator, binaryop_single_operator, BinaryCoeffs};
pub use common::{BoundaryCondition, CarryDirection};
pub use cumsum::cumsum_operator;
pub use flip::flip_operator;
pub use fourier::{quantics_fourier_operator, FTCore, FourierOptions};
pub use phase_rotation::phase_rotation_operator;
pub use shift::shift_operator;
