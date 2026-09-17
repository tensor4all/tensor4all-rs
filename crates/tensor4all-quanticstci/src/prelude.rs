//! Commonly used types and functions for quantics tensor cross interpolation
//! (QTT) on discrete or continuous grids.
//!
//! ```rust
//! use tensor4all_quanticstci::prelude::*;
//!
//! let f = |batch: QuanticsBatch<'_, usize>| -> anyhow::Result<Vec<f64>> {
//!     Ok((0..batch.n_points())
//!         .map(|point| (batch.get(0, point).unwrap() + batch.get(1, point).unwrap()) as f64)
//!         .collect())
//! };
//! let (qtci, _ranks, errors) = quanticscrossinterpolate_discrete_batch(
//!     &[16, 16],
//!     f,
//!     None,
//!     QtciOptions::default().with_tolerance(1e-10),
//! ).unwrap();
//! let value = qtci.evaluate(&[4, 9]).unwrap();
//! assert!((value - 13.0).abs() < 1e-10);
//! assert!(errors.last().copied().unwrap() < 1e-10);
//! ```

// Re-exported while the point-wise entry points remain available: the
// deprecation is reported at each call site, not at this re-export.
#[allow(deprecated)]
pub use crate::{
    pointwise_components_batch, pointwise_coordinate_batch, pointwise_index_batch,
    quanticscrossinterpolate, quanticscrossinterpolate_batch, quanticscrossinterpolate_batched,
    quanticscrossinterpolate_discrete, quanticscrossinterpolate_discrete_batch,
    quanticscrossinterpolate_from_arrays, quanticscrossinterpolate_from_arrays_batch,
    quanticscrossinterpolate_multicomponent, DefaultProposer, DiscretizedGrid,
    InherentDiscreteGrid, QtciOptions, QuanticsBatch, QuanticsTensorCI2, QuanticsTensorCI2Batched,
    SimpleTensorTrain, TreeTciGraph, TreeTciOptions, UnfoldingScheme,
};
