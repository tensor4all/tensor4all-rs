//! Batched (vector/tensor-valued) Quantics TCI interpolation.
//!
//! This module provides [`quanticscrossinterpolate_multicomponent`], which
//! interpolates vector- or tensor-valued functions by interpolating each output
//! component independently and combining the results into a single
//! [`SimpleTensorTrain`] with an additional component site.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use anyhow::{anyhow, Result};
use quanticsgrids::DiscretizedGrid;
use tensor4all_core::TensorElement;
use tensor4all_simplett::{AbstractTensorTrain, SimpleTensorTrain, TTScalar};
use tensor4all_simplett::{Tensor3, Tensor3Ops};
use tensor4all_tensorbackend::FullPivLuScalar;

use crate::batch::QuanticsBatch;
use crate::error::{QuanticsTCIError, Result as QtciResult};
use crate::options::QtciOptions;
use crate::quantics_tci::quanticscrossinterpolate_batch;

/// Result of batched (vector/tensor-valued) Quantics TCI interpolation.
///
/// Wraps a [`SimpleTensorTrain`] where the last site is a component index,
/// plus the output shape and grid information.
///
/// # Examples
///
/// ```
/// use tensor4all_quanticstci::{
///     quanticscrossinterpolate_multicomponent, AbstractTensorTrain, DiscretizedGrid,
///     QtciOptions, QuanticsBatch,
/// };
///
/// let grid = DiscretizedGrid::builder(&[2])
///     .with_lower_bound(&[0.0])
///     .with_upper_bound(&[1.0])
///     .build()
///     .unwrap();
///
/// let (result, _, _) = quanticscrossinterpolate_multicomponent::<f64, _>(
///     &grid,
///     |batch: QuanticsBatch<'_, f64>| -> anyhow::Result<Vec<f64>> {
///         let mut values = Vec::with_capacity(2 * batch.n_points());
///         for point in 0..batch.n_points() {
///             let x = batch.get(0, point).unwrap();
///             values.extend([x + 1.0, 2.0 * x + 1.0]);
///         }
///         Ok(values)
///     },
///     &[2],
///     None,
///     QtciOptions::default(),
/// ).unwrap();
///
/// assert_eq!(result.output_dims(), &[2]);
/// assert_eq!(result.tensor_train().len(), 3); // 2 grid sites + 1 component site
/// ```
#[derive(Clone)]
pub struct QuanticsTensorCI2Batched<V: TTScalar> {
    /// Combined tensor train with component index as the last site.
    tt: SimpleTensorTrain<V>,
    /// Shape of the output (e.g., [3] for 3-vector, [2, 2] for 2x2 matrix).
    output_dims: Vec<usize>,
    /// Grid for coordinate conversion.
    grid: DiscretizedGrid,
}

impl<V> QuanticsTensorCI2Batched<V>
where
    V: TTScalar + Default + Clone,
{
    /// Get the combined tensor train.
    ///
    /// The last site of the tensor train is the component index with
    /// dimension equal to the product of `output_dims`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_quanticstci::{
    ///     quanticscrossinterpolate_multicomponent, AbstractTensorTrain, DiscretizedGrid,
    ///     QtciOptions, QuanticsBatch,
    /// };
    ///
    /// let grid = DiscretizedGrid::builder(&[2])
    ///     .with_lower_bound(&[0.0])
    ///     .with_upper_bound(&[1.0])
    ///     .build()
    ///     .unwrap();
    ///
    /// let (result, _, _) = quanticscrossinterpolate_multicomponent::<f64, _>(
    ///     &grid,
    ///     |batch: QuanticsBatch<'_, f64>| -> anyhow::Result<Vec<f64>> {
    ///         let mut values = Vec::with_capacity(2 * batch.n_points());
    ///         for point in 0..batch.n_points() {
    ///             let x = batch.get(0, point).unwrap();
    ///             values.extend([x + 1.0, x * x + 1.0]);
    ///         }
    ///         Ok(values)
    ///     },
    ///     &[2],
    ///     None,
    ///     QtciOptions::default(),
    /// ).unwrap();
    ///
    /// let tt = result.tensor_train();
    /// assert_eq!(tt.len(), 3); // 2 grid sites + 1 component site
    /// ```
    pub fn tensor_train(&self) -> &SimpleTensorTrain<V> {
        &self.tt
    }

    /// Get the output dimensions (shape of the function output).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_quanticstci::{
    ///     quanticscrossinterpolate_multicomponent, QtciOptions, DiscretizedGrid, QuanticsBatch,
    /// };
    ///
    /// let grid = DiscretizedGrid::builder(&[2])
    ///     .with_lower_bound(&[0.0])
    ///     .with_upper_bound(&[1.0])
    ///     .build()
    ///     .unwrap();
    ///
    /// let (result, _, _) = quanticscrossinterpolate_multicomponent::<f64, _>(
    ///     &grid,
    ///     |batch: QuanticsBatch<'_, f64>| -> anyhow::Result<Vec<f64>> {
    ///         let mut values = Vec::with_capacity(2 * batch.n_points());
    ///         for point in 0..batch.n_points() {
    ///             let x = batch.get(0, point).unwrap();
    ///             values.extend([x + 1.0, x * x + 1.0]);
    ///         }
    ///         Ok(values)
    ///     },
    ///     &[2],
    ///     None,
    ///     QtciOptions::default(),
    /// ).unwrap();
    ///
    /// assert_eq!(result.output_dims(), &[2]);
    /// ```
    pub fn output_dims(&self) -> &[usize] {
        &self.output_dims
    }

    /// Get the discretized grid.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_quanticstci::{
    ///     quanticscrossinterpolate_multicomponent, QtciOptions, DiscretizedGrid, QuanticsBatch,
    /// };
    ///
    /// let grid = DiscretizedGrid::builder(&[2])
    ///     .with_lower_bound(&[0.0])
    ///     .with_upper_bound(&[1.0])
    ///     .build()
    ///     .unwrap();
    ///
    /// let (result, _, _) = quanticscrossinterpolate_multicomponent::<f64, _>(
    ///     &grid,
    ///     |batch: QuanticsBatch<'_, f64>| -> anyhow::Result<Vec<f64>> {
    ///         Ok((0..batch.n_points())
    ///             .map(|point| batch.get(0, point).unwrap() + 1.0)
    ///             .collect())
    ///     },
    ///     &[1],
    ///     None,
    ///     QtciOptions::default(),
    /// ).unwrap();
    ///
    /// assert!(result.grid().grid_step().len() > 0);
    /// ```
    pub fn grid(&self) -> &DiscretizedGrid {
        &self.grid
    }
}

/// Interpolate a vector/tensor-valued function, evaluating `f` in batches.
///
/// Each output component is interpolated independently with
/// [`quanticscrossinterpolate_batch`], and the per-component tensor trains are
/// combined into a single [`SimpleTensorTrain`] with an additional component
/// site at the end. A shared cache means each grid point is evaluated at most
/// once across all components.
///
/// # Arguments
///
/// * `grid` - Discretized grid describing the function domain
/// * `f` - Batched function to interpolate. It receives original coordinates as
///   a [`QuanticsBatch`] and must return `n_points * n_components` values, where
///   `n_components = product(output_dims)`: the components of each requested
///   point consecutively, in point order.
/// * `output_dims` - Shape of the function output (e.g., `&[3]` for 3-vector,
///
///   `&[2, 2]` for 2x2 matrix)
/// * `initial_pivots` - Initial pivot grid indices (0-indexed, optional)
/// * `options` - TCI options
///
/// # Returns
///
/// Tuple of ([`QuanticsTensorCI2Batched`], max_bond_dims_across_components, max_errors_across_components)
///
/// # Errors
///
/// Returns an error when `output_dims` is empty or has a zero factor, the grid
/// or options are invalid, `f` returns the wrong number of values, or a
/// component interpolation fails.
///
/// # Examples
///
/// ```
/// use tensor4all_quanticstci::{
///     quanticscrossinterpolate_multicomponent, AbstractTensorTrain, DiscretizedGrid, QtciOptions,
///     QuanticsBatch,
/// };
///
/// let grid = DiscretizedGrid::builder(&[2])
///     .with_lower_bound(&[0.0])
///     .with_upper_bound(&[1.0])
///     .build()
///     .unwrap();
///
/// let f = |batch: QuanticsBatch<'_, f64>| -> anyhow::Result<Vec<f64>> {
///     let mut values = Vec::with_capacity(2 * batch.n_points());
///     for point in 0..batch.n_points() {
///         let x = batch.get(0, point).unwrap();
///         values.extend([x + 1.0, 2.0 * x + 1.0]);
///     }
///     Ok(values)
/// };
///
/// let (result, ranks, errors) = quanticscrossinterpolate_multicomponent::<f64, _>(
///     &grid,
///     f,
///     &[2],
///     None,
///     QtciOptions::default().with_tolerance(1e-8),
/// ).unwrap();
///
/// assert_eq!(result.tensor_train().len(), 3); // 2 grid sites + 1 component site
/// assert_eq!(result.output_dims(), &[2]);
/// assert!(!ranks.is_empty());
/// assert!(!errors.is_empty());
/// ```
pub fn quanticscrossinterpolate_multicomponent<V, F>(
    grid: &DiscretizedGrid,
    f: F,
    output_dims: &[usize],
    initial_pivots: Option<Vec<Vec<usize>>>,
    options: QtciOptions,
) -> QtciResult<(QuanticsTensorCI2Batched<V>, Vec<usize>, Vec<f64>)>
where
    F: Fn(QuanticsBatch<'_, f64>) -> Result<Vec<V>>,
    V: TTScalar
        + Default
        + Clone
        + 'static
        + TensorElement
        + tensor4all_core::MatrixLuciScalar
        + FullPivLuScalar
        + tensor4all_treetci::globalpivot::ScalarParts,
{
    // Validate output_dims
    if output_dims.is_empty() {
        return Err(QuanticsTCIError::InvalidConfiguration {
            message: "output_dims must not be empty".to_string(),
        });
    }
    let n_components = output_dims.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| QuanticsTCIError::InvalidConfiguration {
                message: format!("product of output_dims overflowed usize: {output_dims:?}"),
            })
    })?;
    if n_components == 0 {
        return Err(QuanticsTCIError::InvalidConfiguration {
            message: format!("product of output_dims must be positive, got 0 from {output_dims:?}"),
        });
    }

    // Shared across components: coordinate bits -> all component values.
    let cache: Rc<RefCell<HashMap<Vec<u64>, Vec<V>>>> = Rc::new(RefCell::new(HashMap::new()));
    let f = Rc::new(f);

    let mut component_tts: Vec<SimpleTensorTrain<V>> = Vec::with_capacity(n_components);
    let mut all_ranks: Vec<Vec<usize>> = Vec::with_capacity(n_components);
    let mut all_errors: Vec<Vec<f64>> = Vec::with_capacity(n_components);

    for component in 0..n_components {
        let cache = Rc::clone(&cache);
        let f = Rc::clone(&f);
        let adapter = move |batch: QuanticsBatch<'_, f64>| -> Result<Vec<V>> {
            let n_points = batch.n_points();
            let n_dims = batch.n_dims();
            let mut values: Vec<Option<V>> = vec![None; n_points];
            let mut missing: Vec<usize> = Vec::new();
            for (point, value) in values.iter_mut().enumerate() {
                let key = coordinate_key(&batch, point)?;
                match cache.borrow().get(&key) {
                    Some(components) => {
                        *value = Some(components.get(component).cloned().ok_or_else(|| {
                            anyhow!("cached point is missing component {component}")
                        })?);
                    }
                    None => missing.push(point),
                }
            }

            if !missing.is_empty() {
                let mut coordinates = Vec::with_capacity(n_dims * missing.len());
                for &point in &missing {
                    let point_values = batch
                        .point(point)
                        .ok_or_else(|| anyhow!("invalid batch point index {point}"))?;
                    coordinates.extend_from_slice(point_values);
                }
                let returned = f(QuanticsBatch::new(&coordinates, n_dims, missing.len())?)?;
                if returned.len() % missing.len() != 0
                    || returned.len() / missing.len() < n_components
                {
                    return Err(anyhow!(
                        "callback returned {} values for {} points, expected at least {} components per point",
                        returned.len(),
                        missing.len(),
                        n_components
                    ));
                }
                for (offset, &point) in missing.iter().enumerate() {
                    let components = returned[offset * n_components..][..n_components].to_vec();
                    values[point] = Some(
                        components
                            .get(component)
                            .cloned()
                            .ok_or_else(|| anyhow!("missing component {component}"))?,
                    );
                    cache
                        .borrow_mut()
                        .insert(coordinate_key(&batch, point)?, components);
                }
            }

            values
                .into_iter()
                .map(|value| value.ok_or_else(|| anyhow!("missing component value")))
                .collect()
        };

        let (qtci, ranks, errors) =
            quanticscrossinterpolate_batch(grid, adapter, initial_pivots.clone(), options.clone())?;

        component_tts.push(qtci.tensor_train());
        all_ranks.push(ranks);
        all_errors.push(errors);
    }

    // Combine component TTs into a single TT with a component selector site.
    let combined_tt = combine_component_tts(&component_tts)?;

    // Compute aggregate ranks and errors (element-wise max across components).
    let max_len = all_ranks.iter().map(|r| r.len()).max().unwrap_or(0);
    let mut max_bond_dims = vec![0usize; max_len];
    for ranks in &all_ranks {
        for (i, &r) in ranks.iter().enumerate() {
            max_bond_dims[i] = max_bond_dims[i].max(r);
        }
    }

    let max_err_len = all_errors.iter().map(|e| e.len()).max().unwrap_or(0);
    let mut max_errors = vec![0.0f64; max_err_len];
    for errors in &all_errors {
        for (i, &e) in errors.iter().enumerate() {
            max_errors[i] = max_errors[i].max(e);
        }
    }

    let result = QuanticsTensorCI2Batched {
        tt: combined_tt,
        output_dims: output_dims.to_vec(),
        grid: grid.clone(),
    };

    Ok((result, max_bond_dims, max_errors))
}

/// Interpolate a vector/tensor-valued function, one point at a time.
///
/// Deprecated: `f` is called once per grid point. Use
/// [`quanticscrossinterpolate_multicomponent`] instead, optionally with
/// [`pointwise_components_batch`](crate::pointwise_components_batch) to keep a
/// point-wise closure.
///
/// # Errors
///
/// Returns the same errors as [`quanticscrossinterpolate_multicomponent`].
#[deprecated(
    note = "calls `f` once per point; use `quanticscrossinterpolate_multicomponent` (optionally with `pointwise_components_batch`) instead"
)]
pub fn quanticscrossinterpolate_batched<V, F>(
    grid: &DiscretizedGrid,
    f: F,
    output_dims: &[usize],
    initial_pivots: Option<Vec<Vec<usize>>>,
    options: QtciOptions,
) -> QtciResult<(QuanticsTensorCI2Batched<V>, Vec<usize>, Vec<f64>)>
where
    F: Fn(&[f64]) -> Vec<V> + 'static,
    V: TTScalar
        + Default
        + Clone
        + 'static
        + TensorElement
        + tensor4all_core::MatrixLuciScalar
        + FullPivLuScalar
        + tensor4all_treetci::globalpivot::ScalarParts,
{
    quanticscrossinterpolate_multicomponent(
        grid,
        crate::batch::pointwise_components_batch(f),
        output_dims,
        initial_pivots,
        options,
    )
}

/// Key for the multi-component evaluation cache.
fn coordinate_key(batch: &QuanticsBatch<'_, f64>, point: usize) -> Result<Vec<u64>> {
    let values = batch
        .point(point)
        .ok_or_else(|| anyhow!("invalid batch point index {point}"))?;
    Ok(values.iter().map(|value| value.to_bits()).collect())
}

/// Combine per-component tensor trains into a single TT with a component
/// selector as the final site.
///
/// The combination strategy ensures SimpleTensorTrain validity:
/// - **First site** (`left_dim = 1` for all components): the component tensors
///
///   are concatenated along the right bond, giving shape `(1, site_dim, sum_right)`.
/// - **Middle sites**: block-diagonal in both bond dimensions.
/// - **Last grid site** (`right_dim = 1` for all components): block-diagonal in
///
///   left bond, concatenated right bond gives `(sum_left, site_dim, n_components)`.
///
/// A final selector site of shape `(total_right_bond, n_components, 1)` is
/// appended so that fixing the component index selects the corresponding
/// component's value.
fn combine_component_tts<V>(component_tts: &[SimpleTensorTrain<V>]) -> Result<SimpleTensorTrain<V>>
where
    V: TTScalar + Default + Clone,
{
    let n_components = component_tts.len();
    if n_components == 0 {
        return Err(anyhow!("no component tensor trains to combine"));
    }

    let n_sites = component_tts[0].len();
    if n_sites == 0 {
        return Err(anyhow!(
            "component tensor trains must have at least one site"
        ));
    }

    // Verify all components have the same number of sites and site dimensions.
    for (c, tt) in component_tts.iter().enumerate() {
        if tt.len() != n_sites {
            return Err(anyhow!(
                "component {} has {} sites, expected {}",
                c,
                tt.len(),
                n_sites
            ));
        }
        for s in 0..n_sites {
            if tt.site_tensor(s).site_dim() != component_tts[0].site_tensor(s).site_dim() {
                return Err(anyhow!(
                    "component {} site {} has site_dim {}, expected {}",
                    c,
                    s,
                    tt.site_tensor(s).site_dim(),
                    component_tts[0].site_tensor(s).site_dim()
                ));
            }
        }
    }

    let mut combined_tensors: Vec<Tensor3<V>> = Vec::with_capacity(n_sites + 1);

    for s in 0..n_sites {
        let site_dim = component_tts[0].site_tensor(s).site_dim();

        let total_right: usize = component_tts
            .iter()
            .map(|tt| tt.site_tensor(s).right_dim())
            .sum();

        if s == 0 {
            // First site: all components have left_dim = 1.
            // Concatenate along right bond: shape (1, site_dim, total_right).
            let mut combined = tensor3_zeros_generic::<V>(1, site_dim, total_right);

            let mut right_offset = 0;
            for tt in component_tts.iter() {
                let t = tt.site_tensor(s);
                let rd = t.right_dim();

                for sd in 0..site_dim {
                    for r in 0..rd {
                        combined.set3(0, sd, right_offset + r, *t.get3(0, sd, r));
                    }
                }

                right_offset += rd;
            }

            combined_tensors.push(combined);
        } else {
            // Middle and last grid sites: block-diagonal in both bond dims.
            let total_left: usize = component_tts
                .iter()
                .map(|tt| tt.site_tensor(s).left_dim())
                .sum();

            let mut combined = tensor3_zeros_generic::<V>(total_left, site_dim, total_right);

            let mut left_offset = 0;
            let mut right_offset = 0;
            for tt in component_tts.iter() {
                let t = tt.site_tensor(s);
                let ld = t.left_dim();
                let rd = t.right_dim();

                for l in 0..ld {
                    for sd in 0..site_dim {
                        for r in 0..rd {
                            combined.set3(left_offset + l, sd, right_offset + r, *t.get3(l, sd, r));
                        }
                    }
                }

                left_offset += ld;
                right_offset += rd;
            }

            combined_tensors.push(combined);
        }
    }

    // Build the component selector site.
    // Shape: (total_right_of_last_grid_site, n_components, 1)
    let total_right: usize = component_tts
        .iter()
        .map(|tt| tt.site_tensor(n_sites - 1).right_dim())
        .sum();

    let mut selector = tensor3_zeros_generic::<V>(total_right, n_components, 1);

    let mut offset = 0;
    for (c, tt) in component_tts.iter().enumerate() {
        let rd = tt.site_tensor(n_sites - 1).right_dim();
        for i in 0..rd {
            selector.set3(offset + i, c, 0, V::one());
        }
        offset += rd;
    }

    combined_tensors.push(selector);

    SimpleTensorTrain::new(combined_tensors)
        .map_err(|e| anyhow!("Failed to build combined TT: {}", e))
}

/// Create a zero-filled Tensor3 using TTScalar bounds (avoids importing tensor3_zeros
/// which may have different trait bounds).
fn tensor3_zeros_generic<V: TTScalar + Default + Clone>(
    left: usize,
    site: usize,
    right: usize,
) -> Tensor3<V> {
    use tensor4all_simplett::tensor3_zeros;
    tensor3_zeros(left, site, right)
}

#[cfg(test)]
mod tests;
