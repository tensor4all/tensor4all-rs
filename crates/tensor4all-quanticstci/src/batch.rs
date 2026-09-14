//! Column-major batch views for batched quantics evaluation.
//!
//! Every batched entry point of this crate hands the target function a
//! [`QuanticsBatch`]. The layout is the column-major `(n_dims, n_points)` order
//! used by `tensor4all_treetci::GlobalIndexBatch` and
//! `tensor4all_treeaci::TreeElementwiseBatch`: point `p` occupies the
//! contiguous block `data()[p * n_dims..][..n_dims]`.
//!
//! A column-major `(n_dims, n_points)` buffer is memory-identical to a
//! C-contiguous `(n_points, n_dims)` array, so language bindings can hand the
//! same bytes to Python as rows of points without transposing.

use std::cell::RefCell;
use std::collections::HashMap;

use crate::error::{QuanticsTCIError, Result as QtciResult};

/// Borrowed view of one batch of grid points in column-major order.
///
/// # Examples
///
/// ```
/// use tensor4all_quanticstci::QuanticsBatch;
///
/// // Three points in two dimensions: (0, 1), (1, 0), (2, 2)
/// let data = [0.0, 1.0, 1.0, 0.0, 2.0, 2.0];
/// let batch = QuanticsBatch::new(&data, 2, 3).unwrap();
///
/// assert_eq!(batch.n_dims(), 2);
/// assert_eq!(batch.n_points(), 3);
/// assert_eq!(batch.get(1, 0), Some(1.0));
/// assert_eq!(batch.point(2), Some(&[2.0, 2.0][..]));
/// ```
#[derive(Clone, Copy, Debug)]
pub struct QuanticsBatch<'a, T> {
    data: &'a [T],
    n_dims: usize,
    n_points: usize,
}

impl<'a, T: Copy> QuanticsBatch<'a, T> {
    /// Create a batch view over column-major `(n_dims, n_points)` data.
    ///
    /// # Errors
    ///
    /// Returns an error when `data.len()` is not `n_dims * n_points` or when
    /// `n_dims` is zero (a shape mismatch).
    pub fn new(data: &'a [T], n_dims: usize, n_points: usize) -> QtciResult<Self> {
        if n_dims == 0 && n_points != 0 {
            return Err(QuanticsTCIError::InvalidConfiguration {
                message: "a batch must have at least one dimension per point".to_string(),
            });
        }
        let expected =
            n_dims
                .checked_mul(n_points)
                .ok_or_else(|| QuanticsTCIError::InvalidConfiguration {
                    message: "batch shape product overflowed usize".to_string(),
                })?;
        if data.len() != expected {
            return Err(QuanticsTCIError::InvalidConfiguration {
                message: format!(
                    "batch data has length {}, expected {expected} for shape ({n_dims}, {n_points})",
                    data.len()
                ),
            });
        }
        Ok(Self {
            data,
            n_dims,
            n_points,
        })
    }

    /// Borrow the column-major backing storage.
    pub fn data(&self) -> &'a [T] {
        self.data
    }

    /// Number of dimensions per point.
    pub fn n_dims(&self) -> usize {
        self.n_dims
    }

    /// Number of points in the batch.
    pub fn n_points(&self) -> usize {
        self.n_points
    }

    /// Coordinate `dim` of `point`, or `None` when out of bounds.
    pub fn get(&self, dim: usize, point: usize) -> Option<T> {
        (dim < self.n_dims && point < self.n_points).then(|| self.data[dim + self.n_dims * point])
    }

    /// Coordinates of `point`, or `None` when out of bounds.
    pub fn point(&self, point: usize) -> Option<&'a [T]> {
        if point >= self.n_points {
            return None;
        }
        self.data
            .get(point * self.n_dims..(point + 1) * self.n_dims)
    }
}

/// Adapt a point-wise function into a batched evaluator.
///
/// Repeated points are evaluated once and cached, which is the behaviour the
/// deprecated point-wise entry points document.
fn pointwise_batch<T, V, F, K>(
    evaluate: F,
    key: K,
) -> impl Fn(QuanticsBatch<'_, T>) -> anyhow::Result<Vec<V>>
where
    T: Copy,
    V: Clone,
    F: Fn(&[T]) -> V,
    K: Fn(&[T]) -> Vec<u64>,
{
    let cache: RefCell<HashMap<Vec<u64>, V>> = RefCell::new(HashMap::new());
    move |batch: QuanticsBatch<'_, T>| {
        let mut values = Vec::with_capacity(batch.n_points());
        for point in 0..batch.n_points() {
            let Some(coordinates) = batch.point(point) else {
                return Err(anyhow::anyhow!("invalid batch point index {point}"));
            };
            let key = key(coordinates);
            if let Some(cached) = cache.borrow().get(&key) {
                values.push(cached.clone());
                continue;
            }
            let value = evaluate(coordinates);
            cache.borrow_mut().insert(key, value.clone());
            values.push(value);
        }
        Ok(values)
    }
}

/// Adapt a point-wise function of original coordinates into a batched evaluator.
///
/// Use this together with [`quanticscrossinterpolate_batch`](crate::quanticscrossinterpolate_batch)
/// when only a scalar function is available. The adapter caches repeated
/// coordinates, so the point-wise function is called at most once per distinct
/// point.
///
/// # Examples
///
/// ```
/// use tensor4all_quanticstci::{
///     pointwise_coordinate_batch, quanticscrossinterpolate_batch, DiscretizedGrid, QtciOptions,
/// };
///
/// let grid = DiscretizedGrid::builder(&[4])
///     .with_lower_bound(&[0.0])
///     .with_upper_bound(&[1.0])
///     .build()
///     .unwrap();
///
/// let (qtci, _, _) = quanticscrossinterpolate_batch::<f64, _>(
///     &grid,
///     pointwise_coordinate_batch(|x: &[f64]| x[0] * x[0]),
///     None,
///     QtciOptions::default(),
/// )
/// .unwrap();
/// assert!(qtci.evaluate(&[3]).is_ok());
/// ```
pub fn pointwise_coordinate_batch<V, F>(
    evaluate: F,
) -> impl Fn(QuanticsBatch<'_, f64>) -> anyhow::Result<Vec<V>>
where
    F: Fn(&[f64]) -> V,
    V: Clone,
{
    pointwise_batch(evaluate, |point| {
        point.iter().map(|value| value.to_bits()).collect()
    })
}

/// Adapt a point-wise function of grid indices into a batched evaluator.
///
/// Use this together with
/// [`quanticscrossinterpolate_discrete_batch`](crate::quanticscrossinterpolate_discrete_batch)
/// when only a scalar function is available.
///
/// # Examples
///
/// ```
/// use tensor4all_quanticstci::{
///     pointwise_index_batch, quanticscrossinterpolate_discrete_batch, QtciOptions,
/// };
///
/// let (qtci, _, _) = quanticscrossinterpolate_discrete_batch::<f64, _>(
///     &[8, 8],
///     pointwise_index_batch(|index: &[usize]| (index[0] + index[1]) as f64),
///     None,
///     QtciOptions::default(),
/// )
/// .unwrap();
/// assert!((qtci.evaluate(&[1, 2]).unwrap() - 3.0).abs() < 1e-10);
/// ```
pub fn pointwise_index_batch<V, F>(
    evaluate: F,
) -> impl Fn(QuanticsBatch<'_, usize>) -> anyhow::Result<Vec<V>>
where
    F: Fn(&[usize]) -> V,
    V: Clone,
{
    pointwise_batch(evaluate, |point| {
        point.iter().map(|&value| value as u64).collect()
    })
}

/// Adapt a point-wise function returning several components into a batched
/// evaluator.
///
/// Use this together with
/// [`quanticscrossinterpolate_multicomponent`](crate::quanticscrossinterpolate_multicomponent)
/// when only a scalar function returning all components is available. Repeated
/// coordinates are evaluated once and cached. Values are emitted point-major:
/// the components of each requested point consecutively.
///
/// # Examples
///
/// ```
/// use tensor4all_quanticstci::{
///     pointwise_components_batch, quanticscrossinterpolate_multicomponent, DiscretizedGrid,
///     QtciOptions,
/// };
///
/// let grid = DiscretizedGrid::builder(&[3])
///     .with_lower_bound(&[0.0])
///     .with_upper_bound(&[1.0])
///     .build()
///     .unwrap();
///
/// let (result, _, _) = quanticscrossinterpolate_multicomponent::<f64, _>(
///     &grid,
///     pointwise_components_batch(|x: &[f64]| vec![x[0] + 1.0, x[0] * x[0] + 1.0]),
///     &[2],
///     None,
///     QtciOptions::default(),
/// )
/// .unwrap();
/// assert_eq!(result.output_dims(), &[2]);
/// ```
pub fn pointwise_components_batch<V, F>(
    evaluate: F,
) -> impl Fn(QuanticsBatch<'_, f64>) -> anyhow::Result<Vec<V>>
where
    F: Fn(&[f64]) -> Vec<V>,
    V: Clone,
{
    let cache: RefCell<HashMap<Vec<u64>, Vec<V>>> = RefCell::new(HashMap::new());
    move |batch: QuanticsBatch<'_, f64>| {
        let mut values = Vec::new();
        for point in 0..batch.n_points() {
            let coordinates = batch
                .point(point)
                .ok_or_else(|| anyhow::anyhow!("invalid batch point index {point}"))?;
            let key: Vec<u64> = coordinates.iter().map(|value| value.to_bits()).collect();
            if let Some(cached) = cache.borrow().get(&key) {
                values.extend(cached.iter().cloned());
                continue;
            }
            let evaluated = evaluate(coordinates);
            cache.borrow_mut().insert(key, evaluated.clone());
            values.extend(evaluated);
        }
        Ok(values)
    }
}
