//! Piecewise Chebyshev interpolation on adaptive dyadic patches.
//!
//! Unlike [`interpolate_patched_nd`](crate::interpolate_patched_nd), this module
//! keeps the Chebyshev-Lobatto node values of each accepted box and builds no
//! QTT: a box is accepted or split purely on its interpolation residual. A
//! failing box divides into `2^D` children, whose node values are exactly the
//! validation points their parent already sampled, so a child inherits its
//! values bit for bit instead of resampling them.
//!
//! See [`interpolate_chebyshev_patches`] for the entry point and
//! [`ChebyshevPatches::evaluate`] for evaluation on the continuous box.

use std::collections::BTreeMap;
use std::error::Error;

use tensor4all_core::ColMajorArrayRef;
use tensor4all_tensorbackend::{explicit::Session, CpuEngine, CpuExecutionContext, Exec, Matrix};

use crate::error::{invalid_argument, InterpolativeQttError, Result};
use crate::interval::NInterval;
use crate::patched::{apply_stencil, checked_power, checked_product, product_indices, sample};
use crate::{get_chebyshev_grid, InterpolativeScalar, LagrangePolynomials};

/// One accepted dyadic box and its Chebyshev-Lobatto node values.
///
/// The box is `[lower, upper)` per axis. `values` holds one value per
/// tensor-product node: `(degree + 1)^D` entries, column-major over the per-axis
/// node index with axis zero fastest, so node `(j_0, .., j_{D-1})` is at
/// `values[sum_d j_d * (degree + 1)^d]`.
///
/// A patch is produced by [`interpolate_chebyshev_patches`] and evaluated
/// through [`ChebyshevPatches::evaluate`].
///
/// # Examples
/// ```
/// use std::convert::Infallible;
/// use tensor4all_interpolativeqtt::{interpolate_chebyshev_patches, ChebyshevPatchOptions};
/// use tensor4all_tensorbackend::{CpuBackend, CpuExecutionContext};
///
/// let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(1)?);
/// let patches = interpolate_chebyshev_patches(
///     &context,
///     |points, _exec| Ok::<_, Infallible>(
///         points.data().iter().map(|&x| 1.0 + x + x * x).collect()),
///     &[0.0], &[1.0], 2, &ChebyshevPatchOptions::default())?;
/// let patch = &patches.patches()[0];
/// assert_eq!(patch.prefix, vec![Vec::<usize>::new()]);
/// assert_eq!(patch.lower, vec![0.0]);
/// assert_eq!(patch.upper, vec![1.0]);
/// assert_eq!(patch.values.len(), 3);
/// assert!((patch.values[2] - 3.0).abs() < 1e-12);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct ChebyshevPatch<T> {
    /// Fixed most-significant-first bits of each axis; empty for the root.
    pub prefix: Vec<Vec<usize>>,
    /// Lower corner of the box per axis.
    pub lower: Vec<f64>,
    /// Upper corner of the box per axis.
    pub upper: Vec<f64>,
    /// Values at the tensor-product Lobatto nodes of the box, `(degree + 1)^D`
    /// entries, column-major over the per-axis node index, axis zero fastest.
    pub values: Vec<T>,
    /// Maximum absolute residual at the sampled validation points of this box.
    /// It is a sampled estimate, not a certificate for arbitrary functions.
    pub error_estimate: f64,
}

/// Accepted patches in canonical prefix order, plus what evaluation needs.
///
/// The patches partition the construction domain into dyadic boxes, and their
/// prefixes form a prefix-free set, so every point of the domain belongs to
/// exactly one patch. A point on an internal boundary belongs to the upper
/// (higher-prefix) patch.
///
/// # Examples
/// ```
/// use std::convert::Infallible;
/// use tensor4all_interpolativeqtt::{interpolate_chebyshev_patches, ChebyshevPatchOptions};
/// use tensor4all_tensorbackend::{CpuBackend, CpuExecutionContext};
///
/// let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(1)?);
/// let patches = interpolate_chebyshev_patches(
///     &context,
///     |points, _exec| Ok::<_, Infallible>(points.data().iter().map(|&x| 1.0 + x).collect()),
///     &[0.0], &[2.0], 8, &ChebyshevPatchOptions::default())?;
/// assert_eq!(patches.patches().len(), 1);
/// assert_eq!(patches.degree(), 8);
/// let coordinates = [0.5_f64, 1.5];
/// let points = tensor4all_core::ColMajorArrayRef::new(&coordinates, &[1, 2])?;
/// let values = patches.evaluate(points)?;
/// assert!((values[0] - 1.5).abs() < 1e-12);
/// assert!((values[1] - 2.5).abs() < 1e-12);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct ChebyshevPatches<T> {
    lower: Vec<f64>,
    upper: Vec<f64>,
    degree: usize,
    basis: LagrangePolynomials,
    /// Node multi-indices of one patch, axis zero fastest.
    node_indices: Vec<Vec<usize>>,
    /// Node multi-index stride per axis.
    node_stride: Vec<usize>,
    patches: Vec<ChebyshevPatch<T>>,
    /// Accepted prefixes, so evaluation stops at the first accepted box.
    by_prefix: BTreeMap<Vec<Vec<usize>>, usize>,
    num_evaluations: usize,
}

impl<T> ChebyshevPatches<T> {
    /// The accepted patches, sorted lexicographically by per-axis prefix.
    ///
    /// # Examples
    /// ```
    /// use std::convert::Infallible;
    /// use tensor4all_interpolativeqtt::{interpolate_chebyshev_patches, ChebyshevPatchOptions};
    /// use tensor4all_tensorbackend::{CpuBackend, CpuExecutionContext};
    ///
    /// let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(1)?);
    /// let patches = interpolate_chebyshev_patches(
    ///     &context,
    ///     |points, _exec| Ok::<_, Infallible>(
    ///         points.data().chunks(2).map(|p| p[0] * p[0] + p[1]).collect()),
    ///     &[0.0, 0.0], &[1.0, 1.0], 3, &ChebyshevPatchOptions::default())?;
    /// assert_eq!(patches.patches().len(), 1);
    /// assert!(patches.patches()[0].error_estimate < 1e-12);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn patches(&self) -> &[ChebyshevPatch<T>] {
        &self.patches
    }

    /// Local polynomial degree of every patch, as passed to construction.
    ///
    /// # Examples
    /// ```
    /// use std::convert::Infallible;
    /// use tensor4all_interpolativeqtt::{interpolate_chebyshev_patches, ChebyshevPatchOptions};
    /// use tensor4all_tensorbackend::{CpuBackend, CpuExecutionContext};
    ///
    /// let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(1)?);
    /// let patches = interpolate_chebyshev_patches(
    ///     &context,
    ///     |points, _exec| Ok::<_, Infallible>(points.data().iter().map(|&x| 1.0 + x).collect()),
    ///     &[0.0], &[1.0], 5, &ChebyshevPatchOptions::default())?;
    /// assert_eq!(patches.degree(), 5);
    /// assert_eq!(patches.patches()[0].values.len(), 6);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Function evaluations spent on nodes and validation points.
    ///
    /// `(degree + 1)^D` for the root's node grid plus `(2 * degree + 1)^D` for
    /// every box, plus the extra validation points of every box whose grid check
    /// passed (whether it was then accepted or split). Use it to compare
    /// construction strategies.
    ///
    /// # Examples
    /// ```
    /// use std::convert::Infallible;
    /// use tensor4all_interpolativeqtt::{interpolate_chebyshev_patches, ChebyshevPatchOptions};
    /// use tensor4all_tensorbackend::{CpuBackend, CpuExecutionContext};
    ///
    /// let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(1)?);
    /// let patches = interpolate_chebyshev_patches(
    ///     &context,
    ///     |points, _exec| Ok::<_, Infallible>(points.data().iter().map(|&x| 1.0 + 2.0 * x).collect()),
    ///     &[0.0], &[1.0], 4, &ChebyshevPatchOptions::default())?;
    /// // Accepted at the root: 5 nodes plus 9 validation points.
    /// assert_eq!(patches.patches().len(), 1);
    /// assert_eq!(patches.num_evaluations(), 5 + 9);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn num_evaluations(&self) -> usize {
        self.num_evaluations
    }
}

impl<T: InterpolativeScalar> ChebyshevPatches<T> {
    /// Evaluate the piecewise interpolant at a batch of points.
    ///
    /// `points` is column-major with shape `[D, n]`: `D` coordinates per point,
    /// point `i` at `data[i * D..(i + 1) * D]`. Every point must lie in the
    /// closed construction domain; a point on an internal boundary is evaluated
    /// by the upper patch. A point whose coordinates are bitwise nodes of the
    /// selected patch returns that patch's stored values, so nodes reproduce.
    /// Across a face where two patches of different depth meet, the two
    /// polynomials need not agree: the point belongs to the upper patch.
    ///
    /// # Errors
    ///
    /// Returns [`InterpolativeQttError::InvalidArgument`] when `points` is not a
    /// rank-two array with `D` rows, or a coordinate is non-finite or outside
    /// the domain.
    ///
    /// # Examples
    /// ```
    /// use std::convert::Infallible;
    /// use tensor4all_core::ColMajorArrayRef;
    /// use tensor4all_interpolativeqtt::{interpolate_chebyshev_patches, ChebyshevPatchOptions};
    /// use tensor4all_tensorbackend::{CpuBackend, CpuExecutionContext};
    ///
    /// let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(1)?);
    /// let patches = interpolate_chebyshev_patches(
    ///     &context,
    ///     |points, _exec| Ok::<_, Infallible>(
    ///         points.data().chunks(2).map(|p| 1.0 + 2.0 * p[0] - 3.0 * p[1]).collect()),
    ///     &[0.0, 0.0], &[1.0, 1.0], 3, &ChebyshevPatchOptions::default())?;
    /// let coordinates = [0.25_f64, 0.5];
    /// let points = ColMajorArrayRef::new(&coordinates, &[2, 1])?;
    /// let values = patches.evaluate(points)?;
    /// assert!(values[0].abs() < 1e-12);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn evaluate(&self, points: ColMajorArrayRef<'_, f64>) -> Result<Vec<T>> {
        let ndims = self.lower.len();
        if points.shape().len() != 2 || points.shape()[0] != ndims {
            return Err(invalid_argument(format!(
                "evaluation points must be a rank-two [{ndims}, n] array, got shape {:?}",
                points.shape()
            )));
        }
        let mut values = Vec::with_capacity(points.shape()[1]);
        for point in points.data().chunks_exact(ndims) {
            values.push(self.evaluate_point(point)?);
        }
        Ok(values)
    }

    fn evaluate_point(&self, point: &[f64]) -> Result<T> {
        let ndims = self.lower.len();
        for (axis, (&x, (&a, &b))) in point
            .iter()
            .zip(self.lower.iter().zip(self.upper.iter()))
            .enumerate()
        {
            if !x.is_finite() || x < a || x > b {
                return Err(invalid_argument(format!(
                    "point coordinate {x} on axis {axis} is outside the domain [{a}, {b}] or not finite"
                )));
            }
        }
        let mut lower = self.lower.clone();
        let mut upper = self.upper.clone();
        let mut prefix: Vec<Vec<usize>> = vec![Vec::new(); ndims];
        loop {
            if let Some(&index) = self.by_prefix.get(&prefix) {
                let patch = &self.patches[index];
                let mut local = Vec::with_capacity(ndims);
                let mut node = Vec::with_capacity(ndims);
                for axis in 0..ndims {
                    let nodes = axis_nodes(lower[axis], upper[axis], self.basis.grid());
                    // A coordinate that is bitwise a node keeps the exact node
                    // value, so the barycentric snap returns the stored value
                    // without relying on a cancellation-sensitive division.
                    let found = nodes.iter().position(|&candidate| candidate == point[axis]);
                    local.push(match found {
                        Some(node) => self.basis.grid()[node],
                        None => (point[axis] - lower[axis]) / (upper[axis] - lower[axis]),
                    });
                    node.push(found);
                }
                // Every coordinate is a node of this patch: return the stored
                // value, which also keeps a negative zero's sign.
                if let Some(nodes) = node.into_iter().collect::<Option<Vec<_>>>() {
                    let flat: usize = nodes
                        .iter()
                        .zip(&self.node_stride)
                        .map(|(&node, &stride)| node * stride)
                        .sum();
                    return Ok(patch.values[flat]);
                }
                let value = interpolate_at(&patch.values, &self.node_indices, &self.basis, &local)?;
                return Ok(value);
            }
            // Same midpoint arithmetic as `NInterval::split`, so the descent
            // reaches the same boxes the driver built.
            for axis in 0..ndims {
                let midpoint = lower[axis] + (upper[axis] - lower[axis]) / 2.0;
                let bit = usize::from(point[axis] >= midpoint);
                prefix[axis].push(bit);
                if bit == 0 {
                    upper[axis] = midpoint;
                } else {
                    lower[axis] = midpoint;
                }
            }
        }
    }
}

/// Accuracy and resource limits for [`interpolate_chebyshev_patches`].
///
/// When in doubt, use the default and choose `tolerance` for the absolute
/// accuracy of the function. `polynomial_degree` sets the local order and is an
/// argument of the constructor rather than an option.
///
/// # Examples
/// ```
/// use tensor4all_interpolativeqtt::{ChebyshevPatchOptions, ExtraValidation};
///
/// let options = ChebyshevPatchOptions {
///     tolerance: 1e-10,
///     extra_validation: Some(ExtraValidation::Fixed { points_per_axis: 2 }),
///     ..Default::default()
/// };
/// assert_eq!(options.tolerance, 1e-10);
/// assert_eq!(options.max_depth, 30);
/// assert_eq!(options.max_patches, 4096);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChebyshevPatchOptions {
    /// Absolute per-value tolerance, finite and nonnegative; default `1e-8`.
    /// A box is accepted when the maximum absolute interpolation residual at its
    /// sampled validation points is at most this value. The residual is sampled,
    /// not a certificate for arbitrary unsampled functions.
    pub tolerance: f64,
    /// Maximum split depth; default `30`. A box that still fails at this depth
    /// returns [`InterpolativeQttError::ChebyshevNotConverged`]. Raising the
    /// degree is usually cheaper than raising the depth.
    pub max_depth: usize,
    /// Maximum number of accepted plus pending boxes; default `4096`. Reaching
    /// it returns [`InterpolativeQttError::PatchLimit`]. A split multiplies the
    /// box count by `2^D`, so this bounds memory.
    pub max_patches: usize,
    /// Additional validation of boxes that passed the inherited grid check;
    /// default `None`, which validates on the inherited grid only. Its points
    /// are never inherited by children and count as evaluations.
    pub extra_validation: Option<ExtraValidation>,
}

impl Default for ChebyshevPatchOptions {
    fn default() -> Self {
        Self {
            tolerance: 1e-8,
            max_depth: 30,
            max_patches: 4096,
            extra_validation: None,
        }
    }
}

/// Additional, non-inherited validation of a box that passed the grid check.
///
/// The inherited validation grid consists of the node coordinates of a box's
/// children, which cluster at the box ends and the midpoint. A residual peak
/// strictly between those points is invisible to it. These modes sample points
/// the children would not inherit, and only for boxes that already passed, so a
/// failing box costs no extra evaluation.
///
/// # Examples
/// ```
/// use tensor4all_interpolativeqtt::ExtraValidation;
///
/// let fixed = ExtraValidation::Fixed { points_per_axis: 2 };
/// let greedy = ExtraValidation::Greedy { starts: 1, max_steps: 8 };
/// assert_ne!(fixed, greedy);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtraValidation {
    /// Evaluate the tensor product of `points_per_axis` evenly spaced interior
    /// points per axis: the centres of `points_per_axis` equal subintervals of
    /// the box, `points_per_axis^D` extra points per accepted box. With
    /// `points_per_axis = 2` these are the quarter points, which include the
    /// midpoints of the next refinement level. Costs `points_per_axis^D`
    /// evaluations per box that passed the grid check.
    Fixed {
        /// Interior points per axis, at least one.
        points_per_axis: usize,
    },
    /// Greedily maximize the residual near its largest sampled values, assuming
    /// a smooth function. Each of the `starts` largest-residual validation
    /// points starts an uphill walk on `|residual|` along the finite-difference
    /// gradient direction, one probe per sweep, halving the step when a probe
    /// does not improve. Costs at most `starts * max_steps` evaluations per box
    /// that passed the grid check. For a non-smooth function the walk only finds
    /// local maxima, and a probe may repeat a coordinate another search already
    /// sampled; the result is still an estimate, not a bound.
    Greedy {
        /// Number of starting points, the largest-residual ones, at least one.
        starts: usize,
        /// Evaluation budget per start, at least one.
        max_steps: usize,
    },
}

/// Construct real or complex Chebyshev patches on a dyadic box.
///
/// `context` is the caller's configured CPU execution context, for example
/// `CpuExecutionContext::from_backend(CpuBackend::with_threads(1)?)`. The whole
/// construction runs in one session of its engine: the evaluator receives that
/// engine's [`Exec`] token, and the validation stencil uses it. No backend is
/// created internally. Calling this from inside a session of the same context
/// fails with [`InterpolativeQttError::Execution`].
///
/// `lower` and `upper` are finite, strictly increasing bounds for each axis and
/// `polynomial_degree` (at least one) sets the local Chebyshev order. The
/// fallible evaluator receives column-major points of shape `[D, n_points]` and
/// the engine's token; it must return one value per point in the same order.
/// Errors retain their original source type.
///
/// The root samples its `(degree + 1)^D` Lobatto nodes plus `(2 * degree + 1)^D`
/// validation points. A box whose maximum absolute residual exceeds `tolerance`
/// splits into `2^D` children; each child's node values are exactly the parent's
/// validation points, so they are inherited rather than resampled and only the
/// child's own `(2 * degree + 1)^D` validation points are evaluated. Boxes are
/// refined breadth first and the returned patches are sorted lexicographically
/// by per-axis prefix. See [`ChebyshevPatchOptions`] for limits and
/// [`ExtraValidation`] for validation beyond the inherited grid.
///
/// # Errors
/// Returns [`InterpolativeQttError::InvalidArgument`] for invalid bounds,
/// degree, options or an overflowing shape; [`InterpolativeQttError::Evaluation`],
/// [`InterpolativeQttError::BatchLength`] or [`InterpolativeQttError::NonFinite`]
/// for invalid evaluator results; [`InterpolativeQttError::ChebyshevNotConverged`]
/// when a box still fails at `max_depth`; [`InterpolativeQttError::PatchLimit`]
/// when splitting would exceed `max_patches`; and
/// [`InterpolativeQttError::Execution`] when the engine refuses a call.
///
/// # Examples
/// ```
/// use std::convert::Infallible;
/// use tensor4all_core::ColMajorArrayRef;
/// use tensor4all_interpolativeqtt::{interpolate_chebyshev_patches, ChebyshevPatchOptions};
/// use tensor4all_tensorbackend::{CpuBackend, CpuExecutionContext};
///
/// let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(1)?);
/// let patches = interpolate_chebyshev_patches(
///     &context,
///     |points, _exec| Ok::<_, Infallible>(
///         points.data().chunks(2).map(|p| (p[0] - 0.3).powi(2) + (p[1] - 0.6).powi(2)).collect()),
///     &[0.0, 0.0], &[1.0, 1.0], 4, &ChebyshevPatchOptions::default())?;
/// // A polynomial below the local order is accepted as a single patch.
/// assert_eq!(patches.patches().len(), 1);
/// let coordinates = [0.3_f64, 0.6];
/// let points = ColMajorArrayRef::new(&coordinates, &[2, 1])?;
/// assert!(patches.evaluate(points)?[0].abs() < 1e-12);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn interpolate_chebyshev_patches<T, F, E>(
    context: &CpuExecutionContext,
    evaluate: F,
    lower: &[f64],
    upper: &[f64],
    polynomial_degree: usize,
    options: &ChebyshevPatchOptions,
) -> Result<ChebyshevPatches<T>>
where
    T: InterpolativeScalar,
    F: Fn(ColMajorArrayRef<'_, f64>, Exec<'_>) -> std::result::Result<Vec<T>, E>,
    E: Into<Box<dyn Error + Send + Sync>>,
{
    let plan = Plan::new::<T>(lower, upper, polynomial_degree, options)?;
    let totals = context
        .with_session(|engine| {
            run_waves(&plan, |wave| {
                wave.into_iter()
                    .map(|work| box_body(&plan, &evaluate, work, engine))
                    .collect()
            })
        })
        .map_err(|error| InterpolativeQttError::Execution {
            source: Box::new(error),
        })??;
    plan.assemble(totals)
}

/// Construct Chebyshev patches on caller-owned lanes, sharing each lane's engine with the evaluator.
///
/// This has the same acceptance rules as [`interpolate_chebyshev_patches`], with
/// `lanes` in place of the single execution context. The caller builds `P`
/// engines with `B` threads each and passes them through `lanes`. Both
/// evaluation and the validation stencil use the engine of the lane that pulls
/// the box, and the inherited node values travel inside the work item, so a
/// child's `(degree + 1)^D` values move to that lane instead of being resampled.
/// No global backend is entered.
///
/// The evaluator must be pure and return the same values independent of
/// scheduling and token width. Results are merged in input order and sorted by
/// prefix, independently of the lane count. Floating-point backend roundoff can
/// matter for a decision exactly on a threshold; choose tolerances with a margin
/// above roundoff.
///
/// # Errors
/// Returns the same errors as [`interpolate_chebyshev_patches`], plus
/// [`InterpolativeQttError::FanOut`] if the lane executor fails. In a wave with
/// several evaluator failures, the first box in canonical wave order wins.
///
/// # Examples
/// ```
/// use std::convert::Infallible;
/// use hataori_tenferro::TenferroLanes;
/// use tensor4all_interpolativeqtt::{
///     interpolate_chebyshev_patches_on_lanes, ChebyshevPatchOptions,
/// };
/// use tensor4all_tensorbackend::CpuBackend;
///
/// let engines = (0..2)
///     .map(|_| CpuBackend::with_threads(2).map(|backend| backend.into_engine()))
///     .collect::<Result<Vec<_>, _>>()?;
/// let mut lanes = TenferroLanes::new(engines)?;
/// let patches = interpolate_chebyshev_patches_on_lanes(
///     &mut lanes,
///     |points, _exec| Ok::<_, Infallible>(points.data().iter().map(|&x| 1.0 + x).collect()),
///     &[0.0], &[1.0], 2, &ChebyshevPatchOptions::default())?;
/// assert_eq!(patches.patches().len(), 1);
/// assert!((patches.patches()[0].values[1] - 1.5).abs() < 1e-12);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[cfg(feature = "parallel")]
pub fn interpolate_chebyshev_patches_on_lanes<T, F, E>(
    lanes: &mut hataori_tenferro::TenferroLanes,
    evaluate: F,
    lower: &[f64],
    upper: &[f64],
    polynomial_degree: usize,
    options: &ChebyshevPatchOptions,
) -> Result<ChebyshevPatches<T>>
where
    T: InterpolativeScalar + Send + Sync,
    F: Fn(ColMajorArrayRef<'_, f64>, Exec<'_>) -> std::result::Result<Vec<T>, E> + Sync,
    E: Into<Box<dyn Error + Send + Sync>>,
{
    let plan = Plan::new::<T>(lower, upper, polynomial_degree, options)?;
    let totals = run_waves(&plan, |wave| {
        // Keep typed errors as ordered output values, so no source is erased by
        // the scheduler and failure selection is independent of completion order.
        hataori::map_in_lanes(lanes, wave, |work, engine| {
            Ok::<_, std::convert::Infallible>(box_body(&plan, &evaluate, work, engine))
        })
        .map_err(InterpolativeQttError::FanOut)?
        .into_iter()
        .collect()
    })?;
    plan.assemble(totals)
}

struct Plan {
    domain: NInterval,
    basis: LagrangePolynomials,
    ndims: usize,
    /// `degree + 1`, the node count per axis.
    order: usize,
    /// `2 * degree + 1`, the validation count per axis.
    test_order: usize,
    node_dim: usize,
    test_dim: usize,
    /// Per-axis node index multi-indices, axis zero fastest.
    nodes: Vec<Vec<usize>>,
    /// Validation multi-index stride per axis.
    test_stride: Vec<usize>,
    /// Node multi-index stride per axis.
    node_stride: Vec<usize>,
    tolerance: f64,
    max_depth: usize,
    max_patches: usize,
    extra_validation: Option<ExtraValidation>,
}

impl Plan {
    fn new<T: InterpolativeScalar>(
        lower: &[f64],
        upper: &[f64],
        degree: usize,
        options: &ChebyshevPatchOptions,
    ) -> Result<Self> {
        validate(lower, upper, degree, options)?;
        let ndims = lower.len();
        let order = degree
            .checked_add(1)
            .ok_or_else(|| invalid_argument("polynomial_degree is too large"))?;
        let test_order = degree
            .checked_mul(2)
            .and_then(|doubled| doubled.checked_add(1))
            .ok_or_else(|| invalid_argument("polynomial_degree is too large"))?;
        let node_dim = checked_power(order, ndims)?;
        let test_dim = checked_power(test_order, ndims)?;
        checked_product(node_dim, std::mem::size_of::<T>())?;
        checked_product(test_dim, std::mem::size_of::<T>())?;
        checked_product(
            ndims,
            checked_product(test_dim, std::mem::size_of::<f64>())?,
        )?;
        let basis = get_chebyshev_grid(degree)?;
        let nodes = product_indices(order, ndims, node_dim);
        let mut test_stride = Vec::with_capacity(ndims);
        let mut node_stride = Vec::with_capacity(ndims);
        let mut stride = 1usize;
        for _ in 0..ndims {
            test_stride.push(stride);
            stride *= test_order;
        }
        let mut stride = 1usize;
        for _ in 0..ndims {
            node_stride.push(stride);
            stride *= order;
        }
        Ok(Self {
            domain: NInterval::new(lower, upper)?,
            basis,
            ndims,
            order,
            test_order,
            node_dim,
            test_dim,
            nodes,
            test_stride,
            node_stride,
            tolerance: options.tolerance,
            max_depth: options.max_depth,
            max_patches: options.max_patches,
            extra_validation: options.extra_validation,
        })
    }

    /// Per-axis validation index multi-index of one flat validation index.
    fn test_multi(&self, flat: usize) -> Vec<usize> {
        self.test_stride
            .iter()
            .map(|&stride| (flat / stride) % self.test_order)
            .collect()
    }

    /// Absolute per-axis node coordinates of `domain`, exactly at the corners.
    fn node_axes(&self, domain: &NInterval) -> Vec<Vec<f64>> {
        (0..self.ndims)
            .map(|axis| axis_nodes(domain.start()[axis], domain.stop()[axis], self.basis.grid()))
            .collect()
    }

    /// Absolute per-axis validation coordinates of `domain`: the node
    /// coordinates of both halves with the shared midpoint taken once.
    fn validation_axes(&self, domain: &NInterval) -> Vec<Vec<f64>> {
        (0..self.ndims)
            .map(|axis| {
                validation_axis(domain.start()[axis], domain.stop()[axis], self.basis.grid())
            })
            .collect()
    }

    /// One-axis stencil per axis, `[2n+1, n+1]`, of the local coordinates of
    /// this box's validation points.
    ///
    /// The coordinates come from the same physical-to-local map that `evaluate`
    /// uses, so the prediction is the interpolant at the point that was actually
    /// sampled. Assuming exact half-widths instead would misplace the midpoint
    /// whenever `a + (b - a) / 2` rounds, and report its own rounding as
    /// interpolation error.
    fn stencils<T: InterpolativeScalar>(
        &self,
        domain: &NInterval,
        axes: &[Vec<f64>],
    ) -> Result<Vec<Matrix<T>>> {
        axes.iter()
            .enumerate()
            .map(|(axis, coordinates)| {
                let (a, b) = (domain.start()[axis], domain.stop()[axis]);
                let grid: Vec<f64> = coordinates.iter().map(|&x| (x - a) / (b - a)).collect();
                stencil(&self.basis, &grid)
            })
            .collect()
    }

    fn assemble<T: InterpolativeScalar>(&self, totals: Totals<T>) -> Result<ChebyshevPatches<T>> {
        let mut by_prefix = BTreeMap::new();
        for (index, patch) in totals.patches.iter().enumerate() {
            by_prefix.insert(patch.prefix.clone(), index);
        }
        let num_evaluations = checked_product(totals.boxes, self.test_dim)
            .and_then(|all| all.checked_add(self.node_dim).ok_or_else(overflow))
            .and_then(|all| {
                all.checked_add(totals.extra_evaluations)
                    .ok_or_else(overflow)
            })?;
        Ok(ChebyshevPatches {
            lower: self.domain.start().to_vec(),
            upper: self.domain.stop().to_vec(),
            degree: self.order - 1,
            basis: self.basis.clone(),
            node_indices: self.nodes.clone(),
            node_stride: self.node_stride.clone(),
            patches: totals.patches,
            by_prefix,
            num_evaluations,
        })
    }
}

fn overflow() -> InterpolativeQttError {
    invalid_argument("evaluation count overflows")
}

/// Accepted patches, processed-box count and extra evaluations of one run.
struct Totals<T> {
    patches: Vec<ChebyshevPatch<T>>,
    boxes: usize,
    extra_evaluations: usize,
}

struct BoxWork<T> {
    domain: NInterval,
    prefix: Vec<Vec<usize>>,
    depth: usize,
    /// Inherited node values; `None` only for the root.
    node_values: Option<Vec<T>>,
}

struct BoxOutcome<T> {
    domain: NInterval,
    prefix: Vec<Vec<usize>>,
    depth: usize,
    node_values: Vec<T>,
    validation_values: Vec<T>,
    error_estimate: f64,
    extra_evaluations: usize,
}

/// One-axis stencil of Lagrange basis values at `grid`, shape `[grid, order]`.
fn stencil<T: InterpolativeScalar>(basis: &LagrangePolynomials, grid: &[f64]) -> Result<Matrix<T>> {
    let order = basis.len();
    let mut weights = Vec::with_capacity(checked_product(grid.len(), order)?);
    for alpha in 0..order {
        for &point in grid {
            weights.push(T::from_f64(basis.evaluate(alpha, point)?));
        }
    }
    Ok(Matrix::from_col_major_vec(grid.len(), order, weights))
}

/// Node coordinates of `[a, b]` along one axis, `a` and `b` exactly at the ends.
///
/// `a + (b - a) * 1.0` need not round to `b`, so both endpoints are returned as
/// given. The interior nodes use the same expression as
/// [`interval_points`](crate::interpolation::interval_points).
fn axis_nodes(a: f64, b: f64, grid: &[f64]) -> Vec<f64> {
    let last = grid.len() - 1;
    grid.iter()
        .enumerate()
        .map(|(j, &x)| {
            if j == 0 {
                a
            } else if j == last {
                b
            } else {
                a + (b - a) * x
            }
        })
        .collect()
}

/// Validation coordinates of `[a, b]`: both halves' node coordinates, the shared
/// midpoint once, `2n + 1` points in total.
fn validation_axis(a: f64, b: f64, grid: &[f64]) -> Vec<f64> {
    let midpoint = a + (b - a) / 2.0;
    let mut coordinates = axis_nodes(a, midpoint, grid);
    coordinates.extend_from_slice(&axis_nodes(midpoint, b, grid)[1..]);
    coordinates
}

/// Tensor product of per-axis coordinates, axis zero fastest.
fn tensor_points(axes: &[Vec<f64>]) -> Vec<f64> {
    let ndims = axes.len();
    let count: usize = axes.iter().map(Vec::len).product();
    let mut points = Vec::with_capacity(count * ndims);
    let mut index = vec![0_usize; ndims];
    for _ in 0..count {
        for (axis, &i) in axes.iter().zip(&index) {
            points.push(axis[i]);
        }
        for axis in 0..ndims {
            index[axis] += 1;
            if index[axis] < axes[axis].len() {
                break;
            }
            index[axis] = 0;
        }
    }
    points
}

fn sample_points<T, F, E>(
    plan: &Plan,
    evaluate: &F,
    engine: &mut CpuEngine,
    points: &[f64],
    count: usize,
) -> Result<Vec<T>>
where
    T: InterpolativeScalar,
    F: Fn(ColMajorArrayRef<'_, f64>, Exec<'_>) -> std::result::Result<Vec<T>, E>,
    E: Into<Box<dyn Error + Send + Sync>>,
{
    let shape = [plan.ndims, count];
    let points = ColMajorArrayRef::new(points, &shape)
        .map_err(|error| invalid_argument(error.to_string()))?;
    sample(evaluate, engine, points)
}

fn box_body<T, F, E>(
    plan: &Plan,
    evaluate: &F,
    work: BoxWork<T>,
    engine: &mut CpuEngine,
) -> Result<BoxOutcome<T>>
where
    T: InterpolativeScalar,
    F: Fn(ColMajorArrayRef<'_, f64>, Exec<'_>) -> std::result::Result<Vec<T>, E>,
    E: Into<Box<dyn Error + Send + Sync>>,
{
    let node_values = match work.node_values {
        Some(values) => values,
        None => {
            let points = tensor_points(&plan.node_axes(&work.domain));
            sample_points(plan, evaluate, engine, &points, plan.node_dim)?
        }
    };
    let axes = plan.validation_axes(&work.domain);
    let points = tensor_points(&axes);
    let validation_values = sample_points(plan, evaluate, engine, &points, plan.test_dim)?;
    let stencils = plan.stencils(&work.domain, &axes)?;
    let predictions = apply_stencil(
        &stencils,
        plan.order,
        plan.ndims,
        &node_values,
        &mut Session::new(engine),
    )?;
    let mut error_estimate = 0.0_f64;
    for (&prediction, &target) in predictions.iter().zip(&validation_values) {
        let residual = (prediction - target).abs_val();
        if !residual.is_finite() {
            return Err(InterpolativeQttError::NonFinite);
        }
        error_estimate = error_estimate.max(residual);
    }
    let mut extra_evaluations = 0_usize;
    // A box that already fails splits without spending extra evaluations.
    if error_estimate <= plan.tolerance
        && let Some(extra) = plan.extra_validation
    {
        let known = Known {
            domain: &work.domain,
            node_values: &node_values,
            axes: &axes,
            validation_values: &validation_values,
            predictions: &predictions,
        };
        let (residual, evaluations) = match extra {
            ExtraValidation::Fixed { points_per_axis } => {
                extra_fixed(plan, evaluate, engine, &known, points_per_axis)?
            }
            ExtraValidation::Greedy { starts, max_steps } => {
                extra_greedy(plan, evaluate, engine, &known, starts, max_steps)?
            }
        };
        extra_evaluations = evaluations;
        error_estimate = error_estimate.max(residual);
    }
    Ok(BoxOutcome {
        domain: work.domain,
        prefix: work.prefix,
        depth: work.depth,
        node_values,
        validation_values,
        error_estimate,
        extra_evaluations,
    })
}

/// What the extra validation of one box already knows: its sampled grid.
struct Known<'a, T> {
    domain: &'a NInterval,
    node_values: &'a [T],
    axes: &'a [Vec<f64>],
    validation_values: &'a [T],
    predictions: &'a [T],
}

fn run_waves<T: InterpolativeScalar>(
    plan: &Plan,
    mut run: impl FnMut(Vec<BoxWork<T>>) -> Result<Vec<BoxOutcome<T>>>,
) -> Result<Totals<T>> {
    let mut wave = vec![BoxWork {
        domain: plan.domain.clone(),
        prefix: vec![Vec::new(); plan.ndims],
        depth: 0,
        node_values: None,
    }];
    let mut accepted = Vec::new();
    let mut boxes = 0_usize;
    let mut extra_evaluations = 0_usize;
    let children = checked_power(2, plan.ndims)?;
    while !wave.is_empty() {
        boxes = boxes
            .checked_add(wave.len())
            .ok_or_else(|| invalid_argument("box count overflows"))?;
        let results = run(wave)?;
        let mut next = Vec::new();
        let mut leaves = accepted
            .len()
            .checked_add(results.len())
            .ok_or_else(|| invalid_argument("patch count overflows"))?;
        for outcome in results {
            extra_evaluations = extra_evaluations
                .checked_add(outcome.extra_evaluations)
                .ok_or_else(|| invalid_argument("evaluation count overflows"))?;
            if outcome.error_estimate <= plan.tolerance {
                accepted.push(ChebyshevPatch {
                    prefix: outcome.prefix,
                    lower: outcome.domain.start().to_vec(),
                    upper: outcome.domain.stop().to_vec(),
                    values: outcome.node_values,
                    error_estimate: outcome.error_estimate,
                });
                continue;
            }
            if outcome.depth >= plan.max_depth {
                return Err(InterpolativeQttError::ChebyshevNotConverged {
                    prefix: outcome.prefix,
                    error_estimate: outcome.error_estimate,
                });
            }
            leaves = leaves
                .checked_add(children - 1)
                .ok_or_else(|| invalid_argument("patch count overflows"))?;
            if leaves > plan.max_patches {
                return Err(InterpolativeQttError::PatchLimit {
                    limit: plan.max_patches,
                });
            }
            for (mask, domain) in outcome.domain.split()?.into_iter().enumerate() {
                let mut prefix = outcome.prefix.clone();
                for (axis, bits) in prefix.iter_mut().enumerate() {
                    bits.push((mask >> axis) & 1);
                }
                next.push(BoxWork {
                    domain,
                    prefix,
                    depth: outcome.depth + 1,
                    node_values: Some(gather_child(plan, &outcome.validation_values, mask)),
                });
            }
        }
        next.sort_by(|a, b| a.prefix.cmp(&b.prefix));
        wave = next;
    }
    accepted.sort_by(|a, b| a.prefix.cmp(&b.prefix));
    Ok(Totals {
        patches: accepted,
        boxes,
        extra_evaluations,
    })
}

/// Gather the node values of child `mask` from its parent's validation values.
///
/// Child node `j` on axis `d` is validation index `j + n * bit`, where `bit` is
/// that axis's bit of `mask`: the child's nodes are a strided subset of the
/// parent's validation grid, which is why the values are inherited bit for bit.
fn gather_child<T: InterpolativeScalar>(plan: &Plan, validation: &[T], mask: usize) -> Vec<T> {
    let degree = plan.order - 1;
    (0..plan.node_dim)
        .map(|flat| {
            let index = plan.nodes[flat]
                .iter()
                .enumerate()
                .map(|(axis, &node)| {
                    (node + degree * ((mask >> axis) & 1)) * plan.test_stride[axis]
                })
                .sum::<usize>();
            validation[index]
        })
        .collect()
}

/// Tensor-product barycentric Lagrange evaluation at one normalized point.
fn interpolate_at<T: InterpolativeScalar>(
    values: &[T],
    nodes: &[Vec<usize>],
    basis: &LagrangePolynomials,
    local: &[f64],
) -> Result<T> {
    let order = basis.len();
    let mut axis_weights = Vec::with_capacity(local.len() * order);
    for &x in local {
        for alpha in 0..order {
            axis_weights.push(basis.evaluate(alpha, x)?);
        }
    }
    let mut value = T::zero();
    for (node, &stored) in nodes.iter().zip(values) {
        let mut weight = 1.0;
        for (axis, &alpha) in node.iter().enumerate() {
            weight *= axis_weights[axis * order + alpha];
        }
        value = value + stored * T::from_f64(weight);
    }
    Ok(value)
}

/// Normalized local coordinates of a point of `domain`.
fn local_coordinates(point: &[f64], domain: &NInterval) -> Vec<f64> {
    point
        .iter()
        .zip(domain.start().iter().zip(domain.stop()))
        .map(|(&x, (&a, &b))| (x - a) / (b - a))
        .collect()
}

/// Residual maximum over `Fixed` extra points, and the evaluations spent.
fn extra_fixed<T, F, E>(
    plan: &Plan,
    evaluate: &F,
    engine: &mut CpuEngine,
    known: &Known<'_, T>,
    points_per_axis: usize,
) -> Result<(f64, usize)>
where
    T: InterpolativeScalar,
    F: Fn(ColMajorArrayRef<'_, f64>, Exec<'_>) -> std::result::Result<Vec<T>, E>,
    E: Into<Box<dyn Error + Send + Sync>>,
{
    let axes: Vec<Vec<f64>> = (0..plan.ndims)
        .map(|axis| {
            let (a, b) = (known.domain.start()[axis], known.domain.stop()[axis]);
            (0..points_per_axis)
                .map(|k| a + (b - a) * ((2 * k + 1) as f64 / (2 * points_per_axis) as f64))
                .collect()
        })
        .collect();
    let count = checked_power(points_per_axis, plan.ndims)?;
    let points = tensor_points(&axes);
    residual_at(plan, known, &points, count, evaluate, engine)
}

/// Residual maximum over the greedy ascent probes, and the evaluations spent.
fn extra_greedy<T, F, E>(
    plan: &Plan,
    evaluate: &F,
    engine: &mut CpuEngine,
    known: &Known<'_, T>,
    starts: usize,
    max_steps: usize,
) -> Result<(f64, usize)>
where
    T: InterpolativeScalar,
    F: Fn(ColMajorArrayRef<'_, f64>, Exec<'_>) -> std::result::Result<Vec<T>, E>,
    E: Into<Box<dyn Error + Send + Sync>>,
{
    let mut scored: Vec<(usize, f64)> = (0..plan.test_dim)
        .map(|flat| {
            let value = (known.predictions[flat] - known.validation_values[flat]).abs_val();
            if !value.is_finite() {
                return Err(InterpolativeQttError::NonFinite);
            }
            Ok((flat, value))
        })
        .collect::<Result<_>>()?;
    // Largest residuals first, ties by grid index, so the walk is deterministic.
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let resolution = |domain: &NInterval| {
        (0..plan.ndims)
            .map(|axis| (domain.stop()[axis] - domain.start()[axis]) / plan.test_order as f64)
            .fold(f64::INFINITY, f64::min)
    };
    let mut walks = Vec::with_capacity(starts.min(scored.len()));
    for &(flat, best) in scored.iter().take(starts) {
        let multi = plan.test_multi(flat);
        let point: Vec<f64> = (0..plan.ndims)
            .map(|axis| known.axes[axis][multi[axis]])
            .collect();
        let step = resolution(known.domain);
        let Some(direction) = ascent_direction(plan, known, &multi)? else {
            continue;
        };
        walks.push(Walk {
            point,
            direction,
            step,
            resolution: step * 1e-3,
            best,
        });
    }
    let mut worst = 0.0_f64;
    let mut spent = 0;
    for _ in 0..max_steps {
        let mut probes: Vec<f64> = Vec::with_capacity(walks.len() * plan.ndims);
        let mut owners = Vec::with_capacity(walks.len());
        for (index, walk) in walks.iter_mut().enumerate() {
            if let Some(point) = walk.probe(known.domain) {
                probes.extend_from_slice(&point);
                owners.push(index);
            } else {
                walk.step *= 0.5;
            }
        }
        if owners.is_empty() {
            break;
        }
        spent += owners.len();
        let values = sample_points(plan, evaluate, engine, &probes, owners.len())?;
        for (probe, &value) in values.iter().enumerate() {
            let point = &probes[probe * plan.ndims..(probe + 1) * plan.ndims];
            let local = local_coordinates(point, known.domain);
            let goal = interpolate_at(known.node_values, &plan.nodes, &plan.basis, &local)?;
            let found = (goal - value).abs_val();
            if !found.is_finite() {
                return Err(InterpolativeQttError::NonFinite);
            }
            let walk = &mut walks[owners[probe]];
            if found > walk.best {
                walk.accept(point.to_vec(), found);
            } else {
                walk.step *= 0.5;
            }
        }
        for walk in &walks {
            worst = worst.max(walk.best);
        }
        walks.retain(|walk| walk.step > walk.resolution);
        if walks.is_empty() {
            break;
        }
    }
    Ok((worst, spent))
}

/// Unit ascent direction of the residual magnitude at a validation point.
fn ascent_direction<T: InterpolativeScalar>(
    plan: &Plan,
    known: &Known<'_, T>,
    multi: &[usize],
) -> Result<Option<Vec<f64>>> {
    let last = plan.test_order - 1;
    let magnitude = |multi: &mut Vec<usize>, axis: usize, index: usize| -> Result<f64> {
        multi[axis] = index;
        let flat = flat_index(plan, multi);
        let value = (known.predictions[flat] - known.validation_values[flat]).abs_val();
        if !value.is_finite() {
            return Err(InterpolativeQttError::NonFinite);
        }
        Ok(value)
    };
    let mut gradient = Vec::with_capacity(plan.ndims);
    let mut neighbour = multi.to_vec();
    for axis in 0..plan.ndims {
        let index = multi[axis];
        // One-sided at the box faces, central otherwise.
        let (low, high) = if index == 0 {
            (0, 1)
        } else if index == last {
            (last - 1, last)
        } else {
            (index - 1, index + 1)
        };
        let low_value = magnitude(&mut neighbour, axis, low)?;
        let high_value = magnitude(&mut neighbour, axis, high)?;
        neighbour[axis] = index;
        let spacing = known.axes[axis][high] - known.axes[axis][low];
        gradient.push((high_value - low_value) / spacing);
    }
    let norm = gradient.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm == 0.0 || !norm.is_finite() {
        return Ok(None);
    }
    Ok(Some(gradient.iter().map(|x| x / norm).collect()))
}

fn flat_index(plan: &Plan, multi: &[usize]) -> usize {
    multi
        .iter()
        .zip(&plan.test_stride)
        .map(|(&index, &stride)| index * stride)
        .sum()
}

/// One uphill walk in the greedy residual search.
struct Walk {
    point: Vec<f64>,
    direction: Vec<f64>,
    step: f64,
    /// Step below which the walk has converged to the validation resolution.
    resolution: f64,
    best: f64,
}

impl Walk {
    /// Next probe, clipped to the box; `None` when the direction leaves the box.
    fn probe(&self, domain: &NInterval) -> Option<Vec<f64>> {
        let mut reach = self.step;
        for axis in 0..self.point.len() {
            let (direction, x) = (self.direction[axis], self.point[axis]);
            if direction == 0.0 {
                continue;
            }
            let limit = if direction > 0.0 {
                (domain.stop()[axis] - x) / direction
            } else {
                (domain.start()[axis] - x) / direction
            };
            reach = reach.min(limit);
        }
        if reach <= 0.0 || reach.is_nan() {
            return None;
        }
        Some(
            self.point
                .iter()
                .zip(&self.direction)
                .map(|(&x, &direction)| x + reach * direction)
                .collect(),
        )
    }

    /// The accepted point lies on the current ray, so the direction is unchanged.
    fn accept(&mut self, point: Vec<f64>, found: f64) {
        self.point = point;
        self.best = found;
    }
}

/// Residual maximum over `points`, and the evaluations spent.
fn residual_at<T, F, E>(
    plan: &Plan,
    known: &Known<'_, T>,
    points: &[f64],
    count: usize,
    evaluate: &F,
    engine: &mut CpuEngine,
) -> Result<(f64, usize)>
where
    T: InterpolativeScalar,
    F: Fn(ColMajorArrayRef<'_, f64>, Exec<'_>) -> std::result::Result<Vec<T>, E>,
    E: Into<Box<dyn Error + Send + Sync>>,
{
    let values = sample_points(plan, evaluate, engine, points, count)?;
    let mut worst = 0.0_f64;
    for (&value, point) in values.iter().zip(points.chunks_exact(plan.ndims)) {
        let local = local_coordinates(point, known.domain);
        let goal = interpolate_at(known.node_values, &plan.nodes, &plan.basis, &local)?;
        let found = (goal - value).abs_val();
        if !found.is_finite() {
            return Err(InterpolativeQttError::NonFinite);
        }
        worst = worst.max(found);
    }
    Ok((worst, count))
}

fn validate(
    lower: &[f64],
    upper: &[f64],
    degree: usize,
    options: &ChebyshevPatchOptions,
) -> Result<()> {
    if lower.is_empty() || lower.len() != upper.len() || lower.len() >= usize::BITS as usize {
        return Err(invalid_argument(
            "bounds must have equal nonzero dimension below usize::BITS",
        ));
    }
    if lower
        .iter()
        .zip(upper)
        .any(|(&a, &b)| !a.is_finite() || !b.is_finite() || a >= b || !(b - a).is_finite())
    {
        return Err(invalid_argument(
            "bounds must be finite, strictly increasing and have finite lengths",
        ));
    }
    if degree == 0 {
        return Err(invalid_argument("polynomial_degree must be at least one"));
    }
    if !options.tolerance.is_finite() || options.tolerance < 0.0 {
        return Err(invalid_argument("tolerance must be finite and nonnegative"));
    }
    if options.max_patches == 0 {
        return Err(invalid_argument("max_patches must be positive"));
    }
    match options.extra_validation {
        Some(ExtraValidation::Fixed { points_per_axis }) => {
            if points_per_axis == 0 {
                return Err(invalid_argument("points_per_axis must be positive"));
            }
            checked_power(points_per_axis, lower.len())?;
        }
        Some(ExtraValidation::Greedy { starts: 0, .. })
        | Some(ExtraValidation::Greedy { max_steps: 0, .. }) => {
            return Err(invalid_argument("starts and max_steps must be positive"));
        }
        Some(ExtraValidation::Greedy { .. }) | None => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests;
