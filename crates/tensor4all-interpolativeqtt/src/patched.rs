//! Deterministic breadth-first construction of independently compressed boxes.

use std::error::Error;
use tensor4all_core::ColMajorArrayRef;
use tensor4all_simplett::{tensor3_from_data, TTCache};
use tensor4all_tensorbackend::{explicit::Session, CpuEngine, CpuExecutionContext, Exec, Matrix};

use crate::error::{invalid_argument, InterpolativeQttError, Result};
use crate::interpolation::{build_refined_cores, dense_test_points, interval_points, SampleBatch};
use crate::interval::NInterval;
use crate::{
    get_chebyshev_grid, AbstractTensorTrain, InterpolativeScalar, LagrangePolynomials,
    SimpleTensorTrain,
};

/// One accepted dyadic box and its local QTT, unlike a full-domain multiscale QTT.
///
/// Every axis fixes the same number of leading bits. The remaining sites fuse
/// one bit per axis, with axis zero the least significant bit of each site.
/// See [`interpolate_patched_nd`] for a runnable construction example.
///
/// # Examples
/// ```
/// use tensor4all_interpolativeqtt::{InterpolativePatch, SimpleTensorTrain};
/// let patch = InterpolativePatch {
///     prefix: vec![vec![1], vec![0]],
///     tt: SimpleTensorTrain::<f64>::constant(&[4, 4], 2.0),
///     error_estimate: 0.0,
/// };
/// assert_eq!(patch.prefix, vec![vec![1], vec![0]]);
/// assert_eq!(patch.error_estimate, 0.0);
/// ```
#[derive(Debug, Clone)]
pub struct InterpolativePatch<T: InterpolativeScalar> {
    /// Fixed most-significant-first bits of each axis; an empty prefix is the root.
    pub prefix: Vec<Vec<usize>>,
    /// QTT on the remaining bits, with fused site dimension `2^D`.
    pub tt: SimpleTensorTrain<T>,
    /// Per-value error estimate of the returned train, compared with
    /// [`PatchedInterpolativeOptions::tolerance`] for acceptance: the maximum
    /// absolute interpolation residual at the Chebyshev validation points plus
    /// the maximum change that SVD compression made at sampled grid points
    /// (every grid point on short trains; otherwise points spread over both
    /// halves of each axis, including the patch midpoint). It is an estimate,
    /// not a certificate for arbitrary unsampled functions.
    pub error_estimate: f64,
}

/// Accuracy and resource limits for patched construction.
///
/// Unlike [`crate::InterpolativeQttOptions`], the bond cap decides whether a
/// box must split, rather than forcing a rank truncation. When in doubt, use
/// the default and choose `tolerance` for the absolute accuracy of the function.
///
/// # Examples
/// ```
/// use tensor4all_interpolativeqtt::PatchedInterpolativeOptions;
/// let options = PatchedInterpolativeOptions { tolerance: 1e-8, ..Default::default() };
/// assert_eq!(options.tolerance, 1e-8);
/// assert_eq!(options.max_bond_dim, None);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PatchedInterpolativeOptions {
    /// Absolute per-value error target, finite and nonnegative; default `1e-8`.
    /// A box is accepted when [`InterpolativePatch::error_estimate`], which
    /// includes the compression change, is at most this value.
    pub tolerance: f64,
    /// Relative SVD cutoff for each candidate train; default `1e-12`. The
    /// change it causes is measured per value at sampled grid points and counts
    /// against `tolerance`, so a cutoff that visibly degrades the train leads
    /// to splitting rather than acceptance. On long trains the check is
    /// sampled, so keep the cutoff well below `tolerance / max|f|`; use zero
    /// to retain every singular value.
    pub compression_tolerance: f64,
    /// Accept only bonds at most this positive size; `None` imposes no cap.
    pub max_bond_dim: Option<usize>,
    /// Maximum fixed prefix length. `None` permits `num_bits - 1` splits.
    /// The last remaining site is sampled exactly, so no zero-site TT is needed.
    pub max_depth: Option<usize>,
    /// Maximum number of accepted plus pending boxes; default 4096.
    pub max_patches: usize,
}

impl Default for PatchedInterpolativeOptions {
    fn default() -> Self {
        Self {
            tolerance: 1e-8,
            compression_tolerance: 1e-12,
            max_bond_dim: None,
            max_depth: None,
            max_patches: 4096,
        }
    }
}

/// Construct real or complex interpolative QTT patches on a dyadic box.
///
/// `context` is the caller's configured CPU execution context, for example
/// `CpuExecutionContext::from_backend(CpuBackend::with_threads(1)?)`. The whole
/// construction runs in one session of its engine: the evaluator receives that
/// engine's [`Exec`] token, and the validation GEMM and SVD compression use it.
/// No backend is created internally. Calling this from inside a session of the
/// same context fails with [`InterpolativeQttError::Execution`]; an evaluator
/// that enters a session of `context` receives the context's re-entry error.
/// `lower` and `upper` are finite, strictly increasing bounds for each axis;
/// `num_bits` (at least one) is the bit depth per axis and `polynomial_degree`
/// (at least one) sets the Chebyshev order. The fallible evaluator receives
/// column-major points of shape `[D, n_points]` and the engine's token.
/// It must return one value per point in the same order. Errors retain their
/// original source type. See [`PatchedInterpolativeOptions`] for limits.
///
/// Each box reuses the multiscale Chebyshev construction with its remaining
/// bits and compresses the result. Boxes whose per-value error estimate
/// (interpolation residual plus compression change) or compressed rank exceeds
/// the limit split along every axis. The last site is sampled exactly. The
/// returned vector is sorted lexicographically by per-axis prefix and
/// partitions the full grid. This sampling algorithm cannot detect arbitrary
/// features missed by its validation points; the error field is an estimate.
///
/// # Errors
/// Returns [`InterpolativeQttError::InvalidArgument`] for invalid bounds,
/// configuration, unrepresentable dyadic boundaries or overflowing sizes;
/// [`InterpolativeQttError::Evaluation`], [`InterpolativeQttError::BatchLength`]
/// or [`InterpolativeQttError::NonFinite`] for invalid evaluator results;
/// [`InterpolativeQttError::NotConverged`] or [`InterpolativeQttError::PatchLimit`]
/// when refinement hits a limit; and execution, multiplication or tensor-train
/// errors when the backend fails.
///
/// # Examples
/// ```
/// use std::convert::Infallible;
/// use tensor4all_tensorbackend::{CpuBackend, CpuExecutionContext};
/// use tensor4all_interpolativeqtt::{interpolate_patched_nd, PatchedInterpolativeOptions,
///     AbstractTensorTrain};
/// let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(1)?);
/// let patches = interpolate_patched_nd(
///     &context,
///     |points, _exec| Ok::<_, Infallible>(points.data().chunks(2)
///         .map(|p| 1.0 + p[0] + 2.0 * p[1]).collect()),
///     &[0.0, 0.0], &[1.0, 1.0], 5, 3, &PatchedInterpolativeOptions::default(),
/// )?;
/// assert_eq!(patches.len(), 1);
/// assert_eq!(patches[0].prefix, vec![Vec::<usize>::new(); 2]);
/// assert!((patches[0].tt.evaluate(&[0; 5])? - 1.0).abs() < 1e-10);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn interpolate_patched_nd<T, F, E>(
    context: &CpuExecutionContext,
    evaluate: F,
    lower: &[f64],
    upper: &[f64],
    num_bits: usize,
    polynomial_degree: usize,
    options: &PatchedInterpolativeOptions,
) -> Result<Vec<InterpolativePatch<T>>>
where
    T: InterpolativeScalar,
    F: Fn(ColMajorArrayRef<'_, f64>, Exec<'_>) -> std::result::Result<Vec<T>, E>,
    E: Into<Box<dyn Error + Send + Sync>>,
{
    let plan = Plan::new(lower, upper, num_bits, polynomial_degree, options)?;
    context
        .with_session(|engine| {
            run_waves(&plan, |wave| {
                wave.into_iter()
                    .map(|patch| patch_body(&plan, &evaluate, patch, engine))
                    .collect()
            })
        })
        .map_err(|e| InterpolativeQttError::Execution {
            source: Box::new(e),
        })?
}

/// Construct patches on caller-owned lanes, sharing each lane's engine with the evaluator.
///
/// This has the same acceptance rules as [`interpolate_patched_nd`], with
/// `lanes` in place of the single execution context. The caller builds `P` engines with
/// `B` threads each and passes them through `lanes`. Both evaluation and compression use the engine of the lane that
/// pulls the box. No global backend is entered. A wave with fewer than `P`
/// boxes leaves lanes idle; the driver does not rebalance their thread budgets.
///
/// The evaluator may ignore its token, use `exec.for_ranges`/`exec.fork`, or
/// enter an external library through `exec.external`. It must be pure and
/// return the same values independent of scheduling and token width. Results
/// are merged in input order and sorted by prefix, independently of lane count.
/// Floating-point backend roundoff can matter for a decision exactly on a
/// threshold; choose tolerances with a margin above roundoff.
///
/// The `parallel` feature adds only the lane executor and builds with any CPU
/// provider, but lane execution is validated on native/faer only, where each
/// lane's linear algebra stays within its `B` threads and results are
/// bit-identical across lane layouts. With a BLAS provider (Accelerate, system
/// BLAS) each of the `P` lanes calls a library that threads on its own, which
/// can oversubscribe the cores and relies on the library's thread safety and
/// determinism; that combination is not tested.
///
/// # Errors
/// Returns the same errors as [`interpolate_patched_nd`], plus
/// [`InterpolativeQttError::FanOut`] if the lane executor fails. In a wave with
/// several evaluator failures, the first box in canonical wave order wins.
///
/// # Examples
/// ```
/// use std::convert::Infallible;
/// use hataori_tenferro::TenferroLanes;
/// use tensor4all_tensorbackend::CpuBackend;
/// use tensor4all_interpolativeqtt::{interpolate_patched_nd_on_lanes,
///     PatchedInterpolativeOptions, AbstractTensorTrain};
/// let engines = (0..2).map(|_| CpuBackend::with_threads(2).map(|b| b.into_engine()))
///     .collect::<Result<Vec<_>, _>>()?;
/// let mut lanes = TenferroLanes::new(engines)?;
/// let patches = interpolate_patched_nd_on_lanes(&mut lanes,
///     |p, _exec| Ok::<_, Infallible>(p.data().iter().map(|x| 1.0 + x).collect()),
///     &[0.0], &[1.0], 4, 3, &PatchedInterpolativeOptions::default())?;
/// assert_eq!(patches.len(), 1);
/// assert!((patches[0].tt.evaluate(&[0; 4])? - 1.0).abs() < 1e-10);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[cfg(feature = "parallel")]
pub fn interpolate_patched_nd_on_lanes<T, F, E>(
    lanes: &mut hataori_tenferro::TenferroLanes,
    evaluate: F,
    lower: &[f64],
    upper: &[f64],
    num_bits: usize,
    polynomial_degree: usize,
    options: &PatchedInterpolativeOptions,
) -> Result<Vec<InterpolativePatch<T>>>
where
    T: InterpolativeScalar + Send + Sync,
    F: Fn(ColMajorArrayRef<'_, f64>, Exec<'_>) -> std::result::Result<Vec<T>, E> + Sync,
    E: Into<Box<dyn Error + Send + Sync>>,
{
    let plan = Plan::new(lower, upper, num_bits, polynomial_degree, options)?;
    run_waves(&plan, |wave| {
        // Keep typed errors as ordered output values, so no source is erased by
        // the scheduler and failure selection is independent of completion order.
        hataori::map_in_lanes(lanes, wave, |patch, engine| {
            Ok::<_, std::convert::Infallible>(patch_body(&plan, &evaluate, patch, engine))
        })
        .map_err(InterpolativeQttError::FanOut)?
        .into_iter()
        .collect()
    })
}

struct Plan<'a, T: InterpolativeScalar> {
    domain: NInterval,
    num_bits: usize,
    max_depth: usize,
    site_dim: usize,
    basis: LagrangePolynomials,
    /// Per-axis validation coordinates in `[0, 1]`, increasing.
    test_points_1d: Vec<f64>,
    num_test_points: usize,
    /// Per-axis basis indices of each flattened Chebyshev node, axis zero fastest.
    nodes: Vec<Vec<usize>>,
    /// One axis of the tensor-product stencil, shape `[2 * order, order]`.
    weights: Matrix<T>,
    options: &'a PatchedInterpolativeOptions,
}

#[derive(Debug)]
struct BoxSpec {
    domain: NInterval,
    prefix: Vec<Vec<usize>>,
    depth: usize,
}

struct PatchOutcome<T: InterpolativeScalar> {
    patch: InterpolativePatch<T>,
    domain: NInterval,
    depth: usize,
}

impl<T: InterpolativeScalar> Plan<'_, T> {
    fn new<'a>(
        lower: &[f64],
        upper: &[f64],
        num_bits: usize,
        degree: usize,
        options: &'a PatchedInterpolativeOptions,
    ) -> Result<Plan<'a, T>> {
        validate(lower, upper, num_bits, degree, options)?;
        let ndims = lower.len();
        let site_dim = 1usize
            .checked_shl(ndims as u32)
            .ok_or_else(|| invalid_argument("too many dimensions for a fused site"))?;
        let order = degree
            .checked_add(1)
            .ok_or_else(|| invalid_argument("degree overflows"))?;
        let tests = order
            .checked_mul(2)
            .ok_or_else(|| invalid_argument("validation order overflows"))?;
        let basis_dim = checked_power(order, ndims)?;
        let test_dim = checked_power(tests, ndims)?;
        let stencil_len = checked_product(order, tests)?;
        checked_product(test_dim, std::mem::size_of::<T>())?;
        checked_product(stencil_len, std::mem::size_of::<T>())?;
        let center_elements = checked_product(site_dim, checked_product(basis_dim, basis_dim)?)?;
        checked_product(center_elements, std::mem::size_of::<T>())?;
        checked_product(
            test_dim,
            checked_product(ndims, std::mem::size_of::<f64>())?,
        )?;
        let basis = get_chebyshev_grid(degree)?;
        let test_points_1d = dense_test_points(order)?;
        let nodes = product_indices(order, ndims, basis_dim);
        let mut weights = Vec::with_capacity(stencil_len);
        for alpha in 0..order {
            for &point in &test_points_1d {
                weights.push(T::from_f64(basis.evaluate(alpha, point)?));
            }
        }
        Ok(Plan {
            domain: NInterval::new(lower, upper)?,
            num_bits,
            max_depth: options.max_depth.unwrap_or(num_bits - 1),
            site_dim,
            basis,
            test_points_1d,
            num_test_points: test_dim,
            nodes,
            weights: Matrix::from_col_major_vec(tests, order, weights),
            options,
        })
    }

    /// Apply the Kronecker stencil without forming its dense product.
    fn apply_stencil(&self, values: &[T], session: &mut Session<'_>) -> Result<Vec<T>> {
        let order = self.basis.len();
        let mut values = values.to_vec();
        for _ in 0..self.domain.ndims() {
            // The current axis is first (contiguous). Rotate its new test axis
            // to the end, exposing the next node axis for the next GEMM.
            // After D rotations the original axis-zero-fastest order is restored.
            let columns = values.len() / order;
            let input = Matrix::from_col_major_vec(order, columns, values);
            let output = session.mat_mul(&self.weights, &input)?;
            values = tensor4all_tensorbackend::transpose(&output).into_col_major_vec();
        }
        Ok(values)
    }
}

fn run_waves<T: InterpolativeScalar>(
    plan: &Plan<'_, T>,
    mut run: impl FnMut(Vec<BoxSpec>) -> Result<Vec<PatchOutcome<T>>>,
) -> Result<Vec<InterpolativePatch<T>>> {
    let mut wave = vec![BoxSpec {
        domain: plan.domain.clone(),
        prefix: vec![vec![]; plan.domain.ndims()],
        depth: 0,
    }];
    let mut accepted = Vec::new();
    while !wave.is_empty() {
        // Save only metadata for refinement; tensors remain owned by the outcome.
        let results = run(wave)?;
        let mut next = Vec::new();
        let mut leaves = accepted
            .len()
            .checked_add(results.len())
            .ok_or_else(|| invalid_argument("patch count overflows"))?;
        for PatchOutcome {
            patch,
            domain,
            depth,
        } in results
        {
            let rank = patch.tt.rank();
            if patch.error_estimate <= plan.options.tolerance
                && plan.options.max_bond_dim.is_none_or(|cap| rank <= cap)
            {
                accepted.push(patch);
                continue;
            }
            if depth >= plan.max_depth {
                return Err(InterpolativeQttError::NotConverged {
                    prefix: patch.prefix,
                    error_estimate: patch.error_estimate,
                    rank,
                });
            }
            leaves = leaves
                .checked_add(plan.site_dim - 1)
                .ok_or_else(|| invalid_argument("patch count overflows"))?;
            if leaves > plan.options.max_patches {
                return Err(InterpolativeQttError::PatchLimit {
                    limit: plan.options.max_patches,
                });
            }
            let children = domain.split()?;
            for (mask, domain) in children.into_iter().enumerate() {
                let mut prefix = patch.prefix.clone();
                for (axis, bits) in prefix.iter_mut().enumerate() {
                    bits.push((mask >> axis) & 1);
                }
                next.push(BoxSpec {
                    domain,
                    prefix,
                    depth: depth + 1,
                });
            }
        }
        next.sort_by(|a, b| a.prefix.cmp(&b.prefix));
        wave = next;
    }
    accepted.sort_by(|a, b| a.prefix.cmp(&b.prefix));
    Ok(accepted)
}

fn patch_body<T, F, E>(
    plan: &Plan<'_, T>,
    evaluate: &F,
    patch: BoxSpec,
    engine: &mut CpuEngine,
) -> Result<PatchOutcome<T>>
where
    T: InterpolativeScalar,
    F: Fn(ColMajorArrayRef<'_, f64>, Exec<'_>) -> std::result::Result<Vec<T>, E>,
    E: Into<Box<dyn Error + Send + Sync>>,
{
    let remaining = plan.num_bits - patch.depth;
    // Chebyshev node values of each immediate child, in sampling order.
    let mut children: Vec<(NInterval, Vec<T>)> = Vec::new();
    let (mut tt, interpolation_error) = if remaining == 1 {
        let pieces = patch.domain.split()?;
        let points: Vec<_> = pieces
            .iter()
            .flat_map(|piece| piece.start().iter().copied())
            .collect();
        let shape = [patch.domain.ndims(), pieces.len()];
        let points =
            ColMajorArrayRef::new(&points, &shape).map_err(|e| invalid_argument(e.to_string()))?;
        let values = sample(evaluate, engine, points)?;
        (
            SimpleTensorTrain::new(vec![tensor3_from_data(values, 1, plan.site_dim, 1)?])?,
            0.0,
        )
    } else {
        let mut error = 0.0_f64;
        let cores = build_refined_cores(
            &mut |batch: SampleBatch<'_>| {
                let values = sample(evaluate, engine, batch.points())?;
                // Only a Chebyshev node set carries the polynomial the stencil
                // validates; exact grid samples have nothing to check.
                let SampleBatch::Nodes { interval, .. } = batch else {
                    return Ok(values);
                };
                children.push((interval.clone(), values.clone()));
                let tests = interval_points(interval, &plan.test_points_1d)?;
                let shape = [interval.ndims(), plan.num_test_points];
                let tests = ColMajorArrayRef::new(&tests, &shape)
                    .map_err(|e| invalid_argument(e.to_string()))?;
                let reference = sample(evaluate, engine, tests)?;
                let predictions = plan.apply_stencil(&values, &mut Session::new(engine))?;
                for (&prediction, &target) in predictions.iter().zip(&reference) {
                    let residual = (prediction - target).abs_val();
                    if !residual.is_finite() {
                        return Err(InterpolativeQttError::NonFinite);
                    }
                    error = error.max(residual);
                }
                Ok(values)
            },
            &patch.domain,
            remaining,
            &plan.basis,
            |_, _| true,
        )?;
        (SimpleTensorTrain::new(cores)?, error)
    };
    // The discarded Frobenius weight covers the whole patch tensor and grows
    // with its grid size, so it is not comparable with the per-value tolerance.
    // Measure the change per value on the train actually returned instead.
    tt.compress_svd_with_session(
        &mut Session::new(engine),
        plan.options.compression_tolerance,
        None,
    )?;
    let error_estimate = if remaining == 1 {
        // A one-site train has no bond to truncate.
        interpolation_error
    } else {
        interpolation_error + compression_change(plan, &patch.domain, &children, &tt)?
    };
    if !error_estimate.is_finite() {
        return Err(InterpolativeQttError::NonFinite);
    }
    Ok(PatchOutcome {
        patch: InterpolativePatch {
            prefix: patch.prefix,
            tt,
            error_estimate,
        },
        domain: patch.domain,
        depth: patch.depth,
    })
}

/// Per-axis grid indices of an `L`-site patch at which compression is checked.
///
/// When the axis has at most `2 * validation` points, every index is returned.
/// Otherwise each of the two child halves contributes the indices nearest to
/// the validation coordinates mapped onto that half, so both child boundaries
/// (including the patch midpoint, where the child interpolants meet) and the
/// child interiors are sampled. The result is sorted and unique.
fn compression_check_indices(test_points_1d: &[f64], sites: usize) -> Vec<u64> {
    // INVARIANT: 1 <= sites <= 52 (validated num_bits), so grid indices fit in
    // u64 and are exactly representable in f64.
    let size = 1u64 << sites;
    if size <= 2 * test_points_1d.len() as u64 {
        return (0..size).collect();
    }
    let half = size / 2;
    let last = (half - 1) as f64;
    let mut indices: Vec<u64> = [0, half]
        .iter()
        .flat_map(|&offset| {
            test_points_1d
                .iter()
                .map(move |&t| offset + (t * last).round() as u64)
        })
        .collect();
    indices.dedup();
    indices
}

/// Maximum `|compressed train - interpolant|` over the grid points from
/// [`compression_check_indices`] on every axis.
///
/// Every box of the patch driver is safe, so the uncompressed multiscale train
/// equals, at each grid point, the Chebyshev interpolant of the immediate child
/// that owns the point. That interpolant costs `(degree + 1)^D` products per
/// point, against `L r^2` for evaluating the uncompressed rank-`r` train.
fn compression_change<T: InterpolativeScalar>(
    plan: &Plan<'_, T>,
    domain: &NInterval,
    children: &[(NInterval, Vec<T>)],
    compressed: &SimpleTensorTrain<T>,
) -> Result<f64> {
    let sites = compressed.len();
    let ndims = domain.ndims();
    // Match sampled node sets to children by their exact intervals: the
    // assembly splits this domain with the same `NInterval::split`, and every
    // box of the driver is safe, so each child was sampled exactly once.
    let pieces = domain.split()?;
    let node_values: Vec<&[T]> = pieces
        .iter()
        .map(|piece| {
            children
                .iter()
                .find(|(interval, _)| interval == piece)
                .map(|(_, values)| values.as_slice())
                .unwrap_or_else(|| {
                    tensor4all_tensorbackend::invariant_violated(
                        "patched compression check",
                        "a child box was not sampled on its Chebyshev nodes",
                    )
                })
        })
        .collect();
    let half = 1u64 << (sites - 1);
    let axis = compression_check_indices(&plan.test_points_1d, sites);
    let count = checked_power(axis.len(), ndims)?;
    checked_product(count, checked_product(sites, std::mem::size_of::<usize>())?)?;
    checked_product(count, std::mem::size_of::<Vec<usize>>())?;
    checked_product(count, std::mem::size_of::<T>())?;
    // Every axis uses the same normalized child coordinates. Compute this
    // table once and share it across axes and all tensor-product points.
    let mut lagrange = Vec::with_capacity(checked_product(axis.len(), plan.basis.len())?);
    for &k in &axis {
        let local = (k % half) as f64 / half as f64;
        for alpha in 0..plan.basis.len() {
            lagrange.push(plan.basis.evaluate(alpha, local)?);
        }
    }
    let mut interpolants = Vec::with_capacity(count);
    let mut indices = Vec::with_capacity(count);
    for multi in product_indices(axis.len(), ndims, count) {
        let child = multi.iter().enumerate().fold(0, |mask, (d, &i)| {
            mask | (usize::from(axis[i] >= half) << d)
        });
        let mut interpolant = T::zero();
        for (node, &value) in plan.nodes.iter().zip(node_values[child]) {
            let weight: f64 = node
                .iter()
                .enumerate()
                .map(|(d, &alpha)| lagrange[multi[d] * plan.basis.len() + alpha])
                .product();
            interpolant = interpolant + value * T::from_f64(weight);
        }
        interpolants.push(interpolant);
        indices.push(
            (0..sites)
                .map(|level| {
                    let shift = sites - 1 - level;
                    multi
                        .iter()
                        .enumerate()
                        .map(|(d, &i)| (((axis[i] >> shift) & 1) as usize) << d)
                        .sum()
                })
                .collect(),
        );
    }
    // A full left prefix preserves evaluate()'s left-to-right accumulation
    // order bit for bit, while the cache reuses all common prefixes. This
    // check owns the cache; its at most count * sites prefixes live only here.
    let values = TTCache::new(compressed).evaluate_many(&indices, Some(sites))?;
    Ok(values
        .into_iter()
        .zip(interpolants)
        .map(|(value, interpolant)| (value - interpolant).abs_val())
        .fold(0.0_f64, f64::max))
}

fn sample<T, F, E>(
    evaluate: &F,
    engine: &mut CpuEngine,
    points: ColMajorArrayRef<'_, f64>,
) -> Result<Vec<T>>
where
    T: InterpolativeScalar,
    F: Fn(ColMajorArrayRef<'_, f64>, Exec<'_>) -> std::result::Result<Vec<T>, E>,
    E: Into<Box<dyn Error + Send + Sync>>,
{
    let values = engine
        .with_exec(|exec| evaluate(points, exec.token()))
        .map_err(|e| InterpolativeQttError::Execution {
            source: Box::new(e),
        })?
        .map_err(|e| InterpolativeQttError::Evaluation { source: e.into() })?;
    if values.len() != points.shape()[1] {
        return Err(InterpolativeQttError::BatchLength {
            expected: points.shape()[1],
            actual: values.len(),
        });
    }
    if values.iter().any(|x| !x.abs_val().is_finite()) {
        return Err(InterpolativeQttError::NonFinite);
    }
    Ok(values)
}

fn checked_product(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .filter(|&n| n <= isize::MAX as usize)
        .ok_or_else(|| invalid_argument("interpolation shape or byte count overflows"))
}

fn checked_power(base: usize, exponent: usize) -> Result<usize> {
    (0..exponent).try_fold(1, |n, _| checked_product(n, base))
}

fn product_indices(base: usize, ndims: usize, count: usize) -> Vec<Vec<usize>> {
    (0..count)
        .map(|mut flat| {
            (0..ndims)
                .map(|_| {
                    let i = flat % base;
                    flat /= base;
                    i
                })
                .collect()
        })
        .collect()
}

fn validate(
    lower: &[f64],
    upper: &[f64],
    num_bits: usize,
    degree: usize,
    options: &PatchedInterpolativeOptions,
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
    if num_bits == 0 || num_bits > 52 || degree == 0 {
        return Err(invalid_argument(
            "num_bits must be in 1..=52 and polynomial_degree at least one",
        ));
    }
    let grid_scale = 2.0_f64.powi(num_bits as i32);
    if lower.iter().zip(upper).any(|(&a, &b)| {
        let step = (b - a) / grid_scale;
        step == 0.0 || a + step <= a || b - step >= b
    }) {
        return Err(invalid_argument(
            "quantics grid spacing is not representable; reduce num_bits or rescale bounds",
        ));
    }
    if options.max_depth.is_some_and(|d| d >= num_bits)
        || options.max_bond_dim == Some(0)
        || options.max_patches == 0
    {
        return Err(invalid_argument(
            "max_depth must be below num_bits and resource limits must be positive",
        ));
    }
    if [options.tolerance, options.compression_tolerance]
        .iter()
        .any(|x| !x.is_finite() || *x < 0.0)
    {
        return Err(invalid_argument(
            "tolerances must be finite and nonnegative",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
