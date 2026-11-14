//! Deterministic breadth-first construction of independently compressed boxes.

use std::error::Error;
use tensor4all_core::ColMajorArrayRef;
use tensor4all_simplett::tensor3_from_data;
use tensor4all_tensorbackend::{explicit::Session, CpuEngine, Exec, Matrix};

use crate::error::{invalid_argument, InterpolativeQttError, Result};
use crate::interpolation::{build_refined_cores, dense_test_points, SampleBatch};
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
///     truncation_bound: 0.0,
/// };
/// assert_eq!(patch.prefix, vec![vec![1], vec![0]]);
/// assert_eq!(patch.error_estimate + patch.truncation_bound, 0.0);
/// ```
#[derive(Debug, Clone)]
pub struct InterpolativePatch<T: InterpolativeScalar> {
    /// Fixed most-significant-first bits of each axis; an empty prefix is the root.
    pub prefix: Vec<Vec<usize>>,
    /// QTT on the remaining bits, with fused site dimension `2^D`.
    pub tt: SimpleTensorTrain<T>,
    /// Maximum absolute interpolation residual at the Chebyshev validation
    /// points; this per-value quantity alone decides acceptance against
    /// [`PatchedInterpolativeOptions::tolerance`]. It is an estimate, not a
    /// certificate for arbitrary unsampled functions.
    pub error_estimate: f64,
    /// Frobenius norm of the singular values discarded by compression, reported
    /// separately from acceptance. It bounds the change of the whole patch
    /// tensor and therefore of every value in it, so a pointwise error estimate
    /// of the returned TT is `error_estimate + truncation_bound`.
    pub truncation_bound: f64,
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
    /// Absolute per-value interpolation error target, finite and nonnegative;
    /// default `1e-8`. A box is accepted when its validation residual
    /// ([`InterpolativePatch::error_estimate`]) is at most this value.
    pub tolerance: f64,
    /// Relative SVD cutoff applied to each accepted patch; default `1e-12`.
    /// It does not take part in acceptance: the discarded weight is reported as
    /// [`InterpolativePatch::truncation_bound`]. Keep it small relative to
    /// `tolerance / max|f|`; use zero to retain every singular value.
    pub compression_tolerance: f64,
    /// Accept only bonds at most this positive size; `None` imposes no cap.
    pub max_bond_dim: Option<usize>,
    /// Maximum fixed prefix length. `None` permits `num_bits - 1` splits.
    /// The last remaining site is sampled exactly, so no zero-site TT is needed.
    pub max_depth: Option<usize>,
    /// Maximum number of accepted plus pending boxes; default 4096.
    pub max_patches: usize,
    /// Maximum entries in the dense Chebyshev validation stencil; default
    /// 1,048,576. Its size is `((degree + 1) * (2 * degree + 2))^D`,
    /// independent of the quantics grid size. Raise explicitly for high degree/dimension.
    pub max_stencil_elements: usize,
}

impl Default for PatchedInterpolativeOptions {
    fn default() -> Self {
        Self {
            tolerance: 1e-8,
            compression_tolerance: 1e-12,
            max_bond_dim: None,
            max_depth: None,
            max_patches: 4096,
            max_stencil_elements: 1 << 20,
        }
    }
}

/// Construct real or complex interpolative QTT patches on a dyadic box.
///
/// `engine` is the caller's configured CPU engine, for example
/// `CpuBackend::with_threads(1)?` followed by `engine_mut()`. Every box runs on
/// it in turn: the evaluator receives its [`Exec`] token, and the validation
/// GEMM and SVD compression use it. No backend is created internally.
/// `lower` and `upper` are finite, strictly increasing bounds for each axis;
/// `num_bits` (at least one) is the bit depth per axis and `polynomial_degree`
/// (at least one) sets the Chebyshev order. The fallible evaluator receives
/// column-major points of shape `[D, n_points]` and the engine's token.
/// It must return one value per point in the same order. Errors retain their
/// original source type. See [`PatchedInterpolativeOptions`] for limits.
///
/// Each box reuses the multiscale Chebyshev construction with its remaining
/// bits. Boxes whose interpolation residual or compressed rank exceeds the
/// limit split along every axis; the SVD truncation bound is reported per
/// patch but does not drive splitting. The last site is sampled exactly. The
/// returned vector is sorted lexicographically by per-axis prefix and
/// partitions the full grid. This sampling algorithm cannot detect arbitrary
/// features missed by its Chebyshev validation points; the error fields are
/// estimates.
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
/// use tensor4all_tensorbackend::CpuBackend;
/// use tensor4all_interpolativeqtt::{interpolate_patched_nd, PatchedInterpolativeOptions,
///     AbstractTensorTrain};
/// let mut backend = CpuBackend::with_threads(1)?;
/// let patches = interpolate_patched_nd(
///     backend.engine_mut(),
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
    engine: &mut CpuEngine,
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
    run_waves(&plan, |wave| {
        wave.into_iter()
            .map(|patch| patch_body(&plan, &evaluate, patch, engine))
            .collect()
    })
}

/// Construct patches on caller-owned lanes, sharing each lane's engine with the evaluator.
///
/// This has the same acceptance rules as [`interpolate_patched_nd`], with
/// `lanes` in place of the single engine. The caller builds `P` engines with
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
/// The `parallel` feature adds only the lane executor and keeps whichever CPU
/// provider the build selects. With native/faer, each lane's linear algebra
/// runs within its `B` threads. A BLAS provider (Accelerate, system BLAS) runs
/// its own threading outside that budget, and bit-identical results across
/// lane counts then rely on that library being deterministic.
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
    test_points: Vec<Vec<f64>>,
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
        let stencil_len = checked_product(basis_dim, test_dim)?;
        if stencil_len > options.max_stencil_elements {
            return Err(invalid_argument("Chebyshev stencil exceeds max_stencil_elements; reduce degree/dimension or raise the limit"));
        }
        checked_product(stencil_len, std::mem::size_of::<T>())?;
        let center_elements = checked_product(site_dim, checked_product(basis_dim, basis_dim)?)?;
        checked_product(center_elements, std::mem::size_of::<T>())?;
        checked_product(
            test_dim,
            checked_product(ndims, std::mem::size_of::<f64>())?,
        )?;
        let basis = get_chebyshev_grid(degree)?;
        let test_points = product_points(&dense_test_points(order)?, ndims, test_dim);
        let nodes = product_indices(order, ndims, basis_dim);
        let mut weights = Vec::with_capacity(stencil_len);
        // INVARIANT: the bounded stencil is independent of num_bits and built once
        // per run, not once per patch. GEMM performs all value contractions.
        for alpha in &nodes {
            for point in &test_points {
                let mut value = 1.0;
                for (&index, &coordinate) in alpha.iter().zip(point) {
                    value *= basis.evaluate(index, coordinate)?;
                }
                weights.push(T::from_f64(value));
            }
        }
        Ok(Plan {
            domain: NInterval::new(lower, upper)?,
            num_bits,
            max_depth: options.max_depth.unwrap_or(num_bits - 1),
            site_dim,
            basis,
            test_points,
            weights: Matrix::from_col_major_vec(test_dim, basis_dim, weights),
            options,
        })
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
            let children = split_box(&domain)?;
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
    let (mut tt, interpolation_error) = if remaining == 1 {
        let pieces = split_box(&patch.domain)?;
        let points: Vec<_> = pieces.iter().map(|piece| piece.start().to_vec()).collect();
        let values = sample(evaluate, engine, &points)?;
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
                let start = interval.start();
                let lengths = interval.lengths();
                let tests: Vec<Vec<f64>> = plan
                    .test_points
                    .iter()
                    .map(|p| {
                        p.iter()
                            .enumerate()
                            .map(|(d, &x)| start[d] + lengths[d] * x)
                            .collect()
                    })
                    .collect();
                let reference = sample(evaluate, engine, &tests)?;
                let values = Matrix::from_col_major_vec(values.len(), 1, values);
                let predictions = Session::new(engine).mat_mul(&plan.weights, &values)?;
                for (&prediction, &target) in
                    predictions.as_col_major_slice().iter().zip(&reference)
                {
                    let residual = (prediction - target).abs_val();
                    if !residual.is_finite() {
                        return Err(InterpolativeQttError::NonFinite);
                    }
                    error = error.max(residual);
                }
                Ok(values.into_col_major_vec())
            },
            &patch.domain,
            remaining,
            &plan.basis,
            |_, _| true,
        )?;
        (SimpleTensorTrain::new(cores)?, error)
    };
    let truncation_bound = tt.compress_svd_with_session(
        &mut Session::new(engine),
        plan.options.compression_tolerance,
        None,
    )?;
    if !interpolation_error.is_finite() || !truncation_bound.is_finite() {
        return Err(InterpolativeQttError::NonFinite);
    }
    Ok(PatchOutcome {
        patch: InterpolativePatch {
            prefix: patch.prefix,
            tt,
            error_estimate: interpolation_error,
            truncation_bound,
        },
        domain: patch.domain,
        depth: patch.depth,
    })
}

fn sample<T, F, E>(evaluate: &F, engine: &mut CpuEngine, points: &[Vec<f64>]) -> Result<Vec<T>>
where
    T: InterpolativeScalar,
    F: Fn(ColMajorArrayRef<'_, f64>, Exec<'_>) -> std::result::Result<Vec<T>, E>,
    E: Into<Box<dyn Error + Send + Sync>>,
{
    let ndims = points.first().map_or(0, Vec::len);
    let len = checked_product(ndims, points.len())?;
    let mut coordinates = Vec::with_capacity(len);
    for point in points {
        coordinates.extend_from_slice(point);
    }
    let shape = [ndims, points.len()];
    let batch =
        ColMajorArrayRef::new(&coordinates, &shape).map_err(|e| invalid_argument(e.to_string()))?;
    let values = engine
        .with_exec(|exec| evaluate(batch, exec.token()))
        .map_err(|e| InterpolativeQttError::Execution {
            source: Box::new(e),
        })?
        .map_err(|e| InterpolativeQttError::Evaluation { source: e.into() })?;
    if values.len() != points.len() {
        return Err(InterpolativeQttError::BatchLength {
            expected: points.len(),
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

fn product_points(grid: &[f64], ndims: usize, count: usize) -> Vec<Vec<f64>> {
    product_indices(grid.len(), ndims, count)
        .into_iter()
        .map(|point| point.into_iter().map(|i| grid[i]).collect())
        .collect()
}

fn split_box(domain: &NInterval) -> Result<Vec<NInterval>> {
    for (&lower, &upper) in domain.start().iter().zip(domain.stop()) {
        let midpoint = lower + (upper - lower) * 0.5;
        if midpoint <= lower || midpoint >= upper {
            return Err(invalid_argument(
                "dyadic midpoint is not representable; reduce num_bits or rescale bounds",
            ));
        }
    }
    domain.split()
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
        || options.max_stencil_elements == 0
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
