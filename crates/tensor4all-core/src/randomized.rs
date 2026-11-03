//! Fixed-rank Gaussian range finding and dense randomized SVD.
//!
//! Composes the backend's contractions, QR and SVD following Halko, Martinsson
//! and Tropp, SIAM Review 53 (2011), Algorithms 4.4 and 5.1. No upstream source
//! code was translated. The input is a materialized local tensor; these routines
//! do not materialize a tensor network.
//!
//! Only dense `f64` and `Complex64` inputs are supported. `f32` and `Complex32`
//! (c32) return [`FactorizeError::UnsupportedStorage`] before consuming RNG state.

use num_complex::Complex64;
use rand::Rng;
use std::time::{Duration, Instant};

use crate::defaults::idx_tensor::{checked_product, unfold_index_sets, RandomScalar};
use crate::defaults::svd::{
    svd_for_factorize, svd_for_factorize_in, svd_for_factorize_on_session, SvdFactorizeResult,
};
use crate::{
    DynIndex, ExecutionContext, FactorizeError, FactorizeOptions, FactorizeResult, IdxTensor,
    IndexLike, PairwiseContractionOptions, SvdOptions,
};
use tensor4all_tensorbackend::explicit::Session;

/// Sketch size and numerical stabilization for a Gaussian range finder.
///
/// Unlike SVD truncation policies, these options do not promise an error
/// tolerance. [`RsvdOptions`] adds the requested output rank for RSVD.
///
/// # Examples
/// ```
/// use tensor4all_core::RangeOptions;
/// let options = RangeOptions::default();
/// assert_eq!(options.oversampling, 8);
/// assert_eq!(options.power_iterations, 1);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RangeOptions {
    /// Extra Gaussian columns beyond the requested rank, capped at the matrix
    /// dimensions. Eight is a useful default; use zero for a rank-`k` projector.
    pub oversampling: usize,
    /// Stabilized power iterations, with QR on both sides of each iteration.
    /// One is the default; zero costs less, two can help slowly decaying spectra.
    pub power_iterations: usize,
}

impl Default for RangeOptions {
    fn default() -> Self {
        Self {
            oversampling: 8,
            power_iterations: 1,
        }
    }
}

/// Fixed output rank and sketch options for [`randomized_svd`].
///
/// Global truncation defaults are ignored. Threshold-based accuracy needs an
/// a posteriori estimator and is deliberately unsupported in this first version.
/// Only dense `f64`/`Complex64` inputs are supported; `f32`/`Complex32` are not.
///
/// # Examples
/// ```
/// use tensor4all_core::RsvdOptions;
/// let options = RsvdOptions::new(3);
/// assert_eq!(options.max_bond_dim, 3);
/// assert_eq!(options.range.oversampling, 8);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct RsvdOptions {
    /// Positive output rank cap. The actual rank is `min(max_bond_dim, m, n)`.
    pub max_bond_dim: usize,
    /// Oversampling and power iterations; when in doubt use the defaults.
    pub range: RangeOptions,
}

impl RsvdOptions {
    /// Request a positive rank cap with default sketch options.
    ///
    /// Zero is rejected by the decomposition before consuming randomness.
    ///
    /// # Examples
    /// ```
    /// use tensor4all_core::RsvdOptions;
    /// assert_eq!(RsvdOptions::new(2).max_bond_dim, 2);
    /// ```
    pub fn new(max_bond_dim: usize) -> Self {
        Self {
            max_bond_dim,
            range: RangeOptions::default(),
        }
    }
}

/// Time spent in one RSVD composite, including sketch construction and index work.
///
/// The GEMM and QR counters allow comparison with the total before considering
/// moving the composite upstream. They include operation dispatch overhead and
/// are observational diagnostics, not a performance guarantee.
///
/// # Examples
/// ```
/// use tensor4all_core::RsvdTimings;
/// let timings = RsvdTimings::default();
/// assert_eq!(timings.gemm + timings.qr + timings.small_svd, timings.total);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct RsvdTimings {
    /// Whole RSVD driver, including validation, sampling and factor assembly.
    pub total: Duration,
    /// Sketch products, power-iteration products, projection and lifting.
    pub gemm: Duration,
    /// Thin QR operations, including reorthogonalization in power iterations.
    pub qr: Duration,
    /// SVD of the projected matrix, including rank slicing and index assembly.
    pub small_svd: Duration,
}

/// Approximate SVD factors with directly contractible index identities.
///
/// Contract `u`, `s`, then `vh` to reconstruct the approximation. Unlike a
/// deterministic SVD, `singular_values` describes the projected matrix only.
///
/// # Examples
/// ```
/// use rand::SeedableRng;
/// use rand_chacha::ChaCha8Rng;
/// use tensor4all_core::{randomized_svd, DynIndex, IdxTensor, RsvdOptions, TensorContractionLike};
/// let i = DynIndex::new_dyn(2);
/// let j = DynIndex::new_dyn(2);
/// let a = IdxTensor::from_dense(vec![i.clone(), j], vec![3., 0., 0., 0.])?;
/// let mut rng = ChaCha8Rng::seed_from_u64(11);
/// let result = randomized_svd(&a, &[i], &RsvdOptions::new(1), &mut rng)?;
/// let reconstructed = result.u.contract_pair(&result.s)?.contract_pair(&result.vh)?;
/// assert!(a.sub(&reconstructed)?.maxabs()? < 1e-12);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct RsvdResult {
    /// Orthonormal left singular vectors with the original left indices.
    pub u: IdxTensor,
    /// Singular values as a diagonal tensor connecting the two factor bonds.
    pub s: IdxTensor,
    /// Conjugate-transposed right singular vectors and original right indices.
    pub vh: IdxTensor,
    /// Retained singular values of `Q† A`, in nonincreasing order.
    pub singular_values: Vec<f64>,
    /// Measured cost attribution for this call; instrumentation is included.
    pub timings: RsvdTimings,
}

fn computation(error: impl Into<anyhow::Error>) -> FactorizeError {
    FactorizeError::ComputationError(error.into())
}

/// Selected execution boundary, shared by both randomized algorithms.
pub(crate) enum Route<'a, 'session> {
    Eager(Option<&'a ExecutionContext>),
    Session(&'a mut Session<'session>),
}

impl Route<'_, '_> {
    fn validate(&self, t: &IdxTensor) -> Result<(), FactorizeError> {
        if t.is_diag()
            || !(t.is_f64() || t.is_c64())
            || t.axis_classes()
                .iter()
                .enumerate()
                .any(|(axis, &class)| axis != class)
        {
            return Err(FactorizeError::UnsupportedStorage(
                "RSVD requires dense f64 or Complex64 tensors",
            ));
        }
        match self {
            Self::Eager(Some(context)) => t.validate_context(context).map_err(computation)?,
            Self::Session(_) if t.tracks_grad() || t.is_cuda_resident() => {
                return Err(FactorizeError::UnsupportedStorage(
                    "explicit-session RSVD requires untracked host tensors",
                ));
            }
            Self::Eager(None) if t.is_cuda_resident() => {
                return Err(FactorizeError::UnsupportedStorage(
                    "resident RSVD requires a caller-owned execution context",
                ));
            }
            _ => {}
        }
        Ok(())
    }

    fn gaussian<T: RandomScalar + tensor4all_tensorbackend::StorageScalar, R: Rng + ?Sized>(
        &self,
        template: &IdxTensor,
        indices: Vec<DynIndex>,
        rng: &mut R,
    ) -> Result<IdxTensor, FactorizeError> {
        let dims: Vec<_> = indices.iter().map(IndexLike::dim).collect();
        let len = checked_product(&dims).map_err(computation)?;
        len.checked_mul(std::mem::size_of::<T>())
            .filter(|&bytes| bytes <= isize::MAX as usize)
            .ok_or(FactorizeError::InvalidOptions(
                "Gaussian sketch byte length overflows",
            ))?;
        let data = (0..len).map(|_| T::random_value(rng)).collect();
        match self {
            Self::Eager(Some(context)) => IdxTensor::from_dense_in(context, indices, data),
            Self::Eager(None) => template.constant_dense_like(indices, data),
            Self::Session(_) => IdxTensor::from_host_dense(indices, data),
        }
        .map_err(computation)
    }

    pub(crate) fn contract(
        &mut self,
        a: &IdxTensor,
        b: &IdxTensor,
    ) -> Result<IdxTensor, FactorizeError> {
        self.contract_options(a, b, PairwiseContractionOptions::new())
    }

    fn contract_options(
        &mut self,
        a: &IdxTensor,
        b: &IdxTensor,
        options: PairwiseContractionOptions,
    ) -> Result<IdxTensor, FactorizeError> {
        match self {
            // Both operands carry the same owning eager runtime. The implicit
            // contraction resolves that runtime's engine, including on `_in`.
            Self::Eager(_) => crate::contract_pair_with_operand_options(a, b, options),
            Self::Session(session) => crate::contract_pair_on_session(session, a, b, options),
        }
        .map_err(computation)
    }

    fn qr(&mut self, t: &IdxTensor, left: &[DynIndex]) -> Result<IdxTensor, FactorizeError> {
        let options = crate::qr::QrOptions::full_rank();
        macro_rules! qr {
            ($scalar:ty) => {
                match self {
                    Self::Eager(None) => crate::qr::qr_with::<$scalar>(t, left, &options),
                    Self::Eager(Some(context)) => {
                        crate::qr::qr_with_in::<$scalar>(t, left, &options, context)
                    }
                    Self::Session(session) => {
                        crate::qr::qr_with_on_session::<$scalar>(session, t, left, &options)
                    }
                }
            };
        }
        let (q, _) = if t.is_f64() { qr!(f64) } else { qr!(Complex64) }?;
        Ok(q)
    }

    fn svd(
        &mut self,
        t: &IdxTensor,
        left: &[DynIndex],
        rank: usize,
    ) -> Result<SvdFactorizeResult, FactorizeError> {
        let options = SvdOptions::full_rank().with_max_bond_dim(rank);
        Ok(match self {
            Self::Eager(None) => svd_for_factorize(t, left, &options),
            Self::Eager(Some(context)) => svd_for_factorize_in(t, left, &options, context),
            Self::Session(session) if t.is_f64() => {
                svd_for_factorize_on_session::<f64>(session, t, left, &options)
            }
            Self::Session(session) => {
                svd_for_factorize_on_session::<Complex64>(session, t, left, &options)
            }
        }?)
    }
}

struct PreparedRange {
    tensor: IdxTensor,
    right: Vec<DynIndex>,
    ell: usize,
}

fn prepare(
    t: &IdxTensor,
    left: &[DynIndex],
    rank: usize,
    options: &RangeOptions,
    route: &Route<'_, '_>,
) -> Result<PreparedRange, FactorizeError> {
    route.validate(t)?;
    if rank == 0 {
        return Err(FactorizeError::InvalidOptions(
            "RSVD rank must be at least 1",
        ));
    }
    let (left, right, _) = unfold_index_sets(t, left).map_err(|_| {
        FactorizeError::InvalidOptions(
        "left indices must be a nonempty proper subset with no duplicates and full index identity")
    })?;
    let rows = checked_product(&left.iter().map(IndexLike::dim).collect::<Vec<_>>())
        .map_err(computation)?;
    let cols = checked_product(&right.iter().map(IndexLike::dim).collect::<Vec<_>>())
        .map_err(computation)?;
    let ell = rank
        .checked_add(options.oversampling)
        .ok_or(FactorizeError::InvalidOptions(
            "rank + oversampling overflows usize",
        ))?
        .min(rows)
        .min(cols);
    if ell == 0 {
        return Err(FactorizeError::InvalidOptions(
            "RSVD requires nonempty matrix dimensions",
        ));
    }
    cols.checked_mul(ell)
        .and_then(|len| len.checked_mul(if t.is_f64() { 8 } else { 16 }))
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .ok_or(FactorizeError::InvalidOptions(
            "Gaussian sketch byte length overflows",
        ))?;
    let tensor = match route {
        Route::Eager(Some(context)) => t.adopt_into(context).map_err(computation)?,
        _ => t.clone(),
    };
    Ok(PreparedRange { tensor, right, ell })
}

fn range<R: Rng + ?Sized>(
    prepared: &PreparedRange,
    left: &[DynIndex],
    options: &RangeOptions,
    rng: &mut R,
    route: &mut Route<'_, '_>,
    timings: &mut RsvdTimings,
) -> Result<IdxTensor, FactorizeError> {
    let t = &prepared.tensor;
    let right = &prepared.right;
    let mut sketch_indices = right.to_vec();
    sketch_indices.push(DynIndex::new_bond(prepared.ell).map_err(computation)?);
    let omega = if t.is_f64() {
        route.gaussian::<f64, _>(t, sketch_indices, rng)
    } else {
        route.gaussian::<Complex64, _>(t, sketch_indices, rng)
    }?;
    let started = Instant::now();
    let y = route.contract(t, &omega)?;
    timings.gemm += started.elapsed();
    let started = Instant::now();
    let mut q = route.qr(&y, left)?;
    timings.qr += started.elapsed();
    for _ in 0..options.power_iterations {
        let started = Instant::now();
        let z =
            route.contract_options(t, &q, PairwiseContractionOptions::new().with_lhs_conj(true))?;
        timings.gemm += started.elapsed();
        let started = Instant::now();
        let z = route.qr(&z, right)?;
        timings.qr += started.elapsed();
        let started = Instant::now();
        let y = route.contract(t, &z)?;
        timings.gemm += started.elapsed();
        let started = Instant::now();
        q = route.qr(&y, left)?;
        timings.qr += started.elapsed();
    }
    Ok(q)
}

/// Find an orthonormal basis for a dense tensor's left-index range.
///
/// `left_inds` must be a nonempty proper subset of the full index values.
/// Returns Q with these indices and a fresh bond of dimension
/// `min(rank + options.oversampling, m, n)`. `Q Q†` is a bond projector.
/// Consumes Gaussian samples directly from `rng`; global QR defaults are ignored.
/// Only dense `f64` and `Complex64` are supported; `f32` and `Complex32` (c32)
/// return [`FactorizeError::UnsupportedStorage`] before advancing the RNG.
///
/// # Errors
/// Returns [`FactorizeError`] for zero rank, invalid/duplicate indices, size
/// overflow, unsupported storage/dtype, or a backend failure.
///
/// # Examples
/// ```
/// use rand::SeedableRng;
/// use rand_chacha::ChaCha8Rng;
/// use tensor4all_core::{randomized_range, DynIndex, IdxTensor, RangeOptions, TensorContractionLike};
/// let i = DynIndex::new_dyn(2);
/// let j = DynIndex::new_dyn(2);
/// let a = IdxTensor::from_dense(vec![i.clone(), j], vec![3., 0., 0., 0.])?;
/// let mut rng = ChaCha8Rng::seed_from_u64(11);
/// let q = randomized_range(&a, &[i], 1, &RangeOptions { oversampling: 0, power_iterations: 0 }, &mut rng)?;
/// let projected = q.contract_pair(&q.conj().contract_pair(&a)?)?;
/// assert!(a.sub(&projected)?.maxabs()? < 1e-12);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn randomized_range<R: Rng + ?Sized>(
    t: &IdxTensor,
    left_inds: &[DynIndex],
    rank: usize,
    options: &RangeOptions,
    rng: &mut R,
) -> Result<IdxTensor, FactorizeError> {
    let mut route = Route::Eager(None);
    let prepared = prepare(t, left_inds, rank, options, &route)?;
    range(
        &prepared,
        left_inds,
        options,
        rng,
        &mut route,
        &mut RsvdTimings::default(),
    )
}

/// Find a Gaussian range basis in a caller-owned execution context.
///
/// The rank, index split, oversampling and return contract match
/// [`randomized_range`]. Inputs and sketches remain in `context`; validation
/// precedes RNG advancement. Use zero oversampling for a rank-`rank` projector.
///
/// # Errors
/// Returns [`FactorizeError`] for invalid rank/indices, unsupported storage,
/// foreign contexts, size overflow, or a backend failure.
///
/// # Examples
/// ```
/// use std::sync::Arc;
/// use rand::SeedableRng;
/// use rand_chacha::ChaCha8Rng;
/// use tensor4all_core::{randomized_range_in, DynIndex, ExecutionContext, IdxTensor, RangeOptions};
/// use tensor4all_tensorbackend::{CpuBackend, CpuExecutionContext};
/// let context = ExecutionContext::Cpu(Arc::new(CpuExecutionContext::from_backend(CpuBackend::with_threads(1)?)));
/// let i = DynIndex::new_dyn(2);
/// let j = DynIndex::new_dyn(2);
/// let a = IdxTensor::from_dense_in(&context, vec![i.clone(), j], vec![3., 0., 0., 0.])?;
/// let mut rng = ChaCha8Rng::seed_from_u64(11);
/// let options = RangeOptions { oversampling: 0, power_iterations: 0 };
/// let q = randomized_range_in(&a, &[i], 1, &options, &mut rng, &context)?;
/// let values = q.to_vec::<f64>()?;
/// assert!((values[0].abs() - 1.).abs() < 1e-12 && values[1].abs() < 1e-12);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn randomized_range_in<R: Rng + ?Sized>(
    t: &IdxTensor,
    left_inds: &[DynIndex],
    rank: usize,
    options: &RangeOptions,
    rng: &mut R,
    context: &ExecutionContext,
) -> Result<IdxTensor, FactorizeError> {
    let mut route = Route::Eager(Some(context));
    let prepared = prepare(t, left_inds, rank, options, &route)?;
    range(
        &prepared,
        left_inds,
        options,
        rng,
        &mut route,
        &mut RsvdTimings::default(),
    )
}

/// Find a Gaussian range basis on a caller-held plain session.
///
/// Same rank, sketch and index contract as [`randomized_range`]. Numeric work
/// stays on `session`; the returned Q is untracked host storage. Global QR
/// defaults are ignored, and no eager runtime or nested session is entered.
///
/// # Errors
/// Returns [`FactorizeError`] for tracked/resident or unsupported operands,
/// invalid rank/indices, size overflow, or a backend failure.
///
/// # Examples
/// ```
/// use rand::SeedableRng;
/// use rand_chacha::ChaCha8Rng;
/// use tensor4all_core::{randomized_range_on_session, DynIndex, IdxTensor, RangeOptions};
/// use tensor4all_tensorbackend::{explicit::Session, CpuBackend};
/// let i = DynIndex::new_dyn(2);
/// let j = DynIndex::new_dyn(2);
/// let a = IdxTensor::from_dense(vec![i.clone(), j], vec![3., 0., 0., 0.])?;
/// let mut rng = ChaCha8Rng::seed_from_u64(11);
/// let mut backend = CpuBackend::with_threads(1)?;
/// let q = {
///     let mut session = Session::new(backend.engine_mut());
///     randomized_range_on_session(&mut session, &a, &[i], 1,
///         &RangeOptions { oversampling: 0, power_iterations: 0 }, &mut rng)?
/// };
/// let values = q.to_vec::<f64>()?;
/// assert!((values[0].abs() - 1.).abs() < 1e-12 && values[1].abs() < 1e-12);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn randomized_range_on_session<R: Rng + ?Sized>(
    session: &mut Session<'_>,
    t: &IdxTensor,
    left_inds: &[DynIndex],
    rank: usize,
    options: &RangeOptions,
    rng: &mut R,
) -> Result<IdxTensor, FactorizeError> {
    let mut route = Route::Session(session);
    let prepared = prepare(t, left_inds, rank, options, &route)?;
    range(
        &prepared,
        left_inds,
        options,
        rng,
        &mut route,
        &mut RsvdTimings::default(),
    )
}

pub(crate) fn svd_result<R: Rng + ?Sized>(
    t: &IdxTensor,
    left_inds: &[DynIndex],
    options: &RsvdOptions,
    rng: &mut R,
    route: &mut Route<'_, '_>,
) -> Result<(SvdFactorizeResult, RsvdTimings), FactorizeError> {
    let total = Instant::now();
    let mut timings = RsvdTimings::default();
    let prepared = prepare(t, left_inds, options.max_bond_dim, &options.range, route)?;
    let t = &prepared.tensor;
    let q = range(
        &prepared,
        left_inds,
        &options.range,
        rng,
        route,
        &mut timings,
    )?;
    let bond = q
        .indices()
        .last()
        .cloned()
        .ok_or(FactorizeError::InvalidOptions("QR returned no bond"))?;
    let started = Instant::now();
    let projected =
        route.contract_options(&q, t, PairwiseContractionOptions::new().with_lhs_conj(true))?;
    timings.gemm += started.elapsed();
    let started = Instant::now();
    let mut result = route.svd(&projected, &[bond], options.max_bond_dim)?;
    timings.small_svd += started.elapsed();
    let started = Instant::now();
    result.u = route.contract(&q, &result.u)?;
    timings.gemm += started.elapsed();
    timings.total = total.elapsed();
    Ok((result, timings))
}

/// Compute a fixed-rank randomized SVD of a materialized local tensor.
///
/// The index split and sketch have the same contract as [`randomized_range`].
/// Returns directly contractible U, S, V† with `min(max_bond_dim, m, n)`
/// triplets. The projected singular values do not certify the full-input error.
/// Uses the supplied RNG directly and ignores global truncation defaults.
/// Supports dense `f64`/`Complex64`; `f32`/`Complex32` (c32) are unsupported.
///
/// # Errors
/// Returns [`FactorizeError`] for invalid rank/indices, size overflow,
/// unsupported storage/dtype, or contraction/decomposition failure.
///
/// # Examples
/// ```
/// use rand::SeedableRng;
/// use rand_chacha::ChaCha8Rng;
/// use tensor4all_core::{randomized_svd, DynIndex, IdxTensor, RsvdOptions, TensorContractionLike};
/// let i = DynIndex::new_dyn(2);
/// let j = DynIndex::new_dyn(2);
/// let a = IdxTensor::from_dense(vec![i.clone(), j], vec![3., 0., 0., 0.])?;
/// let mut rng = ChaCha8Rng::seed_from_u64(11);
/// let result = randomized_svd(&a, &[i], &RsvdOptions::new(1), &mut rng)?;
/// let reconstructed = result.u.contract_pair(&result.s)?.contract_pair(&result.vh)?;
/// assert!(a.sub(&reconstructed)?.maxabs()? < 1e-12);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn randomized_svd<R: Rng + ?Sized>(
    t: &IdxTensor,
    left_inds: &[DynIndex],
    options: &RsvdOptions,
    rng: &mut R,
) -> Result<RsvdResult, FactorizeError> {
    let (result, timings) = svd_result(t, left_inds, options, rng, &mut Route::Eager(None))?;
    let vh = result
        .vh
        .replaceind(&result.bond_index, &result.s.indices()[1])
        .map_err(computation)?;
    Ok(RsvdResult {
        u: result.u,
        s: result.s,
        vh,
        singular_values: result.singular_values,
        timings,
    })
}

pub(crate) fn factorize<R: Rng + ?Sized>(
    t: &IdxTensor,
    left_inds: &[DynIndex],
    options: &FactorizeOptions,
    rng: &mut R,
    route: &mut Route<'_, '_>,
) -> Result<FactorizeResult<IdxTensor>, FactorizeError> {
    options.validate()?;
    let rank = options
        .max_bond_dim
        .ok_or(FactorizeError::InvalidOptions("RSVD requires max_bond_dim"))?;
    let (result, _) = svd_result(
        t,
        left_inds,
        &RsvdOptions {
            max_bond_dim: rank,
            range: options.randomized,
        },
        rng,
        route,
    )?;
    crate::defaults::factorize::assemble_svd_factors(result, options.canonical, &mut |a, b| {
        route.contract(a, b)
    })
}
