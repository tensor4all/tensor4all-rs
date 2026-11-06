//! Options of [`patched_interpolate`](super::patched_interpolate).

use tensor4all_core::DynIndex;

use crate::{ErrorNorm, ErrorTolerance};

/// Options of [`patched_interpolate`](super::patched_interpolate).
///
/// Built with [`PatchedInterpolationOptions::new`], which takes the required
/// bond cap and sets every other field to its default, and refined with the
/// `with_*` builders. There is no `Default`: the bond cap has no sensible
/// default. The fields are validated by
/// [`patched_interpolate`](super::patched_interpolate).
///
/// The accuracy requirement is the pair `error_norm` and `tolerance`. The
/// default norm is the L2 norm, measured by the driver, with
/// [`L2Reference::Required`](crate::L2Reference::Required): unless
/// `tolerance.rtol = 0` or the root patch has at most one site, choose a
/// reference norm with [`ErrorNorm::l2`] (a known L2 norm, or the opt-in
/// Monte Carlo estimate). [`ErrorNorm::sampled_max`] reproduces the M2
/// criterion.
///
/// `tolerance` and `max_bond_dim` trade off: a tighter tolerance or a smaller
/// cap produces more, smaller patches. When in doubt, keep the defaults, give
/// a known reference norm, and choose the cap from the rank the engine can
/// afford per patch. With a tiny or zero tolerance, set `max_patches`: the
/// driver then splits down to exact patches in the worst case, unless
/// `min_patch_bits` stops the splitting earlier.
///
/// `min_patch_bits` and `capped_patches` bound the patch size in generalized
/// bits: every active (unfixed) site of a patch counts as one bit, whatever
/// its dimension, so a fused quantics site of dimension 4 or 8 is one bit.
/// The minimum stops splitting and accepts a failing patch as
/// [`PatchStatus::ToleranceNotMet`](super::PatchStatus::ToleranceNotMet);
/// [`CappedPatches::AcceptUpTo`] accepts small patches that reach the bond
/// cap on their error check. Both default to the M3 behavior.
///
/// # Examples
///
/// ```
/// use tensor4all_core::DynIndex;
/// use tensor4all_partitionedtreetn::adaptive_interpolation::{
///     CappedPatches, PatchedInterpolationOptions, VerificationOptions,
/// };
/// use tensor4all_partitionedtreetn::{ErrorNorm, ErrorTolerance, L2Reference};
///
/// let first = DynIndex::new_dyn(2);
/// let options = PatchedInterpolationOptions::new(16)
///     .with_error_norm(ErrorNorm::l2(L2Reference::Given(2.0)))
///     .with_tolerance(ErrorTolerance { rtol: 1e-6, atol: 0.0 })
///     .with_verification(VerificationOptions::new().with_samples(128))
///     .with_patch_order(vec![first.clone()])
///     .with_recycle_pivots(true);
/// assert_eq!(options.max_bond_dim, 16);
/// assert_eq!(options.error_norm, ErrorNorm::l2(L2Reference::Given(2.0)));
/// assert_eq!(options.tolerance.rtol, 1e-6);
/// assert_eq!(options.verification.samples, 128);
/// assert_eq!(options.patch_order, vec![first]);
/// assert_eq!(options.n_initial_pivots, 5);
/// assert!(options.recycle_pivots);
/// assert_eq!(options.max_patches, None);
/// assert_eq!(options.min_patch_bits, None);
/// assert_eq!(options.capped_patches, CappedPatches::Split);
/// assert!(options.variables.is_empty());
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PatchedInterpolationOptions {
    /// The norm of the accuracy requirement and where its reference comes
    /// from. Default `ErrorNorm::L2 { reference: L2Reference::Required }`.
    pub error_norm: ErrorNorm,
    /// The requirement `max(atol, rtol * reference)` in the units of
    /// `error_norm`. Both fields finite and nonnegative. Default
    /// `rtol = 1e-8`, `atol = 0`.
    pub tolerance: ErrorTolerance,
    /// How patches are measured under [`ErrorNorm::L2`]; validated under
    /// every norm. Default [`VerificationOptions::new`].
    pub verification: VerificationOptions,
    /// Bond cap of every patch, at least 2. A patch is accepted when the
    /// engine converges with a rank strictly below the cap and passes its
    /// error check; with [`CappedPatches::AcceptUpTo`], a small enough patch
    /// that reaches the cap is accepted on its error check as well, and
    /// `min_patch_bits` can retain a patch that did not converge. Required;
    /// a smaller cap means more, smaller patches.
    pub max_bond_dim: usize,
    /// Sites fixed when a patch splits, in order, by full index identity.
    /// A partial order is allowed: a patch that cannot be accepted after
    /// every listed site is fixed fails (the minimum patch size never turns
    /// this into an acceptance) with
    /// [`PatchedInterpolationError::NoSplitIndexLeft`](super::PatchedInterpolationError::NoSplitIndexLeft)
    /// or
    /// [`PatchedInterpolationError::VerificationFailed`](super::PatchedInterpolationError::VerificationFailed).
    /// Empty (the default) means every site in the derived site order
    /// ([`InterpolationProblem::derive_site_order`](tensor4all_treetn::interpolation::InterpolationProblem::derive_site_order)).
    /// For quantics grids, list the most significant bits first.
    pub patch_order: Vec<DynIndex>,
    /// Target number of distinct initial pivots per patch, at least 1.
    /// Compatible user pivots, recycled pivots, the worst points of a failed
    /// parent measurement, and with `cache_candidates` the largest cached
    /// values come first and are all kept, so they can exceed the target;
    /// random points of the patch fill the rest up to the target. Default
    /// `5`.
    pub n_initial_pivots: usize,
    /// Seed each child with the pivots of the parent's last engine outcome.
    /// Default `false`.
    pub recycle_pivots: bool,
    /// Also start each patch from the points of its inherited evaluation
    /// cache with the largest `|f|`: at most `n_initial_pivots` nonzero
    /// points, after the user, recycled, and worst points. The cache holds
    /// every value the parent's engine run and measurements evaluated inside
    /// the patch, so this adds no evaluations. A child with at least
    /// `n_initial_pivots` such points gets no random start points: random
    /// exploration is traded for points where `|f|` is known to be large.
    /// An opt-in mitigation of corner-localized misses under
    /// [`L2Estimator::UniformMonteCarlo`]: a child then starts on a feature
    /// the parent sampled inside it. It cannot help with a feature the parent
    /// never sampled, nor with a miss of the acceptance sample. Default
    /// `false`.
    pub cache_candidates: bool,
    /// Root seed of every per-patch sub-seed and measurement stream. Default
    /// `0`.
    pub seed: u64,
    /// Limit on processed patches (accepted, zero, and split alike); `None`
    /// (the default) means no limit, `Some(0)` is invalid.
    pub max_patches: Option<usize>,
    /// Smallest patch the driver may create, in generalized bits (one bit
    /// per active site, whatever its dimension). A split whose children
    /// would have fewer bits is not made: a patch whose split is blocked
    /// this way is retained instead, with
    /// [`PatchStatus::WithinTolerance`](super::PatchStatus::WithinTolerance)
    /// when it meets its allowance (the measured error under L2, the engine
    /// estimate under `SampledMax`) and
    /// [`PatchStatus::ToleranceNotMet`](super::PatchStatus::ToleranceNotMet)
    /// otherwise; a `ToleranceNotMet` patch is never certified. A last run that was not
    /// judged yet (it did not converge and was not capped-eligible) is
    /// judged once: measured under L2, by its estimate under `SampledMax`.
    /// With `Some(m)` for `m >= 2`, a failing two-site patch is retained
    /// although splitting it would give exact patches, and `m` at least the
    /// number of sites blocks the root. `None` (the default) and `Some(0)` or
    /// `Some(1)` split down to exact patches.
    pub min_patch_bits: Option<usize>,
    /// Whether a patch that reaches the bond cap may be accepted. Default
    /// [`CappedPatches::Split`], which splits every capped patch. Patches
    /// that converge below the cap are never split for their size.
    pub capped_patches: CappedPatches,
    /// Sites of each variable, used only to choose the corner and edge start
    /// points of the residual-TT estimator
    /// ([`L2Estimator::ResidualTt`]). Default empty, which treats every active
    /// site of a patch as one variable. On a quantics problem, list each
    /// variable's bits so that the corners mix whole variables rather than
    /// cutting through a variable's bits. Every index must be a site of the
    /// problem and appear at most once; active sites in no group form one
    /// further group. See [`VerificationOptions::estimator`].
    pub variables: Vec<Vec<DynIndex>>,
}

/// Acceptance of patches whose engine run reaches the bond cap
/// ([`InterpolationTermination::BondCapReached`](tensor4all_treetn::interpolation::InterpolationTermination::BondCapReached)).
///
/// A capped patch can meet its tolerance without compressing, so passing the
/// error check alone does not justify accepting a large one. The bound is in
/// generalized bits, as for
/// [`PatchedInterpolationOptions::min_patch_bits`]: every active site counts
/// as one bit, whatever its dimension. On binary layouts, a bound of at most
/// `log2(max(max_exhaustive_points, samples))` bits (10 with the default
/// [`VerificationOptions`]) keeps every capped acceptance exhaustively
/// measured. `IterationLimit` runs are never accepted through this option.
///
/// # Examples
///
/// ```
/// use tensor4all_partitionedtreetn::adaptive_interpolation::{
///     CappedPatches, PatchedInterpolationOptions,
/// };
///
/// let options = PatchedInterpolationOptions::new(8)
///     .with_capped_patches(CappedPatches::AcceptUpTo { bits: 6 });
/// assert_eq!(options.capped_patches, CappedPatches::AcceptUpTo { bits: 6 });
/// assert_eq!(CappedPatches::default(), CappedPatches::Split);
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum CappedPatches {
    /// Split every patch that reaches the cap (the M3 behavior). The
    /// minimum patch size can still retain one whose split is blocked.
    #[default]
    Split,
    /// Accept a capped patch with at most `bits` active sites when it passes
    /// its error check (the L2 measurement, or the engine estimate under
    /// `SampledMax`); split larger ones. Must be at least `min_patch_bits`.
    AcceptUpTo {
        /// Largest accepted capped patch, in generalized bits.
        bits: usize,
    },
}

impl PatchedInterpolationOptions {
    /// Create options with the given bond cap and the defaults of every
    /// other field: the measured L2 norm with a required reference,
    /// `rtol = 1e-8`, `atol = 0`, default verification, the derived site
    /// order, five initial pivots, no recycling, no cache candidates, seed
    /// `0`, no patch limit,
    /// no minimum patch size, capped patches split, and no variable groups.
    ///
    /// # Arguments
    ///
    /// * `max_bond_dim` - Bond cap of every patch;
    ///   [`patched_interpolate`](super::patched_interpolate) requires at
    ///   least 2.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_partitionedtreetn::adaptive_interpolation::{
    ///     CappedPatches, PatchedInterpolationOptions, VerificationOptions,
    /// };
    /// use tensor4all_partitionedtreetn::{ErrorNorm, ErrorTolerance};
    ///
    /// let options = PatchedInterpolationOptions::new(8);
    /// assert_eq!(options.max_bond_dim, 8);
    /// assert_eq!(options.error_norm, ErrorNorm::default());
    /// assert_eq!(options.tolerance, ErrorTolerance::default());
    /// assert_eq!(options.verification, VerificationOptions::new());
    /// assert!(options.patch_order.is_empty());
    /// assert_eq!(options.n_initial_pivots, 5);
    /// assert!(!options.recycle_pivots);
    /// assert!(!options.cache_candidates);
    /// assert_eq!(options.seed, 0);
    /// assert_eq!(options.max_patches, None);
    /// assert_eq!(options.min_patch_bits, None);
    /// assert_eq!(options.capped_patches, CappedPatches::Split);
    /// assert!(options.variables.is_empty());
    /// ```
    pub fn new(max_bond_dim: usize) -> Self {
        Self {
            error_norm: ErrorNorm::default(),
            tolerance: ErrorTolerance::default(),
            verification: VerificationOptions::new(),
            max_bond_dim,
            patch_order: Vec::new(),
            n_initial_pivots: 5,
            recycle_pivots: false,
            cache_candidates: false,
            seed: 0,
            max_patches: None,
            min_patch_bits: None,
            capped_patches: CappedPatches::Split,
            variables: Vec::new(),
        }
    }

    /// Set the error norm.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_partitionedtreetn::adaptive_interpolation::PatchedInterpolationOptions;
    /// use tensor4all_partitionedtreetn::ErrorNorm;
    ///
    /// let options = PatchedInterpolationOptions::new(4).with_error_norm(ErrorNorm::sampled_max());
    /// assert_eq!(options.error_norm, ErrorNorm::sampled_max());
    /// ```
    pub fn with_error_norm(mut self, error_norm: ErrorNorm) -> Self {
        self.error_norm = error_norm;
        self
    }

    /// Set the tolerance.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_partitionedtreetn::adaptive_interpolation::PatchedInterpolationOptions;
    /// use tensor4all_partitionedtreetn::ErrorTolerance;
    ///
    /// let tolerance = ErrorTolerance { rtol: 1e-4, atol: 1e-9 };
    /// let options = PatchedInterpolationOptions::new(4).with_tolerance(tolerance);
    /// assert_eq!(options.tolerance, tolerance);
    /// ```
    pub fn with_tolerance(mut self, tolerance: ErrorTolerance) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set the verification options.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_partitionedtreetn::adaptive_interpolation::{
    ///     PatchedInterpolationOptions, VerificationOptions,
    /// };
    ///
    /// let verification = VerificationOptions::new().with_retries(2);
    /// let options = PatchedInterpolationOptions::new(4).with_verification(verification);
    /// assert_eq!(options.verification.retries, 2);
    /// ```
    pub fn with_verification(mut self, verification: VerificationOptions) -> Self {
        self.verification = verification;
        self
    }

    /// Set the order in which sites are fixed when a patch splits.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::DynIndex;
    /// use tensor4all_partitionedtreetn::adaptive_interpolation::PatchedInterpolationOptions;
    ///
    /// let (a, b) = (DynIndex::new_dyn(2), DynIndex::new_dyn(2));
    /// let options =
    ///     PatchedInterpolationOptions::new(4).with_patch_order(vec![b.clone(), a.clone()]);
    /// assert_eq!(options.patch_order, vec![b, a]);
    /// ```
    pub fn with_patch_order(mut self, patch_order: Vec<DynIndex>) -> Self {
        self.patch_order = patch_order;
        self
    }

    /// Set the target number of initial pivots per patch.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_partitionedtreetn::adaptive_interpolation::PatchedInterpolationOptions;
    ///
    /// let options = PatchedInterpolationOptions::new(4).with_n_initial_pivots(12);
    /// assert_eq!(options.n_initial_pivots, 12);
    /// ```
    pub fn with_n_initial_pivots(mut self, n_initial_pivots: usize) -> Self {
        self.n_initial_pivots = n_initial_pivots;
        self
    }

    /// Enable or disable pivot recycling.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_partitionedtreetn::adaptive_interpolation::PatchedInterpolationOptions;
    ///
    /// assert!(PatchedInterpolationOptions::new(4).with_recycle_pivots(true).recycle_pivots);
    /// ```
    pub fn with_recycle_pivots(mut self, recycle_pivots: bool) -> Self {
        self.recycle_pivots = recycle_pivots;
        self
    }

    /// Enable or disable the cache candidates of child patches.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_partitionedtreetn::adaptive_interpolation::PatchedInterpolationOptions;
    ///
    /// assert!(PatchedInterpolationOptions::new(4).with_cache_candidates(true).cache_candidates);
    /// ```
    pub fn with_cache_candidates(mut self, cache_candidates: bool) -> Self {
        self.cache_candidates = cache_candidates;
        self
    }

    /// Set the root seed.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_partitionedtreetn::adaptive_interpolation::PatchedInterpolationOptions;
    ///
    /// assert_eq!(PatchedInterpolationOptions::new(4).with_seed(42).seed, 42);
    /// ```
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Limit the number of processed patches.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_partitionedtreetn::adaptive_interpolation::PatchedInterpolationOptions;
    ///
    /// let options = PatchedInterpolationOptions::new(4).with_max_patches(100);
    /// assert_eq!(options.max_patches, Some(100));
    /// ```
    pub fn with_max_patches(mut self, max_patches: usize) -> Self {
        self.max_patches = Some(max_patches);
        self
    }

    /// Set the minimum patch size in generalized bits (one bit per active
    /// site, whatever its dimension).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_partitionedtreetn::adaptive_interpolation::PatchedInterpolationOptions;
    ///
    /// let options = PatchedInterpolationOptions::new(4).with_min_patch_bits(3);
    /// assert_eq!(options.min_patch_bits, Some(3));
    /// ```
    pub fn with_min_patch_bits(mut self, bits: usize) -> Self {
        self.min_patch_bits = Some(bits);
        self
    }

    /// Set the acceptance of patches that reach the bond cap.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_partitionedtreetn::adaptive_interpolation::{
    ///     CappedPatches, PatchedInterpolationOptions,
    /// };
    ///
    /// let options = PatchedInterpolationOptions::new(4)
    ///     .with_capped_patches(CappedPatches::AcceptUpTo { bits: 3 });
    /// assert_eq!(options.capped_patches, CappedPatches::AcceptUpTo { bits: 3 });
    /// ```
    pub fn with_capped_patches(mut self, capped_patches: CappedPatches) -> Self {
        self.capped_patches = capped_patches;
        self
    }

    /// Set the sites of each variable for the residual-TT estimator's corner
    /// and edge start points.
    ///
    /// The groups only affect [`L2Estimator::ResidualTt`]; a
    /// [`L2Estimator::UniformMonteCarlo`] run does not use them. An empty
    /// list (the default) treats every active site of a patch as one
    /// variable. Every index must be a site of the problem and appear at
    /// most once;
    /// [`patched_interpolate`](super::patched_interpolate) validates the
    /// groups before any evaluation.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::DynIndex;
    /// use tensor4all_partitionedtreetn::adaptive_interpolation::PatchedInterpolationOptions;
    ///
    /// let (x, y) = (DynIndex::new_dyn(2), DynIndex::new_dyn(2));
    /// let options = PatchedInterpolationOptions::new(4)
    ///     .with_variables(vec![vec![x.clone()], vec![y.clone()]]);
    /// assert_eq!(options.variables, vec![vec![x], vec![y]]);
    /// ```
    pub fn with_variables(mut self, variables: Vec<Vec<DynIndex>>) -> Self {
        self.variables = variables;
        self
    }
}

/// How a sampled patch measurement estimates the patch's L2 error.
///
/// The estimator applies to sampled measurements only: a patch with at most
/// `max(verification.max_exhaustive_points, verification.samples)` points is
/// measured exhaustively (or exactly), and those methods are unchanged.
/// [`VerificationOptions::estimator`] chooses between the two sampled
/// estimators.
///
/// # Examples
///
/// ```
/// use tensor4all_partitionedtreetn::adaptive_interpolation::{
///     L2Estimator, VerificationOptions,
/// };
///
/// assert_eq!(L2Estimator::default(), L2Estimator::ResidualTt);
/// let options = VerificationOptions::new();
/// assert_eq!(options.estimator, L2Estimator::ResidualTt);
/// let options = options.with_estimator(L2Estimator::UniformMonteCarlo);
/// assert_eq!(options.estimator, L2Estimator::UniformMonteCarlo);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum L2Estimator {
    /// Approximate the patch residual `r = f - f~` itself with the same
    /// interpolation engine and use the exact contraction `||TT_r||_2`
    /// (divided by `sqrt(|P|)`) as the patch's L2 estimate. The default.
    ///
    /// The residual run starts from the largest residuals of the uniform
    /// sample together with the patch's corners and edges, so a feature that
    /// enters the patch only through a corner or an edge is reached even
    /// when the uniform sample misses it. The residual TT is capped at
    /// `max(ceil(chi_patch * verification.residual_bond_ratio),
    /// verification.residual_bond_floor)` with `chi_patch` the largest bond
    /// dimension of the measured patch; when it reaches that cap without
    /// converging, the estimate is the larger of the residual norm and the
    /// uniform sample's RMS. A converged residual estimate is not an
    /// unbiased estimate, so a run with such a contribution is never
    /// certified: [`GlobalL2Error`](super::GlobalL2Error) reports
    /// `Audited` (with an audit) or `AcceptanceOnly` (without one).
    ///
    /// Limits: if every initial pivot of the residual run is exactly zero
    /// the engine returns
    /// [`InterpolationError::AllSamplesZero`](tensor4all_treetn::interpolation::InterpolationError::AllSamplesZero)
    /// before its global pivot search, and the residual estimate is taken as
    /// zero. The norm is read with `TreeTN::log_norm`, so a residual norm
    /// above about `1.34e154` overflows the estimate and a norm below about
    /// `1e-154` underflows it to zero.
    #[default]
    ResidualTt,
    /// Estimate the error from the mean square of the residual at fresh
    /// uniform points of the patch, as in the M3 driver. The estimate is a
    /// decision statistic conditioned on the acceptance it decided, with the
    /// standard error of [`L2Measurement::mean_square_rel_std_error`](super::L2Measurement::mean_square_rel_std_error);
    /// the independent audit sample makes it an unbiased estimate of the mean
    /// square (its RMS is not). Uniform points can miss a residual
    /// concentrated on a small part of the patch, for example a localized
    /// feature that enters it only through a corner or an edge; neither more
    /// samples nor the audit's standard error exclude such a miss.
    UniformMonteCarlo,
}

/// How [`patched_interpolate`](super::patched_interpolate) measures accepted
/// and zero patches under [`ErrorNorm::L2`].
///
/// A patch with at most `max(max_exhaustive_points, samples)` points is
/// measured exhaustively, exact up to rounding (a certificate when the patch
/// meets its allowance, otherwise a bound on its error); a larger one is
/// estimated by [`VerificationOptions::estimator`]. With `audit`, every
/// sampled acceptance gets an independent audit after the decision, which is
/// reported next to the acceptance and used by the audited global error. The
/// audit never changes the acceptance decision.
///
/// The two estimators answer the concentrated-residual case differently.
/// [`L2Estimator::ResidualTt`] (the default) interpolates the residual itself
/// and takes its exact norm, using corner and edge starts so that a localized
/// feature is represented even when uniform points miss it; its bond cap and
/// the saturation fallback are documented on
/// [`L2Estimator::ResidualTt`]. [`L2Estimator::UniformMonteCarlo`] reproduces
/// the M3 behaviour: both its sample and its audit are uniform, so both can
/// miss a localized feature, and the audit's standard error does not reveal
/// such a miss. Only exhaustive (or exact) measurement rules it out.
///
/// The defaults are provisional; their measured cost is a later milestone.
/// When in doubt keep them: `samples = 64`, `max_exhaustive_points = 1024`,
/// `retries = 1`, `audit = true`, `estimator = ResidualTt`,
/// `residual_bond_ratio = 0.25`, `residual_bond_floor = 4`. Raising `samples`
/// lowers the variance of a uniform measurement and the chance of missing a
/// localized residual, at proportional cost, but does not exclude a miss;
/// `max_exhaustive_points` bounds the exhaustive work and cache growth per
/// patch.
///
/// # Examples
///
/// ```
/// use tensor4all_partitionedtreetn::adaptive_interpolation::{
///     L2Estimator, VerificationOptions,
/// };
///
/// let options = VerificationOptions::new()
///     .with_samples(16)
///     .with_max_exhaustive_points(0)
///     .with_retries(0)
///     .with_audit(false)
///     .with_estimator(L2Estimator::UniformMonteCarlo)
///     .with_residual_bond_ratio(0.5)
///     .with_residual_bond_floor(8);
/// assert_eq!(
///     (options.samples, options.max_exhaustive_points, options.retries, options.audit),
///     (16, 0, 0, false)
/// );
/// assert_eq!(options.estimator, L2Estimator::UniformMonteCarlo);
/// assert_eq!(options.residual_bond_ratio, 0.5);
/// assert_eq!(options.residual_bond_floor, 8);
/// assert_eq!(VerificationOptions::default(), VerificationOptions::new());
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct VerificationOptions {
    /// Fresh uniform points per sampled measurement, at least 2. Default 64.
    /// Under [`L2Estimator::ResidualTt`] they seed the residual run's start
    /// points and serve as its saturation fallback; under
    /// [`L2Estimator::UniformMonteCarlo`] they decide the measurement. More
    /// points make a missed localized residual less likely; no finite number
    /// excludes one.
    pub samples: usize,
    /// A patch with at most `max(max_exhaustive_points, samples)` points is
    /// measured exhaustively. Bounds the exhaustive work and cache growth per
    /// patch. Default 1024; 0 means only patches with at most `samples`
    /// points.
    pub max_exhaustive_points: usize,
    /// Engine reruns of one patch after a failed verification of a
    /// converged run, before the patch splits (or, when the minimum patch
    /// size blocks the split, is retained). A failed capped run is not
    /// rerun. Default 1.
    pub retries: usize,
    /// Draw an independent audit measurement for every sampled contribution
    /// after its acceptance decision. Default `true`; without it a run with a
    /// sampled contribution is reported as
    /// [`GlobalL2Error::AcceptanceOnly`](super::GlobalL2Error::AcceptanceOnly).
    /// The audit uses the same estimator as the acceptance and its own
    /// streams; it is never a gate.
    pub audit: bool,
    /// The estimator of a sampled patch measurement. Default
    /// [`L2Estimator::ResidualTt`]; [`L2Estimator::UniformMonteCarlo`]
    /// reproduces the M3 behaviour.
    pub estimator: L2Estimator,
    /// Factor of the residual TT's bond cap under
    /// [`L2Estimator::ResidualTt`]: the cap is
    /// `max(ceil(chi_patch * residual_bond_ratio), residual_bond_floor)` with
    /// `chi_patch` the largest bond dimension of the measured patch (zero
    /// for the zero approximation). Finite and positive. Default `0.25`: the
    /// residual's rank is at most about the target's plus the patch TT's,
    /// and small when the patch has converged. A larger ratio captures more
    /// of a localized residual at a higher cost; a capped residual run falls
    /// back to the larger of its own norm and the uniform sample's RMS.
    pub residual_bond_ratio: f64,
    /// Floor of the residual TT's bond cap under
    /// [`L2Estimator::ResidualTt`], at least 1. Default `4`, which keeps a
    /// small patch (`chi_patch` 1 to 8) able to hold more than one localized
    /// feature. It applies to the zero approximation as well.
    pub residual_bond_floor: usize,
}

impl VerificationOptions {
    /// The defaults: 64 samples, exhaustive up to 1024 points, one retry,
    /// audit on, [`L2Estimator::ResidualTt`], residual bond ratio `0.25`, and
    /// residual bond floor `4`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_partitionedtreetn::adaptive_interpolation::{
    ///     L2Estimator, VerificationOptions,
    /// };
    ///
    /// let options = VerificationOptions::new();
    /// assert_eq!(options.samples, 64);
    /// assert_eq!(options.max_exhaustive_points, 1024);
    /// assert_eq!(options.retries, 1);
    /// assert!(options.audit);
    /// assert_eq!(options.estimator, L2Estimator::ResidualTt);
    /// assert_eq!(options.residual_bond_ratio, 0.25);
    /// assert_eq!(options.residual_bond_floor, 4);
    /// ```
    pub fn new() -> Self {
        Self {
            samples: 64,
            max_exhaustive_points: 1024,
            retries: 1,
            audit: true,
            estimator: L2Estimator::ResidualTt,
            residual_bond_ratio: 0.25,
            residual_bond_floor: 4,
        }
    }

    /// Set the number of points of a sampled measurement (at least 2).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_partitionedtreetn::adaptive_interpolation::VerificationOptions;
    ///
    /// assert_eq!(VerificationOptions::new().with_samples(256).samples, 256);
    /// ```
    pub fn with_samples(mut self, samples: usize) -> Self {
        self.samples = samples;
        self
    }

    /// Set the exhaustive-measurement limit.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_partitionedtreetn::adaptive_interpolation::VerificationOptions;
    ///
    /// let options = VerificationOptions::new().with_max_exhaustive_points(4096);
    /// assert_eq!(options.max_exhaustive_points, 4096);
    /// ```
    pub fn with_max_exhaustive_points(mut self, max_exhaustive_points: usize) -> Self {
        self.max_exhaustive_points = max_exhaustive_points;
        self
    }

    /// Set the number of engine reruns after a failed verification.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_partitionedtreetn::adaptive_interpolation::VerificationOptions;
    ///
    /// assert_eq!(VerificationOptions::new().with_retries(0).retries, 0);
    /// ```
    pub fn with_retries(mut self, retries: usize) -> Self {
        self.retries = retries;
        self
    }

    /// Enable or disable the audit measurement.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_partitionedtreetn::adaptive_interpolation::VerificationOptions;
    ///
    /// assert!(!VerificationOptions::new().with_audit(false).audit);
    /// ```
    pub fn with_audit(mut self, audit: bool) -> Self {
        self.audit = audit;
        self
    }

    /// Set the estimator of a sampled patch measurement.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_partitionedtreetn::adaptive_interpolation::{
    ///     L2Estimator, VerificationOptions,
    /// };
    ///
    /// let options = VerificationOptions::new().with_estimator(L2Estimator::UniformMonteCarlo);
    /// assert_eq!(options.estimator, L2Estimator::UniformMonteCarlo);
    /// ```
    pub fn with_estimator(mut self, estimator: L2Estimator) -> Self {
        self.estimator = estimator;
        self
    }

    /// Set the factor of the residual TT's bond cap (finite and positive).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_partitionedtreetn::adaptive_interpolation::VerificationOptions;
    ///
    /// let options = VerificationOptions::new().with_residual_bond_ratio(0.5);
    /// assert_eq!(options.residual_bond_ratio, 0.5);
    /// ```
    pub fn with_residual_bond_ratio(mut self, residual_bond_ratio: f64) -> Self {
        self.residual_bond_ratio = residual_bond_ratio;
        self
    }

    /// Set the floor of the residual TT's bond cap (at least 1).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_partitionedtreetn::adaptive_interpolation::VerificationOptions;
    ///
    /// let options = VerificationOptions::new().with_residual_bond_floor(8);
    /// assert_eq!(options.residual_bond_floor, 8);
    /// ```
    pub fn with_residual_bond_floor(mut self, residual_bond_floor: usize) -> Self {
        self.residual_bond_floor = residual_bond_floor;
        self
    }
}

impl Default for VerificationOptions {
    /// The same values as [`VerificationOptions::new`].
    fn default() -> Self {
        Self::new()
    }
}
