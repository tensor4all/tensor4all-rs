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
    /// every listed site is fixed fails with
    /// [`PatchedInterpolationError::NoSplitIndexLeft`](super::PatchedInterpolationError::NoSplitIndexLeft)
    /// or
    /// [`PatchedInterpolationError::VerificationFailed`](super::PatchedInterpolationError::VerificationFailed).
    /// Empty (the default) means every site in the derived site order
    /// ([`InterpolationProblem::derive_site_order`](tensor4all_treetn::interpolation::InterpolationProblem::derive_site_order)).
    /// For quantics grids, list the most significant bits first.
    pub patch_order: Vec<DynIndex>,
    /// Target number of distinct initial pivots per patch, at least 1.
    /// Compatible user pivots, recycled pivots, and the worst points of a
    /// failed parent measurement come first; random points of the patch fill
    /// the rest. Default `5`.
    pub n_initial_pivots: usize,
    /// Seed each child with the pivots of the parent's last engine outcome.
    /// Default `false`.
    pub recycle_pivots: bool,
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
    /// otherwise; such a patch is never certified. A patch whose last engine
    /// run did not converge is measured once and judged the same way. With
    /// `Some(m)` for `m >= 2`, a failing two-site patch is retained although
    /// splitting it would give exact patches, and `m` at least the number of
    /// sites blocks the root. `None` (the default) and `Some(0)` or
    /// `Some(1)` split down to exact patches.
    pub min_patch_bits: Option<usize>,
    /// Whether a patch that reaches the bond cap may be accepted. Default
    /// [`CappedPatches::Split`], which splits every capped patch. Patches
    /// that converge below the cap are never split for their size.
    pub capped_patches: CappedPatches,
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
    /// order, five initial pivots, no recycling, seed `0`, no patch limit,
    /// no minimum patch size, and capped patches split.
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
    /// assert_eq!(options.seed, 0);
    /// assert_eq!(options.max_patches, None);
    /// assert_eq!(options.min_patch_bits, None);
    /// assert_eq!(options.capped_patches, CappedPatches::Split);
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
            seed: 0,
            max_patches: None,
            min_patch_bits: None,
            capped_patches: CappedPatches::Split,
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
}

/// How [`patched_interpolate`](super::patched_interpolate) measures accepted
/// and zero patches under [`ErrorNorm::L2`].
///
/// A patch with at most `max(max_exhaustive_points, samples)` points is
/// measured exhaustively, exact up to rounding (a certificate when the patch
/// meets its allowance, otherwise a bound on its error); a larger one on
/// `samples` fresh uniform points, a decision statistic. With `audit`, every
/// sampled acceptance gets an independent audit sample after the decision,
/// whose mean square is an unbiased estimate (its RMS is not; neither is a
/// bound).
///
/// Both samples are uniform, so both can miss a localized feature that
/// enters a patch only through a corner or an edge, and the audit's standard
/// error does not reveal such a miss. Only exhaustive (or exact) measurement
/// rules it out; see "Known limitation" in the
/// [module documentation](super).
///
/// The defaults are provisional; their measured cost is a later milestone.
/// When in doubt keep them: `samples = 64`, `max_exhaustive_points = 1024`,
/// `retries = 1`, `audit = true`. Raising `samples` lowers the variance of a
/// sampled measurement and the chance of missing a localized residual, at
/// proportional cost, but does not exclude a miss; `max_exhaustive_points`
/// bounds the exhaustive work and cache growth per patch.
///
/// # Examples
///
/// ```
/// use tensor4all_partitionedtreetn::adaptive_interpolation::VerificationOptions;
///
/// let options = VerificationOptions::new()
///     .with_samples(16)
///     .with_max_exhaustive_points(0)
///     .with_retries(0)
///     .with_audit(false);
/// assert_eq!(
///     (options.samples, options.max_exhaustive_points, options.retries, options.audit),
///     (16, 0, 0, false)
/// );
/// assert_eq!(VerificationOptions::default(), VerificationOptions::new());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct VerificationOptions {
    /// Fresh uniform points per sampled measurement, at least 2. Default 64.
    /// More points make a missed localized residual less likely; no finite
    /// number excludes one.
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
    /// Draw an independent audit sample for every sampled contribution after
    /// its acceptance decision. Default `true`; without it a run with a
    /// sampled contribution is reported as
    /// [`GlobalL2Error::AcceptanceOnly`](super::GlobalL2Error::AcceptanceOnly).
    /// The audit is uniform like the acceptance sample: it detects a missed
    /// residual only if it draws points of it, and its standard error does
    /// not reveal one it did not draw.
    pub audit: bool,
}

impl VerificationOptions {
    /// The defaults: 64 samples, exhaustive up to 1024 points, one retry,
    /// audit on.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_partitionedtreetn::adaptive_interpolation::VerificationOptions;
    ///
    /// let options = VerificationOptions::new();
    /// assert_eq!(options.samples, 64);
    /// assert_eq!(options.max_exhaustive_points, 1024);
    /// assert_eq!(options.retries, 1);
    /// assert!(options.audit);
    /// ```
    pub fn new() -> Self {
        Self {
            samples: 64,
            max_exhaustive_points: 1024,
            retries: 1,
            audit: true,
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

    /// Enable or disable the audit sample.
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
}

impl Default for VerificationOptions {
    /// The same values as [`VerificationOptions::new`].
    fn default() -> Self {
        Self::new()
    }
}
