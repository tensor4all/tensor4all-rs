//! Error norms and tolerances of accuracy requirements.
//!
//! [`ErrorNorm`] selects the norm in which an accuracy requirement is stated
//! and measured, and [`ErrorTolerance`] gives the requirement in that norm's
//! units as `max(atol, rtol * reference)`. They are used by
//! [`patched_interpolate`](crate::adaptive_interpolation::patched_interpolate)
//! through
//! [`PatchedInterpolationOptions`](crate::adaptive_interpolation::PatchedInterpolationOptions).

/// The norm in which an accuracy requirement is stated and measured.
///
/// - [`ErrorNorm::L2`] (the default) is the unweighted discrete L2 norm over
///   the whole domain, `||g||^2 = sum over x of |g(x)|^2`, the norm of
///   [`TreeTN::norm`](crate::TreeTN::norm) and
///   [`PartitionedTreeTN::norm`](crate::PartitionedTreeTN::norm). The driver
///   measures the error of every accepted and zero patch itself; see
///   [`adaptive_interpolation`](crate::adaptive_interpolation) for what the
///   measurement can claim. Its reference is an L2 norm of the function
///   ([`L2Reference`]).
/// - [`ErrorNorm::SampledMax`] is the engine's own sampled criterion against a
///   max-norm reference, a function value such as `max |f|`. The driver runs
///   no measurement; this is neither a certified bound nor a measured error.
/// - [`ErrorNorm::MaxAbs`] and [`ErrorNorm::WeightedL2`] are placeholders
///   without an implementation. They fail with
///   [`PatchedInterpolationError::UnsupportedNorm`](crate::adaptive_interpolation::PatchedInterpolationError::UnsupportedNorm)
///   before any evaluation and never fall back to another norm.
///
/// The variants with data are built with [`ErrorNorm::l2`],
/// [`ErrorNorm::sampled_max`], and [`ErrorNorm::sampled_max_with_reference`].
///
/// # Examples
///
/// ```
/// use tensor4all_partitionedtreetn::{ErrorNorm, L2Reference};
///
/// assert_eq!(ErrorNorm::default(), ErrorNorm::l2(L2Reference::Required));
/// let given = ErrorNorm::l2(L2Reference::Given(3.0));
/// assert!(matches!(given, ErrorNorm::L2 { reference: L2Reference::Given(s), .. } if s == 3.0));
/// let max = ErrorNorm::sampled_max_with_reference(2.0);
/// assert!(matches!(max, ErrorNorm::SampledMax { max_reference: Some(r), .. } if r == 2.0));
/// assert_ne!(ErrorNorm::MaxAbs, ErrorNorm::WeightedL2);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum ErrorNorm {
    /// Unweighted discrete L2 norm over the whole domain (the default), with
    /// the driver's own measurement of every accepted and zero patch.
    #[non_exhaustive]
    L2 {
        /// Where the L2 reference norm `S` comes from.
        reference: L2Reference,
    },
    /// The engine's own sampled criterion against a max-norm reference: the
    /// M2 behavior, with no measurement by the driver. Neither a certified
    /// bound nor a measured error.
    #[non_exhaustive]
    SampledMax {
        /// A known `max |f|`, finite and positive. `None` pins it to the
        /// largest magnitude among the root patch's candidate samples (or its
        /// exact values when the root has at most one site), a sampled lower
        /// bound on `max |f|`.
        max_reference: Option<f64>,
    },
    /// Placeholder: the maximum norm over the whole domain. For a black-box
    /// function it can be certified only exhaustively, by evaluating every
    /// point. Not implemented.
    MaxAbs,
    /// Placeholder: an L2 norm with caller-supplied weights. Not implemented.
    WeightedL2,
}

impl ErrorNorm {
    /// The L2 norm, measured by the driver, with the given reference.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_partitionedtreetn::{ErrorNorm, L2Reference};
    ///
    /// let norm = ErrorNorm::l2(L2Reference::MonteCarlo);
    /// assert!(matches!(norm, ErrorNorm::L2 { reference: L2Reference::MonteCarlo, .. }));
    /// ```
    pub fn l2(reference: L2Reference) -> Self {
        Self::L2 { reference }
    }

    /// The M2 sampled max-norm criterion with the reference pinned from the
    /// root patch.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_partitionedtreetn::ErrorNorm;
    ///
    /// assert!(matches!(
    ///     ErrorNorm::sampled_max(),
    ///     ErrorNorm::SampledMax { max_reference: None, .. }
    /// ));
    /// ```
    pub fn sampled_max() -> Self {
        Self::SampledMax {
            max_reference: None,
        }
    }

    /// The M2 sampled max-norm criterion with a known `max |f|`.
    ///
    /// # Arguments
    ///
    /// * `max_reference` - A function value, typically `max |f|`; finite and
    ///   positive (checked by the driver). It is not an L2 norm.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_partitionedtreetn::ErrorNorm;
    ///
    /// assert!(matches!(
    ///     ErrorNorm::sampled_max_with_reference(4.0),
    ///     ErrorNorm::SampledMax { max_reference: Some(r), .. } if r == 4.0
    /// ));
    /// ```
    pub fn sampled_max_with_reference(max_reference: f64) -> Self {
        Self::SampledMax {
            max_reference: Some(max_reference),
        }
    }
}

impl Default for ErrorNorm {
    /// `ErrorNorm::L2 { reference: L2Reference::Required }`.
    fn default() -> Self {
        Self::l2(L2Reference::Required)
    }
}

/// Where the L2 reference norm `S` of [`ErrorNorm::L2`] comes from.
///
/// `S` is an unweighted discrete L2 norm of the function over the whole
/// domain. For a uniform quantics grid of `|X|` points on a domain of volume
/// `V`, it is approximately the continuum norm times `sqrt(|X| / V)`. A
/// max-norm value passed here would be wrong by `sqrt(|X|)`.
///
/// The reference is not needed when `rtol = 0` or when the root patch has at
/// most one site (it is then computed exactly from all values).
///
/// # Examples
///
/// ```
/// use tensor4all_partitionedtreetn::L2Reference;
///
/// assert_eq!(L2Reference::default(), L2Reference::Required);
/// assert_ne!(L2Reference::Given(1.0), L2Reference::MonteCarlo);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[non_exhaustive]
pub enum L2Reference {
    /// The caller's L2 norm of the function, finite and positive.
    Given(f64),
    /// Estimate it from uniform root samples, an explicit opt-in. The
    /// estimate is heavy-tailed for localized functions: it can make the
    /// allowance looser than requested as well as tighter.
    MonteCarlo,
    /// No reference (the default): allowed when `rtol = 0` or the root is
    /// exact, otherwise an error before any evaluation.
    #[default]
    Required,
}

/// Accuracy requirement in the units of the selected [`ErrorNorm`]:
/// `allowance = max(atol, rtol * reference)`.
///
/// Under [`ErrorNorm::L2`] the allowance is a global L2 norm `delta` and the
/// reference is the L2 norm `S`; under [`ErrorNorm::SampledMax`] it is the
/// engine's absolute tolerance and the reference is a function value. Both
/// fields must be finite and nonnegative; `rtol = atol = 0` is allowed and
/// accepts only exactly reproduced patches.
///
/// # Examples
///
/// ```
/// use tensor4all_partitionedtreetn::ErrorTolerance;
///
/// let tolerance = ErrorTolerance { rtol: 1e-6, atol: 1e-3 };
/// assert_eq!(tolerance.allowance(10.0), 1e-3);
/// assert_eq!(tolerance.allowance(1e4), 1e-6 * 1e4);
/// assert_eq!(ErrorTolerance::default(), ErrorTolerance { rtol: 1e-8, atol: 0.0 });
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ErrorTolerance {
    /// Tolerance relative to the reference of the selected norm. Default
    /// `1e-8`.
    pub rtol: f64,
    /// Absolute floor of the allowance, in the units of the selected norm.
    /// Default `0`.
    pub atol: f64,
}

impl ErrorTolerance {
    /// The allowance `max(atol, rtol * reference)` for a given reference.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_partitionedtreetn::ErrorTolerance;
    ///
    /// let tolerance = ErrorTolerance { rtol: 0.5, atol: 0.0 };
    /// assert_eq!(tolerance.allowance(3.0), 1.5);
    /// ```
    pub fn allowance(&self, reference: f64) -> f64 {
        self.atol.max(self.rtol * reference)
    }
}

impl Default for ErrorTolerance {
    /// `rtol = 1e-8` (the M2 default value), `atol = 0`.
    fn default() -> Self {
        Self {
            rtol: 1e-8,
            atol: 0.0,
        }
    }
}
