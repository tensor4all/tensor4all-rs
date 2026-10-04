//! Global pivot finder for the TCI2 algorithm.
//!
//! After each two-site sweep, the TCI2 algorithm calls a
//! [`GlobalPivotFinder`] to locate regions of high interpolation error
//! that the local sweeps may have missed. The default implementation
//! ([`DefaultGlobalPivotFinder`]) uses random starting points with local
//! optimization.

use rand::Rng;
use tensor4all_core::{floating_zone_walk, MultiIndex, Scalar};
use tensor4all_simplett::{AbstractTensorTrain, SimpleTensorTrain, TTCache, TTScalar, Tensor3Ops};

use crate::error::{validate_nonnegative_finite, Result, TCIError};

/// Snapshot of the current TCI state, passed to [`GlobalPivotFinder`].
pub struct GlobalPivotSearchInput<T: Scalar + TTScalar> {
    /// Local dimensions of each tensor index.
    pub local_dims: Vec<usize>,
    /// Current tensor train approximation.
    pub current_tt: SimpleTensorTrain<T>,
    /// Maximum absolute function value encountered so far.
    pub max_sample_value: f64,
    /// Left index sets (I) for each site.
    pub i_set: Vec<Vec<MultiIndex>>,
    /// Right index sets (J) for each site.
    pub j_set: Vec<Vec<MultiIndex>>,
}

/// Trait for global pivot finders.
///
/// Implementors search for multi-indices where the interpolation error
/// `|f(idx) - tt(idx)|` is large. Found pivots are added to the TCI state
/// to improve the approximation in the next sweep.
///
/// The default implementation is [`DefaultGlobalPivotFinder`]. Implement
/// this trait to supply domain-specific search strategies.
///
/// # Examples
///
/// Using the default implementation via [`DefaultGlobalPivotFinder`]:
///
/// ```
/// use rand::SeedableRng;
/// use tensor4all_tensorci::{DefaultGlobalPivotFinder, GlobalPivotFinder,
///     GlobalPivotSearchInput};
/// use tensor4all_simplett::SimpleTensorTrain;
///
/// // Constant-zero TT on a 4×4 grid
/// let tt = SimpleTensorTrain::<f64>::constant(&[4, 4], 0.0);
///
/// let input = GlobalPivotSearchInput {
///     local_dims: vec![4, 4],
///     current_tt: tt,
///     max_sample_value: 9.0,
///     i_set: vec![vec![vec![]], vec![vec![0]]],
///     j_set: vec![vec![vec![0]], vec![vec![]]],
/// };
///
/// let finder = DefaultGlobalPivotFinder::default();
/// let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
///
/// // Every coordinate walk on f(i,j) = i+j reaches (3,3).
/// let pivots = finder.find_global_pivots(
///     &input,
///     &|idx: &Vec<usize>| (idx[0] + idx[1]) as f64,
///     0.5,
///     &mut rng,
/// )?;
///
/// assert_eq!(pivots, vec![vec![3, 3]; 5]);
/// # Ok::<(), tensor4all_tensorci::TCIError>(())
/// ```
pub trait GlobalPivotFinder {
    /// Find multi-indices with high interpolation error.
    ///
    /// # Arguments
    ///
    /// * `input` -- current TCI state (tensor train, index sets, etc.)
    /// * `f` -- the function being interpolated
    /// * `abs_tol` -- absolute tolerance; pivots with error above this
    ///
    ///   threshold (times `tol_margin`) are interesting
    /// * `rng` -- random number generator for stochastic search
    ///
    /// # Returns
    ///
    /// Multi-indices where the interpolation error is large, up to the
    /// implementation's maximum count.
    ///
    /// # Errors
    ///
    /// The default finder returns [`TCIError::InvalidConfiguration`] for empty
    /// or zero dimensions, or non-finite/negative tolerances or thresholds;
    /// [`TCIError::DimensionMismatch`] if the grid and TT shapes differ;
    /// [`TCIError::InvalidOperation`] for non-finite residuals; and
    /// [`TCIError::SimpleTensorTrain`] for failed TT evaluation. Configuration
    /// is validated even when search is disabled. Custom finders may return
    /// other [`TCIError`] variants; the optimizer propagates them unchanged.
    fn find_global_pivots<T, F, R>(
        &self,
        input: &GlobalPivotSearchInput<T>,
        f: &F,
        abs_tol: f64,
        rng: &mut R,
    ) -> Result<Vec<MultiIndex>>
    where
        T: Scalar + TTScalar,
        F: Fn(&MultiIndex) -> T,
        R: Rng + ?Sized;
}

/// Default global pivot finder using random search with local optimization.
///
/// Algorithm:
///
/// 1. Generate `nsearch` random initial points.
/// 2. For each point, retain each maximizing coordinate while sweeping the
///    dimensions. Repeat until a full sweep no longer improves the error,
///    the error exceeds `10 * abs_tol * tol_margin`, or 100 sweeps complete.
/// 3. Keep points where the error exceeds `abs_tol * tol_margin`.
/// 4. Return at most `max_nglobal_pivot` results.
///
/// The walk and stopping thresholds follow TensorCrossInterpolation.jl's
/// default floating-zone search. Starts use the supplied RNG; repeated pivots
/// are retained in search order. This does not guarantee identical Julia
/// results, which also depend on RNGs, deduplication, and outer sweeps.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorci::DefaultGlobalPivotFinder;
///
/// // Default configuration
/// let finder = DefaultGlobalPivotFinder::default();
/// assert_eq!(finder.nsearch, 5);
/// assert_eq!(finder.max_nglobal_pivot, 5);
/// assert!((finder.tol_margin - 10.0).abs() < 1e-15);
///
/// // Custom configuration
/// let custom = DefaultGlobalPivotFinder::new(20, 10, 5.0);
/// assert_eq!(custom.nsearch, 20);
/// assert_eq!(custom.max_nglobal_pivot, 10);
/// assert!((custom.tol_margin - 5.0).abs() < 1e-15);
/// ```
#[derive(Debug, Clone)]
pub struct DefaultGlobalPivotFinder {
    /// Number of random initial points to search from
    pub nsearch: usize,
    /// Maximum number of pivots to add per iteration
    pub max_nglobal_pivot: usize,
    /// Search for pivots with error > abs_tol × tol_margin.
    /// Must be finite and nonnegative; the default is 10.
    pub tol_margin: f64,
}

impl Default for DefaultGlobalPivotFinder {
    fn default() -> Self {
        Self {
            nsearch: 5,
            max_nglobal_pivot: 5,
            tol_margin: 10.0,
        }
    }
}

impl DefaultGlobalPivotFinder {
    /// Create a new DefaultGlobalPivotFinder with the given parameters.
    pub fn new(nsearch: usize, max_nglobal_pivot: usize, tol_margin: f64) -> Self {
        Self {
            nsearch,
            max_nglobal_pivot,
            tol_margin,
        }
    }
}

impl GlobalPivotFinder for DefaultGlobalPivotFinder {
    fn find_global_pivots<T, F, R>(
        &self,
        input: &GlobalPivotSearchInput<T>,
        f: &F,
        abs_tol: f64,
        rng: &mut R,
    ) -> Result<Vec<MultiIndex>>
    where
        T: Scalar + TTScalar,
        F: Fn(&MultiIndex) -> T,
        R: Rng + ?Sized,
    {
        validate_nonnegative_finite("abs_tol", abs_tol)?;
        validate_nonnegative_finite("tol_margin", self.tol_margin)?;
        let threshold = abs_tol * self.tol_margin;
        validate_nonnegative_finite("abs_tol * tol_margin", threshold)?;
        if input.local_dims.is_empty() || input.local_dims.contains(&0) {
            return Err(TCIError::InvalidConfiguration {
                message: "local_dims must contain positive dimensions".to_string(),
            });
        }
        if input.local_dims.len() != input.current_tt.len()
            || input
                .local_dims
                .iter()
                .enumerate()
                .any(|(site, &dim)| dim != input.current_tt.site_tensor(site).site_dim())
        {
            return Err(TCIError::DimensionMismatch {
                message: "local_dims must match the tensor train's site dimensions".to_string(),
            });
        }
        if self.nsearch == 0 || self.max_nglobal_pivot == 0 {
            return Ok(Vec::new());
        }

        // Generate random initial points
        let initial_points: Vec<MultiIndex> = (0..self.nsearch)
            .map(|_| {
                input
                    .local_dims
                    .iter()
                    .map(|&dim| rng.random_range(0..dim))
                    .collect()
            })
            .collect();

        // Reuse contractions across coordinate batches and random starts.
        let mut tt_cache = TTCache::new(&input.current_tt);
        let mut found_pivots: Vec<MultiIndex> = Vec::new();

        for point in &initial_points {
            // Match searchglobalpivots -> floatingzone in
            // TensorCrossInterpolation.jl v0.9.14, src/tensorci2.jl:
            // at most 100 sweeps, with early stopping at 10 * threshold.
            // Overflow in this *early-stop* bound means no early stop;
            // the acceptance threshold itself was validated above.
            let (best_point, best_error) = floating_zone_walk(
                &input.local_dims,
                point,
                100,
                10.0 * threshold,
                |_site, points| {
                    let tt_values = tt_cache.evaluate_many(points, None)?;
                    points
                        .iter()
                        .zip(tt_values)
                        .map(|(point, tt_value)| {
                            let error = Scalar::abs_val(f(point) - tt_value);
                            if !error.is_finite() {
                                return Err(TCIError::InvalidOperation {
                                    message: format!(
                                        "non-finite global pivot residual at {point:?}"
                                    ),
                                });
                            }
                            Ok(error)
                        })
                        .collect::<Result<Vec<_>>>()
                },
            )?;

            // Add point if error exceeds threshold
            if best_error > threshold {
                found_pivots.push(best_point);
            }
        }

        // Limit number of pivots
        found_pivots.truncate(self.max_nglobal_pivot);

        Ok(found_pivots)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_global_pivot_finder() {
        // Simple function: f(i, j) = i * j
        let f = |idx: &MultiIndex| (idx[0] * idx[1]) as f64;
        let local_dims = vec![4, 4];

        // Build a deliberately bad TT (constant 0) to ensure large errors
        let tensors = vec![
            tensor4all_simplett::tensor3_zeros(1, 4, 1),
            tensor4all_simplett::tensor3_zeros(1, 4, 1),
        ];
        let tt = SimpleTensorTrain::new(tensors).unwrap();

        let input = GlobalPivotSearchInput {
            local_dims: local_dims.clone(),
            current_tt: tt,
            max_sample_value: 9.0,
            i_set: vec![vec![vec![]], vec![vec![0]]],
            j_set: vec![vec![vec![0]], vec![vec![]]],
        };

        let finder = DefaultGlobalPivotFinder::new(10, 3, 1.0);
        let mut rng = <rand_chacha::ChaCha8Rng as rand::SeedableRng>::seed_from_u64(7);

        let pivots = finder
            .find_global_pivots(&input, &f, 0.1, &mut rng)
            .unwrap();

        // Should find some pivots since the TT is zero but f is not
        // (except at i=0 or j=0)
        assert!(
            pivots.len() <= 3,
            "Should limit to max_nglobal_pivot=3, got {}",
            pivots.len()
        );
    }

    #[test]
    fn test_custom_global_pivot_finder() {
        struct FixedPivotFinder;

        impl GlobalPivotFinder for FixedPivotFinder {
            fn find_global_pivots<T, F, R>(
                &self,
                _input: &GlobalPivotSearchInput<T>,
                _f: &F,
                _abs_tol: f64,
                _rng: &mut R,
            ) -> Result<Vec<MultiIndex>>
            where
                T: Scalar + TTScalar,
                F: Fn(&MultiIndex) -> T,
                R: Rng + ?Sized,
            {
                // Always return a fixed pivot
                Ok(vec![vec![1, 2]])
            }
        }

        let finder = FixedPivotFinder;
        let f = |_: &MultiIndex| 1.0f64;
        let tensors = vec![
            tensor4all_simplett::tensor3_zeros(1, 3, 1),
            tensor4all_simplett::tensor3_zeros(1, 3, 1),
        ];
        let tt = SimpleTensorTrain::new(tensors).unwrap();

        let input = GlobalPivotSearchInput {
            local_dims: vec![3, 3],
            current_tt: tt,
            max_sample_value: 1.0,
            i_set: vec![],
            j_set: vec![],
        };

        let mut rng = <rand_chacha::ChaCha8Rng as rand::SeedableRng>::seed_from_u64(7);
        let pivots = finder
            .find_global_pivots(&input, &f, 0.0, &mut rng)
            .unwrap();
        assert_eq!(pivots, vec![vec![1, 2]]);
    }
}
