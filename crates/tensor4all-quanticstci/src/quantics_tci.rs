//! QuanticsTensorCI2 and interpolation functions.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use anyhow::{anyhow, Result};
use quanticsgrids::{DiscretizedGrid, InherentDiscreteGrid};
use rand::Rng;
use tensor4all_simplett::{AbstractTensorTrain, TTScalar, TensorTrain};
use tensor4all_tensorci::Scalar;
use tensor4all_tensorci::{crossinterpolate2, TensorCI2};

use crate::options::QtciOptions;

/// TCI result wrapped with grid information.
///
/// This struct combines a TensorCI2 result with grid information for
/// seamless conversion between grid indices and quantics indices.
pub struct QuanticsTensorCI2<V: Scalar + TTScalar> {
    /// Underlying TCI result
    tci: TensorCI2<V>,
    /// Grid for coordinate conversion (DiscretizedGrid)
    discretized_grid: Option<DiscretizedGrid>,
    /// Grid for coordinate conversion (InherentDiscreteGrid)
    inherent_grid: Option<InherentDiscreteGrid>,
    /// Cached function values (quantics index -> value)
    cache: HashMap<Vec<i64>, V>,
}

impl<V: Scalar + TTScalar + Default + Clone> QuanticsTensorCI2<V> {
    /// Create a new QuanticsTensorCI2 from TCI result and discretized grid.
    pub fn from_discretized(
        tci: TensorCI2<V>,
        grid: DiscretizedGrid,
        cache: HashMap<Vec<i64>, V>,
    ) -> Self {
        Self {
            tci,
            discretized_grid: Some(grid),
            inherent_grid: None,
            cache,
        }
    }

    /// Create a new QuanticsTensorCI2 from TCI result and inherent discrete grid.
    pub fn from_inherent(
        tci: TensorCI2<V>,
        grid: InherentDiscreteGrid,
        cache: HashMap<Vec<i64>, V>,
    ) -> Self {
        Self {
            tci,
            discretized_grid: None,
            inherent_grid: Some(grid),
            cache,
        }
    }

    /// Get the underlying TensorCI2.
    pub fn tci(&self) -> &TensorCI2<V> {
        &self.tci
    }

    /// Get the discretized grid (if available).
    pub fn discretized_grid(&self) -> Option<&DiscretizedGrid> {
        self.discretized_grid.as_ref()
    }

    /// Get the inherent discrete grid (if available).
    pub fn inherent_grid(&self) -> Option<&InherentDiscreteGrid> {
        self.inherent_grid.as_ref()
    }

    /// Get the bond dimension (maximum rank).
    pub fn rank(&self) -> usize {
        self.tci.rank()
    }

    /// Get link dimensions.
    pub fn link_dims(&self) -> Vec<usize> {
        self.tci.link_dims()
    }

    /// Convert grid indices to quantics indices.
    fn grididx_to_quantics(&self, indices: &[i64]) -> Result<Vec<i64>> {
        if let Some(grid) = &self.discretized_grid {
            grid.grididx_to_quantics(indices)
                .map_err(|e| anyhow!("Grid index conversion error: {}", e))
        } else if let Some(grid) = &self.inherent_grid {
            grid.grididx_to_quantics(indices)
                .map_err(|e| anyhow!("Grid index conversion error: {}", e))
        } else {
            Err(anyhow!("No grid available"))
        }
    }

    /// Evaluate at grid indices.
    ///
    /// # Arguments
    /// * `indices` - Grid indices (1-indexed as in Julia)
    ///
    /// # Returns
    /// Value at the specified grid point
    pub fn evaluate(&self, indices: &[i64]) -> Result<V> {
        let quantics = self.grididx_to_quantics(indices)?;
        let tt = self.tci.to_tensor_train()?;
        // Convert 1-indexed i64 quantics to 0-indexed usize for tensor train evaluation
        let quantics_usize: Vec<usize> = quantics.iter().map(|&x| (x - 1) as usize).collect();
        tt.evaluate(&quantics_usize)
            .map_err(|e| anyhow!("Evaluation error: {}", e))
    }

    /// Factorized sum over all grid points.
    ///
    /// This computes the sum efficiently using the tensor train structure.
    pub fn sum(&self) -> Result<V> {
        let tt = self.tci.to_tensor_train()?;
        Ok(tt.sum())
    }

    /// Integral over continuous domain.
    ///
    /// Returns the sum multiplied by the grid step sizes.
    /// Only available for discretized grids.
    pub fn integral(&self) -> Result<V>
    where
        V: std::ops::Mul<f64, Output = V>,
    {
        let sum_val = self.sum()?;
        if let Some(grid) = &self.discretized_grid {
            let step_product: f64 = grid.grid_step().iter().product();
            Ok(sum_val * step_product)
        } else {
            // For inherent discrete grids, just return the sum
            Ok(sum_val)
        }
    }

    /// Get the underlying TensorTrain.
    pub fn tensor_train(&self) -> Result<TensorTrain<V>> {
        Ok(self.tci.to_tensor_train()?)
    }

    /// Access cached evaluation points.
    ///
    /// Returns a map from quantics indices to function values.
    pub fn cachedata(&self) -> &HashMap<Vec<i64>, V> {
        &self.cache
    }

    /// Access cached evaluation points with original coordinates.
    ///
    /// Only available for discretized grids.
    /// Returns a vector of (coordinates, value) pairs since f64 is not hashable.
    pub fn cachedata_origcoord(&self) -> Result<Vec<(Vec<f64>, V)>>
    where
        V: Clone,
    {
        if let Some(grid) = &self.discretized_grid {
            let mut result = Vec::new();
            for (quantics, value) in &self.cache {
                let coord = grid
                    .quantics_to_origcoord(quantics)
                    .map_err(|e| anyhow!("Coordinate conversion error: {}", e))?;
                #[allow(clippy::clone_on_copy)]
                result.push((coord, value.clone()));
            }
            Ok(result)
        } else {
            Err(anyhow!(
                "Original coordinates only available for discretized grids"
            ))
        }
    }
}

/// Interpolate a function with an explicit Grid.
///
/// # Arguments
/// * `grid` - Discretized grid describing the function domain
/// * `f` - Function to interpolate, takes original coordinates
/// * `initial_pivots` - Initial pivot grid indices (optional)
/// * `options` - TCI options
///
/// # Returns
/// Tuple of (QuanticsTensorCI2, ranks, errors)
pub fn quanticscrossinterpolate<V, F>(
    grid: &DiscretizedGrid,
    f: F,
    initial_pivots: Option<Vec<Vec<i64>>>,
    options: QtciOptions,
) -> Result<(QuanticsTensorCI2<V>, Vec<usize>, Vec<f64>)>
where
    V: Scalar + TTScalar + Default + Clone + 'static,
    F: Fn(&[f64]) -> V + 'static,
{
    let local_dims = grid.local_dimensions();

    // Use RefCell to allow mutation from within the closure
    let cache: Rc<RefCell<HashMap<Vec<i64>, V>>> = Rc::new(RefCell::new(HashMap::new()));
    let cache_clone = cache.clone();

    // Wrap function to accept quantics indices (usize 0-indexed for TCI)
    let grid_clone = grid.clone();
    let qf = move |q: &Vec<usize>| -> V {
        // Convert 0-indexed TCI values to 1-indexed for quanticsgrids
        let q_i64: Vec<i64> = q.iter().map(|&x| (x + 1) as i64).collect();

        // Check cache first
        if let Some(v) = cache_clone.borrow().get(&q_i64) {
            #[allow(clippy::clone_on_copy)]
            return v.clone();
        }

        // Compute and cache
        let coords = match grid_clone.quantics_to_origcoord(&q_i64) {
            Ok(coords) => coords,
            Err(err) => {
                debug_assert!(
                    false,
                    "Quantics index conversion failed for {:?}: {}",
                    q_i64, err
                );
                return V::default();
            }
        };
        let value = f(&coords);
        #[allow(clippy::clone_on_copy)]
        cache_clone.borrow_mut().insert(q_i64, value.clone());
        value
    };

    // Prepare initial pivots
    let mut qinitialpivots: Vec<Vec<usize>> = if let Some(pivots) = initial_pivots {
        pivots
            .iter()
            .filter_map(|p| {
                grid.grididx_to_quantics(p)
                    .ok()
                    // Convert 1-indexed quantics to 0-indexed for TCI
                    .map(|q| q.iter().map(|&x| (x - 1) as usize).collect())
            })
            .collect()
    } else {
        // Default to first grid point (0-indexed for TCI)
        vec![vec![0; local_dims.len()]]
    };

    // Add random initial pivots (0-indexed for TCI)
    let mut rng = rand::rng();
    for _ in 0..options.nrandominitpivot {
        let pivot: Vec<usize> = local_dims.iter().map(|&d| rng.random_range(0..d)).collect();
        qinitialpivots.push(pivot);
    }

    // Run TCI
    let tci_opts = options.to_tci2_options();
    let (tci, ranks, errors) = crossinterpolate2(
        qf,
        None::<fn(&[Vec<usize>]) -> Vec<V>>,
        local_dims,
        qinitialpivots,
        tci_opts,
    )?;

    // Extract cache
    let final_cache = Rc::try_unwrap(cache)
        .map_err(|_| anyhow!("Failed to extract cache"))?
        .into_inner();

    Ok((
        QuanticsTensorCI2::from_discretized(tci, grid.clone(), final_cache),
        ranks,
        errors,
    ))
}

/// Interpolate from grid point arrays.
///
/// # Arguments
/// * `xvals` - Arrays of grid points for each dimension
/// * `f` - Function to interpolate
/// * `initial_pivots` - Initial pivot grid indices (optional)
/// * `options` - TCI options
///
/// # Returns
/// Tuple of (QuanticsTensorCI2, ranks, errors)
pub fn quanticscrossinterpolate_from_arrays<V, F>(
    xvals: &[Vec<f64>],
    f: F,
    initial_pivots: Option<Vec<Vec<i64>>>,
    options: QtciOptions,
) -> Result<(QuanticsTensorCI2<V>, Vec<usize>, Vec<f64>)>
where
    V: Scalar + TTScalar + Default + Clone + 'static,
    F: Fn(&[f64]) -> V + 'static,
{
    if xvals.is_empty() {
        return Err(anyhow!("xvals must not be empty"));
    }
    if xvals.iter().any(|x| x.is_empty()) {
        return Err(anyhow!("xvals must not contain empty dimensions"));
    }

    // Validate inputs
    let dimensions: Vec<f64> = xvals.iter().map(|x| (x.len() as f64).log2()).collect();

    // Check all dimensions are equal (current limitation)
    if !dimensions.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10) {
        return Err(anyhow!(
            "This method only supports grids with equal number of points in each direction"
        ));
    }

    // Check dimensions are powers of 2
    if !dimensions.iter().all(|&d| (d - d.round()).abs() < 1e-10) {
        return Err(anyhow!(
            "This method only supports grid sizes that are powers of 2"
        ));
    }

    let rs: Vec<usize> = dimensions.iter().map(|&d| d as usize).collect();
    let lower: Vec<f64> = xvals
        .iter()
        .map(|x| {
            x.first()
                .copied()
                .ok_or_else(|| anyhow!("xvals must not be empty"))
        })
        .collect::<Result<Vec<f64>>>()?;
    let upper: Vec<f64> = xvals
        .iter()
        .map(|x| {
            x.last()
                .copied()
                .ok_or_else(|| anyhow!("xvals must not be empty"))
        })
        .collect::<Result<Vec<f64>>>()?;

    // Build grid
    let grid = DiscretizedGrid::builder(&rs)
        .with_lower_bound(&lower)
        .with_upper_bound(&upper)
        .with_unfolding_scheme(options.unfoldingscheme)
        .include_endpoint(true)
        .build()
        .map_err(|e| anyhow!("Failed to build grid: {}", e))?;

    quanticscrossinterpolate(&grid, f, initial_pivots, options)
}

/// Interpolate with discrete integer grid.
///
/// # Arguments
/// * `size` - Grid size in each dimension (must be powers of 2)
/// * `f` - Function to interpolate, takes grid indices (1-indexed)
/// * `initial_pivots` - Initial pivot grid indices (optional)
/// * `options` - TCI options
///
/// # Returns
/// Tuple of (QuanticsTensorCI2, ranks, errors)
pub fn quanticscrossinterpolate_discrete<V, F>(
    size: &[usize],
    f: F,
    initial_pivots: Option<Vec<Vec<i64>>>,
    options: QtciOptions,
) -> Result<(QuanticsTensorCI2<V>, Vec<usize>, Vec<f64>)>
where
    V: Scalar + TTScalar + Default + Clone + 'static,
    F: Fn(&[i64]) -> V + 'static,
{
    // Validate sizes are powers of 2
    let dimensions: Vec<f64> = size.iter().map(|&s| (s as f64).log2()).collect();

    if !dimensions.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10) {
        return Err(anyhow!(
            "This method only supports grids with equal number of points in each direction"
        ));
    }

    if !dimensions.iter().all(|&d| (d - d.round()).abs() < 1e-10) {
        return Err(anyhow!(
            "This method only supports grid sizes that are powers of 2"
        ));
    }

    let r = dimensions[0] as usize;
    let n = size.len();

    // Build inherent discrete grid - rs is the number of bits per variable
    let rs: Vec<usize> = vec![r; n];
    let grid = InherentDiscreteGrid::builder(&rs)
        .with_unfolding_scheme(options.unfoldingscheme)
        .build()
        .map_err(|e| anyhow!("Failed to build grid: {}", e))?;

    let local_dims = grid.local_dimensions();

    // Use RefCell to allow mutation from within the closure
    let cache: Rc<RefCell<HashMap<Vec<i64>, V>>> = Rc::new(RefCell::new(HashMap::new()));
    let cache_clone = cache.clone();

    // Wrap function to accept quantics indices (usize 0-indexed for TCI)
    let grid_clone = grid.clone();
    let qf = move |q: &Vec<usize>| -> V {
        // Convert 0-indexed TCI values to 1-indexed for quanticsgrids
        let q_i64: Vec<i64> = q.iter().map(|&x| (x + 1) as i64).collect();

        // Check cache first
        if let Some(v) = cache_clone.borrow().get(&q_i64) {
            #[allow(clippy::clone_on_copy)]
            return v.clone();
        }

        // Compute and cache
        let grididx = grid_clone.quantics_to_grididx(&q_i64).unwrap();
        let value = f(&grididx);
        #[allow(clippy::clone_on_copy)]
        cache_clone.borrow_mut().insert(q_i64, value.clone());
        value
    };

    // Prepare initial pivots
    let mut qinitialpivots: Vec<Vec<usize>> = if let Some(pivots) = initial_pivots {
        pivots
            .iter()
            .filter_map(|p| {
                grid.grididx_to_quantics(p)
                    .ok()
                    // Convert 1-indexed quantics to 0-indexed for TCI
                    .map(|q| q.iter().map(|&x| (x - 1) as usize).collect())
            })
            .collect()
    } else {
        // Default to first grid point (0-indexed for TCI)
        vec![vec![0; local_dims.len()]]
    };

    // Add random initial pivots (0-indexed for TCI)
    let mut rng = rand::rng();
    for _ in 0..options.nrandominitpivot {
        let pivot: Vec<usize> = local_dims.iter().map(|&d| rng.random_range(0..d)).collect();
        qinitialpivots.push(pivot);
    }

    // Run TCI
    let tci_opts = options.to_tci2_options();
    let (tci, ranks, errors) = crossinterpolate2(
        qf,
        None::<fn(&[Vec<usize>]) -> Vec<V>>,
        local_dims,
        qinitialpivots,
        tci_opts,
    )?;

    // Extract cache
    let final_cache = Rc::try_unwrap(cache)
        .map_err(|_| anyhow!("Failed to extract cache"))?
        .into_inner();

    Ok((
        QuanticsTensorCI2::from_inherent(tci, grid, final_cache),
        ranks,
        errors,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use quanticsgrids::UnfoldingScheme;

    #[test]
    fn test_discrete_simple_function() {
        // f(i, j) = i + j (grididx are 1-indexed)
        // Use 4x4 grid which gives 2 sites with Fused scheme
        let f = |idx: &[i64]| (idx[0] + idx[1]) as f64;
        let sizes = vec![4, 4];

        // Use Fused to get 2 sites (4x4 = 2 bits, so 2 sites for 2D)
        let opts = QtciOptions::default()
            .with_tolerance(1e-10)
            .with_nrandominitpivot(3)
            .with_unfoldingscheme(UnfoldingScheme::Fused);

        let result = quanticscrossinterpolate_discrete(&sizes, f, None, opts);
        assert!(result.is_ok(), "Error: {:?}", result.err());

        let (qtci, _ranks, _errors) = result.unwrap();

        // Verify some evaluations (grididx are 1-indexed)
        let val = qtci.evaluate(&[3, 4]).unwrap();
        assert_relative_eq!(val, 7.0, epsilon = 1e-8);

        let val = qtci.evaluate(&[1, 1]).unwrap();
        assert_relative_eq!(val, 2.0, epsilon = 1e-8);

        // Rank should be low for this simple function (i + j is rank 2)
        assert!(qtci.rank() <= 3);
    }

    #[test]
    fn test_discrete_tci_structure() {
        // Test that the QTCI structure (bonds, rank, cache, sum) works correctly.
        // f(i,j) = i + j on a 4x4 grid with Fused scheme.
        let f = |idx: &[i64]| (idx[0] + idx[1]) as f64;
        let sizes = vec![4, 4];

        let opts = QtciOptions::default()
            .with_tolerance(1e-10)
            .with_nrandominitpivot(3)
            .with_unfoldingscheme(UnfoldingScheme::Fused);

        let result = quanticscrossinterpolate_discrete(&sizes, f, None, opts);
        assert!(result.is_ok(), "Error: {:?}", result.err());

        let (qtci, _ranks, _errors) = result.unwrap();

        // Verify the structure is created correctly
        assert_eq!(qtci.link_dims().len(), 1); // 2 sites = 1 bond
        assert!(qtci.rank() > 0);

        // Verify that the function was called with correct grid indices by checking
        // all cached values. The cache maps quantics indices to function values.
        let grid = qtci.inherent_grid().unwrap();
        for (quantics_idx, &cached_val) in qtci.cachedata() {
            let grid_idx = grid.quantics_to_grididx(quantics_idx).unwrap();
            let expected = (grid_idx[0] + grid_idx[1]) as f64;
            assert_relative_eq!(cached_val, expected, epsilon = 1e-10);
        }
        assert!(!qtci.cachedata().is_empty());

        // Verify evaluate() matches f at known-exact points (same block in
        // quantics representation).
        let val = qtci.evaluate(&[1, 1]).unwrap();
        assert_relative_eq!(val, 2.0, epsilon = 1e-8);
        let val = qtci.evaluate(&[3, 4]).unwrap();
        assert_relative_eq!(val, 7.0, epsilon = 1e-8);
        let val = qtci.evaluate(&[4, 4]).unwrap();
        assert_relative_eq!(val, 8.0, epsilon = 1e-8);
    }

    #[test]
    fn test_size_validation() {
        let f = |_idx: &[i64]| 1.0_f64;

        // Non-power of 2 should fail
        let sizes = vec![5, 5];
        let result = quanticscrossinterpolate_discrete(&sizes, f, None, QtciOptions::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_from_arrays_empty_inputs() {
        let f = |_coords: &[f64]| 1.0_f64;
        let result =
            quanticscrossinterpolate_from_arrays::<f64, _>(&[], f, None, QtciOptions::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_from_arrays_empty_dimension() {
        let f = |_coords: &[f64]| 1.0_f64;
        let xvals = vec![vec![], vec![0.0, 1.0]];
        let result =
            quanticscrossinterpolate_from_arrays::<f64, _>(&xvals, f, None, QtciOptions::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_options_builder() {
        let opts = QtciOptions::default()
            .with_tolerance(1e-6)
            .with_maxbonddim(50)
            .with_unfoldingscheme(UnfoldingScheme::Fused);

        assert!((opts.tolerance - 1e-6).abs() < 1e-15);
        assert_eq!(opts.maxbonddim, Some(50));
        assert_eq!(opts.unfoldingscheme, UnfoldingScheme::Fused);
    }

    #[test]
    fn test_discrete_inherent_grid_accessor() {
        // quanticscrossinterpolate_discrete uses from_inherent internally.
        // Verify that inherent_grid() returns Some and discretized_grid() returns None.
        let f = |idx: &[i64]| (idx[0] + idx[1]) as f64;
        let sizes = vec![4, 4];

        let opts = QtciOptions::default()
            .with_tolerance(1e-10)
            .with_nrandominitpivot(3)
            .with_unfoldingscheme(UnfoldingScheme::Fused);

        let (qtci, _ranks, _errors) =
            quanticscrossinterpolate_discrete(&sizes, f, None, opts).unwrap();

        // inherent_grid should be Some, discretized_grid should be None
        assert!(qtci.inherent_grid().is_some());
        assert!(qtci.discretized_grid().is_none());

        // Verify that cached function values are correct, proving the grid
        // coordinate mapping works for inherent discrete grids.
        let grid = qtci.inherent_grid().unwrap();
        for (quantics_idx, &cached_val) in qtci.cachedata() {
            let grid_idx = grid.quantics_to_grididx(quantics_idx).unwrap();
            let expected = (grid_idx[0] + grid_idx[1]) as f64;
            assert_relative_eq!(cached_val, expected, epsilon = 1e-10);
        }

        // Verify evaluate() at known-exact points
        let val = qtci.evaluate(&[1, 1]).unwrap();
        assert_relative_eq!(val, 2.0, epsilon = 1e-8);
        let val = qtci.evaluate(&[4, 4]).unwrap();
        assert_relative_eq!(val, 8.0, epsilon = 1e-8);
    }

    #[test]
    fn test_discrete_cachedata_origcoord_error() {
        // cachedata_origcoord() should return an error for inherent discrete grids
        // because there are no original continuous coordinates.
        let f = |idx: &[i64]| idx[0] as f64;
        let sizes = vec![4];

        let opts = QtciOptions::default()
            .with_tolerance(1e-10)
            .with_nrandominitpivot(1)
            .with_unfoldingscheme(UnfoldingScheme::Fused);

        let (qtci, _ranks, _errors) =
            quanticscrossinterpolate_discrete(&sizes, f, None, opts).unwrap();

        let result = qtci.cachedata_origcoord();
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Original coordinates only available for discretized grids"));
    }

    #[test]
    fn test_discrete_integral_returns_sum() {
        // For inherent discrete grids, integral() should just return the sum.
        // f(i) = 1 for all i, on a grid of size 4 => sum = 4
        let f = |_idx: &[i64]| 1.0_f64;
        let sizes = vec![4];

        let opts = QtciOptions::default()
            .with_tolerance(1e-10)
            .with_nrandominitpivot(3)
            .with_unfoldingscheme(UnfoldingScheme::Fused);

        let (qtci, _ranks, _errors) =
            quanticscrossinterpolate_discrete(&sizes, f, None, opts).unwrap();

        let integral = qtci.integral().unwrap();
        let sum = qtci.sum().unwrap();
        // For inherent grids, integral == sum
        assert_relative_eq!(integral, sum, epsilon = 1e-10);
        assert_relative_eq!(integral, 4.0, epsilon = 1e-8);
    }

    #[test]
    fn test_continuous_grid_interpolation() {
        // Test quanticscrossinterpolate with a DiscretizedGrid.
        // f(x) = x^2 on [0, 1], 8 grid points (3 bits)
        let grid = DiscretizedGrid::builder(&[3])
            .with_lower_bound(&[0.0])
            .with_upper_bound(&[1.0])
            .include_endpoint(true)
            .build()
            .unwrap();

        let f = |coords: &[f64]| coords[0] * coords[0];

        let opts = QtciOptions::default()
            .with_tolerance(1e-12)
            .with_nrandominitpivot(5)
            .with_unfoldingscheme(UnfoldingScheme::Interleaved);

        let (qtci, _ranks, _errors) = quanticscrossinterpolate(&grid, f, None, opts).unwrap();

        // Verify accessors
        assert!(qtci.discretized_grid().is_some());
        assert!(qtci.inherent_grid().is_none());
        assert!(qtci.rank() > 0);

        // cachedata should have some entries
        assert!(!qtci.cachedata().is_empty());

        // Verify cached function values via cachedata_origcoord.
        // Each cached point should have the correct original coordinate and
        // function value f(x) = x^2.
        let origcoord_data = qtci.cachedata_origcoord().unwrap();
        assert!(!origcoord_data.is_empty());
        for (coord, val) in &origcoord_data {
            assert_eq!(coord.len(), 1);
            let x = coord[0];
            let expected = x * x;
            assert!(
                (val - expected).abs() < 1e-10,
                "cached f({}) = {}, expected {}",
                x,
                val,
                expected
            );
        }

        // Verify evaluate() produces finite values at grid endpoints
        let val = qtci.evaluate(&[1]).unwrap();
        assert!(val.is_finite());
        let val = qtci.evaluate(&[8]).unwrap();
        assert!(val.is_finite());
    }

    #[test]
    fn test_continuous_grid_integral() {
        // Integral of f(x) = 1 over [0, 1] with 16 points should be ~1.0
        // (sum of 16 ones * step = 16 * (1/16) = 1.0 for non-endpoint grids)
        let grid = DiscretizedGrid::builder(&[4]) // 2^4 = 16 points
            .with_lower_bound(&[0.0])
            .with_upper_bound(&[1.0])
            .build()
            .unwrap();

        let f = |_coords: &[f64]| 1.0_f64;

        let opts = QtciOptions::default()
            .with_tolerance(1e-12)
            .with_nrandominitpivot(3);

        let (qtci, _ranks, _errors) = quanticscrossinterpolate(&grid, f, None, opts).unwrap();

        let integral = qtci.integral().unwrap();
        // integral = sum * step_size
        // sum = 16.0, step_size = 1/16 = 0.0625 => integral = 1.0
        assert_relative_eq!(integral, 1.0, epsilon = 1e-8);
    }

    #[test]
    fn test_discrete_with_initial_pivots() {
        // Test that initial pivots are correctly converted and the TCI runs successfully.
        let f = |idx: &[i64]| (idx[0] * idx[1]) as f64;
        let sizes = vec![4, 4];

        let opts = QtciOptions::default()
            .with_tolerance(1e-10)
            .with_nrandominitpivot(3)
            .with_unfoldingscheme(UnfoldingScheme::Fused);

        // Provide explicit initial pivots (1-indexed grid indices)
        let pivots = vec![vec![1, 1], vec![2, 3]];
        let result = quanticscrossinterpolate_discrete(&sizes, f, Some(pivots), opts);
        assert!(result.is_ok(), "Error: {:?}", result.err());

        let (qtci, _ranks, _errors) = result.unwrap();

        // Verify cached values match f(i,j) = i*j, proving the function was
        // called with correct grid indices from the initial pivots and TCI sweep.
        let grid = qtci.inherent_grid().unwrap();
        for (quantics_idx, &cached_val) in qtci.cachedata() {
            let grid_idx = grid.quantics_to_grididx(quantics_idx).unwrap();
            let expected = (grid_idx[0] * grid_idx[1]) as f64;
            assert_relative_eq!(cached_val, expected, epsilon = 1e-10);
        }
        assert!(!qtci.cachedata().is_empty());

        // Verify evaluate() at known-exact points
        let val = qtci.evaluate(&[1, 1]).unwrap();
        assert_relative_eq!(val, 1.0, epsilon = 1e-8);
        let val = qtci.evaluate(&[4, 4]).unwrap();
        assert_relative_eq!(val, 16.0, epsilon = 1e-8);
    }

    #[test]
    fn test_continuous_grid_with_initial_pivots() {
        // Test quanticscrossinterpolate with initial pivots.
        let grid = DiscretizedGrid::builder(&[3])
            .with_lower_bound(&[0.0])
            .with_upper_bound(&[1.0])
            .include_endpoint(true)
            .build()
            .unwrap();

        let f = |coords: &[f64]| coords[0];

        let opts = QtciOptions::default()
            .with_tolerance(1e-12)
            .with_nrandominitpivot(3);

        let pivots = vec![vec![1], vec![4]];
        let result = quanticscrossinterpolate(&grid, f, Some(pivots), opts);
        assert!(result.is_ok(), "Error: {:?}", result.err());

        let (qtci, _ranks, _errors) = result.unwrap();

        // Verify cached function values via cachedata_origcoord.
        // Each cached point should store f(x) = x correctly.
        let origcoord_data = qtci.cachedata_origcoord().unwrap();
        assert!(!origcoord_data.is_empty());
        for (coord, val) in &origcoord_data {
            assert_eq!(coord.len(), 1);
            let x = coord[0];
            assert!(
                (val - x).abs() < 1e-10,
                "cached f({}) = {}, expected {}",
                x,
                val,
                x
            );
        }

        // Verify evaluate() produces finite values
        let val = qtci.evaluate(&[1]).unwrap();
        assert!(val.is_finite());
        let val = qtci.evaluate(&[8]).unwrap();
        assert!(val.is_finite());
    }

    #[test]
    fn test_from_arrays_non_power_of_two() {
        let f = |_coords: &[f64]| 1.0_f64;
        let xvals = vec![vec![0.0, 1.0, 2.0]]; // 3 points, not power of 2
        let result =
            quanticscrossinterpolate_from_arrays::<f64, _>(&xvals, f, None, QtciOptions::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_from_arrays_unequal_dimensions() {
        let f = |_coords: &[f64]| 1.0_f64;
        // 4 points vs 8 points => different dimensions
        let xvals = vec![
            vec![0.0, 1.0, 2.0, 3.0],
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        ];
        let result =
            quanticscrossinterpolate_from_arrays::<f64, _>(&xvals, f, None, QtciOptions::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_from_arrays_valid() {
        let f = |coords: &[f64]| coords[0] + coords[1];
        let xvals = vec![vec![0.0, 1.0, 2.0, 3.0], vec![0.0, 1.0, 2.0, 3.0]];

        let opts = QtciOptions::default()
            .with_tolerance(1e-10)
            .with_nrandominitpivot(3)
            .with_unfoldingscheme(UnfoldingScheme::Fused);

        let result = quanticscrossinterpolate_from_arrays::<f64, _>(&xvals, f, None, opts);
        assert!(result.is_ok(), "Error: {:?}", result.err());

        let (qtci, _ranks, _errors) = result.unwrap();
        assert!(qtci.discretized_grid().is_some());
        assert!(qtci.rank() > 0);

        // Verify cached function values via cachedata_origcoord.
        // Each cached point should store f(x,y) = x + y correctly.
        let origcoord_data = qtci.cachedata_origcoord().unwrap();
        assert!(!origcoord_data.is_empty());
        for (coord, val) in &origcoord_data {
            assert_eq!(coord.len(), 2);
            let expected = coord[0] + coord[1];
            assert!(
                (val - expected).abs() < 1e-10,
                "cached f({},{}) = {}, expected {}",
                coord[0],
                coord[1],
                val,
                expected
            );
        }

        // Verify evaluate() at known-exact points
        // xvals = [0,1,2,3], so grid (1,1) -> (0,0), f=0 and (4,4) -> (3,3), f=6
        let val = qtci.evaluate(&[1, 1]).unwrap();
        assert_relative_eq!(val, 0.0, epsilon = 1e-8);
        let val = qtci.evaluate(&[4, 4]).unwrap();
        assert_relative_eq!(val, 6.0, epsilon = 1e-8);
    }

    #[test]
    fn test_discrete_unequal_dimensions_error() {
        let f = |_idx: &[i64]| 1.0_f64;
        // 4 vs 8 => unequal
        let sizes = vec![4, 8];
        let result = quanticscrossinterpolate_discrete(&sizes, f, None, QtciOptions::default());
        assert!(result.is_err());
    }
}
