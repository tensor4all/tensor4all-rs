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
        let coords = grid_clone.quantics_to_origcoord(&q_i64).unwrap();
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
    let mut rng = rand::thread_rng();
    for _ in 0..options.nrandominitpivot {
        let pivot: Vec<usize> = local_dims.iter().map(|&d| rng.gen_range(0..d)).collect();
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
    let lower: Vec<f64> = xvals.iter().map(|x| *x.first().unwrap()).collect();
    let upper: Vec<f64> = xvals.iter().map(|x| *x.last().unwrap()).collect();

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
    let mut rng = rand::thread_rng();
    for _ in 0..options.nrandominitpivot {
        let pivot: Vec<usize> = local_dims.iter().map(|&d| rng.gen_range(0..d)).collect();
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
        // Test that the QTCI structure works correctly.
        // Note: TCI approximation accuracy depends on the function's rank
        // in quantics representation, which may differ from grid-space rank.
        let f = |idx: &[i64]| (idx[0] + idx[1]) as f64;
        let sizes = vec![4, 4];

        // Use Fused to get 2 sites (tested configuration for TCI2)
        let opts = QtciOptions::default()
            .with_tolerance(1e-12)
            .with_maxiter(100)
            .with_nrandominitpivot(10)
            .with_unfoldingscheme(UnfoldingScheme::Fused);

        let result = quanticscrossinterpolate_discrete(&sizes, f, None, opts);
        assert!(result.is_ok(), "Error: {:?}", result.err());

        let (qtci, _ranks, _errors) = result.unwrap();

        // Verify the structure is created correctly
        assert_eq!(qtci.link_dims().len(), 1); // 2 sites = 1 bond
        assert!(qtci.rank() > 0);

        // sum() should return a finite value
        let sum = qtci.sum().unwrap();
        assert!(sum.is_finite());
        assert!(sum > 0.0);

        // evaluate() should work at any grid point
        let val = qtci.evaluate(&[2, 3]).unwrap();
        assert!(val.is_finite());

        // cachedata should contain some evaluated points
        let cache = qtci.cachedata();
        assert!(!cache.is_empty());
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
    fn test_options_builder() {
        let opts = QtciOptions::default()
            .with_tolerance(1e-6)
            .with_maxbonddim(50)
            .with_unfoldingscheme(UnfoldingScheme::Fused);

        assert!((opts.tolerance - 1e-6).abs() < 1e-15);
        assert_eq!(opts.maxbonddim, Some(50));
        assert_eq!(opts.unfoldingscheme, UnfoldingScheme::Fused);
    }
}
