//! QuanticsTensorCI2 and interpolation functions.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use anyhow::{anyhow, Result};
use quanticsgrids::{DiscretizedGrid, InherentDiscreteGrid};
use rand::Rng;
use tensor4all_core::TensorDynLen;
use tensor4all_simplett::{tensor3_from_data, AbstractTensorTrain, TTScalar, TensorTrain};
use tensor4all_tcicore::{DenseFaerLuKernel, PivotKernel};
use tensor4all_treetci::materialize::{to_treetn, FullPivLuScalar};
use tensor4all_treetci::{
    optimize_with_proposer, DefaultProposer, GlobalIndexBatch, TreeTCI2, TreeTciGraph,
};

use crate::options::QtciOptions;

/// TCI result wrapped with grid information.
///
/// This struct combines a TensorTrain result with grid information for
/// seamless conversion between grid indices and quantics indices.
///
/// # Examples
///
/// ```
/// use tensor4all_quanticstci::{quanticscrossinterpolate, QtciOptions};
/// use quanticsgrids::DiscretizedGrid;
///
/// // Interpolate constant f(x) = 2.0 on [0, 1) with 2^3 = 8 grid points
/// let grid = DiscretizedGrid::builder(&[3])
///     .with_lower_bound(&[0.0])
///     .with_upper_bound(&[1.0])
///     .build()
///     .unwrap();
///
/// let f = |_coords: &[f64]| 2.0_f64;
/// let (qtci, _ranks, _errors) =
///     quanticscrossinterpolate::<f64, _>(&grid, f, None, QtciOptions::default()).unwrap();
///
/// // rank() gives the maximum bond dimension
/// assert!(qtci.rank() >= 1);
///
/// // Sum over all 8 grid points: 2.0 * 8 = 16.0
/// let sum = qtci.sum().unwrap();
/// assert!((sum - 16.0).abs() < 1e-8);
/// ```
#[derive(Clone)]
pub struct QuanticsTensorCI2<V: TTScalar> {
    /// Underlying tensor train
    tt: TensorTrain<V>,
    /// TreeTCI2 state (pivot sets, graph, etc.)
    tci_state: TreeTCI2<V>,
    /// Grid for coordinate conversion (DiscretizedGrid)
    discretized_grid: Option<DiscretizedGrid>,
    /// Grid for coordinate conversion (InherentDiscreteGrid)
    inherent_grid: Option<InherentDiscreteGrid>,
    /// Cached function values (quantics index -> value)
    cache: HashMap<Vec<i64>, V>,
}

impl<V> QuanticsTensorCI2<V>
where
    V: TTScalar + Default + Clone,
{
    /// Create a new QuanticsTensorCI2 from a TensorTrain, TreeTCI2 state, and discretized grid.
    pub fn from_discretized(
        tt: TensorTrain<V>,
        tci_state: TreeTCI2<V>,
        grid: DiscretizedGrid,
        cache: HashMap<Vec<i64>, V>,
    ) -> Self {
        Self {
            tt,
            tci_state,
            discretized_grid: Some(grid),
            inherent_grid: None,
            cache,
        }
    }

    /// Create a new QuanticsTensorCI2 from a TensorTrain, TreeTCI2 state, and inherent discrete grid.
    pub fn from_inherent(
        tt: TensorTrain<V>,
        tci_state: TreeTCI2<V>,
        grid: InherentDiscreteGrid,
        cache: HashMap<Vec<i64>, V>,
    ) -> Self {
        Self {
            tt,
            tci_state,
            discretized_grid: None,
            inherent_grid: Some(grid),
            cache,
        }
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
        self.tt.rank()
    }

    /// Get link dimensions.
    pub fn link_dims(&self) -> Vec<usize> {
        self.tt.link_dims()
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
        // Convert 1-indexed i64 quantics to 0-indexed usize for tensor train evaluation
        let quantics_usize: Vec<usize> = quantics.iter().map(|&x| (x - 1) as usize).collect();
        self.tt
            .evaluate(&quantics_usize)
            .map_err(|e| anyhow!("Evaluation error: {}", e))
    }

    /// Factorized sum over all grid points.
    ///
    /// This computes the sum efficiently using the tensor train structure.
    pub fn sum(&self) -> Result<V> {
        Ok(self.tt.sum())
    }

    /// Integral over continuous domain (left Riemann sum).
    ///
    /// Computes `sum(f(x_i)) * product(step_sizes)`, which is a left Riemann sum
    /// with O(h) convergence where h is the grid spacing. The result depends on the
    /// `include_endpoint` setting of the `DiscretizedGrid`.
    ///
    /// For inherent discrete grids (no continuous domain), returns the plain sum.
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
    pub fn tensor_train(&self) -> TensorTrain<V> {
        self.tt.clone()
    }

    /// Access the TreeTCI2 state.
    pub fn tci(&self) -> &TreeTCI2<V> {
        &self.tci_state
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

/// Convert a linear-chain TreeTN to a SimpleTT TensorTrain.
///
/// The TreeTN must have been produced by treetci::crossinterpolate2 with a
/// linear chain graph and center_site=0. Nodes are numbered 0..n-1.
///
/// TreeTN tensors from `to_treetn` have index order:
///   [site_dim, incoming_bond_dims..., outgoing_bond_dims...]
/// where incoming = children, outgoing = parent in BFS from root=0.
///
/// For a linear chain rooted at 0:
///   - Node 0 (root): [site, bond_01]
///   - Node k (middle): [site, bond_{k,k+1}, bond_{k-1,k}]
///   - Node n-1 (leaf): [site, bond_{n-2,n-1}]
///
/// For SimpleTT we need (left_bond, site_dim, right_bond).
fn treetn_to_tensor_train<V>(
    treetn: &tensor4all_treetn::TreeTN<TensorDynLen, usize>,
    n_sites: usize,
    local_dims: &[usize],
) -> Result<TensorTrain<V>>
where
    V: TTScalar + Default + Clone + tensor4all_core::TensorElement,
{
    let mut tensors = Vec::with_capacity(n_sites);

    for (site, &site_dim) in local_dims.iter().enumerate().take(n_sites) {
        let node_idx = treetn
            .node_index(&site)
            .ok_or_else(|| anyhow!("node {} not found in TreeTN", site))?;
        let tensor = treetn
            .tensor(node_idx)
            .ok_or_else(|| anyhow!("tensor not found at node {}", site))?;

        let dims = tensor.dims();
        let data: Vec<V> = tensor
            .to_vec::<V>()
            .map_err(|e| anyhow!("failed to extract tensor data at node {}: {}", site, e))?;

        if n_sites == 1 {
            // Single site: tensor has only site index, shape (site_dim,)
            // Need: (1, site_dim, 1)
            tensors.push(tensor3_from_data(data, 1, site_dim, 1));
        } else if site == 0 {
            // Root (leftmost): indices = [site, bond_01]
            // Data is column-major with shape (site_dim, bond_dim)
            // Need: (1, site_dim, bond_dim)
            assert_eq!(dims.len(), 2, "root node should have 2 indices");
            let bond_dim = dims[1];
            // Column-major (site, bond): data[s + site_dim * b]
            // Target (1, site, bond): data[0 + 1*(s + site_dim * b)] — same layout
            tensors.push(tensor3_from_data(data, 1, site_dim, bond_dim));
        } else if site == n_sites - 1 {
            // Leaf (rightmost): indices = [site, bond_{n-2,n-1}]
            // Data is column-major with shape (site_dim, left_bond)
            // Need: (left_bond, site_dim, 1)
            assert_eq!(dims.len(), 2, "leaf node should have 2 indices");
            let left_bond = dims[1];
            // Permute: (site, left) → (left, site)
            let mut permuted = vec![V::default(); data.len()];
            for l in 0..left_bond {
                for s in 0..site_dim {
                    permuted[l + left_bond * s] = data[s + site_dim * l];
                }
            }
            tensors.push(tensor3_from_data(permuted, left_bond, site_dim, 1));
        } else {
            // Middle node: indices = [site, bond_{k,k+1}, bond_{k-1,k}]
            // Data is column-major with shape (site_dim, right_bond, left_bond)
            // Need: (left_bond, site_dim, right_bond)
            assert_eq!(dims.len(), 3, "middle node should have 3 indices");
            let right_bond = dims[1];
            let left_bond = dims[2];
            let total = data.len();
            assert_eq!(
                total,
                site_dim * right_bond * left_bond,
                "data size mismatch for middle node {}: expected {}*{}*{}={}, got {}",
                site,
                site_dim,
                right_bond,
                left_bond,
                site_dim * right_bond * left_bond,
                total
            );
            // Permute: (site, right, left) → (left, site, right)
            // Source col-major: data[s + site_dim * (r + right_bond * l)]
            // Target col-major: permuted[l + left_bond * (s + site_dim * r)]
            let mut permuted = vec![V::default(); total];
            for l in 0..left_bond {
                for s in 0..site_dim {
                    for r in 0..right_bond {
                        let src = s + site_dim * (r + right_bond * l);
                        let dst = l + left_bond * (s + site_dim * r);
                        permuted[dst] = data[src];
                    }
                }
            }
            tensors.push(tensor3_from_data(permuted, left_bond, site_dim, right_bond));
        }
    }

    TensorTrain::new(tensors).map_err(|e| anyhow!("Failed to build TensorTrain: {}", e))
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
///
/// # Examples
///
/// ```
/// use tensor4all_quanticstci::{quanticscrossinterpolate, QtciOptions};
/// use quanticsgrids::DiscretizedGrid;
///
/// // Interpolate f(x) = sin(x) on [0, pi) with 2^4 = 16 points
/// let grid = DiscretizedGrid::builder(&[4])
///     .with_lower_bound(&[0.0])
///     .with_upper_bound(&[std::f64::consts::PI])
///     .build()
///     .unwrap();
///
/// let f = |coords: &[f64]| coords[0].sin();
/// let opts = QtciOptions::default().with_tolerance(1e-8);
/// let (qtci, _ranks, errors) =
///     quanticscrossinterpolate::<f64, _>(&grid, f, None, opts).unwrap();
///
/// // Last error should be within tolerance
/// assert!(*errors.last().unwrap() < 1e-6);
///
/// // Sum (integral * step) approximates the Riemann sum of sin(x)
/// let sum = qtci.sum().unwrap();
/// assert!(sum > 0.0); // sin(x) > 0 on (0, pi)
/// ```
pub fn quanticscrossinterpolate<V, F>(
    grid: &DiscretizedGrid,
    f: F,
    initial_pivots: Option<Vec<Vec<i64>>>,
    options: QtciOptions,
) -> Result<(QuanticsTensorCI2<V>, Vec<usize>, Vec<f64>)>
where
    V: TTScalar + Default + Clone + 'static + tensor4all_core::TensorElement + FullPivLuScalar,
    DenseFaerLuKernel: PivotKernel<V>,
    F: Fn(&[f64]) -> V + 'static,
{
    let local_dims = grid.local_dimensions();
    let n_sites = local_dims.len();

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

    // Batch adapter: treetci expects Fn(GlobalIndexBatch) -> Result<Vec<V>>
    let batch_eval = move |batch: GlobalIndexBatch<'_>| -> Result<Vec<V>> {
        let n_points = batch.n_points();
        let n = batch.n_sites();
        let mut results = Vec::with_capacity(n_points);
        for p in 0..n_points {
            let point: Vec<usize> = (0..n)
                .map(|s| batch.get(s, p).expect("valid batch index"))
                .collect();
            results.push(qf(&point));
        }
        Ok(results)
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
        vec![vec![0; n_sites]]
    };

    // Add random initial pivots (0-indexed for TCI)
    let mut rng = rand::rng();
    for _ in 0..options.nrandominitpivot {
        let pivot: Vec<usize> = local_dims.iter().map(|&d| rng.random_range(0..d)).collect();
        qinitialpivots.push(pivot);
    }

    // Run TreeTCI with linear chain (lower-level API)
    let graph = TreeTciGraph::linear_chain(n_sites)?;
    let tree_opts = options.to_treetci_options();
    let proposer = DefaultProposer;

    let pivots = if qinitialpivots.is_empty() {
        vec![vec![0; local_dims.len()]]
    } else {
        qinitialpivots
    };

    let mut tci = TreeTCI2::<V>::new(local_dims.clone(), graph)?;
    tci.add_global_pivots(&pivots)?;

    // Initialize max_sample_value via batch evaluate
    let flat: Vec<usize> = pivots.iter().flat_map(|p| p.iter().copied()).collect();
    let init_batch = GlobalIndexBatch::new(&flat, n_sites, pivots.len())?;
    let init_vals = batch_eval(init_batch)?;
    tci.max_sample_value = init_vals
        .iter()
        .map(|v| <V as tensor4all_tcicore::MatrixLuciScalar>::abs_val(*v))
        .fold(0.0f64, f64::max);
    anyhow::ensure!(
        tci.max_sample_value > 0.0,
        "initial pivots must not all evaluate to zero"
    );

    let (ranks, errors) = optimize_with_proposer(&mut tci, &batch_eval, &tree_opts, &proposer)?;
    let treetn = to_treetn(&tci, &batch_eval, Some(0))?;

    // Convert TreeTN → TensorTrain<V>
    let tt = treetn_to_tensor_train::<V>(&treetn, n_sites, &local_dims)?;

    // Drop batch_eval (and its captured Rc clone) before extracting the cache
    drop(batch_eval);

    let final_cache = Rc::try_unwrap(cache)
        .map_err(|_| anyhow!("Failed to extract cache"))?
        .into_inner();

    Ok((
        QuanticsTensorCI2::from_discretized(tt, tci, grid.clone(), final_cache),
        ranks,
        errors,
    ))
}

/// Interpolate from grid point arrays.
///
/// # Arguments
/// * `xvals` - Arrays of grid points for each dimension. All dimensions must have the
///   **same** number of points and each must be a power of 2.
/// * `f` - Function to interpolate, takes original coordinates as `&[f64]`
/// * `initial_pivots` - Initial pivot grid indices (1-indexed, optional)
/// * `options` - TCI options
///
/// # Returns
/// Tuple of (QuanticsTensorCI2, ranks, errors)
///
/// # Errors
/// Returns an error if dimensions are not equal or not powers of 2.
pub fn quanticscrossinterpolate_from_arrays<V, F>(
    xvals: &[Vec<f64>],
    f: F,
    initial_pivots: Option<Vec<Vec<i64>>>,
    options: QtciOptions,
) -> Result<(QuanticsTensorCI2<V>, Vec<usize>, Vec<f64>)>
where
    V: TTScalar + Default + Clone + 'static + tensor4all_core::TensorElement + FullPivLuScalar,
    DenseFaerLuKernel: PivotKernel<V>,
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
/// * `size` - Grid size in each dimension. All dimensions must have the **same** number of
///   points and each must be a power of 2 (e.g., `vec![16, 16]`).
/// * `f` - Function to interpolate, taking **1-indexed** grid indices as `&[i64]`
/// * `initial_pivots` - Initial pivot grid indices (1-indexed, optional)
/// * `options` - TCI options
///
/// # Returns
/// Tuple of (QuanticsTensorCI2, ranks, errors)
///
/// # Errors
/// Returns an error if dimensions are not equal or not powers of 2.
pub fn quanticscrossinterpolate_discrete<V, F>(
    size: &[usize],
    f: F,
    initial_pivots: Option<Vec<Vec<i64>>>,
    options: QtciOptions,
) -> Result<(QuanticsTensorCI2<V>, Vec<usize>, Vec<f64>)>
where
    V: TTScalar + Default + Clone + 'static + tensor4all_core::TensorElement + FullPivLuScalar,
    DenseFaerLuKernel: PivotKernel<V>,
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
    let n_sites = local_dims.len();

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

    // Batch adapter: treetci expects Fn(GlobalIndexBatch) -> Result<Vec<V>>
    let batch_eval = move |batch: GlobalIndexBatch<'_>| -> Result<Vec<V>> {
        let n_points = batch.n_points();
        let n = batch.n_sites();
        let mut results = Vec::with_capacity(n_points);
        for p in 0..n_points {
            let point: Vec<usize> = (0..n)
                .map(|s| batch.get(s, p).expect("valid batch index"))
                .collect();
            results.push(qf(&point));
        }
        Ok(results)
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

    // Run TreeTCI with linear chain (lower-level API)
    let graph = TreeTciGraph::linear_chain(n_sites)?;
    let tree_opts = options.to_treetci_options();
    let proposer = DefaultProposer;

    let pivots = if qinitialpivots.is_empty() {
        vec![vec![0; local_dims.len()]]
    } else {
        qinitialpivots
    };

    let mut tci = TreeTCI2::<V>::new(local_dims.clone(), graph)?;
    tci.add_global_pivots(&pivots)?;

    // Initialize max_sample_value via batch evaluate
    let flat: Vec<usize> = pivots.iter().flat_map(|p| p.iter().copied()).collect();
    let init_batch = GlobalIndexBatch::new(&flat, n_sites, pivots.len())?;
    let init_vals = batch_eval(init_batch)?;
    tci.max_sample_value = init_vals
        .iter()
        .map(|v| <V as tensor4all_tcicore::MatrixLuciScalar>::abs_val(*v))
        .fold(0.0f64, f64::max);
    anyhow::ensure!(
        tci.max_sample_value > 0.0,
        "initial pivots must not all evaluate to zero"
    );

    let (ranks, errors) = optimize_with_proposer(&mut tci, &batch_eval, &tree_opts, &proposer)?;
    let treetn = to_treetn(&tci, &batch_eval, Some(0))?;

    // Convert TreeTN → TensorTrain<V>
    let tt = treetn_to_tensor_train::<V>(&treetn, n_sites, &local_dims)?;

    // Drop batch_eval (and its captured Rc clone) before extracting the cache
    drop(batch_eval);

    let final_cache = Rc::try_unwrap(cache)
        .map_err(|_| anyhow!("Failed to extract cache"))?
        .into_inner();

    Ok((
        QuanticsTensorCI2::from_inherent(tt, tci, grid, final_cache),
        ranks,
        errors,
    ))
}

#[cfg(test)]
mod tests;
