//! Discretized grid implementation with continuous coordinate support

use crate::error::{QuanticsGridError, Result};
use crate::inherent_discrete_grid::InherentDiscreteGridBuilder;
use crate::{IndexTable, InherentDiscreteGrid, UnfoldingScheme};

/// A discretized grid with continuous domain and floating-point coordinates.
///
/// This structure wraps an [`InherentDiscreteGrid`] and adds support for
/// continuous coordinate systems with specified lower and upper bounds.
///
/// # Example
/// ```
/// use quanticsgrids::{DiscretizedGrid, UnfoldingScheme};
///
/// // Create a 2D grid discretizing [0, 1) x [0, 1)
/// let grid = DiscretizedGrid::builder(&[3, 2])
///     .with_variable_names(&["x", "y"])
///     .build()
///     .unwrap();
///
/// // Convert between coordinate systems
/// let quantics = grid.origcoord_to_quantics(&[0.5, 0.25]).unwrap();
/// let coords = grid.quantics_to_origcoord(&quantics).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct DiscretizedGrid {
    /// The underlying discrete grid
    discrete_grid: InherentDiscreteGrid,
    /// Lower bounds for each dimension
    lower_bound: Vec<f64>,
    /// Public upper bounds for each dimension
    upper_bound: Vec<f64>,
    /// Internal upper bounds used for grid spacing calculations
    effective_upper_bound: Vec<f64>,
    /// Whether each dimension includes the upper endpoint
    include_endpoint: Vec<bool>,
}

impl DiscretizedGrid {
    /// Create a new builder for a DiscretizedGrid.
    ///
    /// # Example
    /// ```
    /// use quanticsgrids::DiscretizedGrid;
    ///
    /// let grid = DiscretizedGrid::builder(&[3, 2])
    ///     .with_variable_names(&["x", "y"])
    ///     .build()
    ///     .unwrap();
    /// ```
    pub fn builder(rs: &[usize]) -> DiscretizedGridBuilder {
        DiscretizedGridBuilder::new(rs)
    }

    /// Create a grid from an explicit index table.
    pub fn from_index_table(
        variable_names: &[&str],
        index_table: IndexTable,
    ) -> DiscretizedGridBuilder {
        DiscretizedGridBuilder::from_index_table(variable_names, index_table)
    }

    /// Number of dimensions
    pub fn ndims(&self) -> usize {
        self.discrete_grid.ndims()
    }

    /// Number of tensor sites (cores)
    pub fn len(&self) -> usize {
        self.discrete_grid.len()
    }

    /// Returns true if the grid has no tensor sites
    pub fn is_empty(&self) -> bool {
        self.discrete_grid.is_empty()
    }

    /// Resolution (number of bits) per dimension
    pub fn rs(&self) -> &[usize] {
        self.discrete_grid.rs()
    }

    /// Variable names
    pub fn variable_names(&self) -> &[String] {
        self.discrete_grid.variable_names()
    }

    /// Numeric base
    pub fn base(&self) -> usize {
        self.discrete_grid.base()
    }

    /// Index table
    pub fn index_table(&self) -> &IndexTable {
        self.discrete_grid.index_table()
    }

    /// Lower bounds for each dimension
    pub fn lower_bound(&self) -> &[f64] {
        &self.lower_bound
    }

    /// Upper bounds for each dimension
    pub fn upper_bound(&self) -> &[f64] {
        &self.upper_bound
    }

    /// Local dimension of a tensor site
    pub fn site_dim(&self, site: usize) -> Result<usize> {
        self.discrete_grid.site_dim(site)
    }

    /// Local dimensions of all tensor sites
    pub fn local_dimensions(&self) -> Vec<usize> {
        self.discrete_grid.local_dimensions()
    }

    /// Grid step size in each dimension
    pub fn grid_step(&self) -> Vec<f64> {
        let rs = self.discrete_grid.rs();
        let base = self.discrete_grid.base() as f64;
        self.lower_bound
            .iter()
            .zip(self.effective_upper_bound.iter())
            .zip(rs.iter())
            .map(|((&lo, &hi), &r)| (hi - lo) / base.powi(r as i32))
            .collect()
    }

    /// Minimum grid coordinates (same as lower_bound)
    pub fn grid_min(&self) -> &[f64] {
        &self.lower_bound
    }

    /// Maximum representable grid coordinates in each dimension
    pub fn grid_max(&self) -> Vec<f64> {
        let step = self.grid_step();
        self.effective_upper_bound
            .iter()
            .zip(step.iter())
            .map(|(&hi, &s)| hi - s)
            .collect()
    }

    /// Get original coordinates for a dimension
    pub fn grid_origcoords(&self, dim: usize) -> Result<Vec<f64>> {
        if dim >= self.ndims() {
            return Err(QuanticsGridError::GridIndexOutOfBounds {
                dim,
                value: dim as i64,
                max: self.ndims() as i64,
            });
        }

        let rs = self.discrete_grid.rs();
        let base = self.discrete_grid.base();
        let n = base.pow(rs[dim] as u32);
        let step = self.grid_step()[dim];
        let start = self.lower_bound[dim];

        Ok((0..n).map(|i| start + (i as f64) * step).collect())
    }

    // ========================================================================
    // Core conversion functions
    // ========================================================================

    /// Convert quantics indices to grid indices.
    pub fn quantics_to_grididx(&self, quantics: &[i64]) -> Result<Vec<i64>> {
        self.discrete_grid.quantics_to_grididx(quantics)
    }

    /// Convert grid indices to quantics indices.
    pub fn grididx_to_quantics(&self, grididx: &[i64]) -> Result<Vec<i64>> {
        self.discrete_grid.grididx_to_quantics(grididx)
    }

    /// Convert grid indices to original coordinates.
    pub fn grididx_to_origcoord(&self, grididx: &[i64]) -> Result<Vec<f64>> {
        let grididx = self.expand_grididx(grididx)?;
        self.validate_grididx(&grididx)?;

        let step = self.grid_step();
        Ok(self
            .lower_bound
            .iter()
            .zip(grididx.iter())
            .zip(step.iter())
            .map(|((&lo, &g), &s)| lo + ((g - 1) as f64) * s)
            .collect())
    }

    /// Convert original coordinates to grid indices.
    pub fn origcoord_to_grididx(&self, coord: &[f64]) -> Result<Vec<i64>> {
        let coord = self.expand_coord(coord)?;
        self.validate_origcoord(&coord)?;

        let step = self.grid_step();
        let rs = self.discrete_grid.rs();
        let base = self.discrete_grid.base() as i64;

        let indices: Vec<i64> = self
            .lower_bound
            .iter()
            .zip(coord.iter())
            .zip(step.iter())
            .zip(rs.iter())
            .map(|(((&lo, &c), &s), &r)| {
                let continuous_idx = (c - lo) / s + 1.0;
                let discrete_idx = continuous_idx.round() as i64;
                discrete_idx.clamp(1, base.pow(r as u32))
            })
            .collect();

        Ok(indices)
    }

    /// Convert original coordinates to quantics indices.
    pub fn origcoord_to_quantics(&self, coord: &[f64]) -> Result<Vec<i64>> {
        let grididx = self.origcoord_to_grididx(coord)?;
        self.grididx_to_quantics(&grididx)
    }

    /// Convert quantics indices to original coordinates.
    pub fn quantics_to_origcoord(&self, quantics: &[i64]) -> Result<Vec<f64>> {
        let grididx = self.quantics_to_grididx(quantics)?;
        self.grididx_to_origcoord(&grididx)
    }

    // ========================================================================
    // Private helper methods
    // ========================================================================

    fn validate_grididx(&self, grididx: &[i64]) -> Result<()> {
        let max_grididx = self.discrete_grid.max_grididx();
        for (dim, (&val, &max)) in grididx.iter().zip(max_grididx.iter()).enumerate() {
            if val < 1 || val > max {
                return Err(QuanticsGridError::GridIndexOutOfBounds {
                    dim,
                    value: val,
                    max,
                });
            }
        }
        Ok(())
    }

    fn validate_origcoord(&self, coord: &[f64]) -> Result<()> {
        for (dim, (((&c, &lo), &hi), &include_endpoint)) in coord
            .iter()
            .zip(self.lower_bound.iter())
            .zip(self.upper_bound.iter())
            .zip(self.include_endpoint.iter())
            .enumerate()
        {
            let out_of_bounds = if include_endpoint {
                c < lo || c > hi
            } else {
                c < lo || c >= hi
            };

            if out_of_bounds {
                return Err(QuanticsGridError::CoordinateOutOfBounds {
                    dim,
                    value: c,
                    lower: lo,
                    upper: hi,
                });
            }
        }
        Ok(())
    }

    fn expand_grididx(&self, grididx: &[i64]) -> Result<Vec<i64>> {
        let ndims = self.ndims();
        if grididx.len() == 1 && ndims > 1 {
            Ok(vec![grididx[0]; ndims])
        } else if grididx.len() == ndims {
            Ok(grididx.to_vec())
        } else {
            Err(QuanticsGridError::DimensionMismatch {
                expected: ndims,
                actual: grididx.len(),
            })
        }
    }

    fn expand_coord(&self, coord: &[f64]) -> Result<Vec<f64>> {
        let ndims = self.ndims();
        if coord.len() == 1 && ndims > 1 {
            Ok(vec![coord[0]; ndims])
        } else if coord.len() == ndims {
            Ok(coord.to_vec())
        } else {
            Err(QuanticsGridError::DimensionMismatch {
                expected: ndims,
                actual: coord.len(),
            })
        }
    }
}

impl std::fmt::Display for DiscretizedGrid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ndims = self.ndims();
        let rs = self.discrete_grid.rs();
        let base = self.discrete_grid.base();

        // Calculate total points
        let total_points: u64 = rs.iter().map(|&r| (base as u64).pow(r as u32)).product();

        if ndims <= 1 {
            write!(
                f,
                "DiscretizedGrid{{{}}} with {} grid points",
                ndims, total_points
            )?;
        } else {
            let sizes: Vec<String> = rs
                .iter()
                .map(|&r| format!("{}", (base as u64).pow(r as u32)))
                .collect();
            write!(
                f,
                "DiscretizedGrid{{{}}} with {} = {} grid points",
                ndims,
                sizes.join(" x "),
                total_points
            )?;
        }

        // Variable names
        let var_names = self.discrete_grid.variable_names();
        if var_names
            .iter()
            .any(|n| !n.chars().all(|c| c.is_ascii_digit()))
        {
            write!(f, "\n  Variables: ({})", var_names.join(", "))?;
        }

        // Resolutions
        if ndims == 1 {
            write!(f, "\n  Resolution: {} bits", rs[0])?;
        } else {
            let res_str: Vec<String> = var_names
                .iter()
                .zip(rs.iter())
                .map(|(n, &r)| format!("{}: {}", n, r))
                .collect();
            write!(f, "\n  Resolutions: ({})", res_str.join(", "))?;
        }

        // Domain
        let step = self.grid_step();
        if ndims == 1 {
            let right_bracket = if self.include_endpoint[0] { ']' } else { ')' };
            write!(
                f,
                "\n  Domain: [{}, {}{}",
                self.lower_bound[0], self.upper_bound[0], right_bracket
            )?;
            write!(f, "\n  Grid spacing: {}", step[0])?;
        } else {
            let bounds_str: Vec<String> = self
                .lower_bound
                .iter()
                .zip(self.upper_bound.iter())
                .zip(self.include_endpoint.iter())
                .map(|((&lo, &hi), &include_endpoint)| {
                    let right_bracket = if include_endpoint { ']' } else { ')' };
                    format!("[{}, {}{}", lo, hi, right_bracket)
                })
                .collect();
            write!(f, "\n  Domain: {}", bounds_str.join(" x "))?;

            let step_str: Vec<String> = var_names
                .iter()
                .zip(step.iter())
                .map(|(n, &s)| format!("d{} = {}", n, s))
                .collect();
            write!(f, "\n  Grid spacing: ({})", step_str.join(", "))?;
        }

        // Base (if not binary)
        if base != 2 {
            write!(f, "\n  Base: {}", base)?;
        }

        // Tensor structure
        let num_sites = self.len();
        let sitedims = self.local_dimensions();
        write!(f, "\n  Tensor train: {} sites", num_sites)?;
        if !sitedims.is_empty() {
            if sitedims.iter().all(|&d| d == sitedims[0]) {
                write!(f, " (uniform dimension {})", sitedims[0])?;
            } else {
                let dims_str: Vec<String> = sitedims.iter().map(|d| d.to_string()).collect();
                write!(f, " (dimensions: {})", dims_str.join("-"))?;
            }
        }

        Ok(())
    }
}

/// Builder for DiscretizedGrid
#[derive(Debug, Clone)]
pub struct DiscretizedGridBuilder {
    inner: InherentDiscreteGridBuilder,
    lower_bound: Option<Vec<f64>>,
    upper_bound: Option<Vec<f64>>,
    include_endpoint: Option<Vec<bool>>,
}

impl DiscretizedGridBuilder {
    /// Create a new builder with given resolutions
    pub fn new(rs: &[usize]) -> Self {
        Self {
            inner: InherentDiscreteGridBuilder::new(rs),
            lower_bound: None,
            upper_bound: None,
            include_endpoint: None,
        }
    }

    /// Create a builder from an explicit index table
    pub fn from_index_table(variable_names: &[&str], index_table: IndexTable) -> Self {
        Self {
            inner: InherentDiscreteGridBuilder::from_index_table(variable_names, index_table),
            lower_bound: None,
            upper_bound: None,
            include_endpoint: None,
        }
    }

    /// Set the lower bounds for each dimension
    pub fn with_lower_bound(mut self, lower_bound: &[f64]) -> Self {
        self.lower_bound = Some(lower_bound.to_vec());
        self
    }

    /// Set the upper bounds for each dimension
    pub fn with_upper_bound(mut self, upper_bound: &[f64]) -> Self {
        self.upper_bound = Some(upper_bound.to_vec());
        self
    }

    /// Set bounds for 1D case
    pub fn with_bounds(mut self, lower: f64, upper: f64) -> Self {
        self.lower_bound = Some(vec![lower]);
        self.upper_bound = Some(vec![upper]);
        self
    }

    /// Set whether to include the endpoint for each dimension
    pub fn with_include_endpoint(mut self, include: &[bool]) -> Self {
        self.include_endpoint = Some(include.to_vec());
        self
    }

    /// Set whether to include the endpoint (single value for all dimensions)
    pub fn include_endpoint(mut self, include: bool) -> Self {
        // Will be expanded to match dimensions during build
        self.include_endpoint = Some(vec![include]);
        self
    }

    /// Set variable names
    pub fn with_variable_names(mut self, names: &[&str]) -> Self {
        self.inner = self.inner.with_variable_names(names);
        self
    }

    /// Set the numeric base (default 2)
    pub fn with_base(mut self, base: usize) -> Self {
        self.inner = self.inner.with_base(base);
        self
    }

    /// Set the unfolding scheme
    pub fn with_unfolding_scheme(mut self, scheme: UnfoldingScheme) -> Self {
        self.inner = self.inner.with_unfolding_scheme(scheme);
        self
    }

    /// Build the DiscretizedGrid
    pub fn build(self) -> Result<DiscretizedGrid> {
        let discrete_grid = self.inner.build()?;
        let ndims = discrete_grid.ndims();
        let rs = discrete_grid.rs();
        let base = discrete_grid.base();

        // Default bounds: [0, 1) for each dimension
        let lower_bound = self.lower_bound.unwrap_or_else(|| vec![0.0; ndims]);
        let mut upper_bound = self.upper_bound.unwrap_or_else(|| vec![1.0; ndims]);

        // Expand bounds if needed
        let lower_bound = if lower_bound.len() == 1 && ndims > 1 {
            vec![lower_bound[0]; ndims]
        } else {
            lower_bound
        };

        let upper_bound_raw = if upper_bound.len() == 1 && ndims > 1 {
            vec![upper_bound[0]; ndims]
        } else {
            upper_bound.clone()
        };
        upper_bound = upper_bound_raw;

        // Validate dimensions
        if lower_bound.len() != ndims {
            return Err(QuanticsGridError::DimensionMismatch {
                expected: ndims,
                actual: lower_bound.len(),
            });
        }
        if upper_bound.len() != ndims {
            return Err(QuanticsGridError::DimensionMismatch {
                expected: ndims,
                actual: upper_bound.len(),
            });
        }

        // Validate bounds
        for d in 0..ndims {
            if lower_bound[d] >= upper_bound[d] {
                return Err(QuanticsGridError::InvalidBounds {
                    dim: d,
                    lower: lower_bound[d],
                    upper: upper_bound[d],
                });
            }
        }

        // Handle endpoint inclusion
        let include_endpoint = match self.include_endpoint {
            Some(v) if v.len() == 1 && ndims > 1 => vec![v[0]; ndims],
            Some(v) if v.len() == ndims => v,
            Some(v) if v.len() != ndims => {
                return Err(QuanticsGridError::DimensionMismatch {
                    expected: ndims,
                    actual: v.len(),
                });
            }
            _ => vec![false; ndims],
        };

        let mut effective_upper_bound = upper_bound.clone();

        // Adjust upper bounds for endpoint inclusion
        for d in 0..ndims {
            if include_endpoint[d] {
                if rs[d] == 0 {
                    return Err(QuanticsGridError::EndpointWithZeroResolution { dim: d });
                }
                let n_points = (base as f64).powi(rs[d] as i32);
                effective_upper_bound[d] +=
                    (effective_upper_bound[d] - lower_bound[d]) / (n_points - 1.0);
            }
        }

        Ok(DiscretizedGrid {
            discrete_grid,
            lower_bound,
            upper_bound,
            effective_upper_bound,
            include_endpoint,
        })
    }
}

/// Wrap a function to accept quantics indices
pub fn quantics_function<F>(grid: &DiscretizedGrid, f: F) -> impl Fn(&[i64]) -> Result<f64> + '_
where
    F: Fn(&[f64]) -> f64 + 'static,
{
    move |quantics: &[i64]| {
        let coords = grid.quantics_to_origcoord(quantics)?;
        Ok(f(&coords))
    }
}

#[cfg(test)]
mod tests;
