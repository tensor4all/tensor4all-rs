# quanticsgrids

## src/discretized_grid.rs

### `pub fn builder(rs: & [usize]) -> DiscretizedGridBuilder` (impl DiscretizedGrid)

Create a new builder for a DiscretizedGrid.

### `pub fn from_index_table(variable_names: & [& str], index_table: IndexTable) -> DiscretizedGridBuilder` (impl DiscretizedGrid)

Create a grid from an explicit index table.

### `pub fn ndims(&self) -> usize` (impl DiscretizedGrid)

Number of dimensions

### `pub fn len(&self) -> usize` (impl DiscretizedGrid)

Number of tensor sites (cores)

### `pub fn is_empty(&self) -> bool` (impl DiscretizedGrid)

Returns true if the grid has no tensor sites

### `pub fn rs(&self) -> & [usize]` (impl DiscretizedGrid)

Resolution (number of bits) per dimension

### `pub fn variable_names(&self) -> & [String]` (impl DiscretizedGrid)

Variable names

### `pub fn base(&self) -> usize` (impl DiscretizedGrid)

Numeric base

### `pub fn index_table(&self) -> & IndexTable` (impl DiscretizedGrid)

Index table

### `pub fn lower_bound(&self) -> & [f64]` (impl DiscretizedGrid)

Lower bounds for each dimension

### `pub fn upper_bound(&self) -> & [f64]` (impl DiscretizedGrid)

Upper bounds for each dimension

### `pub fn site_dim(&self, site: usize) -> Result < usize >` (impl DiscretizedGrid)

Local dimension of a tensor site

### `pub fn local_dimensions(&self) -> Vec < usize >` (impl DiscretizedGrid)

Local dimensions of all tensor sites

### `pub fn grid_step(&self) -> Vec < f64 >` (impl DiscretizedGrid)

Grid step size in each dimension

### `pub fn grid_min(&self) -> & [f64]` (impl DiscretizedGrid)

Minimum grid coordinates (same as lower_bound)

### `pub fn grid_max(&self) -> Vec < f64 >` (impl DiscretizedGrid)

Maximum grid coordinates (upper_bound - grid_step)

### `pub fn grid_origcoords(&self, dim: usize) -> Result < Vec < f64 > >` (impl DiscretizedGrid)

Get original coordinates for a dimension

### `pub fn quantics_to_grididx(&self, quantics: & [i64]) -> Result < Vec < i64 > >` (impl DiscretizedGrid)

Convert quantics indices to grid indices.

### `pub fn grididx_to_quantics(&self, grididx: & [i64]) -> Result < Vec < i64 > >` (impl DiscretizedGrid)

Convert grid indices to quantics indices.

### `pub fn grididx_to_origcoord(&self, grididx: & [i64]) -> Result < Vec < f64 > >` (impl DiscretizedGrid)

Convert grid indices to original coordinates.

### `pub fn origcoord_to_grididx(&self, coord: & [f64]) -> Result < Vec < i64 > >` (impl DiscretizedGrid)

Convert original coordinates to grid indices.

### `pub fn origcoord_to_quantics(&self, coord: & [f64]) -> Result < Vec < i64 > >` (impl DiscretizedGrid)

Convert original coordinates to quantics indices.

### `pub fn quantics_to_origcoord(&self, quantics: & [i64]) -> Result < Vec < f64 > >` (impl DiscretizedGrid)

Convert quantics indices to original coordinates.

### ` fn validate_grididx(&self, grididx: & [i64]) -> Result < () >` (impl DiscretizedGrid)

### ` fn validate_origcoord(&self, coord: & [f64]) -> Result < () >` (impl DiscretizedGrid)

### ` fn expand_grididx(&self, grididx: & [i64]) -> Result < Vec < i64 > >` (impl DiscretizedGrid)

### ` fn expand_coord(&self, coord: & [f64]) -> Result < Vec < f64 > >` (impl DiscretizedGrid)

### ` fn fmt(&self, f: & mut std :: fmt :: Formatter < '_ >) -> std :: fmt :: Result` (impl DiscretizedGrid)

### `pub fn new(rs: & [usize]) -> Self` (impl DiscretizedGridBuilder)

Create a new builder with given resolutions

### `pub fn from_index_table(variable_names: & [& str], index_table: IndexTable) -> Self` (impl DiscretizedGridBuilder)

Create a builder from an explicit index table

### `pub fn with_lower_bound(mut self, lower_bound: & [f64]) -> Self` (impl DiscretizedGridBuilder)

Set the lower bounds for each dimension

### `pub fn with_upper_bound(mut self, upper_bound: & [f64]) -> Self` (impl DiscretizedGridBuilder)

Set the upper bounds for each dimension

### `pub fn with_bounds(mut self, lower: f64, upper: f64) -> Self` (impl DiscretizedGridBuilder)

Set bounds for 1D case

### `pub fn with_include_endpoint(mut self, include: & [bool]) -> Self` (impl DiscretizedGridBuilder)

Set whether to include the endpoint for each dimension

### `pub fn include_endpoint(mut self, include: bool) -> Self` (impl DiscretizedGridBuilder)

Set whether to include the endpoint (single value for all dimensions)

### `pub fn with_variable_names(mut self, names: & [& str]) -> Self` (impl DiscretizedGridBuilder)

Set variable names

### `pub fn with_base(mut self, base: usize) -> Self` (impl DiscretizedGridBuilder)

Set the numeric base (default 2)

### `pub fn with_unfolding_scheme(mut self, scheme: UnfoldingScheme) -> Self` (impl DiscretizedGridBuilder)

Set the unfolding scheme

### `pub fn build(self) -> Result < DiscretizedGrid >` (impl DiscretizedGridBuilder)

Build the DiscretizedGrid

### `pub fn quantics_function(grid: & DiscretizedGrid, f: F) -> impl Fn (& [i64]) -> Result < f64 > + '_`

Wrap a function to accept quantics indices

### ` fn test_basic_1d_grid()`

### ` fn test_basic_2d_grid()`

### ` fn test_custom_bounds()`

### ` fn test_origcoord_to_grididx()`

### ` fn test_grididx_to_origcoord()`

### ` fn test_roundtrip_all_points()`

### ` fn test_include_endpoint()`

### ` fn test_grid_origcoords()`

### ` fn test_display()`

### ` fn test_error_invalid_bounds()`

### ` fn test_error_coordinate_out_of_bounds()`

### ` fn test_from_index_table()`

### ` fn test_quantics_function()`

## src/inherent_discrete_grid.rs

### `pub fn builder(rs: & [usize]) -> InherentDiscreteGridBuilder` (impl InherentDiscreteGrid)

Create a new builder for an InherentDiscreteGrid.

### `pub fn from_index_table(variable_names: & [& str], index_table: IndexTable) -> InherentDiscreteGridBuilder` (impl InherentDiscreteGrid)

Create a grid from an explicit index table.

### `pub fn ndims(&self) -> usize` (impl InherentDiscreteGrid)

Number of dimensions

### `pub fn len(&self) -> usize` (impl InherentDiscreteGrid)

Number of tensor sites (cores)

### `pub fn is_empty(&self) -> bool` (impl InherentDiscreteGrid)

Returns true if the grid has no tensor sites

### `pub fn rs(&self) -> & [usize]` (impl InherentDiscreteGrid)

Resolution (number of bits) per dimension

### `pub fn origin(&self) -> & [i64]` (impl InherentDiscreteGrid)

Origin in each dimension

### `pub fn step(&self) -> & [i64]` (impl InherentDiscreteGrid)

Step size in each dimension

### `pub fn variable_names(&self) -> & [String]` (impl InherentDiscreteGrid)

Variable names

### `pub fn base(&self) -> usize` (impl InherentDiscreteGrid)

Numeric base

### `pub fn index_table(&self) -> & IndexTable` (impl InherentDiscreteGrid)

Index table

### `pub fn max_grididx(&self) -> & [i64]` (impl InherentDiscreteGrid)

Maximum grid index per dimension

### `pub fn site_dim(&self, site: usize) -> Result < usize >` (impl InherentDiscreteGrid)

Local dimension of a tensor site

### `pub fn local_dimensions(&self) -> Vec < usize >` (impl InherentDiscreteGrid)

Local dimensions of all tensor sites

### `pub fn grid_min(&self) -> Vec < i64 >` (impl InherentDiscreteGrid)

Minimum grid coordinate (origin)

### `pub fn grid_max(&self) -> Vec < i64 >` (impl InherentDiscreteGrid)

Maximum grid coordinate

### `pub fn quantics_to_grididx(&self, quantics: & [i64]) -> Result < Vec < i64 > >` (impl InherentDiscreteGrid)

Convert quantics indices to grid indices.

### `pub fn grididx_to_quantics(&self, grididx: & [i64]) -> Result < Vec < i64 > >` (impl InherentDiscreteGrid)

Convert grid indices to quantics indices.

### `pub fn grididx_to_origcoord(&self, grididx: & [i64]) -> Result < Vec < i64 > >` (impl InherentDiscreteGrid)

Convert grid indices to original coordinates.

### `pub fn origcoord_to_grididx(&self, coord: & [i64]) -> Result < Vec < i64 > >` (impl InherentDiscreteGrid)

Convert original coordinates to grid indices.

### `pub fn origcoord_to_quantics(&self, coord: & [i64]) -> Result < Vec < i64 > >` (impl InherentDiscreteGrid)

Convert original coordinates to quantics indices.

### `pub fn quantics_to_origcoord(&self, quantics: & [i64]) -> Result < Vec < i64 > >` (impl InherentDiscreteGrid)

Convert quantics indices to original coordinates.

### ` fn validate_quantics(&self, quantics: & [i64]) -> Result < () >` (impl InherentDiscreteGrid)

### ` fn validate_grididx(&self, grididx: & [i64]) -> Result < () >` (impl InherentDiscreteGrid)

### ` fn validate_origcoord(&self, coord: & [i64]) -> Result < () >` (impl InherentDiscreteGrid)

### ` fn expand_grididx(&self, grididx: & [i64]) -> Result < Vec < i64 > >` (impl InherentDiscreteGrid)

Expand a scalar or lower-dimensional input to full dimensions

### ` fn expand_coord(&self, coord: & [i64]) -> Result < Vec < i64 > >` (impl InherentDiscreteGrid)

### ` fn quantics_to_grididx_base2(&self, quantics: & [i64]) -> Vec < i64 >` (impl InherentDiscreteGrid)

### ` fn quantics_to_grididx_general(&self, quantics: & [i64]) -> Vec < i64 >` (impl InherentDiscreteGrid)

### ` fn grididx_to_quantics_base2(&self, result: & mut [i64], grididx: & [i64])` (impl InherentDiscreteGrid)

### ` fn grididx_to_quantics_general(&self, result: & mut [i64], grididx: & [i64])` (impl InherentDiscreteGrid)

### `pub fn new(rs: & [usize]) -> Self` (impl InherentDiscreteGridBuilder)

Create a new builder with given resolutions

### `pub fn from_index_table(variable_names: & [& str], index_table: IndexTable) -> Self` (impl InherentDiscreteGridBuilder)

Create a builder from an explicit index table

### `pub fn with_origin(mut self, origin: & [i64]) -> Self` (impl InherentDiscreteGridBuilder)

Set the origin for each dimension

### `pub fn with_step(mut self, step: & [i64]) -> Self` (impl InherentDiscreteGridBuilder)

Set the step size for each dimension

### `pub fn with_variable_names(mut self, names: & [& str]) -> Self` (impl InherentDiscreteGridBuilder)

Set variable names

### `pub fn with_base(mut self, base: usize) -> Self` (impl InherentDiscreteGridBuilder)

Set the numeric base (default 2)

### `pub fn with_unfolding_scheme(mut self, scheme: UnfoldingScheme) -> Self` (impl InherentDiscreteGridBuilder)

Set the unfolding scheme

### `pub fn build(self) -> Result < InherentDiscreteGrid >` (impl InherentDiscreteGridBuilder)

Build the InherentDiscreteGrid

### ` fn rangecheck_r(r: usize, base: usize) -> bool`

Check if base^R fits in i64

### ` fn build_index_table(variable_names: & [String], rs: & [usize], scheme: UnfoldingScheme) -> IndexTable`

Build an index table from variable names and resolutions

### ` fn add_interleaved_indices(index_table: & mut IndexTable, variable_names: & [String], rs: & [usize], bitnumber: usize)`

### ` fn add_fused_indices(index_table: & mut IndexTable, variable_names: & [String], rs: & [usize], bitnumber: usize)`

### ` fn build_lookup_table(rs: & [usize], index_table: & IndexTable, variable_names: & [String]) -> Result < Vec < Vec < LookupEntry > > >`

Build lookup table from index table

### ` fn test_basic_1d_grid()`

### ` fn test_basic_2d_grid()`

### ` fn test_grididx_to_quantics_roundtrip()`

### ` fn test_all_grididx_roundtrip()`

### ` fn test_interleaved_scheme()`

### ` fn test_base3_grid()`

### ` fn test_origcoord_conversion()`

### ` fn test_error_invalid_base()`

### ` fn test_error_duplicate_variable_names()`

### ` fn test_error_quantics_out_of_range()`

### ` fn test_error_grididx_out_of_bounds()`

### ` fn test_local_dimensions()`

### ` fn test_from_index_table()`

## src/lib.rs

### ` fn test_unfolding_scheme_default()`

