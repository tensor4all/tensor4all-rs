# tensor4all-quanticstci

## src/options.rs

### ` fn default() -> Self` (impl QtciOptions)

### `pub fn with_tolerance(mut self, tolerance: f64) -> Self` (impl QtciOptions)

Create new options with specified tolerance.

### `pub fn with_maxbonddim(mut self, maxbonddim: usize) -> Self` (impl QtciOptions)

Set maximum bond dimension.

### `pub fn with_maxiter(mut self, maxiter: usize) -> Self` (impl QtciOptions)

Set maximum number of iterations.

### `pub fn with_nrandominitpivot(mut self, n: usize) -> Self` (impl QtciOptions)

Set number of random initial pivots.

### `pub fn with_unfoldingscheme(mut self, scheme: UnfoldingScheme) -> Self` (impl QtciOptions)

Set unfolding scheme.

### `pub fn with_verbosity(mut self, verbosity: usize) -> Self` (impl QtciOptions)

Set verbosity level.

### `pub fn with_nsearchglobalpivot(mut self, n: usize) -> Self` (impl QtciOptions)

Set number of global pivots to search per iteration.

### `pub fn with_nsearch(mut self, n: usize) -> Self` (impl QtciOptions)

Set number of random searches for global pivots.

### `pub fn with_pivot_search(mut self, strategy: PivotSearchStrategy) -> Self` (impl QtciOptions)

Set pivot search strategy.

### `pub fn to_tci2_options(&self) -> TCI2Options` (impl QtciOptions)

Convert to TCI2Options for the underlying algorithm.

### ` fn test_default_options()`

### ` fn test_builder_pattern()`

### ` fn test_to_tci2_options()`

## src/quantics_tci.rs

### `pub fn from_discretized(tci: TensorCI2 < V >, grid: DiscretizedGrid, cache: HashMap < Vec < i64 > , V >) -> Self` (impl QuanticsTensorCI2 < V >)

Create a new QuanticsTensorCI2 from TCI result and discretized grid.

### `pub fn from_inherent(tci: TensorCI2 < V >, grid: InherentDiscreteGrid, cache: HashMap < Vec < i64 > , V >) -> Self` (impl QuanticsTensorCI2 < V >)

Create a new QuanticsTensorCI2 from TCI result and inherent discrete grid.

### `pub fn tci(&self) -> & TensorCI2 < V >` (impl QuanticsTensorCI2 < V >)

Get the underlying TensorCI2.

### `pub fn discretized_grid(&self) -> Option < & DiscretizedGrid >` (impl QuanticsTensorCI2 < V >)

Get the discretized grid (if available).

### `pub fn inherent_grid(&self) -> Option < & InherentDiscreteGrid >` (impl QuanticsTensorCI2 < V >)

Get the inherent discrete grid (if available).

### `pub fn rank(&self) -> usize` (impl QuanticsTensorCI2 < V >)

Get the bond dimension (maximum rank).

### `pub fn link_dims(&self) -> Vec < usize >` (impl QuanticsTensorCI2 < V >)

Get link dimensions.

### ` fn grididx_to_quantics(&self, indices: & [i64]) -> Result < Vec < i64 > >` (impl QuanticsTensorCI2 < V >)

Convert grid indices to quantics indices.

### `pub fn evaluate(&self, indices: & [i64]) -> Result < V >` (impl QuanticsTensorCI2 < V >)

Evaluate at grid indices.

### `pub fn sum(&self) -> Result < V >` (impl QuanticsTensorCI2 < V >)

Factorized sum over all grid points. This computes the sum efficiently using the tensor train structure.

### `pub fn integral(&self) -> Result < V >` (impl QuanticsTensorCI2 < V >)

Integral over continuous domain. Returns the sum multiplied by the grid step sizes. Only available for discretized grids.

### `pub fn tensor_train(&self) -> Result < TensorTrain < V > >` (impl QuanticsTensorCI2 < V >)

Get the underlying TensorTrain.

### `pub fn cachedata(&self) -> & HashMap < Vec < i64 > , V >` (impl QuanticsTensorCI2 < V >)

Access cached evaluation points. Returns a map from quantics indices to function values.

### `pub fn cachedata_origcoord(&self) -> Result < Vec < (Vec < f64 > , V) > >` (impl QuanticsTensorCI2 < V >)

Access cached evaluation points with original coordinates. Only available for discretized grids. Returns a vector of (coordinates, value) pairs since f64 is not hashable.

### `pub fn quanticscrossinterpolate(grid: & DiscretizedGrid, f: F, initial_pivots: Option < Vec < Vec < i64 > > >, options: QtciOptions) -> Result < (QuanticsTensorCI2 < V > , Vec < usize > , Vec < f64 >) >`

Interpolate a function with an explicit Grid.

### `pub fn quanticscrossinterpolate_from_arrays(xvals: & [Vec < f64 >], f: F, initial_pivots: Option < Vec < Vec < i64 > > >, options: QtciOptions) -> Result < (QuanticsTensorCI2 < V > , Vec < usize > , Vec < f64 >) >`

Interpolate from grid point arrays.

### `pub fn quanticscrossinterpolate_discrete(size: & [usize], f: F, initial_pivots: Option < Vec < Vec < i64 > > >, options: QtciOptions) -> Result < (QuanticsTensorCI2 < V > , Vec < usize > , Vec < f64 >) >`

Interpolate with discrete integer grid.

### ` fn test_discrete_simple_function()`

### ` fn test_discrete_tci_structure()`

### ` fn test_size_validation()`

### ` fn test_options_builder()`

