# tensor4all-tensorci

## src/cached_function.rs

### ` fn new(local_dims: & [usize]) -> Self` (impl InnerCache < V >)

Create a new cache, automatically selecting the key type

### ` fn compute_coeffs_u64(local_dims: & [usize]) -> Vec < u64 >` (impl InnerCache < V >)

### ` fn compute_coeffs_u128(local_dims: & [usize]) -> Vec < u128 >` (impl InnerCache < V >)

### ` fn flat_index_u64(idx: & [usize], coeffs: & [u64]) -> u64` (impl InnerCache < V >)

### ` fn flat_index_u128(idx: & [usize], coeffs: & [u128]) -> u128` (impl InnerCache < V >)

### ` fn get(&self, idx: & [usize]) -> Option < & V >` (impl InnerCache < V >)

### ` fn insert(&mut self, idx: & [usize], value: V)` (impl InnerCache < V >)

### ` fn contains(&self, idx: & [usize]) -> bool` (impl InnerCache < V >)

### ` fn len(&self) -> usize` (impl InnerCache < V >)

### ` fn clear(&mut self)` (impl InnerCache < V >)

### ` fn key_type_name(&self) -> & 'static str` (impl InnerCache < V >)

### `pub fn new(func: F, local_dims: & [usize]) -> Self` (impl CachedFunction < V , F >)

Create a new cached function wrapper. The key type is automatically selected based on `local_dims`: - `u64` if total index space fits in 64 bits

### `pub fn eval(&mut self, idx: & [usize]) -> V` (impl CachedFunction < V , F >)

Evaluate the function at a given index, using cache if available.

### `pub fn eval_no_cache(&self, idx: & [usize]) -> V` (impl CachedFunction < V , F >)

Evaluate the function at a given index, bypassing the cache.

### `pub fn local_dims(&self) -> & [usize]` (impl CachedFunction < V , F >)

Get the local dimensions.

### `pub fn num_sites(&self) -> usize` (impl CachedFunction < V , F >)

Get the number of sites (length of index).

### `pub fn num_evals(&self) -> usize` (impl CachedFunction < V , F >)

Get the number of actual function evaluations.

### `pub fn num_cache_hits(&self) -> usize` (impl CachedFunction < V , F >)

Get the number of cache hits.

### `pub fn total_calls(&self) -> usize` (impl CachedFunction < V , F >)

Get the total number of calls (evals + cache hits).

### `pub fn cache_hit_ratio(&self) -> f64` (impl CachedFunction < V , F >)

Get the cache hit ratio.

### `pub fn clear_cache(&mut self)` (impl CachedFunction < V , F >)

Clear the cache.

### `pub fn cache_size(&self) -> usize` (impl CachedFunction < V , F >)

Get the number of cached entries.

### `pub fn is_cached(&self, idx: & [usize]) -> bool` (impl CachedFunction < V , F >)

Check if an index is cached.

### `pub fn key_type(&self) -> & 'static str` (impl CachedFunction < V , F >)

Get the internal key type name (for debugging).

### ` fn test_cached_function_basic()`

### ` fn test_auto_key_selection_small()`

### ` fn test_auto_key_selection_large()`

### ` fn test_cached_function_clear()`

### ` fn test_local_dims()`

## src/indexset.rs

### ` fn default() -> Self` (impl IndexSet < T >)

### `pub fn new() -> Self` (impl IndexSet < T >)

Create an empty index set

### `pub fn from_vec(values: Vec < T >) -> Self` (impl IndexSet < T >)

Create an index set from a vector

### `pub fn get(&self, i: usize) -> Option < & T >` (impl IndexSet < T >)

Get the value at integer index

### `pub fn pos(&self, value: & T) -> Option < usize >` (impl IndexSet < T >)

Get the integer position of a value

### `pub fn positions(&self, values: & [T]) -> Option < Vec < usize > >` (impl IndexSet < T >)

Get positions for a slice of values

### `pub fn push(&mut self, value: T)` (impl IndexSet < T >)

Push a new value to the set

### `pub fn contains(&self, value: & T) -> bool` (impl IndexSet < T >)

Check if the set contains a value

### `pub fn len(&self) -> usize` (impl IndexSet < T >)

Number of elements in the set

### `pub fn is_empty(&self) -> bool` (impl IndexSet < T >)

Check if the set is empty

### `pub fn iter(&self) -> impl Iterator < Item = & T >` (impl IndexSet < T >)

Iterate over values

### `pub fn values(&self) -> & [T]` (impl IndexSet < T >)

Get all values as a slice

### ` fn index(&self, i: usize) -> & Self :: Output` (impl IndexSet < T >)

### ` fn into_iter(self) -> Self :: IntoIter` (impl IndexSet < T >)

### ` fn into_iter(self) -> Self :: IntoIter` (impl & 'a IndexSet < T >)

### ` fn test_indexset_basic()`

### ` fn test_indexset_from_vec()`

### ` fn test_indexset_contains()`

### ` fn test_indexset_iter()`

## src/tensorci1.rs

### ` fn forward_sweep(strategy: SweepStrategy, iter: usize) -> bool`

Returns true if this iteration should be a forward sweep

### ` fn default() -> Self` (impl TCI1Options)

### `pub fn new(local_dims: Vec < usize >) -> Self` (impl TensorCI1 < T >)

Create a new empty TensorCI1

### `pub fn len(&self) -> usize` (impl TensorCI1 < T >)

Number of sites

### `pub fn is_empty(&self) -> bool` (impl TensorCI1 < T >)

Check if empty

### `pub fn local_dims(&self) -> & [usize]` (impl TensorCI1 < T >)

Get local dimensions

### `pub fn rank(&self) -> usize` (impl TensorCI1 < T >)

Get current rank (maximum bond dimension)

### `pub fn link_dims(&self) -> Vec < usize >` (impl TensorCI1 < T >)

Get bond dimensions

### `pub fn last_sweep_pivot_error(&self) -> f64` (impl TensorCI1 < T >)

Get the maximum pivot error from the last sweep

### `pub fn site_tensor(&self, p: usize) -> Tensor3 < T >` (impl TensorCI1 < T >)

Get site tensor at position p (T * P^{-1})

### `pub fn site_tensors(&self) -> Vec < Tensor3 < T > >` (impl TensorCI1 < T >)

Get all site tensors

### `pub fn to_tensor_train(&self) -> Result < TensorTrain < T > >` (impl TensorCI1 < T >)

Convert to TensorTrain

### `pub fn max_sample_value(&self) -> f64` (impl TensorCI1 < T >)

Get maximum sample value

### ` fn update_max_sample(&mut self, values: & [T])` (impl TensorCI1 < T >)

Update maximum sample value from a slice of values

### ` fn update_max_sample_matrix(&mut self, mat: & Matrix < T >)` (impl TensorCI1 < T >)

Update maximum sample value from a matrix

### `pub fn evaluate(&self, indices: & [usize]) -> Result < T >` (impl TensorCI1 < T >)

Evaluate the TCI at a specific set of indices

### `pub fn i_set(&self, p: usize) -> & IndexSet < MultiIndex >` (impl TensorCI1 < T >)

Get I set for a site

### `pub fn j_set(&self, p: usize) -> & IndexSet < MultiIndex >` (impl TensorCI1 < T >)

Get J set for a site

### ` fn get_pi_i_set(&self, p: usize) -> IndexSet < MultiIndex >` (impl TensorCI1 < T >)

Build the Pi I set for site p PiIset[p] = { [i..., up] : i in Iset[p], up in 1..localdims[p] }

### ` fn get_pi_j_set(&self, p: usize) -> IndexSet < MultiIndex >` (impl TensorCI1 < T >)

Build the Pi J set for site p PiJset[p] = { [up+1, j...] : up+1 in 1..localdims[p], j in Jset[p] }

### ` fn get_pi(&mut self, p: usize, f: & F) -> Matrix < T >` (impl TensorCI1 < T >)

Build the Pi matrix at bond p Pi[p][i, j] = f([PiIset[p][i]..., PiJset[p+1][j]...])

### ` fn update_pi_rows(&mut self, p: usize, f: & F)` (impl TensorCI1 < T >)

Update Pi rows at site p (after I set changed at p+1)

### ` fn update_pi_cols(&mut self, p: usize, f: & F)` (impl TensorCI1 < T >)

Update Pi cols at site p (after J set changed at p)

### ` fn add_pivot_row(&mut self, p: usize, new_i: usize, f: & F)` (impl TensorCI1 < T >)

Add a pivot row at bond p

### ` fn add_pivot_col(&mut self, p: usize, new_j: usize, f: & F)` (impl TensorCI1 < T >)

Add a pivot col at bond p

### ` fn update_p_matrix(&mut self, p: usize)` (impl TensorCI1 < T >)

Update P matrix at bond p from current I and J sets

### ` fn add_pivot(&mut self, p: usize, f: & F, tolerance: f64)` (impl TensorCI1 < T >)

Add a pivot at bond p

### ` fn initialize_from_pivot(&mut self, f: & F, first_pivot: & MultiIndex) -> Result < () >` (impl TensorCI1 < T >)

Initialize from function with first pivot

### ` fn vec_to_row_matrix(v: & [T]) -> Matrix < T >`

Convert a vector to a row matrix

### ` fn vec_to_col_matrix(v: & [T]) -> Matrix < T >`

Convert a vector to a column matrix

### ` fn row_matrix_to_vec(mat: & Matrix < T >) -> Vec < T >`

Convert a row matrix to a vector

### ` fn tensor3_to_matrix(tensor: & Tensor3 < T >) -> Matrix < T >`

Convert Tensor3 to Matrix (reshape for columns: (left*site, right))

### ` fn tensor3_to_matrix_cols(tensor: & Tensor3 < T >, rows: usize, cols: usize) -> Matrix < T >`

Convert Tensor3 to Matrix for columns (left*site, right)

### ` fn tensor3_to_matrix_rows(tensor: & Tensor3 < T >, rows: usize, cols: usize) -> Matrix < T >`

Convert Tensor3 to Matrix for rows (left, site*right)

### ` fn matrix_to_tensor3(mat: & Matrix < T >, left_dim: usize, site_dim: usize, right_dim: usize) -> Tensor3 < T >`

Convert Matrix to Tensor3

### `pub fn crossinterpolate1(f: F, local_dims: Vec < usize >, first_pivot: MultiIndex, options: TCI1Options) -> Result < (TensorCI1 < T > , Vec < usize > , Vec < f64 >) >`

Cross interpolate a function using TCI1 algorithm

### ` fn test_tensorci1_new()`

### ` fn test_crossinterpolate1_constant()`

### ` fn test_crossinterpolate1_simple()`

### ` fn test_crossinterpolate1_evaluate_at_pivot()`

### ` fn test_crossinterpolate1_evaluate_on_cross()`

### ` fn test_crossinterpolate1_to_tensor_train()`

### ` fn test_crossinterpolate1_3d()`

### ` fn test_crossinterpolate1_rank2_function()`

### ` fn test_crossinterpolate1_converges()`

## src/tensorci2.rs

### ` fn default() -> Self` (impl TCI2Options)

### `pub fn new(local_dims: Vec < usize >) -> Result < Self >` (impl TensorCI2 < T >)

Create a new empty TensorCI2

### `pub fn len(&self) -> usize` (impl TensorCI2 < T >)

Number of sites

### `pub fn is_empty(&self) -> bool` (impl TensorCI2 < T >)

Check if empty

### `pub fn local_dims(&self) -> & [usize]` (impl TensorCI2 < T >)

Get local dimensions

### `pub fn rank(&self) -> usize` (impl TensorCI2 < T >)

Get current rank (maximum bond dimension)

### `pub fn link_dims(&self) -> Vec < usize >` (impl TensorCI2 < T >)

Get bond dimensions

### `pub fn max_sample_value(&self) -> f64` (impl TensorCI2 < T >)

Get maximum sample value

### `pub fn max_bond_error(&self) -> f64` (impl TensorCI2 < T >)

Get maximum bond error

### `pub fn pivot_errors(&self) -> & [f64]` (impl TensorCI2 < T >)

Get pivot errors from back-truncation

### `pub fn is_site_tensors_available(&self) -> bool` (impl TensorCI2 < T >)

Check if site tensors are available

### `pub fn site_tensor(&self, p: usize) -> & Tensor3 < T >` (impl TensorCI2 < T >)

Get site tensor at position p

### `pub fn to_tensor_train(&self) -> Result < TensorTrain < T > >` (impl TensorCI2 < T >)

Convert to TensorTrain

### `pub fn add_global_pivots(&mut self, pivots: & [MultiIndex]) -> Result < () >` (impl TensorCI2 < T >)

Add global pivots to the TCI

### ` fn invalidate_site_tensors(&mut self)` (impl TensorCI2 < T >)

Invalidate all site tensors

### ` fn kronecker_i(&self, p: usize) -> Vec < MultiIndex >` (impl TensorCI2 < T >)

Expand indices by Kronecker product with local dimension

### ` fn kronecker_j(&self, p: usize) -> Vec < MultiIndex >` (impl TensorCI2 < T >)

### `pub fn crossinterpolate2(f: F, batched_f: Option < B >, local_dims: Vec < usize >, initial_pivots: Vec < MultiIndex >, options: TCI2Options) -> Result < (TensorCI2 < T > , Vec < usize > , Vec < f64 >) >`

Cross interpolate a function using TCI2 algorithm

### ` fn update_pivots(tci: & mut TensorCI2 < T >, b: usize, f: & F, batched_f: & Option < B >, left_orthogonal: bool, options: & TCI2Options) -> Result < () >`

Update pivots at bond b using LU-based cross interpolation

### ` fn test_tensorci2_new()`

### ` fn test_tensorci2_requires_two_sites()`

### ` fn test_crossinterpolate2_constant()`

### ` fn test_crossinterpolate2_with_batch_function()`

### ` fn test_crossinterpolate2_rank2_function()`

