# tensor4all-itensorlike

## src/options.rs

### ` fn default() -> Self` (impl TruncateOptions)

### `pub fn svd() -> Self` (impl TruncateOptions)

Create options for SVD-based truncation.

### `pub fn lu() -> Self` (impl TruncateOptions)

Create options for LU-based truncation.

### `pub fn ci() -> Self` (impl TruncateOptions)

Create options for CI-based truncation.

### `pub fn with_rtol(mut self, rtol: f64) -> Self` (impl TruncateOptions)

Set the relative tolerance for truncation.

### `pub fn with_max_rank(mut self, max_rank: usize) -> Self` (impl TruncateOptions)

Set the maximum rank (bond dimension).

### `pub fn with_site_range(mut self, range: Range < usize >) -> Self` (impl TruncateOptions)

Set the site range for truncation. The range is 0-indexed with exclusive end. For example, `0..5` truncates bonds between sites 0-1, 1-2, 2-3, 3-4.

### ` fn default() -> Self` (impl ContractOptions)

### `pub fn zipup() -> Self` (impl ContractOptions)

Create options for zipup contraction.

### `pub fn fit() -> Self` (impl ContractOptions)

Create options for fit contraction.

### `pub fn naive() -> Self` (impl ContractOptions)

Create options for naive contraction. Note: Naive contraction is O(exp(n)) in memory and is primarily useful for debugging and testing.

### `pub fn with_max_rank(mut self, max_rank: usize) -> Self` (impl ContractOptions)

Set maximum bond dimension.

### `pub fn with_rtol(mut self, rtol: f64) -> Self` (impl ContractOptions)

Set relative tolerance.

### `pub fn with_nsweeps(mut self, nsweeps: usize) -> Self` (impl ContractOptions)

Set number of sweeps for Fit method.

## src/tensortrain.rs

### `pub fn new(tensors: Vec < TensorDynLen < Id , Symm > >) -> Result < Self >` (impl TensorTrain < Id , Symm >)

Create a new tensor train from a vector of tensors. The tensor train is created with no assumed orthogonality.

### `pub fn with_ortho(tensors: Vec < TensorDynLen < Id , Symm > >, llim: i32, rlim: i32, canonical_form: Option < CanonicalForm >) -> Result < Self >` (impl TensorTrain < Id , Symm >)

Create a new tensor train with specified orthogonality center. This is useful when constructing a tensor train that is already in canonical form.

### `pub fn len(&self) -> usize` (impl TensorTrain < Id , Symm >)

Number of sites (tensors) in the tensor train.

### `pub fn is_empty(&self) -> bool` (impl TensorTrain < Id , Symm >)

Check if the tensor train is empty.

### `pub fn llim(&self) -> i32` (impl TensorTrain < Id , Symm >)

Left orthogonality limit. Sites `0..llim` are guaranteed to be left-orthogonal. Returns -1 if no sites are left-orthogonal.

### `pub fn rlim(&self) -> i32` (impl TensorTrain < Id , Symm >)

Right orthogonality limit. Sites `rlim..len()` are guaranteed to be right-orthogonal. Returns `len() + 1` if no sites are right-orthogonal.

### `pub fn set_llim(&mut self, llim: i32)` (impl TensorTrain < Id , Symm >)

Set the left orthogonality limit.

### `pub fn set_rlim(&mut self, rlim: i32)` (impl TensorTrain < Id , Symm >)

Set the right orthogonality limit.

### `pub fn ortho_lims(&self) -> Range < usize >` (impl TensorTrain < Id , Symm >)

Get the orthogonality center range. Returns the range of sites that may not be orthogonal. If the tensor train is fully left-orthogonal, returns an empty range at the end.

### `pub fn isortho(&self) -> bool` (impl TensorTrain < Id , Symm >)

Check if the tensor train has a single orthogonality center. Returns true if there is exactly one site that is not guaranteed to be orthogonal.

### `pub fn orthocenter(&self) -> Option < usize >` (impl TensorTrain < Id , Symm >)

Get the orthogonality center (0-indexed). Returns `Some(site)` if the tensor train has a single orthogonality center, `None` otherwise.

### `pub fn canonical_form(&self) -> Option < CanonicalForm >` (impl TensorTrain < Id , Symm >)

Get the canonicalization method used.

### `pub fn set_canonical_form(&mut self, method: Option < CanonicalForm >)` (impl TensorTrain < Id , Symm >)

Set the canonicalization method.

### `pub fn tensor(&self, site: usize) -> & TensorDynLen < Id , Symm >` (impl TensorTrain < Id , Symm >)

Get a reference to the tensor at the given site.

### `pub fn tensor_checked(&self, site: usize) -> Result < & TensorDynLen < Id , Symm > >` (impl TensorTrain < Id , Symm >)

Get a reference to the tensor at the given site. Returns `Err` if `site >= len()`.

### `pub fn tensor_mut(&mut self, site: usize) -> & mut TensorDynLen < Id , Symm >` (impl TensorTrain < Id , Symm >)

Get a mutable reference to the tensor at the given site.

### `pub fn tensors(&self) -> Vec < & TensorDynLen < Id , Symm > >` (impl TensorTrain < Id , Symm >)

Get a reference to all tensors.

### `pub fn tensors_mut(&mut self) -> Vec < & mut TensorDynLen < Id , Symm > >` (impl TensorTrain < Id , Symm >)

Get a mutable reference to all tensors.

### `pub fn linkind(&self, i: usize) -> Option < Index < Id , Symm > >` (impl TensorTrain < Id , Symm >)

Get the link index between sites `i` and `i+1`. Returns `None` if `i >= len() - 1` or if no common index exists.

### `pub fn linkinds(&self) -> Vec < Index < Id , Symm > >` (impl TensorTrain < Id , Symm >)

Get all link indices. Returns a vector of length `len() - 1` containing the link indices.

### `pub fn sim_linkinds(&self) -> Self` (impl TensorTrain < Id , Symm >)

Create a copy with all link indices replaced by new unique IDs. This is useful for computing inner products where two tensor trains share link indices. By simulating (replacing) the link indices in one

### `pub fn siteinds(&self) -> Vec < Vec < Index < Id , Symm > > >` (impl TensorTrain < Id , Symm >)

Get the site indices (non-link indices) for all sites. For each site, returns a vector of indices that are not shared with adjacent tensors (i.e., the "physical" or "site" indices).

### `pub fn bond_dim(&self, i: usize) -> Option < usize >` (impl TensorTrain < Id , Symm >)

Get the bond dimension at link `i` (between sites `i` and `i+1`). Returns `None` if `i >= len() - 1`.

### `pub fn bond_dims(&self) -> Vec < usize >` (impl TensorTrain < Id , Symm >)

Get all bond dimensions. Returns a vector of length `len() - 1`.

### `pub fn maxbonddim(&self) -> usize` (impl TensorTrain < Id , Symm >)

Get the maximum bond dimension.

### `pub fn haslink(&self, i: usize) -> bool` (impl TensorTrain < Id , Symm >)

Check if two adjacent tensors share an index.

### `pub fn set_tensor(&mut self, site: usize, tensor: TensorDynLen < Id , Symm >)` (impl TensorTrain < Id , Symm >)

Replace the tensor at the given site. This invalidates orthogonality tracking.

### `pub fn orthogonalize(&mut self, site: usize) -> Result < () >` (impl TensorTrain < Id , Symm >)

Orthogonalize the tensor train to have orthogonality center at the given site. This function performs a series of factorizations to make the tensor train canonical with orthogonality center at `site`.

### `pub fn orthogonalize_with(&mut self, site: usize, form: CanonicalForm) -> Result < () >` (impl TensorTrain < Id , Symm >)

Orthogonalize with a specified canonical form.

### `pub fn truncate(&mut self, options: & TruncateOptions) -> Result < () >` (impl TensorTrain < Id , Symm >)

Truncate the tensor train bond dimensions. This delegates to the TreeTN's truncate_mut method, which performs a two-site sweep with Euler tour traversal for optimal truncation.

### `pub fn inner(&self, other: & Self) -> AnyScalar` (impl TensorTrain < Id , Symm >)

Compute the inner product (dot product) of two tensor trains. Computes `<self | other>` = sum over all indices of `conj(self) * other`. Both tensor trains must have the same site indices (same IDs).

### `pub fn norm_squared(&self) -> f64` (impl TensorTrain < Id , Symm >)

Compute the squared norm of the tensor train. Returns `<self | self>` = ||self||^2.

### `pub fn norm(&self) -> f64` (impl TensorTrain < Id , Symm >)

Compute the norm of the tensor train. Returns ||self|| = sqrt(<self | self>).

### `pub fn contract(&self, other: & Self, options: & ContractOptions) -> Result < Self >` (impl TensorTrain < Id , Symm >)

Contract two tensor trains with the same site indices. This contracts two tensor trains that share the same site indices, resulting in a new tensor train. The contraction is performed using

### ` fn default() -> Self` (impl TensorTrain < Id , Symm >)

### ` fn truncate_alg_to_form(alg: TruncateAlg) -> CanonicalForm`

Convert TruncateAlg to CanonicalForm. Note: SVD truncation algorithm corresponds to Unitary canonical form because both produce orthogonal/isometric tensors.

### ` fn make_tensor(indices: Vec < Index < DynId , NoSymmSpace > >) -> TensorDynLen < DynId , NoSymmSpace >`

Helper to create a simple tensor for testing using DynId

### ` fn idx(id: u128, size: usize) -> Index < DynId , NoSymmSpace >`

Helper to create an index with DynId

### ` fn test_empty_tt()`

### ` fn test_single_site_tt()`

### ` fn test_two_site_tt()`

### ` fn test_multi_site_indices()`

### ` fn test_ortho_tracking()`

### ` fn test_ortho_lims_range()`

### ` fn test_no_common_index_error()`

### ` fn test_orthogonalize_two_site()`

### ` fn test_orthogonalize_three_site()`

### ` fn test_orthogonalize_with_lu()`

### ` fn test_orthogonalize_with_ci()`

### ` fn test_truncate_with_max_rank()`

### ` fn test_inner_product()`

