# tensor4all-core

## src/algorithm.rs

### `pub fn from_i32(value: i32) -> Option < Self >` (impl FactorizeAlgorithm)

Create from C API integer representation. Returns `None` for invalid values.

### `pub fn to_i32(self) -> i32` (impl FactorizeAlgorithm)

Convert to C API integer representation.

### `pub fn name(&self) -> & 'static str` (impl FactorizeAlgorithm)

Get algorithm name as string.

### `pub fn from_i32(value: i32) -> Option < Self >` (impl ContractionAlgorithm)

Create from C API integer representation. Returns `None` for invalid values.

### `pub fn to_i32(self) -> i32` (impl ContractionAlgorithm)

Convert to C API integer representation.

### `pub fn name(&self) -> & 'static str` (impl ContractionAlgorithm)

Get algorithm name as string.

### `pub fn from_i32(value: i32) -> Option < Self >` (impl CanonicalForm)

Create from C API integer representation. Returns `None` for invalid values.

### `pub fn to_i32(self) -> i32` (impl CanonicalForm)

Convert to C API integer representation.

### `pub fn name(&self) -> & 'static str` (impl CanonicalForm)

Get form name as string.

### `pub fn from_i32(value: i32) -> Option < Self >` (impl CompressionAlgorithm)

Create from C API integer representation. Returns `None` for invalid values.

### `pub fn to_i32(self) -> i32` (impl CompressionAlgorithm)

Convert to C API integer representation.

### `pub fn name(&self) -> & 'static str` (impl CompressionAlgorithm)

Get algorithm name as string.

### ` fn test_factorize_algorithm_roundtrip()`

### ` fn test_contraction_algorithm_roundtrip()`

### ` fn test_compression_algorithm_roundtrip()`

### ` fn test_canonical_form_roundtrip()`

### ` fn test_invalid_values()`

### ` fn test_default()`

## src/any_scalar.rs

### ` fn sum_from_storage(storage: & Storage) -> Self` (impl AnyScalar)

### `pub fn new_real(x: f64) -> Self` (impl AnyScalar)

Create a real scalar value.

### `pub fn new_complex(re: f64, im: f64) -> Self` (impl AnyScalar)

Create a complex scalar value from real and imaginary parts.

### `pub fn is_complex(&self) -> bool` (impl AnyScalar)

Check if this scalar is complex.

### `pub fn real(&self) -> f64` (impl AnyScalar)

Get the real part of the scalar.

### `pub fn abs(&self) -> f64` (impl AnyScalar)

Get the absolute value (magnitude).

### `pub fn sqrt(&self) -> Self` (impl AnyScalar)

Compute square root. For negative real numbers, returns a complex number with the principal value.

### `pub fn powf(&self, exp: f64) -> Self` (impl AnyScalar)

Raise to a floating-point power. For negative real numbers, returns a complex number with the principal value.

### `pub fn powi(&self, exp: i32) -> Self` (impl AnyScalar)

Raise to an integer power.

### ` fn add(self, rhs: Self) -> Self :: Output` (impl AnyScalar)

### ` fn sub(self, rhs: Self) -> Self :: Output` (impl AnyScalar)

### ` fn mul(self, rhs: Self) -> Self :: Output` (impl AnyScalar)

### ` fn div(self, rhs: Self) -> Self :: Output` (impl AnyScalar)

### ` fn neg(self) -> Self :: Output` (impl AnyScalar)

### ` fn from(x: f64) -> Self` (impl AnyScalar)

### ` fn from(z: Complex64) -> Self` (impl AnyScalar)

### ` fn try_from(value: AnyScalar) -> Result < Self , Self :: Error >` (impl f64)

### ` fn from(value: AnyScalar) -> Self` (impl Complex64)

### ` fn default() -> Self` (impl AnyScalar)

### ` fn zero() -> Self` (impl AnyScalar)

### ` fn is_zero(&self) -> bool` (impl AnyScalar)

### ` fn one() -> Self` (impl AnyScalar)

### ` fn partial_cmp(&self, other: & Self) -> Option < std :: cmp :: Ordering >` (impl AnyScalar)

### ` fn fmt(&self, f: & mut fmt :: Formatter < '_ >) -> fmt :: Result` (impl AnyScalar)

## src/backend.rs

### `pub(crate) fn svd_backend(a: & mut DSlice < T , 2 >) -> Result < SVDDecomp < T > >`

Compute SVD decomposition using the selected backend.

### `pub(crate) fn qr_backend(a: & mut DSlice < T , 2 >) -> (DTensor < T , 2 > , DTensor < T , 2 >)`

Compute QR decomposition using the selected backend.

## src/contract.rs

### `pub fn contract_multi(tensors: & [TensorDynLen < Id , Symm >]) -> Result < TensorDynLen < Id , Symm > >`

Contract multiple tensors into a single tensor. Uses omeco's GreedyMethod to find the optimal contraction order for N>=3 tensors.

### ` fn contract_pair(a: & TensorDynLen < Id , Symm >, b: & TensorDynLen < Id , Symm >) -> Result < TensorDynLen < Id , Symm > >`

Contract two tensors over their common indices. If there are no common indices, performs outer product.

### ` fn contract_multi_optimized(tensors: & [TensorDynLen < Id , Symm >]) -> Result < TensorDynLen < Id , Symm > >`

Contract multiple tensors using omeco's GreedyMethod for optimal ordering.

### ` fn execute_contraction_tree(tensors: & [TensorDynLen < Id , Symm >], tree: & NestedEinsum < usize >) -> Result < TensorDynLen < Id , Symm > >`

Execute a contraction tree by recursively contracting tensors.

### ` fn make_test_tensor(shape: & [usize], ids: & [u128]) -> TensorDynLen < DynId , NoSymmSpace >`

### ` fn test_contract_multi_empty()`

### ` fn test_contract_multi_single()`

### ` fn test_contract_multi_pair()`

### ` fn test_contract_multi_three()`

### ` fn test_contract_multi_four()`

### ` fn test_contract_multi_no_contraction()`

## src/factorize.rs

### ` fn default() -> Self` (impl FactorizeOptions)

### `pub fn svd() -> Self` (impl FactorizeOptions)

Create options for SVD factorization.

### `pub fn qr() -> Self` (impl FactorizeOptions)

Create options for QR factorization.

### `pub fn lu() -> Self` (impl FactorizeOptions)

Create options for LU factorization.

### `pub fn ci() -> Self` (impl FactorizeOptions)

Create options for CI factorization.

### `pub fn with_canonical(mut self, canonical: Canonical) -> Self` (impl FactorizeOptions)

Set canonical direction.

### `pub fn with_rtol(mut self, rtol: f64) -> Self` (impl FactorizeOptions)

Set relative tolerance.

### `pub fn with_max_rank(mut self, max_rank: usize) -> Self` (impl FactorizeOptions)

Set maximum rank.

### `pub fn factorize(t: & TensorDynLen < Id , Symm >, left_inds: & [Index < Id , Symm >], options: & FactorizeOptions) -> Result < FactorizeResult < Id , Symm > , FactorizeError >`

Factorize a tensor into left and right factors. This function dispatches to the appropriate algorithm based on `options.alg`: - `SVD`: Singular Value Decomposition

### ` fn factorize_impl(t: & TensorDynLen < Id , Symm >, left_inds: & [Index < Id , Symm >], options: & FactorizeOptions) -> Result < FactorizeResult < Id , Symm > , FactorizeError >`

Internal implementation with scalar type.

### ` fn factorize_svd(t: & TensorDynLen < Id , Symm >, left_inds: & [Index < Id , Symm >], options: & FactorizeOptions) -> Result < FactorizeResult < Id , Symm > , FactorizeError >`

SVD factorization implementation.

### ` fn factorize_qr(t: & TensorDynLen < Id , Symm >, left_inds: & [Index < Id , Symm >], options: & FactorizeOptions) -> Result < FactorizeResult < Id , Symm > , FactorizeError >`

QR factorization implementation.

### ` fn factorize_lu(t: & TensorDynLen < Id , Symm >, left_inds: & [Index < Id , Symm >], options: & FactorizeOptions) -> Result < FactorizeResult < Id , Symm > , FactorizeError >`

LU factorization implementation.

### ` fn factorize_ci(t: & TensorDynLen < Id , Symm >, left_inds: & [Index < Id , Symm >], options: & FactorizeOptions) -> Result < FactorizeResult < Id , Symm > , FactorizeError >`

CI (Cross Interpolation) factorization implementation.

### ` fn extract_singular_values(s: & TensorDynLen < Id , Symm >) -> Vec < f64 >`

Extract singular values from a diagonal tensor.

### ` fn dtensor_to_matrix(tensor: & mdarray :: DTensor < T , 2 >, m: usize, n: usize) -> matrixci :: Matrix < T >`

Convert DTensor to Matrix (tensor4all-matrixci format).

### ` fn matrix_to_vec(matrix: & matrixci :: Matrix < T >) -> Vec < T >`

Convert Matrix to Vec for storage.

## src/index.rs

### `pub fn total_dim(&self) -> usize` (trait Symmetry)

Return the total dimension of the space. For no symmetry, this is just the dimension. For quantum number spaces, this is the sum of all block dimensions.

### `pub fn new(dim: usize) -> Self` (impl NoSymmSpace)

Create a new no-symmetry space with the given dimension.

### `pub fn dim(&self) -> usize` (impl NoSymmSpace)

Get the dimension.

### ` fn total_dim(&self) -> usize` (impl NoSymmSpace)

### `pub fn new() -> Self` (impl TagSet)

Create an empty tag set.

### `pub fn from_str(s: & str) -> Result < Self , TagSetError >` (impl TagSet)

Create a tag set from a comma-separated string.

### `pub fn from_tags(tags: & [& str]) -> Result < Self , TagSetError >` (impl TagSet)

Create a tag set from a slice of tag strings. Returns an error if any tag contains a comma (reserved as separator in `from_str`).

### `pub fn has_tag(&self, tag: & str) -> bool` (impl TagSet)

Check if a tag is present.

### `pub fn len(&self) -> usize` (impl TagSet)

Get the number of tags.

### `pub fn is_empty(&self) -> bool` (impl TagSet)

Check if the tag set is empty.

### `pub fn inner(&self) -> & Arc < InlineTagSet >` (impl TagSet)

Get the inner Arc for advanced use.

### ` fn deref(&self) -> & Self :: Target` (impl TagSet)

### ` fn len(&self) -> usize` (impl TagSet)

### ` fn capacity(&self) -> usize` (impl TagSet)

### ` fn get(&self, index: usize) -> Option < String >` (impl TagSet)

### ` fn iter(&self) -> TagSetIterator < '_ >` (impl TagSet)

### ` fn has_tag(&self, tag: & str) -> bool` (impl TagSet)

### ` fn add_tag(&mut self, tag: & str) -> Result < () , TagSetError >` (impl TagSet)

### ` fn remove_tag(&mut self, tag: & str) -> bool` (impl TagSet)

### `pub fn new(id: Id, symm: Symm) -> Self` (impl Index < Id , Symm , Tags >)

Create a new index with the given identity and symmetry.

### `pub fn new_with_tags(id: Id, symm: Symm, tags: Tags) -> Self` (impl Index < Id , Symm , Tags >)

Create a new index with the given identity, symmetry, and tags.

### `pub fn size(&self) -> usize` (impl Index < Id , Symm , Tags >)

Get the total dimension (size) of the index. This is computed from the symmetry information.

### `pub fn tags(&self) -> & Tags` (impl Index < Id , Symm , Tags >)

Get a reference to the tags.

### `pub fn new_with_size(id: Id, size: usize) -> Self` (impl Index < Id , NoSymmSpace , Tags >)

Create a new index with no symmetry from dimension. This is a convenience constructor for the common case of no symmetry.

### `pub fn new_with_size_and_tags(id: Id, size: usize, tags: Tags) -> Self` (impl Index < Id , NoSymmSpace , Tags >)

Create a new index with no symmetry from dimension and tags.

### `pub fn new_dyn(size: usize) -> Self` (impl Index < DynId , NoSymmSpace , TagSet >)

Create a new index with a generated dynamic ID and no tags.

### `pub fn new_dyn_with_tags(size: usize, tags: TagSet) -> Self` (impl Index < DynId , NoSymmSpace , TagSet >)

Create a new index with a generated dynamic ID and shared tags. This is the most efficient way to create many indices with the same tags. The `Arc` is cloned (reference count increment only), not the underlying data.

### `pub fn new_dyn_with_tag(size: usize, tag: & str) -> Result < Self , TagSetError >` (impl Index < DynId , NoSymmSpace , TagSet >)

Create a new index with a generated dynamic ID and a single tag. This creates a new `TagSet` with the given tag. For sharing the same tag across many indices, create the `TagSet`

### `pub fn new_link(size: usize) -> Result < Self , TagSetError >` (impl Index < DynId , NoSymmSpace , TagSet >)

Create a new bond index with "Link" tag (for SVD, QR, etc.). This is a convenience method for creating bond indices commonly used in tensor decompositions like SVD and QR factorization.

### ` fn eq(&self, other: & Self) -> bool` (impl Index < Id , Symm , Tags >)

### ` fn hash(&self, state: & mut H)` (impl Index < Id , Symm , Tags >)

### `pub(crate) fn generate_id() -> u128`

Generate a unique random ID for dynamic indices (thread-safe). Uses thread-local random number generator to generate UInt128 IDs, providing extremely low collision probability (see design.md for analysis).

### ` fn test_id_generation()`

### ` fn test_thread_local_rng_different_seeds()`

## src/index_ops.rs

### `pub fn sim(i: & Index < Id , Symm , Tags >) -> Index < Id , Symm , Tags >`

Create a similar index with the same space and tags but a new ID. This corresponds to ITensors.jl's `sim(i::Index)` function. It creates a new index with the same symmetry space (dimension/QN structure) and tags, but with

### `pub fn sim_owned(i: Index < Id , Symm , Tags >) -> Index < Id , Symm , Tags >`

Create a similar index with the same space and tags but a new ID (consumes input). This is an owned variant of `sim` that consumes the input index, avoiding unnecessary clones when you already own the index.

### ` fn fmt(&self, f: & mut std :: fmt :: Formatter < '_ >) -> std :: fmt :: Result` (impl ReplaceIndsError)

### `pub fn check_unique_indices(indices: & [Index < Id , Symm , Tags >]) -> Result < () , ReplaceIndsError >`

Check if a collection of indices contains any duplicates (by ID).

### `pub fn replaceinds(indices: Vec < Index < Id , Symm , Tags > >, replacements: & [(Index < Id , Symm , Tags > , Index < Id , Symm , Tags >)]) -> Result < Vec < Index < Id , Symm , Tags > > , ReplaceIndsError >`

Replace indices in a collection based on ID matching. This corresponds to ITensors.jl's `replaceinds` function. It replaces indices in `indices` that match (by ID) any of the `(old, new)` pairs in `replacements`.

### `pub fn replaceinds_in_place(indices: & mut [Index < Id , Symm , Tags >], replacements: & [(Index < Id , Symm , Tags > , Index < Id , Symm , Tags >)]) -> Result < () , ReplaceIndsError >`

Replace indices in-place based on ID matching. This is an in-place variant of `replaceinds` that modifies the input slice directly. Useful for performance-critical code where you want to avoid allocations.

### `pub fn unique_inds(indices_a: & [Index < Id , Symm , Tags >], indices_b: & [Index < Id , Symm , Tags >]) -> Vec < Index < Id , Symm , Tags > >`

Find indices that are unique to the first collection (set difference A \ B). Returns indices that appear in `indices_a` but not in `indices_b` (matched by ID). This corresponds to ITensors.jl's `uniqueinds` function.

### `pub fn noncommon_inds(indices_a: & [Index < Id , Symm , Tags >], indices_b: & [Index < Id , Symm , Tags >]) -> Vec < Index < Id , Symm , Tags > >`

Find indices that are not common between two collections (symmetric difference). Returns indices that appear in either `indices_a` or `indices_b` but not in both (matched by ID). This corresponds to ITensors.jl's `noncommoninds` function.

### `pub fn union_inds(indices_a: & [Index < Id , Symm , Tags >], indices_b: & [Index < Id , Symm , Tags >]) -> Vec < Index < Id , Symm , Tags > >`

Find the union of two index collections. Returns all unique indices from both collections (matched by ID). This corresponds to ITensors.jl's `unioninds` function.

### `pub fn hasind(indices: & [Index < Id , Symm , Tags >], index: & Index < Id , Symm , Tags >) -> bool`

Check if a collection contains a specific index (by ID). This corresponds to ITensors.jl's `hasind` function.

### `pub fn hasinds(indices: & [Index < Id , Symm , Tags >], targets: & [Index < Id , Symm , Tags >]) -> bool`

Check if a collection contains all of the specified indices (by ID). This corresponds to ITensors.jl's `hasinds` function.

### `pub fn hascommoninds(indices_a: & [Index < Id , Symm , Tags >], indices_b: & [Index < Id , Symm , Tags >]) -> bool`

Check if two collections have any common indices (by ID). This corresponds to ITensors.jl's `hascommoninds` function.

### `pub fn common_inds(indices_a: & [Index < Id , Symm , Tags >], indices_b: & [Index < Id , Symm , Tags >]) -> Vec < Index < Id , Symm , Tags > >`

Find common indices between two index collections. Returns a vector of indices that appear in both `indices_a` and `indices_b` (set intersection). This is similar to ITensors.jl's `commoninds` function.

## src/physical_indices.rs

### `pub fn new() -> Self` (impl PhysicalIndices < Id , Symm , Tags >)

Create a new empty PhysicalIndices manager.

### `pub fn with_capacity(sites: usize) -> Self` (impl PhysicalIndices < Id , Symm , Tags >)

Create a new PhysicalIndices manager with the given capacity for sites.

### ` fn update_flattened_indices(&mut self)` (impl PhysicalIndices < Id , Symm , Tags >)

Update the flattened index lists from the current physical_indices. This should be called whenever physical_indices are modified to keep the flattened lists in sync.

### `pub fn add_site_indices(&mut self, site_index: usize, indices: Vec < Index < Id , Symm , Tags > >, tensor_id: usize)` (impl PhysicalIndices < Id , Symm , Tags >)

Add physical indices to a site and bind the site to a tensor ID. If the site already has a tensor ID set, it must match `tensor_id`. The new indices are appended in order.

### `pub fn set_site_indices(&mut self, site_index: usize, indices: Vec < Index < Id , Symm , Tags > >, tensor_id: usize)` (impl PhysicalIndices < Id , Symm , Tags >)

Set physical indices for a site, replacing any existing indices.

### `pub fn get_site_indices(&self, site_index: usize) -> Option < & [Index < Id , Symm , Tags >] >` (impl PhysicalIndices < Id , Symm , Tags >)

Get the physical indices for a site. Returns `None` if the site doesn't exist.

### `pub fn get_site_tensor_id(&self, site_index: usize) -> Option < usize >` (impl PhysicalIndices < Id , Symm , Tags >)

Get the tensor integer ID for a site. Returns `None` if the site doesn't exist.

### `pub fn num_sites(&self) -> usize` (impl PhysicalIndices < Id , Symm , Tags >)

Get the number of sites.

### `pub fn total_indices(&self) -> usize` (impl PhysicalIndices < Id , Symm , Tags >)

Get the total number of physical indices across all sites.

### `pub fn all_indices(&self) -> & [Vec < Index < Id , Symm , Tags > >]` (impl PhysicalIndices < Id , Symm , Tags >)

Get a reference to all physical indices (organized by site).

### `pub fn all_tensor_ids_by_site(&self) -> & [Option < usize >]` (impl PhysicalIndices < Id , Symm , Tags >)

Get a reference to all tensor IDs (organized by site).

### `pub fn unsorted_indices(&self) -> & [Index < Id , Symm , Tags >]` (impl PhysicalIndices < Id , Symm , Tags >)

Get a reference to the unsorted flattened indices (in tensor order).

### `pub fn sorted_indices(&self) -> & [Index < Id , Symm , Tags >]` (impl PhysicalIndices < Id , Symm , Tags >)

Get a reference to the sorted flattened indices (sorted by ID).

### `pub fn clear(&mut self)` (impl PhysicalIndices < Id , Symm , Tags >)

Clear all physical indices and tensor IDs.

### `pub fn remove_site(&mut self, site_index: usize) -> bool` (impl PhysicalIndices < Id , Symm , Tags >)

Remove a site and all its physical indices. Returns `true` if the site existed and was removed, `false` otherwise.

### ` fn default() -> Self` (impl PhysicalIndices < Id , Symm , Tags >)

### ` fn eq(&self, other: & Self) -> bool` (impl PhysicalIndices < Id , Symm , Tags >)

Two `PhysicalIndices` are equal if and only if their sorted flattened index lists match.

## src/qr.rs

### ` fn default() -> Self` (impl QrOptions)

### `pub fn with_rtol(rtol: f64) -> Self` (impl QrOptions)

Create new QR options with the specified rtol.

### `pub fn default_qr_rtol() -> f64`

Get the global default rtol for QR truncation. The default value is 1e-15 (very strict, near machine precision).

### `pub fn set_default_qr_rtol(rtol: f64) -> Result < () , QrError >`

Set the global default rtol for QR truncation.

### ` fn compute_retained_rank_qr(r_full: & DTensor < T , 2 >, k: usize, n: usize, rtol: f64) -> usize`

Compute the retained rank based on rtol truncation for QR. This checks R's diagonal elements and truncates columns where |R[i, i]| < rtol.

### `pub fn qr(t: & TensorDynLen < Id , Symm >, left_inds: & [Index < Id , Symm >]) -> Result < (TensorDynLen < Id , Symm > , TensorDynLen < Id , Symm >) , QrError >`

Compute QR decomposition of a tensor with arbitrary rank, returning (Q, R). This function uses the global default rtol for truncation. See `qr_with` for per-call rtol control.

### `pub fn qr_with(t: & TensorDynLen < Id , Symm >, left_inds: & [Index < Id , Symm >], options: & QrOptions) -> Result < (TensorDynLen < Id , Symm > , TensorDynLen < Id , Symm >) , QrError >`

Compute QR decomposition of a tensor with arbitrary rank, returning (Q, R). This function allows per-call control of the truncation tolerance via `QrOptions`. If `options.rtol` is `None`, uses the global default rtol.

### `pub fn qr_c64(t: & TensorDynLen < Id , Symm >, left_inds: & [Index < Id , Symm >]) -> Result < (TensorDynLen < Id , Symm > , TensorDynLen < Id , Symm >) , QrError >`

Compute QR decomposition of a complex tensor with arbitrary rank, returning (Q, R). This is a convenience wrapper around the generic `qr` function for `Complex64` tensors. For the mathematical convention:

## src/smallstring.rs

### `pub fn from_char(c: char) -> Option < Self >` (trait SmallChar)

Convert from a Rust char. Returns None if the character cannot be represented in this type.

### `pub fn to_char(self) -> char` (trait SmallChar)

Convert to a Rust char.

### ` fn from_char(c: char) -> Option < Self >` (impl u16)

### ` fn to_char(self) -> char` (impl u16)

### ` fn from_char(c: char) -> Option < Self >` (impl char)

### ` fn to_char(self) -> char` (impl char)

### `pub fn new() -> Self` (impl SmallString < MAX_LEN , C >)

Create an empty SmallString.

### `pub fn from_str(s: & str) -> Result < Self , SmallStringError >` (impl SmallString < MAX_LEN , C >)

Create a SmallString from a string slice. Returns an error if: - The string is longer than MAX_LEN characters

### `pub fn as_str(&self) -> String` (impl SmallString < MAX_LEN , C >)

Convert to a String.

### `pub fn is_empty(&self) -> bool` (impl SmallString < MAX_LEN , C >)

Check if the string is empty.

### `pub fn len(&self) -> usize` (impl SmallString < MAX_LEN , C >)

Get the length of the string.

### `pub fn capacity(&self) -> usize` (impl SmallString < MAX_LEN , C >)

Get the maximum capacity.

### `pub fn get(&self, index: usize) -> Option < char >` (impl SmallString < MAX_LEN , C >)

Get a character at the given index.

### `pub fn as_slice(&self) -> & [C]` (impl SmallString < MAX_LEN , C >)

Get a reference to the internal data slice.

### ` fn default() -> Self` (impl SmallString < MAX_LEN , C >)

### ` fn eq(&self, other: & Self) -> bool` (impl SmallString < MAX_LEN , C >)

### ` fn partial_cmp(&self, other: & Self) -> Option < std :: cmp :: Ordering >` (impl SmallString < MAX_LEN , C >)

### ` fn cmp(&self, other: & Self) -> std :: cmp :: Ordering` (impl SmallString < MAX_LEN , C >)

### ` fn fmt(&self, f: & mut std :: fmt :: Formatter < '_ >) -> std :: fmt :: Result` (impl SmallString < MAX_LEN , C >)

### ` fn test_smallstring_u16_basic()`

### ` fn test_smallstring_u16_japanese()`

### ` fn test_smallstring_u16_emoji_fails()`

### ` fn test_smallstring_char_emoji()`

### ` fn test_smallstring_too_long()`

### ` fn test_smallstring_ordering()`

### ` fn test_smallstring_size()`

## src/storage.rs

### `pub fn with_capacity(capacity: usize) -> Self` (impl DenseStorageF64)

### `pub fn from_vec(vec: Vec < f64 >) -> Self` (impl DenseStorageF64)

### `pub fn random(rng: & mut R, size: usize) -> Self` (impl DenseStorageF64)

Create storage with random values from standard normal distribution.

### `pub fn as_slice(&self) -> & [f64]` (impl DenseStorageF64)

### `pub fn as_mut_slice(&mut self) -> & mut [f64]` (impl DenseStorageF64)

### `pub fn into_vec(self) -> Vec < f64 >` (impl DenseStorageF64)

### `pub fn len(&self) -> usize` (impl DenseStorageF64)

### `pub fn capacity(&self) -> usize` (impl DenseStorageF64)

### `pub fn push(&mut self, val: f64)` (impl DenseStorageF64)

### `pub fn extend_from_slice(&mut self, other: & [f64])` (impl DenseStorageF64)

### `pub fn extend(&mut self, iter: I)` (impl DenseStorageF64)

### `pub fn get(&self, i: usize) -> f64` (impl DenseStorageF64)

### `pub fn set(&mut self, i: usize, val: f64)` (impl DenseStorageF64)

### `pub fn iter(&self) -> std :: slice :: Iter < '_ , f64 >` (impl DenseStorageF64)

### `pub fn permute(&self, dims: & [usize], perm: & [usize]) -> Self` (impl DenseStorageF64)

Permute the dense storage data according to the given permutation.

### `pub fn contract(&self, dims: & [usize], axes: & [usize], other: & Self, other_dims: & [usize], other_axes: & [usize]) -> Self` (impl DenseStorageF64)

Contract this dense storage with another dense storage. This method handles non-contiguous contracted axes by permuting the tensors to make the contracted axes contiguous before calling mdarray-linalg's contract.

### ` fn contract_via_gemm(a: & [f64], dims_a: & [usize], axes_a: & [usize], b: & [f64], dims_b: & [usize], axes_b: & [usize]) -> Vec < f64 >`

Contract two tensors via GEMM (matrix multiplication). This function assumes that contracted axes are already contiguous: - For `a`: contracted axes are at the END (axes_a are the last naxes positions)

### ` fn contract_via_gemm_c64(a: & [Complex64], dims_a: & [usize], axes_a: & [usize], b: & [Complex64], dims_b: & [usize], axes_b: & [usize]) -> Vec < Complex64 >`

Contract two Complex64 tensors via GEMM (matrix multiplication). Same as contract_via_gemm but for complex numbers.

### ` fn compute_contraction_permutation(dims: & [usize], axes: & [usize], axes_at_front: bool) -> (Vec < usize > , Vec < usize > , Vec < usize >)`

Compute permutation to make contracted axes contiguous. If `axes_at_front` is true, contracted axes are moved to the front (maintaining original order). If false, contracted axes are moved to the end (maintaining original order).

### `pub fn with_capacity(capacity: usize) -> Self` (impl DenseStorageC64)

### `pub fn from_vec(vec: Vec < Complex64 >) -> Self` (impl DenseStorageC64)

### `pub fn random(rng: & mut R, size: usize) -> Self` (impl DenseStorageC64)

Create storage with random complex values (re, im both from standard normal).

### `pub fn as_slice(&self) -> & [Complex64]` (impl DenseStorageC64)

### `pub fn as_mut_slice(&mut self) -> & mut [Complex64]` (impl DenseStorageC64)

### `pub fn into_vec(self) -> Vec < Complex64 >` (impl DenseStorageC64)

### `pub fn len(&self) -> usize` (impl DenseStorageC64)

### `pub fn capacity(&self) -> usize` (impl DenseStorageC64)

### `pub fn push(&mut self, val: Complex64)` (impl DenseStorageC64)

### `pub fn extend_from_slice(&mut self, other: & [Complex64])` (impl DenseStorageC64)

### `pub fn extend(&mut self, iter: I)` (impl DenseStorageC64)

### `pub fn get(&self, i: usize) -> Complex64` (impl DenseStorageC64)

### `pub fn set(&mut self, i: usize, val: Complex64)` (impl DenseStorageC64)

### `pub fn permute(&self, dims: & [usize], perm: & [usize]) -> Self` (impl DenseStorageC64)

Permute the dense storage data according to the given permutation.

### `pub fn contract(&self, dims: & [usize], axes: & [usize], other: & Self, other_dims: & [usize], other_axes: & [usize]) -> Self` (impl DenseStorageC64)

Contract this dense storage with another dense storage. This method handles non-contiguous contracted axes by permuting the tensors to make the contracted axes contiguous before calling mdarray-linalg's contract.

### `pub fn from_vec(vec: Vec < f64 >) -> Self` (impl DiagStorageF64)

### `pub fn as_slice(&self) -> & [f64]` (impl DiagStorageF64)

### `pub fn as_mut_slice(&mut self) -> & mut [f64]` (impl DiagStorageF64)

### `pub fn into_vec(self) -> Vec < f64 >` (impl DiagStorageF64)

### `pub fn len(&self) -> usize` (impl DiagStorageF64)

### `pub fn get(&self, i: usize) -> f64` (impl DiagStorageF64)

### `pub fn set(&mut self, i: usize, val: f64)` (impl DiagStorageF64)

### `pub fn to_dense_vec(&self, dims: & [usize]) -> Vec < f64 >` (impl DiagStorageF64)

Convert diagonal storage to a dense vector representation. Creates a dense vector with diagonal elements set and off-diagonal elements as zero.

### `pub fn contract_diag_diag(&self, dims: & [usize], other: & Self, other_dims: & [usize], result_dims: & [usize]) -> Storage` (impl DiagStorageF64)

Contract this diagonal storage with another diagonal storage. Returns either a scalar (DenseStorageF64 with one element) or a diagonal storage.

### `pub fn from_vec(vec: Vec < Complex64 >) -> Self` (impl DiagStorageC64)

### `pub fn as_slice(&self) -> & [Complex64]` (impl DiagStorageC64)

### `pub fn as_mut_slice(&mut self) -> & mut [Complex64]` (impl DiagStorageC64)

### `pub fn into_vec(self) -> Vec < Complex64 >` (impl DiagStorageC64)

### `pub fn len(&self) -> usize` (impl DiagStorageC64)

### `pub fn get(&self, i: usize) -> Complex64` (impl DiagStorageC64)

### `pub fn set(&mut self, i: usize, val: Complex64)` (impl DiagStorageC64)

### `pub fn to_dense_vec(&self, dims: & [usize]) -> Vec < Complex64 >` (impl DiagStorageC64)

Convert diagonal storage to a dense vector representation. Creates a dense vector with diagonal elements set and off-diagonal elements as zero.

### `pub fn contract_diag_diag(&self, dims: & [usize], other: & Self, other_dims: & [usize], result_dims: & [usize]) -> Storage` (impl DiagStorageC64)

Contract this diagonal storage with another diagonal storage. Returns either a scalar (DenseStorageC64 with one element) or a diagonal storage.

### `pub fn new_dense(capacity: usize) -> Storage` (trait DenseStorageFactory)

### ` fn new_dense(capacity: usize) -> Storage` (impl f64)

### ` fn new_dense(capacity: usize) -> Storage` (impl Complex64)

### `pub fn sum_from_storage(storage: & Storage) -> Self` (trait SumFromStorage)

### ` fn sum_from_storage(storage: & Storage) -> Self` (impl f64)

### ` fn sum_from_storage(storage: & Storage) -> Self` (impl Complex64)

### `pub fn new_dense_f64(capacity: usize) -> Self` (impl Storage)

Create a new DenseF64 storage with the given capacity.

### `pub fn new_dense_c64(capacity: usize) -> Self` (impl Storage)

Create a new DenseC64 storage with the given capacity.

### `pub fn new_diag_f64(diag_data: Vec < f64 >) -> Self` (impl Storage)

Create a new DiagF64 storage with the given diagonal data.

### `pub fn new_diag_c64(diag_data: Vec < Complex64 >) -> Self` (impl Storage)

Create a new DiagC64 storage with the given diagonal data.

### `pub fn is_diag(&self) -> bool` (impl Storage)

Check if this storage is a Diag storage type.

### `pub fn len(&self) -> usize` (impl Storage)

Get the length of the storage (number of elements).

### `pub fn sum_f64(&self) -> f64` (impl Storage)

Sum all elements as f64.

### `pub fn sum_c64(&self) -> Complex64` (impl Storage)

Sum all elements as Complex64.

### `pub fn to_dense_storage(&self, dims: & [usize]) -> Storage` (impl Storage)

Convert this storage to dense storage. For Diag storage, creates a Dense storage with diagonal elements set and off-diagonal elements as zero.

### `pub fn permute_storage(&self, dims: & [usize], perm: & [usize]) -> Storage` (impl Storage)

Permute the storage data according to the given permutation.

### `pub fn extract_real_part(&self) -> Storage` (impl Storage)

Extract real part from Complex64 storage as f64 storage. For f64 storage, returns a copy.

### `pub fn extract_imag_part(&self, dims: & [usize]) -> Storage` (impl Storage)

Extract imaginary part from Complex64 storage as f64 storage. For f64 storage, returns zero storage (will be resized appropriately).

### `pub fn to_complex_storage(&self) -> Storage` (impl Storage)

Convert f64 storage to Complex64 storage (real part only, imaginary part is zero). For Complex64 storage, returns a copy.

### `pub fn conj(&self) -> Self` (impl Storage)

Complex conjugate of all elements. For real (f64) storage, returns a copy (conjugate of real is identity). For complex (Complex64) storage, conjugates each element.

### `pub fn combine_to_complex(real_storage: & Storage, imag_storage: & Storage) -> Storage` (impl Storage)

Combine two f64 storages into Complex64 storage. real_storage becomes the real part, imag_storage becomes the imaginary part. Formula: real + i * imag

### `pub fn try_add(&self, other: & Storage) -> Result < Storage , String >` (impl Storage)

Add two storages element-wise, returning `Result` on error instead of panicking. Both storages must have the same type and length.

### `pub fn make_mut_storage(arc: & mut Arc < Storage >) -> & mut Storage`

Helper to get a mutable reference to storage, cloning if needed (COW).

### `pub fn mindim(dims: & [usize]) -> usize`

Get the minimum dimension from a slice of dimensions. This is used for DiagTensor where all indices must have the same dimension.

### `pub fn contract_storage(storage_a: & Storage, dims_a: & [usize], axes_a: & [usize], storage_b: & Storage, dims_b: & [usize], axes_b: & [usize], result_dims: & [usize]) -> Storage`

Contract two storage tensors along specified axes. This is an internal helper function that contracts two `Storage` tensors. For Dense tensors, uses mdarray-linalg's contract method.

### `pub fn extract_dense_view(storage: & 'a Storage) -> Result < & 'a [Self] , String >` (trait StorageScalar)

Extract a borrowed view of dense storage data (no copy). Returns an error if the storage is not the matching dense type.

### `pub fn extract_dense_cow(storage: & 'a Storage) -> Result < Cow < 'a , [Self] > , String >` (trait StorageScalar default)

Extract dense storage data as `Cow` (borrowed if possible, owned if needed). For dense storage, returns `Cow::Borrowed` (no copy). For other storage types, may need to convert to dense first (copy).

### `pub fn extract_dense(storage: & Storage) -> Result < Vec < Self > , String >` (trait StorageScalar default)

Extract dense storage data as owned `Vec` (always copies). This is a convenience method that calls `extract_dense_cow` and converts to owned.

### `pub fn dense_storage(data: Vec < Self >) -> Arc < Storage >` (trait StorageScalar)

Create `Storage` from owned dense data.

### `pub fn storage_to_dtensor(storage: & Storage, shape: [usize ; 2]) -> Result < DTensor < T , 2 > , String >`

Convert dense storage to a DTensor with rank 2. This function extracts data from dense storage and reshapes it into a `DTensor<T, 2>` with the specified shape `[m, n]`. The data length must match `m * n`.

### ` fn extract_dense_view(storage: & 'a Storage) -> Result < & 'a [Self] , String >` (impl f64)

### ` fn dense_storage(data: Vec < Self >) -> Arc < Storage >` (impl f64)

### ` fn extract_dense_view(storage: & 'a Storage) -> Result < & 'a [Self] , String >` (impl Complex64)

### ` fn dense_storage(data: Vec < Self >) -> Arc < Storage >` (impl Complex64)

### ` fn add(self, rhs: & Storage) -> Self :: Output` (impl & Storage)

### ` fn mul(self, scalar: f64) -> Self :: Output` (impl & Storage)

### ` fn mul(self, scalar: Complex64) -> Self :: Output` (impl & Storage)

### ` fn mul(self, scalar: AnyScalar) -> Self :: Output` (impl & Storage)

## src/svd.rs

### ` fn default() -> Self` (impl SvdOptions)

### `pub fn with_rtol(rtol: f64) -> Self` (impl SvdOptions)

Create new SVD options with the specified rtol.

### `pub fn default_svd_rtol() -> f64`

Get the global default rtol for SVD truncation. The default value is 1e-12 (near machine precision).

### `pub fn set_default_svd_rtol(rtol: f64) -> Result < () , SvdError >`

Set the global default rtol for SVD truncation.

### ` fn compute_retained_rank(s_vec: & [f64], rtol: f64) -> usize`

Compute the retained rank based on rtol (TSVD truncation). This implements the truncation criterion: sum_{i>r} σ_i² / sum_i σ_i² <= rtol²

### ` fn extract_usv_from_svd_decomp(decomp: SVDDecomp < T >, m: usize, n: usize, k: usize) -> (Vec < T > , Vec < f64 > , Vec < T >)`

Extract U, S, V from mdarray-linalg's SVDDecomp (which returns U, S, Vt). This helper function converts the backend's SVD result to our desired format: - Extracts singular values from the diagonal view (first row)

### `pub fn svd(t: & TensorDynLen < Id , Symm >, left_inds: & [Index < Id , Symm >]) -> Result < (TensorDynLen < Id , Symm > , TensorDynLen < Id , Symm > , TensorDynLen < Id , Symm > ,) , SvdError , >`

Compute SVD decomposition of a tensor with arbitrary rank, returning (U, S, V). This function uses the global default rtol for truncation. See `svd_with` for per-call rtol control.

### `pub fn svd_with(t: & TensorDynLen < Id , Symm >, left_inds: & [Index < Id , Symm >], options: & SvdOptions) -> Result < (TensorDynLen < Id , Symm > , TensorDynLen < Id , Symm > , TensorDynLen < Id , Symm > ,) , SvdError , >`

Compute SVD decomposition of a tensor with arbitrary rank, returning (U, S, V). This function allows per-call control of the truncation tolerance via `SvdOptions`. If `options.rtol` is `None`, uses the global default rtol.

### `pub fn svd_c64(t: & TensorDynLen < Id , Symm >, left_inds: & [Index < Id , Symm >]) -> Result < (TensorDynLen < Id , Symm > , TensorDynLen < Id , Symm > , TensorDynLen < Id , Symm > ,) , SvdError , >`

Compute SVD decomposition of a complex tensor with arbitrary rank, returning (U, S, V). This is a convenience wrapper around the generic `svd` function for `Complex64` tensors. For complex-valued matrices, the mathematical convention is:

## src/tagset.rs

### `pub fn len(&self) -> usize` (trait TagSetLike)

Get the number of tags.

### `pub fn is_empty(&self) -> bool` (trait TagSetLike default)

Check if the tag set is empty.

### `pub fn capacity(&self) -> usize` (trait TagSetLike)

Get the maximum capacity (if applicable). For unbounded implementations, this may return `usize::MAX` or a reasonable upper bound.

### `pub fn get(&self, index: usize) -> Option < String >` (trait TagSetLike)

Get a tag at the given index as a string. Returns `None` if the index is out of bounds.

### `pub fn iter(&self) -> TagSetIterator < '_ >` (trait TagSetLike)

Iterate over tags as strings.

### `pub fn has_tag(&self, tag: & str) -> bool` (trait TagSetLike)

Check if a tag is present.

### `pub fn has_tags(&self, other: & T) -> bool` (trait TagSetLike default)

Check if all tags in another tag set are present. This allows comparing different tag set implementations.

### `pub fn add_tag(&mut self, tag: & str) -> Result < () , TagSetError >` (trait TagSetLike)

Add a tag (maintains sorted order). Returns an error if the tag cannot be added (e.g., capacity exceeded, invalid tag).

### `pub fn remove_tag(&mut self, tag: & str) -> bool` (trait TagSetLike)

Remove a tag. Returns `true` if the tag was present and removed, `false` otherwise.

### `pub fn common_tags(&self, other: & T) -> Self` (trait TagSetLike default)

Get common tags between this tag set and another. Returns a new tag set containing only tags present in both.

### `pub fn from_str(s: & str) -> Result < Self , TagSetError >` (trait TagSetLike default)

Create a tag set from a comma-separated string. Whitespace is ignored (similar to ITensors.jl). Tags are automatically sorted.

### `pub fn new() -> Self` (impl TagSet < MAX_TAGS , MAX_TAG_LEN , C >)

Create an empty TagSet.

### `pub fn from_str(s: & str) -> Result < Self , TagSetError >` (impl TagSet < MAX_TAGS , MAX_TAG_LEN , C >)

Create a TagSet from a comma-separated string. Whitespace is ignored (similar to ITensors.jl). Tags are automatically sorted.

### `pub fn len(&self) -> usize` (impl TagSet < MAX_TAGS , MAX_TAG_LEN , C >)

Get the number of tags.

### `pub fn is_empty(&self) -> bool` (impl TagSet < MAX_TAGS , MAX_TAG_LEN , C >)

Check if the tag set is empty.

### `pub fn capacity(&self) -> usize` (impl TagSet < MAX_TAGS , MAX_TAG_LEN , C >)

Get the maximum capacity.

### `pub fn get(&self, index: usize) -> Option < & SmallString < MAX_TAG_LEN , C > >` (impl TagSet < MAX_TAGS , MAX_TAG_LEN , C >)

Get a tag at the given index.

### `pub fn iter(&self) -> impl Iterator < Item = & SmallString < MAX_TAG_LEN , C > >` (impl TagSet < MAX_TAGS , MAX_TAG_LEN , C >)

Iterate over tags.

### `pub fn has_tag(&self, tag: & str) -> bool` (impl TagSet < MAX_TAGS , MAX_TAG_LEN , C >)

Check if a tag is present.

### `pub fn has_tags(&self, tags: & TagSet < MAX_TAGS , MAX_TAG_LEN , C >) -> bool` (impl TagSet < MAX_TAGS , MAX_TAG_LEN , C >)

Check if all tags in another TagSet are present.

### `pub fn add_tag(&mut self, tag: & str) -> Result < () , TagSetError >` (impl TagSet < MAX_TAGS , MAX_TAG_LEN , C >)

Add a tag (maintains sorted order).

### `pub fn remove_tag(&mut self, tag: & str) -> bool` (impl TagSet < MAX_TAGS , MAX_TAG_LEN , C >)

Remove a tag.

### `pub fn common_tags(&self, other: & Self) -> Self` (impl TagSet < MAX_TAGS , MAX_TAG_LEN , C >)

Get common tags between two TagSets.

### ` fn len(&self) -> usize` (impl TagSet < MAX_TAGS , MAX_TAG_LEN , C >)

### ` fn capacity(&self) -> usize` (impl TagSet < MAX_TAGS , MAX_TAG_LEN , C >)

### ` fn get(&self, index: usize) -> Option < String >` (impl TagSet < MAX_TAGS , MAX_TAG_LEN , C >)

### ` fn iter(&self) -> TagSetIterator < '_ >` (impl TagSet < MAX_TAGS , MAX_TAG_LEN , C >)

### ` fn has_tag(&self, tag: & str) -> bool` (impl TagSet < MAX_TAGS , MAX_TAG_LEN , C >)

### ` fn add_tag(&mut self, tag: & str) -> Result < () , TagSetError >` (impl TagSet < MAX_TAGS , MAX_TAG_LEN , C >)

### ` fn remove_tag(&mut self, tag: & str) -> bool` (impl TagSet < MAX_TAGS , MAX_TAG_LEN , C >)

### ` fn _add_tag_ordered(&mut self, tag: SmallString < MAX_TAG_LEN , C >) -> Result < () , TagSetError >` (impl TagSet < MAX_TAGS , MAX_TAG_LEN , C >)

Internal: Add a tag in sorted order (similar to ITensors.jl's `_addtag_ordered!`).

### ` fn _has_tag(&self, tag: & SmallString < MAX_TAG_LEN , C >) -> bool` (impl TagSet < MAX_TAGS , MAX_TAG_LEN , C >)

Internal: Check if a tag is present (binary search).

### ` fn default() -> Self` (impl TagSet < MAX_TAGS , MAX_TAG_LEN , C >)

### ` fn eq(&self, other: & Self) -> bool` (impl TagSet < MAX_TAGS , MAX_TAG_LEN , C >)

## src/tensor.rs

### `pub fn compute_permutation_from_indices(original_indices: & [Index < Id , Symm >], new_indices: & [Index < Id , Symm >]) -> Vec < usize >`

Compute the permutation array from original indices to new indices. This function finds the mapping from new indices to original indices by matching index IDs. The result is a permutation array `perm` such that

### `pub fn indices(&self) -> & [Index < Self :: Id , Self :: Symm >]` (trait TensorAccess)

Get a reference to the indices.

### `pub fn storage(&self) -> & Storage` (trait TensorAccess)

Get a reference to the storage.

### ` fn indices(&self) -> & [Index < Self :: Id , Self :: Symm >]` (impl TensorDynLen < Id , Symm >)

### ` fn storage(&self) -> & Storage` (impl TensorDynLen < Id , Symm >)

### `pub fn new(indices: Vec < Index < Id , Symm > >, dims: Vec < usize >, storage: Arc < Storage >) -> Self` (impl TensorDynLen < Id , Symm >)

Create a new tensor with dynamic rank. Dimensions are automatically computed from the indices using `Index::size()`.

### `pub fn from_indices(indices: Vec < Index < Id , Symm > >, storage: Arc < Storage >) -> Self` (impl TensorDynLen < Id , Symm >)

Create a new tensor with dynamic rank, automatically computing dimensions from indices. This is a convenience constructor that extracts dimensions from indices using `Index::size()`.

### `pub fn storage_mut(&mut self) -> & mut Storage` (impl TensorDynLen < Id , Symm >)

Get a mutable reference to storage (COW: clones if shared).

### `pub fn sum(&self) -> AnyScalar` (impl TensorDynLen < Id , Symm >)

Sum all elements, returning `AnyScalar`.

### `pub fn sum_f64(&self) -> f64` (impl TensorDynLen < Id , Symm >)

Sum all elements as f64.

### `pub fn only(&self) -> AnyScalar` (impl TensorDynLen < Id , Symm >)

Extract the scalar value from a 0-dimensional tensor (or 1-element tensor). This is similar to Julia's `only()` function.

### `pub fn permute_indices(&self, new_indices: & [Index < Id , Symm >]) -> Self` (impl TensorDynLen < Id , Symm >)

Permute the tensor dimensions using the given new indices order. This is the main permutation method that takes the desired new indices and automatically computes the corresponding permutation of dimensions

### `pub fn permute(&self, perm: & [usize]) -> Self` (impl TensorDynLen < Id , Symm >)

Permute the tensor dimensions, returning a new tensor. This method reorders the indices, dimensions, and data according to the given permutation. The permutation specifies which old axis each new

### `pub fn contract(&self, other: & Self) -> Self` (impl TensorDynLen < Id , Symm >)

Contract this tensor with another tensor along common indices. This method finds common indices between `self` and `other`, then contracts along those indices. The result tensor contains all non-contracted indices

### `pub fn tensordot(&self, other: & Self, pairs: & [(Index < Id , Symm > , Index < Id , Symm >)]) -> Result < Self >` (impl TensorDynLen < Id , Symm >)

Contract this tensor with another tensor along explicitly specified index pairs. Similar to NumPy's `tensordot`, this method contracts only along the explicitly specified pairs of indices. Unlike `contract()` which automatically contracts

### `pub fn outer_product(&self, other: & Self) -> Result < Self >` (impl TensorDynLen < Id , Symm >)

Compute the outer product (tensor product) of two tensors. Creates a new tensor whose indices are the concatenation of the indices from both input tensors. The result has shape `[...self.dims, ...other.dims]`.

### `pub fn random_f64(rng: & mut R, indices: Vec < Index < Id , Symm > >) -> Self` (impl TensorDynLen < Id , Symm >)

Create a random f64 tensor with values from standard normal distribution.

### `pub fn random_c64(rng: & mut R, indices: Vec < Index < Id , Symm > >) -> Self` (impl TensorDynLen < Id , Symm >)

Create a random Complex64 tensor with values from standard normal distribution. Both real and imaginary parts are drawn from standard normal distribution.

### ` fn mul(self, other: & TensorDynLen < Id , Symm >) -> Self :: Output` (impl & TensorDynLen < Id , Symm >)

### ` fn mul(self, other: TensorDynLen < Id , Symm >) -> Self :: Output` (impl TensorDynLen < Id , Symm >)

### ` fn mul(self, other: TensorDynLen < Id , Symm >) -> Self :: Output` (impl & TensorDynLen < Id , Symm >)

### ` fn mul(self, other: & TensorDynLen < Id , Symm >) -> Self :: Output` (impl TensorDynLen < Id , Symm >)

### `pub fn is_diag_tensor(tensor: & TensorDynLen < Id , Symm >) -> bool`

Check if a tensor is a DiagTensor (has Diag storage).

### `pub fn add(&self, other: & Self) -> Result < Self >` (impl TensorDynLen < Id , Symm >)

Add two tensors element-wise. The tensors must have the same index set (matched by ID). If the indices are in a different order, the other tensor will be permuted to match `self`.

### ` fn clone(&self) -> Self` (impl TensorDynLen < Id , Symm >)

### `pub fn replaceind(&self, old_index: & Index < Id , Symm >, new_index: & Index < Id , Symm >) -> Self` (impl TensorDynLen < Id , Symm >)

Replace an index in the tensor with a new index. This replaces the index matching `old_index` by ID with `new_index`. The storage data is not modified, only the index metadata is changed.

### `pub fn replaceinds(&self, old_indices: & [Index < Id , Symm >], new_indices: & [Index < Id , Symm >]) -> Self` (impl TensorDynLen < Id , Symm >)

Replace multiple indices in the tensor. This replaces each index in `old_indices` (matched by ID) with the corresponding index in `new_indices`. The storage data is not modified.

### `pub fn conj(&self) -> Self` (impl TensorDynLen < Id , Symm >)

Complex conjugate of all tensor elements. For real (f64) tensors, returns a copy (conjugate of real is identity). For complex (Complex64) tensors, conjugates each element.

### `pub fn norm_squared(&self) -> f64` (impl TensorDynLen < Id , Symm >)

Compute the squared Frobenius norm of the tensor: ||T||² = Σ|T_ijk...|² For real tensors: sum of squares of all elements. For complex tensors: sum of |z|² = z * conj(z) for all elements.

### `pub fn norm(&self) -> f64` (impl TensorDynLen < Id , Symm >)

Compute the Frobenius norm of the tensor: ||T|| = sqrt(Σ|T_ijk...|²)

### `pub fn distance(&self, other: & Self) -> f64` (impl TensorDynLen < Id , Symm >)

Compute the relative distance between two tensors. Returns `||A - B|| / ||A||` (Frobenius norm). If `||A|| = 0`, returns `||B||` instead to avoid division by zero.

### ` fn fmt(&self, f: & mut std :: fmt :: Formatter < '_ >) -> std :: fmt :: Result` (impl TensorDynLen < Id , Symm >)

### `pub fn diag_tensor_dyn_len(indices: Vec < Index < Id , Symm > >, diag_data: Vec < f64 >) -> TensorDynLen < Id , Symm >`

Create a DiagTensor with dynamic rank from diagonal data.

### `pub fn diag_tensor_dyn_len_c64(indices: Vec < Index < Id , Symm > >, diag_data: Vec < Complex64 >) -> TensorDynLen < Id , Symm >`

Create a DiagTensor with dynamic rank from complex diagonal data.

### `pub fn unfold_split(t: & TensorDynLen < Id , Symm >, left_inds: & [Index < Id , Symm >]) -> Result < (DTensor < T , 2 > , usize , usize , usize , Vec < Index < Id , Symm > > , Vec < Index < Id , Symm > > ,) >`

Unfold a tensor into a matrix by splitting indices into left and right groups. This function validates the split, permutes the tensor so that left indices come first, and returns a 2D matrix tensor (`DTensor<T, 2>`) along with metadata.

## src/tensor_like.rs

### `pub fn external_indices(&self) -> Vec < Index < Self :: Id , Self :: Symm , Self :: Tags > >` (trait TensorLike)

Return flattened external indices for this object. - For `TensorDynLen`: returns the tensor's indices - For `TreeTN`: returns union of all site/physical indices across nodes

### `pub fn num_external_indices(&self) -> usize` (trait TensorLike default)

Number of external indices. Default implementation calls `external_indices().len()`, but implementations SHOULD override this for efficiency when the count can be computed without

### `pub fn to_tensor(&self) -> Result < TensorDynLen < Self :: Id , Self :: Symm > >` (trait TensorLike)

Convert this object to a dense tensor. - For `TensorDynLen`: returns a clone of self - For `TreeTN`: contracts all nodes to produce a single tensor

### `pub fn as_any(&self) -> & dyn Any` (trait TensorLike)

Return `self` as `Any` for optional downcasting / runtime type inspection. This allows callers to attempt downcasting a trait object back to its concrete type when needed (similar to C++'s `dynamic_cast`).

### `pub fn replaceind(&self, old_index: & Index < Self :: Id , Self :: Symm , Self :: Tags >, new_index: & Index < Self :: Id , Self :: Symm , Self :: Tags >) -> Result < TensorDynLen < Self :: Id , Self :: Symm > >` (trait TensorLike default)

Replace an index in this tensor-like object. This replaces the index matching `old_index` by ID with `new_index`. The storage data is not modified, only the index metadata is changed.

### `pub fn replaceinds(&self, old_indices: & [Index < Self :: Id , Self :: Symm , Self :: Tags >], new_indices: & [Index < Self :: Id , Self :: Symm , Self :: Tags >]) -> Result < TensorDynLen < Self :: Id , Self :: Symm > >` (trait TensorLike default)

Replace multiple indices in this tensor-like object. This replaces each index in `old_indices` (matched by ID) with the corresponding index in `new_indices`. The storage data is not modified.

### `pub fn tensordot(&self, other: & dyn TensorLike < Id = Self :: Id , Symm = Self :: Symm , Tags = Self :: Tags >, pairs: & [(Index < Self :: Id , Self :: Symm , Self :: Tags > , Index < Self :: Id , Self :: Symm , Self :: Tags > ,)]) -> Result < TensorDynLen < Self :: Id , Self :: Symm > >` (trait TensorLike default)

Explicit contraction between two tensor-like objects. This performs binary contraction over the specified index pairs. Each pair `(idx_self, idx_other)` specifies:

### `pub fn is(&self) -> bool` (trait TensorLikeDowncast)

Check if the underlying type is `T`.

### `pub fn downcast_ref(&self) -> Option < & T >` (trait TensorLikeDowncast)

Attempt to downcast to a reference of type `T`.

### ` fn is(&self) -> bool` (impl dyn TensorLike < Id = Id , Symm = Symm , Tags = Tags >)

### ` fn downcast_ref(&self) -> Option < & T >` (impl dyn TensorLike < Id = Id , Symm = Symm , Tags = Tags >)

### ` fn is(&self) -> bool` (impl dyn TensorLike < Id = Id , Symm = Symm , Tags = Tags > + Send)

### ` fn downcast_ref(&self) -> Option < & T >` (impl dyn TensorLike < Id = Id , Symm = Symm , Tags = Tags > + Send)

### ` fn is(&self) -> bool` (impl dyn TensorLike < Id = Id , Symm = Symm , Tags = Tags > + Send + Sync)

### ` fn downcast_ref(&self) -> Option < & T >` (impl dyn TensorLike < Id = Id , Symm = Symm , Tags = Tags > + Send + Sync)

### ` fn external_indices(&self) -> Vec < Index < Self :: Id , Self :: Symm , Self :: Tags > >` (impl TensorDynLen < Id , Symm >)

### ` fn num_external_indices(&self) -> usize` (impl TensorDynLen < Id , Symm >)

### ` fn to_tensor(&self) -> Result < TensorDynLen < Self :: Id , Self :: Symm > >` (impl TensorDynLen < Id , Symm >)

### ` fn as_any(&self) -> & dyn Any` (impl TensorDynLen < Id , Symm >)

### ` fn _assert_object_safe()`

