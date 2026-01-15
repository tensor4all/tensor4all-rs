# tensor4all-core

## src/defaults/contract.rs

### `pub fn contract_multi(tensors: & [TensorDynLen], allowed: AllowedPairs < '_ >) -> Result < TensorDynLen >`

Contract multiple tensors into a single tensor, handling disconnected components. This function automatically handles disconnected tensor graphs by: 1. Finding connected components based on contractable indices

### `pub fn contract_connected(tensors: & [TensorDynLen], allowed: AllowedPairs < '_ >) -> Result < TensorDynLen >`

Contract multiple tensors that form a connected graph. Uses omeco's GreedyMethod to find the optimal contraction order.

### ` fn has_contractable_indices(a: & TensorDynLen, b: & TensorDynLen) -> bool`

Check if two tensors have any contractable indices.

### ` fn find_tensor_connected_components(tensors: & [TensorDynLen], allowed: AllowedPairs < '_ >) -> Vec < Vec < usize > >`

Find connected components of tensors based on contractable indices. Uses petgraph for O(V+E) connected component detection.

### ` fn remap_allowed_pairs(allowed: AllowedPairs < '_ >, component: & [usize]) -> RemappedAllowedPairs`

Remap AllowedPairs for a subset of tensors. Given original tensor indices in `component`, returns AllowedPairs with indices remapped to the component's local indices.

### ` fn as_ref(&self) -> AllowedPairs < '_ >` (impl RemappedAllowedPairs)

### ` fn contract_pair(a: & TensorDynLen, b: & TensorDynLen) -> Result < TensorDynLen >`

Contract two tensors over their common indices. If there are no common indices, performs outer product.

### ` fn contract_connected_optimized(tensors: & [TensorDynLen], allowed: AllowedPairs < '_ >) -> Result < TensorDynLen >`

Contract multiple tensors using omeco's GreedyMethod for optimal ordering. Uses internal IDs to control which indices are contracted based on `allowed`.

### ` fn build_internal_ids(tensors: & [TensorDynLen], allowed: AllowedPairs < '_ >) -> (Vec < Vec < usize > > , HashMap < usize , (usize , usize) >)`

Build internal IDs for contraction. Internal IDs are integers that represent indices during contraction: - Contractable pairs in allowed tensor pairs share the same internal ID

### ` fn validate_connected_graph(num_tensors: usize, pairs: & [(usize , usize)]) -> Result < () >`

Validate that the specified tensor pairs form a connected graph. Returns an error if the graph is disconnected.

### ` fn execute_contraction_tree(tensors: & [TensorDynLen], tree: & NestedEinsum < usize >) -> Result < TensorDynLen >`

Execute a contraction tree by recursively contracting tensors.

### ` fn make_test_tensor(shape: & [usize], ids: & [u128]) -> TensorDynLen`

### ` fn test_contract_multi_empty()`

### ` fn test_contract_multi_single()`

### ` fn test_contract_multi_pair()`

### ` fn test_contract_multi_three()`

### ` fn test_contract_multi_four()`

### ` fn test_contract_multi_outer_product()`

### ` fn test_contract_multi_vector_outer_product()`

### ` fn test_contract_connected_disconnected_error()`

### ` fn test_contract_connected_specified_no_contractable_error()`

### ` fn test_contract_specified_pairs()`

### ` fn test_contract_specified_no_contractable_indices_error()`

### ` fn test_contract_specified_disconnected_outer_product()`

### ` fn test_validate_connected_graph()`

### ` fn test_omeco_hyperedge_delta()`

Test omeco's handling of hyperedges. Simulates: A(i, I) * B(j, J) * C(k, K) * delta_{IJK} where delta is a 3D superdiagonal (I==J==K).

### ` fn test_omeco_hyperedge_svd()`

Test omeco with a simple hyperedge case: U * s * V (SVD-like)

## src/defaults/direct_sum.rs

### `pub fn direct_sum(a: & TensorDynLen, b: & TensorDynLen, pairs: & [(DynIndex , DynIndex)]) -> Result < (TensorDynLen , Vec < DynIndex >) >`

Compute the direct sum of two tensors along specified index pairs. For tensors A and B with indices to be summed specified as pairs, creates a new tensor C where each paired index has dimension = dim_A + dim_B.

### ` fn setup_direct_sum(a: & TensorDynLen, b: & TensorDynLen, pairs: & [(DynIndex , DynIndex)]) -> Result < DirectSumSetup >`

### ` fn linear_to_multi(linear: usize, dims: & [usize]) -> Vec < usize >`

### ` fn multi_to_linear(multi: & [usize], strides: & [usize]) -> usize`

### ` fn direct_sum_f64(a: & TensorDynLen, b: & TensorDynLen, pairs: & [(DynIndex , DynIndex)]) -> Result < (TensorDynLen , Vec < DynIndex >) >`

### ` fn direct_sum_c64(a: & TensorDynLen, b: & TensorDynLen, pairs: & [(DynIndex , DynIndex)]) -> Result < (TensorDynLen , Vec < DynIndex >) >`

### ` fn test_direct_sum_simple()`

### ` fn test_direct_sum_multiple_pairs()`

## src/defaults/factorize.rs

### `pub fn factorize(t: & TensorDynLen, left_inds: & [DynIndex], options: & FactorizeOptions) -> Result < FactorizeResult < TensorDynLen > , FactorizeError >`

Factorize a tensor into left and right factors. This function dispatches to the appropriate algorithm based on `options.alg`: - `SVD`: Singular Value Decomposition

### ` fn factorize_impl(t: & TensorDynLen, left_inds: & [DynIndex], options: & FactorizeOptions) -> Result < FactorizeResult < TensorDynLen > , FactorizeError >`

Internal implementation with scalar type.

### ` fn factorize_svd(t: & TensorDynLen, left_inds: & [DynIndex], options: & FactorizeOptions) -> Result < FactorizeResult < TensorDynLen > , FactorizeError >`

SVD factorization implementation.

### ` fn factorize_qr(t: & TensorDynLen, left_inds: & [DynIndex], options: & FactorizeOptions) -> Result < FactorizeResult < TensorDynLen > , FactorizeError >`

QR factorization implementation.

### ` fn factorize_lu(t: & TensorDynLen, left_inds: & [DynIndex], options: & FactorizeOptions) -> Result < FactorizeResult < TensorDynLen > , FactorizeError >`

LU factorization implementation.

### ` fn factorize_ci(t: & TensorDynLen, left_inds: & [DynIndex], options: & FactorizeOptions) -> Result < FactorizeResult < TensorDynLen > , FactorizeError >`

CI (Cross Interpolation) factorization implementation.

### ` fn extract_singular_values(s: & TensorDynLen) -> Vec < f64 >`

Extract singular values from a diagonal tensor.

### ` fn dtensor_to_matrix(tensor: & tensor4all_tensorbackend :: mdarray :: DTensor < T , 2 >, m: usize, n: usize) -> matrixci :: Matrix < T >`

Convert DTensor to Matrix (tensor4all-matrixci format).

### ` fn matrix_to_vec(matrix: & matrixci :: Matrix < T >) -> Vec < T >`

Convert Matrix to Vec for storage.

## src/defaults/index.rs

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

### `pub fn new(id: Id, dim: usize) -> Self` (impl Index < Id , Tags >)

Create a new index with the given identity and dimension.

### `pub fn new_with_tags(id: Id, dim: usize, tags: Tags) -> Self` (impl Index < Id , Tags >)

Create a new index with the given identity, dimension, and tags.

### `pub fn size(&self) -> usize` (impl Index < Id , Tags >)

Get the dimension (size) of the index.

### `pub fn tags(&self) -> & Tags` (impl Index < Id , Tags >)

Get a reference to the tags.

### `pub fn new_with_size(id: Id, size: usize) -> Self` (impl Index < Id , Tags >)

Create a new index from dimension (convenience constructor).

### `pub fn new_with_size_and_tags(id: Id, size: usize, tags: Tags) -> Self` (impl Index < Id , Tags >)

Create a new index from dimension and tags.

### `pub fn new_dyn(size: usize) -> Self` (impl Index < DynId , TagSet >)

Create a new index with a generated dynamic ID and no tags.

### `pub fn new_dyn_with_tags(size: usize, tags: TagSet) -> Self` (impl Index < DynId , TagSet >)

Create a new index with a generated dynamic ID and shared tags. This is the most efficient way to create many indices with the same tags. The `Arc` is cloned (reference count increment only), not the underlying data.

### `pub fn new_dyn_with_tag(size: usize, tag: & str) -> Result < Self , TagSetError >` (impl Index < DynId , TagSet >)

Create a new index with a generated dynamic ID and a single tag. This creates a new `TagSet` with the given tag. For sharing the same tag across many indices, create the `TagSet`

### `pub fn new_link(size: usize) -> Result < Self , TagSetError >` (impl Index < DynId , TagSet >)

Create a new bond index with "Link" tag (for SVD, QR, etc.). This is a convenience method for creating bond indices commonly used in tensor decompositions like SVD and QR factorization.

### ` fn eq(&self, other: & Self) -> bool` (impl Index < Id , Tags >)

### ` fn hash(&self, state: & mut H)` (impl Index < Id , Tags >)

### `pub(crate) fn generate_id() -> u128`

Generate a unique random ID for dynamic indices (thread-safe). Uses thread-local random number generator to generate UInt128 IDs, providing extremely low collision probability (see design.md for analysis).

### ` fn id(&self) -> & Self :: Id` (impl DynIndex)

### ` fn dim(&self) -> usize` (impl DynIndex)

### ` fn conj_state(&self) -> crate :: ConjState` (impl DynIndex)

### ` fn conj(&self) -> Self` (impl DynIndex)

### ` fn sim(&self) -> Self` (impl DynIndex)

### ` fn create_dummy_link_pair() -> (Self , Self)` (impl DynIndex)

### `pub fn new_bond(dim: usize) -> Result < Self >` (impl DynIndex)

Create a new bond index with a fresh identity and the specified dimension. This is used by factorization operations (SVD, QR) to create new internal bond indices connecting the factors.

### ` fn test_id_generation()`

### ` fn test_thread_local_rng_different_seeds()`

### ` fn test_index_like_basic()`

### ` fn test_index_like_id_methods()`

### ` fn test_index_like_equality()`

### ` fn test_index_like_in_hashset()`

### ` fn test_new_bond()`

### ` fn test_sim()`

### ` fn _assert_index_like_bounds()`

### ` fn test_index_satisfies_index_like()`

### ` fn test_conj_state_undirected()`

### ` fn test_conj_undirected_noop()`

### ` fn test_is_contractable_undirected()`

### ` fn test_is_contractable_same_id_dim()`

## src/defaults/qr.rs

### `pub fn with_rtol(rtol: f64) -> Self` (impl QrOptions)

Create new QR options with the specified rtol.

### `pub fn rtol(&self) -> Option < f64 >` (impl QrOptions)

Get rtol from options (for backwards compatibility).

### `pub fn default_qr_rtol() -> f64`

Get the global default rtol for QR truncation. The default value is 1e-15 (very strict, near machine precision).

### `pub fn set_default_qr_rtol(rtol: f64) -> Result < () , QrError >`

Set the global default rtol for QR truncation.

### ` fn compute_retained_rank_qr(r_full: & DTensor < T , 2 >, k: usize, n: usize, rtol: f64) -> usize`

Compute the retained rank based on rtol truncation for QR. This checks R's diagonal elements and truncates columns where |R[i, i]| < rtol.

### `pub fn qr(t: & TensorDynLen, left_inds: & [DynIndex]) -> Result < (TensorDynLen , TensorDynLen) , QrError >`

Compute QR decomposition of a tensor with arbitrary rank, returning (Q, R). This function uses the global default rtol for truncation. See `qr_with` for per-call rtol control.

### `pub fn qr_with(t: & TensorDynLen, left_inds: & [DynIndex], options: & QrOptions) -> Result < (TensorDynLen , TensorDynLen) , QrError >`

Compute QR decomposition of a tensor with arbitrary rank, returning (Q, R). This function allows per-call control of the truncation tolerance via `QrOptions`. If `options.rtol` is `None`, uses the global default rtol.

### `pub fn qr_c64(t: & TensorDynLen, left_inds: & [DynIndex]) -> Result < (TensorDynLen , TensorDynLen) , QrError >`

Compute QR decomposition of a complex tensor with arbitrary rank, returning (Q, R). This is a convenience wrapper around the generic `qr` function for `Complex64` tensors. For the mathematical convention:

## src/defaults/svd.rs

### `pub fn with_rtol(rtol: f64) -> Self` (impl SvdOptions)

Create new SVD options with the specified rtol.

### `pub fn with_max_rank(max_rank: usize) -> Self` (impl SvdOptions)

Create new SVD options with the specified max_rank.

### `pub fn rtol(&self) -> Option < f64 >` (impl SvdOptions)

Get rtol from options (for backwards compatibility).

### `pub fn max_rank(&self) -> Option < usize >` (impl SvdOptions)

Get max_rank from options (for backwards compatibility).

### ` fn truncation_params(&self) -> & TruncationParams` (impl SvdOptions)

### ` fn truncation_params_mut(&mut self) -> & mut TruncationParams` (impl SvdOptions)

### `pub fn default_svd_rtol() -> f64`

Get the global default rtol for SVD truncation. The default value is 1e-12 (near machine precision).

### `pub fn set_default_svd_rtol(rtol: f64) -> Result < () , SvdError >`

Set the global default rtol for SVD truncation.

### ` fn compute_retained_rank(s_vec: & [f64], rtol: f64) -> usize`

Compute the retained rank based on rtol (TSVD truncation). This implements the truncation criterion: sum_{i>r} σ_i² / sum_i σ_i² <= rtol²

### ` fn extract_usv_from_svd_result(decomp: SvdResult < T >, m: usize, n: usize, k: usize) -> (Vec < T > , Vec < f64 > , Vec < T >)`

Extract U, S, V from tensorbackend's SvdResult (which returns U, S, Vt). This helper function converts the backend's SVD result to our desired format: - Extracts singular values from the diagonal view (first row)

### `pub fn svd(t: & TensorDynLen, left_inds: & [DynIndex]) -> Result < (TensorDynLen , TensorDynLen , TensorDynLen) , SvdError >`

Compute SVD decomposition of a tensor with arbitrary rank, returning (U, S, V). This function uses the global default rtol for truncation. See `svd_with` for per-call rtol control.

### `pub fn svd_with(t: & TensorDynLen, left_inds: & [DynIndex], options: & SvdOptions) -> Result < (TensorDynLen , TensorDynLen , TensorDynLen) , SvdError >`

Compute SVD decomposition of a tensor with arbitrary rank, returning (U, S, V). This function allows per-call control of the truncation tolerance via `SvdOptions`. If `options.rtol` is `None`, uses the global default rtol.

### `pub fn svd_c64(t: & TensorDynLen, left_inds: & [DynIndex]) -> Result < (TensorDynLen , TensorDynLen , TensorDynLen) , SvdError >`

Compute SVD decomposition of a complex tensor with arbitrary rank, returning (U, S, V). This is a convenience wrapper around the generic `svd` function for `Complex64` tensors. For complex-valued matrices, the mathematical convention is:

## src/defaults/tensor_data.rs

### `pub fn new(storage: Arc < Storage >, index_ids: Vec < DynId >, dims: Vec < usize >) -> Self` (impl TensorComponent)

Create a new TensorComponent.

### `pub fn ndim(&self) -> usize` (impl TensorComponent)

Get the number of dimensions.

### `pub fn numel(&self) -> usize` (impl TensorComponent)

Get the total number of elements.

### `pub fn new(storage: Arc < Storage >, index_ids: Vec < DynId >, dims: Vec < usize >) -> Self` (impl TensorData)

Create a new TensorData from a single storage.

### `pub fn from_components(components: Vec < TensorComponent >, external_index_ids: Vec < DynId >, external_dims: Vec < usize >) -> Self` (impl TensorData)

Create TensorData from components with explicit external order.

### `pub fn is_simple(&self) -> bool` (impl TensorData)

Check if this is a simple tensor (single component, no permutation needed).

### `pub fn storage(&self) -> Option < & Arc < Storage > >` (impl TensorData)

Get the underlying storage if this is a simple tensor.

### `pub fn ndim(&self) -> usize` (impl TensorData)

Get the number of external dimensions.

### `pub fn numel(&self) -> usize` (impl TensorData)

Get the total number of elements.

### `pub fn dims(&self) -> & [usize]` (impl TensorData)

Get the external dimensions.

### `pub fn index_ids(&self) -> & [DynId]` (impl TensorData)

Get the external index IDs.

### `pub fn outer_product(a: & Self, b: & Self) -> Self` (impl TensorData)

Compute outer product of two TensorData (lazy). This just concatenates the components and index lists without actually computing the outer product data.

### `pub fn permute(&self, new_order: & [DynId]) -> Self` (impl TensorData)

Permute the external index order (lazy). This only updates the external_index_ids order without touching the underlying storage data.

### `pub fn permute_by_perm(&self, perm: & [usize]) -> Self` (impl TensorData)

Permute using a permutation array.

### `pub fn materialize(&self) -> anyhow :: Result < (Arc < Storage > , Vec < usize >) >` (impl TensorData)

Materialize the tensor into a single Storage with the external index order. This contracts all components and permutes the result to match the external_index_ids order.

### `pub fn into_components(self) -> Vec < TensorComponent >` (impl TensorData)

Get all components (for passing to contraction).

### `pub fn components(&self) -> & [TensorComponent]` (impl TensorData)

Get a reference to all components.

### ` fn make_test_storage(data: Vec < f64 >) -> Arc < Storage >`

### ` fn new_id() -> DynId`

### ` fn test_tensor_data_simple()`

### ` fn test_outer_product()`

### ` fn test_permute()`

### ` fn test_permute_outer_product()`

### ` fn test_materialize_simple()`

### ` fn test_materialize_outer_product()`

### ` fn test_materialize_with_permute()`

## src/defaults/tensordynlen.rs

### `pub fn compute_permutation_from_indices(original_indices: & [DynIndex], new_indices: & [DynIndex]) -> Vec < usize >`

Compute the permutation array from original indices to new indices. This function finds the mapping from new indices to original indices by matching index IDs. The result is a permutation array `perm` such that

### `pub fn indices(&self) -> & [DynIndex]` (trait TensorAccess)

Get a reference to the indices.

### `pub fn data(&self) -> & Storage` (trait TensorAccess)

Get a reference to the underlying data (Storage).

### ` fn indices(&self) -> & [DynIndex]` (impl TensorDynLen)

### ` fn data(&self) -> & Storage` (impl TensorDynLen)

### `pub fn new(indices: Vec < DynIndex >, dims: Vec < usize >, storage: Arc < Storage >) -> Self` (impl TensorDynLen)

Create a new tensor with dynamic rank.

### `pub fn from_indices(indices: Vec < DynIndex >, storage: Arc < Storage >) -> Self` (impl TensorDynLen)

Create a new tensor with dynamic rank, automatically computing dimensions from indices. This is a convenience constructor that extracts dimensions from indices using `IndexLike::dim()`.

### `pub fn is_simple(&self) -> bool` (impl TensorDynLen)

Check if this tensor is simple (single storage, no lazy operations pending).

### `pub fn storage(&self) -> & Arc < Storage >` (impl TensorDynLen)

Get the storage (for simple tensors only).

### `pub fn try_storage(&self) -> Option < & Arc < Storage > >` (impl TensorDynLen)

Try to get the storage without materializing. Returns `None` if the tensor has pending lazy operations. Use `materialize_storage()` to force materialization.

### `pub fn materialize_storage(&self) -> Result < Arc < Storage > >` (impl TensorDynLen)

Get the storage, materializing if necessary. For simple tensors, returns the underlying storage without copying. For lazy tensors, performs any pending operations and returns the result.

### `pub fn tensor_data(&self) -> & TensorData` (impl TensorDynLen)

Get the internal TensorData reference.

### ` fn from_data(indices: Vec < DynIndex >, dims: Vec < usize >, data: TensorData) -> Self` (impl TensorDynLen)

Create TensorDynLen directly from TensorData and indices. This is an internal constructor for building tensors from lazy operations.

### `pub fn sum(&self) -> AnyScalar` (impl TensorDynLen)

Sum all elements, returning `AnyScalar`.

### `pub fn sum_f64(&self) -> f64` (impl TensorDynLen)

Sum all elements as f64.

### `pub fn only(&self) -> AnyScalar` (impl TensorDynLen)

Extract the scalar value from a 0-dimensional tensor (or 1-element tensor). This is similar to Julia's `only()` function.

### `pub fn permute_indices(&self, new_indices: & [DynIndex]) -> Self` (impl TensorDynLen)

Permute the tensor dimensions using the given new indices order. This is the main permutation method that takes the desired new indices and automatically computes the corresponding permutation of dimensions

### `pub fn permute(&self, perm: & [usize]) -> Self` (impl TensorDynLen)

Permute the tensor dimensions, returning a new tensor. This method reorders the indices, dimensions, and data according to the given permutation. The permutation specifies which old axis each new

### `pub fn contract(&self, other: & Self) -> Self` (impl TensorDynLen)

Contract this tensor with another tensor along common indices. This method finds common indices between `self` and `other`, then contracts along those indices. The result tensor contains all non-contracted indices

### `pub fn tensordot(&self, other: & Self, pairs: & [(DynIndex , DynIndex)]) -> Result < Self >` (impl TensorDynLen)

Contract this tensor with another tensor along explicitly specified index pairs. Similar to NumPy's `tensordot`, this method contracts only along the explicitly specified pairs of indices. Unlike `contract()` which automatically contracts

### `pub fn outer_product(&self, other: & Self) -> Result < Self >` (impl TensorDynLen)

Compute the outer product (tensor product) of two tensors. Creates a new tensor whose indices are the concatenation of the indices from both input tensors. The result has shape `[...self.dims, ...other.dims]`.

### `pub fn random_f64(rng: & mut R, indices: Vec < DynIndex >) -> Self` (impl TensorDynLen)

Create a random f64 tensor with values from standard normal distribution.

### `pub fn random_c64(rng: & mut R, indices: Vec < DynIndex >) -> Self` (impl TensorDynLen)

Create a random Complex64 tensor with values from standard normal distribution. Both real and imaginary parts are drawn from standard normal distribution.

### ` fn mul(self, other: & TensorDynLen) -> Self :: Output` (impl & TensorDynLen)

### ` fn mul(self, other: TensorDynLen) -> Self :: Output` (impl TensorDynLen)

### ` fn mul(self, other: TensorDynLen) -> Self :: Output` (impl & TensorDynLen)

### ` fn mul(self, other: & TensorDynLen) -> Self :: Output` (impl TensorDynLen)

### `pub fn is_diag_tensor(tensor: & TensorDynLen) -> bool`

Check if a tensor is a DiagTensor (has Diag storage).

### `pub fn add(&self, other: & Self) -> Result < Self >` (impl TensorDynLen)

Add two tensors element-wise. The tensors must have the same index set (matched by ID). If the indices are in a different order, the other tensor will be permuted to match `self`.

### `pub fn axpby(&self, a: AnyScalar, other: & Self, b: AnyScalar) -> Result < Self >` (impl TensorDynLen)

Compute a linear combination: `a * self + b * other`.

### `pub fn scale(&self, scalar: AnyScalar) -> Result < Self >` (impl TensorDynLen)

Scalar multiplication.

### `pub fn inner_product(&self, other: & Self) -> Result < AnyScalar >` (impl TensorDynLen)

Inner product (dot product) of two tensors. Computes `⟨self, other⟩ = Σ conj(self)_i * other_i`.

### ` fn clone(&self) -> Self` (impl TensorDynLen)

### `pub fn replaceind(&self, old_index: & DynIndex, new_index: & DynIndex) -> Self` (impl TensorDynLen)

Replace an index in the tensor with a new index. This replaces the index matching `old_index` by ID with `new_index`. The storage data is not modified, only the index metadata is changed.

### `pub fn replaceinds(&self, old_indices: & [DynIndex], new_indices: & [DynIndex]) -> Self` (impl TensorDynLen)

Replace multiple indices in the tensor. This replaces each index in `old_indices` (matched by ID) with the corresponding index in `new_indices`. The storage data is not modified.

### `pub fn conj(&self) -> Self` (impl TensorDynLen)

Complex conjugate of all tensor elements. For real (f64) tensors, returns a copy (conjugate of real is identity). For complex (Complex64) tensors, conjugates each element.

### `pub fn norm_squared(&self) -> f64` (impl TensorDynLen)

Compute the squared Frobenius norm of the tensor: ||T||² = Σ|T_ijk...|² For real tensors: sum of squares of all elements. For complex tensors: sum of |z|² = z * conj(z) for all elements.

### `pub fn norm(&self) -> f64` (impl TensorDynLen)

Compute the Frobenius norm of the tensor: ||T|| = sqrt(Σ|T_ijk...|²)

### `pub fn distance(&self, other: & Self) -> f64` (impl TensorDynLen)

Compute the relative distance between two tensors. Returns `||A - B|| / ||A||` (Frobenius norm). If `||A|| = 0`, returns `||B||` instead to avoid division by zero.

### ` fn fmt(&self, f: & mut std :: fmt :: Formatter < '_ >) -> std :: fmt :: Result` (impl TensorDynLen)

### `pub fn diag_tensor_dyn_len(indices: Vec < DynIndex >, diag_data: Vec < f64 >) -> TensorDynLen`

Create a DiagTensor with dynamic rank from diagonal data.

### `pub fn diag_tensor_dyn_len_c64(indices: Vec < DynIndex >, diag_data: Vec < Complex64 >) -> TensorDynLen`

Create a DiagTensor with dynamic rank from complex diagonal data.

### `pub fn unfold_split(t: & TensorDynLen, left_inds: & [DynIndex]) -> Result < (DTensor < T , 2 > , usize , usize , usize , Vec < DynIndex > , Vec < DynIndex > ,) >`

Unfold a tensor into a matrix by splitting indices into left and right groups. This function validates the split, permutes the tensor so that left indices come first, and returns a 2D matrix tensor (`DTensor<T, 2>`) along with metadata.

### ` fn external_indices(&self) -> Vec < DynIndex >` (impl TensorDynLen)

### ` fn num_external_indices(&self) -> usize` (impl TensorDynLen)

### ` fn replaceind(&self, old_index: & DynIndex, new_index: & DynIndex) -> Result < Self >` (impl TensorDynLen)

### ` fn replaceinds(&self, old_indices: & [DynIndex], new_indices: & [DynIndex]) -> Result < Self >` (impl TensorDynLen)

### ` fn factorize(&self, left_inds: & [DynIndex], options: & FactorizeOptions) -> std :: result :: Result < FactorizeResult < Self > , FactorizeError >` (impl TensorDynLen)

### ` fn conj(&self) -> Self` (impl TensorDynLen)

### ` fn direct_sum(&self, other: & Self, pairs: & [(DynIndex , DynIndex)]) -> Result < crate :: tensor_like :: DirectSumResult < Self > >` (impl TensorDynLen)

### ` fn outer_product(&self, other: & Self) -> Result < Self >` (impl TensorDynLen)

### ` fn norm_squared(&self) -> f64` (impl TensorDynLen)

### ` fn permuteinds(&self, new_order: & [DynIndex]) -> Result < Self >` (impl TensorDynLen)

### ` fn contract(tensors: & [Self], allowed: crate :: AllowedPairs < '_ >) -> Result < Self >` (impl TensorDynLen)

### ` fn contract_connected(tensors: & [Self], allowed: crate :: AllowedPairs < '_ >) -> Result < Self >` (impl TensorDynLen)

### ` fn axpby(&self, a: crate :: AnyScalar, other: & Self, b: crate :: AnyScalar) -> Result < Self >` (impl TensorDynLen)

### ` fn scale(&self, scalar: crate :: AnyScalar) -> Result < Self >` (impl TensorDynLen)

### ` fn inner_product(&self, other: & Self) -> Result < crate :: AnyScalar >` (impl TensorDynLen)

### ` fn diagonal(input_index: & DynIndex, output_index: & DynIndex) -> Result < Self >` (impl TensorDynLen)

### ` fn scalar_one() -> Result < Self >` (impl TensorDynLen)

### ` fn ones(indices: & [DynIndex]) -> Result < Self >` (impl TensorDynLen)

## src/global_default.rs

### `pub fn new(initial: f64) -> Self` (impl GlobalDefault)

Create a new global default with the given initial value. This is a const fn, so it can be used in static declarations.

### `pub fn get(&self) -> f64` (impl GlobalDefault)

Get the current default value.

### `pub fn set(&self, value: f64) -> Result < () , InvalidRtolError >` (impl GlobalDefault)

Set a new default value.

### `pub fn set_unchecked(&self, value: f64)` (impl GlobalDefault)

Set a new default value without validation.

### ` fn test_global_default()`

### ` fn test_invalid_values()`

## src/index_like.rs

### `pub fn id(&self) -> & Self :: Id` (trait IndexLike)

Get the identifier of this index. The ID is used as the pairing key during contraction. **Contractable indices must have the same ID** — this is enforced by `is_contractable()`.

### `pub fn dim(&self) -> usize` (trait IndexLike)

Get the total dimension (state-space dimension) of the index.

### `pub fn conj_state(&self) -> ConjState` (trait IndexLike)

Get the conjugate state (direction) of this index. Returns `ConjState::Undirected` for directionless indices (ITensors.jl-like default), or `ConjState::Ket`/`ConjState::Bra` for directed indices (QSpace-compatible).

### `pub fn conj(&self) -> Self` (trait IndexLike)

Create the conjugate of this index. For directed indices, this toggles between `Ket` and `Bra`. For `Undirected` indices, this returns `self` unchanged (no-op).

### `pub fn is_contractable(&self, other: & Self) -> bool` (trait IndexLike default)

Check if this index can be contracted with another index. Two indices are contractable if: - They have the same `id()` and `dim()`

### `pub fn same_id(&self, other: & Self) -> bool` (trait IndexLike default)

Check if this index has the same ID as another. Default implementation compares IDs directly. This is a convenience method for pure ID comparison (does not check contractability).

### `pub fn has_id(&self, id: & Self :: Id) -> bool` (trait IndexLike default)

Check if this index has the given ID. Default implementation compares with the given ID.

### `pub fn sim(&self) -> Self` (trait IndexLike)

Create a similar index with a new identity but the same structure (dimension, tags, etc.). This is used to create "equivalent" indices that have the same properties but different identities, commonly needed in index replacement operations.

### `pub fn create_dummy_link_pair() -> (Self , Self)` (trait IndexLike)

Create a pair of contractable dummy indices with dimension 1. These are used for structural connections that don't carry quantum numbers, such as connecting components in a tree tensor network.

## src/index_ops.rs

### ` fn fmt(&self, f: & mut std :: fmt :: Formatter < '_ >) -> std :: fmt :: Result` (impl ReplaceIndsError)

### `pub fn check_unique_indices(indices: & [I]) -> Result < () , ReplaceIndsError >`

Check if a collection of indices contains any duplicates (by ID).

### `pub fn replaceinds(indices: Vec < I >, replacements: & [(I , I)]) -> Result < Vec < I > , ReplaceIndsError >`

Replace indices in a collection based on ID matching. This corresponds to ITensors.jl's `replaceinds` function. It replaces indices in `indices` that match (by ID) any of the `(old, new)` pairs in `replacements`.

### `pub fn replaceinds_in_place(indices: & mut [I], replacements: & [(I , I)]) -> Result < () , ReplaceIndsError >`

Replace indices in-place based on ID matching. This is an in-place variant of `replaceinds` that modifies the input slice directly. Useful for performance-critical code where you want to avoid allocations.

### `pub fn unique_inds(indices_a: & [I], indices_b: & [I]) -> Vec < I >`

Find indices that are unique to the first collection (set difference A \ B). Returns indices that appear in `indices_a` but not in `indices_b` (matched by ID). This corresponds to ITensors.jl's `uniqueinds` function.

### `pub fn noncommon_inds(indices_a: & [I], indices_b: & [I]) -> Vec < I >`

Find indices that are not common between two collections (symmetric difference). Returns indices that appear in either `indices_a` or `indices_b` but not in both (matched by ID). This corresponds to ITensors.jl's `noncommoninds` function.

### `pub fn union_inds(indices_a: & [I], indices_b: & [I]) -> Vec < I >`

Find the union of two index collections. Returns all unique indices from both collections (matched by ID). This corresponds to ITensors.jl's `unioninds` function.

### `pub fn hasind(indices: & [I], index: & I) -> bool`

Check if a collection contains a specific index (by ID). This corresponds to ITensors.jl's `hasind` function.

### `pub fn hasinds(indices: & [I], targets: & [I]) -> bool`

Check if a collection contains all of the specified indices (by ID). This corresponds to ITensors.jl's `hasinds` function.

### `pub fn hascommoninds(indices_a: & [I], indices_b: & [I]) -> bool`

Check if two collections have any common indices (by ID). This corresponds to ITensors.jl's `hascommoninds` function.

### `pub fn common_inds(indices_a: & [I], indices_b: & [I]) -> Vec < I >`

Find common indices between two index collections. Returns a vector of indices that appear in both `indices_a` and `indices_b` (set intersection). This is similar to ITensors.jl's `commoninds` function.

### `pub fn common_ind_positions(indices_a: & [I], indices_b: & [I]) -> Vec < (usize , usize) >`

Find contractable indices between two slices and return their positions. Returns a vector of `(pos_a, pos_b)` tuples where each tuple indicates that `indices_a[pos_a]` and `indices_b[pos_b]` are contractable

### ` fn fmt(&self, f: & mut std :: fmt :: Formatter < '_ >) -> std :: fmt :: Result` (impl ContractionError)

### `pub fn prepare_contraction(indices_a: & [I], dims_a: & [usize], indices_b: & [I], dims_b: & [usize]) -> Result < ContractionSpec < I > , ContractionError >`

Prepare contraction data for two tensors that share common indices. This function finds common indices and computes the axes to contract and the resulting indices/dimensions.

### `pub fn prepare_contraction_pairs(indices_a: & [I], dims_a: & [usize], indices_b: & [I], dims_b: & [usize], pairs: & [(I , I)]) -> Result < ContractionSpec < I > , ContractionError >`

Prepare contraction data for explicit index pairs (like tensordot). Unlike `prepare_contraction`, this function takes explicit pairs of indices to contract, allowing contraction of indices with different IDs.

## src/krylov.rs

### ` fn default() -> Self` (impl GmresOptions)

### `pub fn gmres(apply_a: F, b: & T, x0: & T, options: & GmresOptions) -> Result < GmresResult < T > >`

Solve `A x = b` using GMRES (Generalized Minimal Residual Method). This implements the restarted GMRES algorithm that works with abstract tensor types through the [`TensorLike`] trait's vector space operations.

### ` fn compute_givens_rotation(a: & AnyScalar, b: & AnyScalar) -> (AnyScalar , AnyScalar)`

Compute Givens rotation coefficients to eliminate b in (a, b).

### ` fn apply_givens_rotation(c: & AnyScalar, s: & AnyScalar, x: & AnyScalar, y: & AnyScalar) -> (AnyScalar , AnyScalar)`

Apply Givens rotation: (c, s) @ (x, y) -> (c*x + s*y, -conj(s)*x + c*y) for complex or (c*x + s*y, -s*x + c*y) for real.

### ` fn solve_upper_triangular(h: & [Vec < AnyScalar >], g: & [AnyScalar]) -> Result < Vec < AnyScalar > >`

Solve upper triangular system R y = g using back substitution.

### ` fn update_solution(x: & T, v_basis: & [T], y: & [AnyScalar]) -> Result < T >`

Update solution: x_new = x + sum_i y_i * v_i

### ` fn make_vector_with_index(data: Vec < f64 >, idx: & DynIndex) -> TensorDynLen`

Helper to create a 1D tensor (vector) with given data and shared index.

### ` fn test_givens_rotation_real()`

### ` fn test_apply_givens_rotation_real()`

### ` fn test_gmres_identity_operator()`

### ` fn test_gmres_diagonal_matrix()`

### ` fn test_gmres_nonsymmetric_matrix()`

### ` fn test_gmres_with_good_initial_guess()`

### ` fn test_gmres_zero_rhs()`

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

## src/tensor_index.rs

### `pub fn external_indices(&self) -> Vec < Self :: Index >` (trait TensorIndex)

Return flattened external indices for this object.

### `pub fn num_external_indices(&self) -> usize` (trait TensorIndex default)

Number of external indices. Default implementation calls `external_indices().len()`, but implementations SHOULD override this for efficiency when the count can be computed without

### `pub fn replaceind(&self, old_index: & Self :: Index, new_index: & Self :: Index) -> Result < Self >` (trait TensorIndex)

Replace an index in this object. This replaces the index matching `old_index` by ID with `new_index`. The storage data is not modified, only the index metadata is changed.

### `pub fn replaceinds(&self, old_indices: & [Self :: Index], new_indices: & [Self :: Index]) -> Result < Self >` (trait TensorIndex)

Replace multiple indices in this object. This replaces each index in `old_indices` (matched by ID) with the corresponding index in `new_indices`. The storage data is not modified.

### `pub fn replaceinds_pairs(&self, pairs: & [(Self :: Index , Self :: Index)]) -> Result < Self >` (trait TensorIndex default)

Replace indices using pairs of (old, new). This is a convenience method that wraps `replaceinds`.

## src/tensor_like.rs

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

### `pub fn factorize(&self, left_inds: & [< Self as TensorIndex > :: Index], options: & FactorizeOptions) -> std :: result :: Result < FactorizeResult < Self > , FactorizeError >` (trait TensorLike)

Factorize this tensor into left and right factors. This function dispatches to the appropriate algorithm based on `options.alg`: - `SVD`: Singular Value Decomposition

### `pub fn conj(&self) -> Self` (trait TensorLike)

Tensor conjugate operation. This is a generalized conjugate operation that depends on the tensor type: - For dense tensors (TensorDynLen): element-wise complex conjugate

### `pub fn direct_sum(&self, other: & Self, pairs: & [(< Self as TensorIndex > :: Index , < Self as TensorIndex > :: Index)]) -> Result < DirectSumResult < Self > >` (trait TensorLike)

Direct sum of two tensors along specified index pairs. For tensors A and B with indices to be summed specified as pairs, creates a new tensor C where each paired index has dimension = dim_A + dim_B.

### `pub fn outer_product(&self, other: & Self) -> Result < Self >` (trait TensorLike)

Outer product (tensor product) of two tensors. Computes the tensor product of `self` and `other`, resulting in a tensor with all indices from both tensors. No indices are contracted.

### `pub fn norm_squared(&self) -> f64` (trait TensorLike)

Compute the squared Frobenius norm of the tensor. The squared Frobenius norm is defined as the sum of squared absolute values of all tensor elements: `||T||_F^2 = sum_i |T_i|^2`.

### `pub fn permuteinds(&self, new_order: & [< Self as TensorIndex > :: Index]) -> Result < Self >` (trait TensorLike)

Permute tensor indices to match the specified order. This reorders the tensor's axes to match the order specified by `new_order`. The indices in `new_order` are matched by ID with the tensor's current indices.

### `pub fn contract(tensors: & [Self], allowed: AllowedPairs < '_ >) -> Result < Self >` (trait TensorLike)

Contract multiple tensors over their contractable indices. This method contracts 2 or more tensors. Pairs of indices that satisfy `is_contractable()` (same ID, same dimension, compatible ConjState)

### `pub fn contract_connected(tensors: & [Self], allowed: AllowedPairs < '_ >) -> Result < Self >` (trait TensorLike)

Contract multiple tensors that must form a connected graph. This is the core contraction method that requires all tensors to be connected through contractable indices. Use [`contract`] if you want

### `pub fn axpby(&self, a: AnyScalar, other: & Self, b: AnyScalar) -> Result < Self >` (trait TensorLike)

Compute a linear combination: `a * self + b * other`. This is the fundamental vector space operation.

### `pub fn scale(&self, scalar: AnyScalar) -> Result < Self >` (trait TensorLike)

Scalar multiplication.

### `pub fn inner_product(&self, other: & Self) -> Result < AnyScalar >` (trait TensorLike)

Inner product (dot product) of two tensors. Computes `⟨self, other⟩ = Σ conj(self)_i * other_i`.

### `pub fn norm(&self) -> f64` (trait TensorLike default)

Compute the Frobenius norm of the tensor.

### `pub fn diagonal(input_index: & < Self as TensorIndex > :: Index, output_index: & < Self as TensorIndex > :: Index) -> Result < Self >` (trait TensorLike)

Create a diagonal (Kronecker delta) tensor for a single index pair. Creates a 2D tensor `T[i, o]` where `T[i, o] = δ_{i,o}` (1 if i==o, 0 otherwise).

### `pub fn delta(input_indices: & [< Self as TensorIndex > :: Index], output_indices: & [< Self as TensorIndex > :: Index]) -> Result < Self >` (trait TensorLike default)

Create a delta (identity) tensor as outer product of diagonals. For paired indices `(i1, o1), (i2, o2), ...`, creates a tensor where: `T[i1, o1, i2, o2, ...] = δ_{i1,o1} × δ_{i2,o2} × ...`

### `pub fn scalar_one() -> Result < Self >` (trait TensorLike)

Create a scalar tensor with value 1.0. This is used as the identity element for outer products.

### `pub fn ones(indices: & [< Self as TensorIndex > :: Index]) -> Result < Self >` (trait TensorLike)

Create a tensor filled with 1.0 for the given indices. This is useful for adding indices to tensors via outer product without changing tensor values (since multiplying by 1 is identity).

### ` fn _assert_sized()`

## src/truncation.rs

### `pub fn is_svd_based(&self) -> bool` (impl DecompositionAlg)

Check if this algorithm is SVD-based (SVD or RSVD).

### `pub fn is_orthogonal(&self) -> bool` (impl DecompositionAlg)

Check if this algorithm provides orthogonal factors.

### `pub fn new() -> Self` (impl TruncationParams)

Create new truncation parameters with default values.

### `pub fn with_rtol(mut self, rtol: f64) -> Self` (impl TruncationParams)

Set the relative tolerance.

### `pub fn with_max_rank(mut self, max_rank: usize) -> Self` (impl TruncationParams)

Set the maximum rank.

### `pub fn effective_rtol(&self, default: f64) -> f64` (impl TruncationParams)

Get the effective rtol, using the provided default if not set.

### `pub fn effective_max_rank(&self) -> usize` (impl TruncationParams)

Get the effective max_rank, using usize::MAX if not set.

### `pub fn merge(&self, other: & Self) -> Self` (impl TruncationParams)

Merge with another set of parameters, preferring self's values.

### `pub fn truncation_params(&self) -> & TruncationParams` (trait HasTruncationParams)

Get a reference to the truncation parameters.

### `pub fn truncation_params_mut(&mut self) -> & mut TruncationParams` (trait HasTruncationParams)

Get a mutable reference to the truncation parameters.

### `pub fn rtol(&self) -> Option < f64 >` (trait HasTruncationParams default)

Get the rtol value.

### `pub fn max_rank(&self) -> Option < usize >` (trait HasTruncationParams default)

Get the max_rank value.

### `pub fn with_rtol(mut self, rtol: f64) -> Self` (trait HasTruncationParams default)

Set the rtol value (builder pattern).

### `pub fn with_max_rank(mut self, max_rank: usize) -> Self` (trait HasTruncationParams default)

Set the max_rank value (builder pattern).

### ` fn truncation_params(&self) -> & TruncationParams` (impl TruncationParams)

### ` fn truncation_params_mut(&mut self) -> & mut TruncationParams` (impl TruncationParams)

### ` fn test_truncation_params_builder()`

### ` fn test_effective_values()`

### ` fn test_decomposition_alg()`

