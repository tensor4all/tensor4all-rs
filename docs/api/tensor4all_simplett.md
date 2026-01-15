# tensor4all-simplett

## src/arithmetic.rs

### `pub fn add(&self, other: & Self) -> Result < Self >` (impl TensorTrain < T >)

Add two tensor trains element-wise The result has bond dimension equal to the sum of the input bond dimensions. Use `compress` to reduce the bond dimension afterward.

### `pub fn sub(&self, other: & Self) -> Result < Self >` (impl TensorTrain < T >)

Subtract another tensor train from this one

### `pub fn negate(&self) -> Self` (impl TensorTrain < T >)

Negate the tensor train (multiply by -1)

### ` fn add(self, other: Self) -> Self :: Output` (impl TensorTrain < T >)

### ` fn add(self, other: Self) -> Self :: Output` (impl & TensorTrain < T >)

### ` fn sub(self, other: Self) -> Self :: Output` (impl TensorTrain < T >)

### ` fn sub(self, other: Self) -> Self :: Output` (impl & TensorTrain < T >)

### ` fn neg(self) -> Self :: Output` (impl TensorTrain < T >)

### ` fn neg(self) -> Self :: Output` (impl & TensorTrain < T >)

### ` fn mul(self, scalar: T) -> Self :: Output` (impl TensorTrain < T >)

### ` fn mul(self, scalar: T) -> Self :: Output` (impl & TensorTrain < T >)

### ` fn test_add_constant_tensors()`

### ` fn test_sub_constant_tensors()`

### ` fn test_negate()`

### ` fn test_add_operator()`

### ` fn test_add_preserves_evaluation()`

## src/cache.rs

### `pub fn new(tt: & TT) -> Self` (impl TTCache < T >)

Create a new TTCache from a tensor train

### `pub fn with_site_dims(tt: & TT, site_dims: Vec < Vec < usize > >) -> Result < Self >` (impl TTCache < T >)

Create a new TTCache with custom site dimensions This allows treating a single tensor site as multiple logical indices.

### `pub fn len(&self) -> usize` (impl TTCache < T >)

Number of sites

### `pub fn is_empty(&self) -> bool` (impl TTCache < T >)

Check if empty

### `pub fn site_dims(&self) -> & [Vec < usize >]` (impl TTCache < T >)

Get site dimensions

### `pub fn link_dims(&self) -> Vec < usize >` (impl TTCache < T >)

Get link dimensions

### `pub fn link_dim(&self, i: usize) -> usize` (impl TTCache < T >)

Get link dimension at position i (between site i and i+1)

### `pub fn clear_cache(&mut self)` (impl TTCache < T >)

Clear all cached values

### ` fn multi_to_flat(&self, site: usize, indices: & [LocalIndex]) -> LocalIndex` (impl TTCache < T >)

Convert multi-index to flat index for a site

### `pub fn evaluate_left(&mut self, indices: & [LocalIndex]) -> Vec < T >` (impl TTCache < T >)

Evaluate from the left up to (but not including) site `end` Returns a vector of size `link_dim(end-1)` (or 1 if end == 0)

### `pub fn evaluate_right(&mut self, indices: & [LocalIndex]) -> Vec < T >` (impl TTCache < T >)

Evaluate from the right starting at site `start` `indices` contains indices for sites `start` to `n-1` Returns a vector of size `link_dim(start-1)` (or 1 if start == n)

### `pub fn evaluate(&mut self, indices: & [LocalIndex]) -> Result < T >` (impl TTCache < T >)

Evaluate the tensor train at a given index set using cache

### `pub fn batch_evaluate(&mut self, left_indices: & [MultiIndex], right_indices: & [MultiIndex], n_center: usize) -> Result < (Vec < T > , Vec < usize >) >` (impl TTCache < T >)

Batch evaluate the tensor train Evaluates for all combinations of left_indices and right_indices, with `n_center` free indices in the middle.

### ` fn test_ttcache_evaluate()`

### ` fn test_ttcache_caching()`

### ` fn test_ttcache_batch_evaluate()`

### ` fn test_ttcache_clear()`

## src/canonical.rs

### ` fn qr_decomp(matrix: & Matrix < T >) -> (Matrix < T > , Matrix < T >)`

Compute QR decomposition using rank-revealing LU with left-orthogonal output

### ` fn lq_decomp(matrix: & Matrix < T >) -> (Matrix < T > , Matrix < T >)`

Compute LQ decomposition (transpose, QR, transpose)

### ` fn tensor3_to_left_matrix(tensor: & Tensor3 < T >) -> Matrix < T >`

Convert Tensor3 to Matrix with left dimensions flattened

### ` fn tensor3_to_right_matrix(tensor: & Tensor3 < T >) -> Matrix < T >`

Convert Tensor3 to Matrix with right dimensions flattened

### `pub fn new(tensors: Vec < Tensor3 < T > >, center: usize) -> Result < Self >` (impl SiteTensorTrain < T >)

Create a new SiteTensorTrain from tensors with specified center

### `pub fn from_tensor_train(tt: & TensorTrain < T >, center: usize) -> Result < Self >` (impl SiteTensorTrain < T >)

Create from TensorTrain with specified center

### `pub fn center(&self) -> usize` (impl SiteTensorTrain < T >)

Get the center index

### `pub fn partition(&self) -> & Range < usize >` (impl SiteTensorTrain < T >)

Get the partition range

### `pub fn site_tensors_mut(&mut self) -> & mut [Tensor3 < T >]` (impl SiteTensorTrain < T >)

Get mutable access to site tensors

### ` fn canonicalize(&mut self)` (impl SiteTensorTrain < T >)

Canonicalize the tensor train around the center

### ` fn make_left_orthogonal(&mut self, i: usize)` (impl SiteTensorTrain < T >)

Make tensor at site i left-orthogonal, pushing R to site i+1

### ` fn make_right_orthogonal(&mut self, i: usize)` (impl SiteTensorTrain < T >)

Make tensor at site i right-orthogonal, pushing L to site i-1

### `pub fn move_center_right(&mut self) -> Result < () >` (impl SiteTensorTrain < T >)

Move the center one position to the right

### `pub fn move_center_left(&mut self) -> Result < () >` (impl SiteTensorTrain < T >)

Move the center one position to the left

### `pub fn set_center(&mut self, new_center: usize) -> Result < () >` (impl SiteTensorTrain < T >)

Move the center to a specific position

### `pub fn to_tensor_train(&self) -> TensorTrain < T >` (impl SiteTensorTrain < T >)

Convert to a regular TensorTrain

### `pub fn set_site_tensor(&mut self, i: usize, tensor: Tensor3 < T >)` (impl SiteTensorTrain < T >)

Set the tensor at a specific site Note: This may invalidate the canonical form. Use with caution.

### `pub fn set_two_site_tensors(&mut self, i: usize, tensor1: Tensor3 < T >, tensor2: Tensor3 < T >) -> Result < () >` (impl SiteTensorTrain < T >)

Set two adjacent site tensors (useful for TEBD-like algorithms)

### ` fn len(&self) -> usize` (impl SiteTensorTrain < T >)

### ` fn site_tensor(&self, i: usize) -> & Tensor3 < T >` (impl SiteTensorTrain < T >)

### ` fn site_tensors(&self) -> & [Tensor3 < T >]` (impl SiteTensorTrain < T >)

### `pub fn center_canonicalize(tensors: & mut [Tensor3 < T >], center: usize)`

Center canonicalize a vector of tensors in place

### ` fn test_site_tensor_train_creation()`

### ` fn test_site_tensor_train_preserves_values()`

### ` fn test_move_center()`

### ` fn test_set_center()`

### ` fn test_to_tensor_train()`

### ` fn test_center_canonicalize_function()`

### ` fn test_evaluate_matches_original()`

## src/compression.rs

### ` fn default() -> Self` (impl CompressionOptions)

### ` fn tensor3_to_left_matrix(tensor: & Tensor3 < T >) -> Matrix < T >`

Convert Tensor3 to Matrix for factorization (left matrix view)

### ` fn tensor3_to_right_matrix(tensor: & Tensor3 < T >) -> Matrix < T >`

Convert Tensor3 to Matrix for factorization (right matrix view)

### ` fn factorize(matrix: & Matrix < T >, method: CompressionMethod, tolerance: f64, max_bond_dim: usize, left_orthogonal: bool) -> (Matrix < T > , Matrix < T > , usize)`

Factorize a matrix into left and right factors

### `pub fn compress(&mut self, options: & CompressionOptions) -> Result < () >` (impl TensorTrain < T >)

Compress the tensor train in-place using the specified method This performs a two-sweep compression: 1. Left-to-right sweep with left-orthogonal factorization (no truncation)

### `pub fn compressed(&self, options: & CompressionOptions) -> Result < Self >` (impl TensorTrain < T >)

Create a compressed copy of the tensor train

### ` fn test_compress_constant()`

### ` fn test_compress_preserves_values()`

### ` fn test_compress_with_max_bond_dim()`

## src/contraction.rs

### ` fn default() -> Self` (impl ContractionOptions)

### `pub fn dot(&self, other: & Self) -> Result < T >` (impl TensorTrain < T >)

Compute the inner product (dot product) of two tensor trains Returns: sum over all indices i of self[i] * other[i]

### `pub fn dot(a: & TensorTrain < T >, b: & TensorTrain < T >) -> Result < T >`

Convenience function to compute dot product

### ` fn test_dot_constant()`

### ` fn test_dot_different_tensors()`

## src/decomposition.rs

### `pub fn qr_decomp(matrix: & Matrix < T >) -> (Matrix < T > , Matrix < T >)`

Compute QR decomposition using rank-revealing LU with left-orthogonal output Returns (Q, R) where Q is left-orthogonal and A ≈ Q * R

### `pub fn lq_decomp(matrix: & Matrix < T >) -> (Matrix < T > , Matrix < T >)`

Compute LQ decomposition (transpose, QR, transpose) Returns (L, Q) where Q is right-orthogonal and A ≈ L * Q

### `pub fn tensor3_to_left_matrix(tensor: & Tensor3 < T >) -> Matrix < T >`

Convert Tensor3 to Matrix with left dimensions flattened Reshapes tensor of shape (left, site, right) to matrix of shape (left * site, right)

### `pub fn tensor3_to_right_matrix(tensor: & Tensor3 < T >) -> Matrix < T >`

Convert Tensor3 to Matrix with right dimensions flattened Reshapes tensor of shape (left, site, right) to matrix of shape (left, site * right)

### ` fn test_qr_decomp_identity()`

### ` fn test_tensor3_to_left_matrix()`

### ` fn test_tensor3_to_right_matrix()`

## src/mpo/contract_fit.rs

### ` fn default() -> Self` (impl FitOptions)

### `pub fn contract_fit(mpo_a: & MPO < T >, mpo_b: & MPO < T >, options: & FitOptions, initial: Option < MPO < T > >) -> Result < MPO < T > >`

Perform variational fitting contraction of two MPOs This computes C = A * B using a variational (DMRG-like) algorithm that alternates between sweeping left-to-right and right-to-left,

### ` fn identity(dim_result: usize, dim_a: usize, dim_b: usize) -> Self` (impl Environment < T >)

### ` fn get(&self, r: usize, a: usize, b: usize) -> T` (impl Environment < T >)

### ` fn set(&mut self, r: usize, a: usize, b: usize, val: T)` (impl Environment < T >)

### ` fn build_left_environment(tensor_a: & Tensor4 < T >, tensor_b: & Tensor4 < T >, tensor_result: & Tensor4 < T >, prev_env: & Environment < T >) -> Result < Environment < T > >`

Build left environment by extending from previous environment

### ` fn build_right_environment(tensor_a: & Tensor4 < T >, tensor_b: & Tensor4 < T >, tensor_result: & Tensor4 < T >, next_env: & Environment < T >) -> Result < Environment < T > >`

Build right environment by extending from next environment

### ` fn update_two_site_core(_mpo_a: & MPO < T >, _mpo_b: & MPO < T >, _result: & mut SiteMPO < T >, _site: usize, _left_envs: & [Option < Environment < T > >], _right_envs: & [Option < Environment < T > >], _options: & FitOptions) -> Result < bool >`

Update the two-site core tensor at positions site and site+1

### ` fn test_contract_fit_identity()`

### ` fn test_contract_fit_constant()`

## src/mpo/contract_naive.rs

### `pub fn contract_naive(mpo_a: & MPO < T >, mpo_b: & MPO < T >, options: Option < ContractionOptions >) -> Result < MPO < T > >`

Perform naive contraction of two MPOs This computes C = A * B where the contraction is over the shared physical index (s2 of A contracts with s1 of B).

### ` fn compress_mpo(mpo: & mut MPO < T >, options: & ContractionOptions) -> Result < () >`

Compress an MPO using the specified options

### ` fn test_contract_naive_identity()`

### ` fn test_contract_naive_constant()`

### ` fn test_contract_naive_dimension_mismatch()`

### ` fn test_contract_naive_with_compression()`

## src/mpo/contract_zipup.rs

### `pub fn contract_zipup(mpo_a: & MPO < T >, mpo_b: & MPO < T >, options: & ContractionOptions) -> Result < MPO < T > >`

Perform zip-up contraction of two MPOs This computes C = A * B where the contraction is over the shared physical index (s2 of A contracts with s1 of B), with on-the-fly

### ` fn test_contract_zipup_identity()`

### ` fn test_contract_zipup_constant()`

### ` fn test_contract_zipup_compresses()`

## src/mpo/contraction.rs

### ` fn default() -> Self` (impl ContractionOptions)

### `pub fn new(mpo_a: MPO < T >, mpo_b: MPO < T >) -> Result < Self >` (impl Contraction < T >)

Create a new Contraction from two MPOs

### `pub fn with_transform(mpo_a: MPO < T >, mpo_b: MPO < T >, f: F) -> Result < Self >` (impl Contraction < T >)

Create a new Contraction with a transformation function

### `pub fn len(&self) -> usize` (impl Contraction < T >)

Get the number of sites

### `pub fn is_empty(&self) -> bool` (impl Contraction < T >)

Check if empty

### `pub fn result_site_dims(&self) -> Vec < (usize , usize) >` (impl Contraction < T >)

Get site dimensions for the contracted result Returns (s1_result, s2_result) at each site where: - s1_result = s1_a (first physical index of A)

### `pub fn clear_cache(&mut self)` (impl Contraction < T >)

Clear all cached environments

### `pub fn evaluate(&mut self, indices: & [(usize , usize)]) -> Result < T >` (impl Contraction < T >)

Evaluate the contraction at a specific set of indices indices should be [(i1, j1), (i2, j2), ...] where: - i_k is the index for s1 of MPO A at site k

### `pub fn evaluate_left(&mut self, n: usize, indices: & [(usize , usize)]) -> Result < Matrix2 < T > >` (impl Contraction < T >)

Evaluate the left environment up to site n (exclusive) Returns L[n] = product of sites 0..n

### `pub fn evaluate_right(&mut self, n: usize, indices: & [(usize , usize)]) -> Result < Matrix2 < T > >` (impl Contraction < T >)

Evaluate the right environment from site n (exclusive) to the end Returns R[n] = product of sites n..L

### ` fn test_contraction_new()`

### ` fn test_contraction_evaluate()`

### ` fn test_contraction_with_transform()`

## src/mpo/dispatch.rs

### `pub fn contract(mpo_a: & MPO < T >, mpo_b: & MPO < T >, algorithm: ContractionAlgorithm, options: & ContractionOptions) -> Result < MPO < T > >`

Unified contraction function with algorithm dispatch Contracts two MPOs using the specified algorithm.

### ` fn test_contract_dispatch_naive()`

### ` fn test_contract_dispatch_zipup()`

### ` fn test_contract_dispatch_fit()`

### ` fn test_contract_algorithms_consistent()`

## src/mpo/environment.rs

### ` fn matrix2_zeros(rows: usize, cols: usize) -> Matrix2 < T >`

Helper function to create a zero-filled 2D tensor

### `pub fn contract_tensors(_a: & [T], _a_shape: & [usize], _b: & [T], _b_shape: & [usize], _idx_a: & [usize], _idx_b: & [usize]) -> Result < (Vec < T > , Vec < usize >) >`

Contract two general tensors over specified indices This is the Rust equivalent of `_contract` from Julia.

### `pub fn contract_site_tensors(a: & Tensor4 < T >, b: & Tensor4 < T >) -> Result < Tensor4 < T > >`

Contract two 4D site tensors over their shared physical index Given two 4D tensors: - A: (left_a, s1_a, s2_a, right_a) where s2_a is the shared index

### `pub fn left_environment(mpo_a: & MPO < T >, mpo_b: & MPO < T >, site: usize, cache: & mut Vec < Option < Matrix2 < T > > >) -> Result < Matrix2 < T > >`

Compute the left environment at site i for MPO contraction The left environment L[i] represents the contraction of all sites 0..i for the product of two MPOs A and B.

### `pub fn right_environment(mpo_a: & MPO < T >, mpo_b: & MPO < T >, site: usize, cache: & mut Vec < Option < Matrix2 < T > > >) -> Result < Matrix2 < T > >`

Compute the right environment at site i for MPO contraction The right environment R[i] represents the contraction of all sites i+1..L for the product of two MPOs A and B.

### ` fn test_contract_site_tensors()`

### ` fn test_left_environment()`

### ` fn test_right_environment()`

## src/mpo/factorize.rs

### ` fn default() -> Self` (impl FactorizeOptions)

### `pub fn factorize(matrix: & Matrix2 < T >, options: & FactorizeOptions) -> Result < FactorizeResult < T > >`

Factorize a matrix into left and right factors Returns (L, R, rank, discarded) where: - L: left factor matrix (rows x rank)

### ` fn factorize_svd(matrix: & Matrix2 < T >, options: & FactorizeOptions) -> Result < FactorizeResult < T > >`

Factorize using SVD

### ` fn factorize_rsvd(_matrix: & Matrix2 < T >, _options: & FactorizeOptions) -> Result < FactorizeResult < T > >`

Factorize using randomized SVD

### `pub fn factorize_lu(matrix: & Matrix2 < T >, options: & FactorizeOptions) -> Result < FactorizeResult < T > >`

Factorize using LU decomposition This function requires the matrixci::Scalar trait. Use this directly when you need LU-based factorization.

### `pub fn factorize_ci(matrix: & Matrix2 < T >, options: & FactorizeOptions) -> Result < FactorizeResult < T > >`

Factorize using Cross Interpolation This function requires the matrixci::Scalar trait. Use this directly when you need CI-based factorization.

### ` fn test_factorize_svd()`

### ` fn test_factorize_lu()`

### ` fn test_factorize_with_truncation()`

### ` fn test_factorize_svd_complex64()`

## src/mpo/inverse_mpo.rs

### `pub fn from_mpo(_mpo: MPO < T >) -> Result < Self >` (impl InverseMPO < T >)

Create an InverseMPO from an MPO

### `pub(crate) fn from_parts_unchecked(tensors: Vec < Tensor4 < T > >, inv_lambdas: Vec < Vec < f64 > >) -> Self` (impl InverseMPO < T >)

Create an InverseMPO from parts without validation

### `pub fn len(&self) -> usize` (impl InverseMPO < T >)

Get the number of sites

### `pub fn is_empty(&self) -> bool` (impl InverseMPO < T >)

Check if empty

### `pub fn site_tensor(&self, i: usize) -> & Tensor4 < T >` (impl InverseMPO < T >)

Get the tensor at position i

### `pub fn site_tensor_mut(&mut self, i: usize) -> & mut Tensor4 < T >` (impl InverseMPO < T >)

Get mutable reference to the tensor at position i

### `pub fn site_tensors(&self) -> & [Tensor4 < T >]` (impl InverseMPO < T >)

Get all tensors

### `pub fn inv_lambda(&self, i: usize) -> & [f64]` (impl InverseMPO < T >)

Get the inverse Lambda vector at bond i (between sites i and i+1)

### `pub fn inv_lambdas(&self) -> & [Vec < f64 >]` (impl InverseMPO < T >)

Get all inverse Lambda vectors

### `pub fn link_dims(&self) -> Vec < usize >` (impl InverseMPO < T >)

Bond dimensions

### `pub fn site_dims(&self) -> Vec < (usize , usize) >` (impl InverseMPO < T >)

Site dimensions

### `pub fn rank(&self) -> usize` (impl InverseMPO < T >)

Maximum bond dimension

### `pub fn into_mpo(self) -> Result < MPO < T > >` (impl InverseMPO < T >)

Convert to basic MPO

### ` fn test_inverse_mpo_placeholder()`

## src/mpo/mod.rs

### `pub(crate) fn matrix2_zeros(rows: usize, cols: usize) -> Matrix2 < T >`

Helper function to create a zero-filled 2D tensor. This is a shared utility used across multiple MPO modules.

## src/mpo/mpo.rs

### `pub fn new(tensors: Vec < Tensor4 < T > >) -> Result < Self >` (impl MPO < T >)

Create a new MPO from a list of 4D tensors Each tensor should have shape (left_bond, site_dim_1, site_dim_2, right_bond) where the right_bond of tensor i equals the left_bond of tensor i+1.

### `pub(crate) fn from_tensors_unchecked(tensors: Vec < Tensor4 < T > >) -> Self` (impl MPO < T >)

Create an MPO from tensors without dimension validation (for internal use when dimensions are known to be correct)

### `pub fn zeros(site_dims: & [(usize , usize)]) -> Self` (impl MPO < T >)

Create an MPO representing the zero operator

### `pub fn constant(site_dims: & [(usize , usize)], value: T) -> Self` (impl MPO < T >)

Create an MPO representing a constant operator Each element O[i1, j1, i2, j2, ..., iL, jL] = value

### `pub fn identity(site_dims: & [usize]) -> Result < Self >` (impl MPO < T >)

Create an identity MPO (only when site_dim_1 == site_dim_2 at each site) The identity operator: O[i1, j1, ...] = delta(i1, j1) * delta(i2, j2) * ...

### `pub fn len(&self) -> usize` (impl MPO < T >)

Number of sites (tensors) in the MPO

### `pub fn is_empty(&self) -> bool` (impl MPO < T >)

Check if the MPO is empty

### `pub fn site_tensor(&self, i: usize) -> & Tensor4 < T >` (impl MPO < T >)

Get the site tensor at position i

### `pub fn site_tensor_mut(&mut self, i: usize) -> & mut Tensor4 < T >` (impl MPO < T >)

Get mutable reference to the site tensor at position i

### `pub fn site_tensors(&self) -> & [Tensor4 < T >]` (impl MPO < T >)

Get all site tensors

### `pub fn site_tensors_mut(&mut self) -> & mut [Tensor4 < T >]` (impl MPO < T >)

Get mutable access to the site tensors

### `pub fn link_dims(&self) -> Vec < usize >` (impl MPO < T >)

Bond dimensions along the links between tensors Returns a vector of length L-1 where L is the number of sites

### `pub fn link_dim(&self, i: usize) -> usize` (impl MPO < T >)

Bond dimension at the link between tensor i and i+1

### `pub fn site_dims(&self) -> Vec < (usize , usize) >` (impl MPO < T >)

Site dimensions (physical dimensions) for each tensor Returns a vector of (site_dim_1, site_dim_2) tuples

### `pub fn site_dim(&self, i: usize) -> (usize , usize)` (impl MPO < T >)

Site dimensions at position i

### `pub fn rank(&self) -> usize` (impl MPO < T >)

Maximum bond dimension (rank) of the MPO

### `pub fn evaluate(&self, indices: & [LocalIndex]) -> Result < T >` (impl MPO < T >)

Evaluate the MPO at a given index set indices should have length 2*L where L is the number of sites alternating between site_dim_1 and site_dim_2 indices:

### `pub fn sum(&self) -> T` (impl MPO < T >)

Sum over all indices of the MPO

### `pub fn scale(&mut self, factor: T)` (impl MPO < T >)

Multiply the MPO by a scalar

### `pub fn scaled(&self, factor: T) -> Self` (impl MPO < T >)

Create a scaled copy of the MPO

### `pub fn fulltensor(&self) -> (Vec < T > , Vec < usize >)` (impl MPO < T >)

Convert the MPO to a full tensor Returns a flat vector containing all tensor elements in row-major order, along with the shape (alternating site_dim_1, site_dim_2 dimensions).

### ` fn test_mpo_zeros()`

### ` fn test_mpo_constant()`

### ` fn test_mpo_identity()`

### ` fn test_mpo_evaluate()`

### ` fn test_mpo_scale()`

### ` fn test_mpo_link_dims()`

### ` fn test_mpo_fulltensor()`

## src/mpo/site_mpo.rs

### `pub fn from_mpo(mpo: MPO < T >, center: usize) -> Result < Self >` (impl SiteMPO < T >)

Create a SiteMPO from an MPO, placing the center at the given position

### `pub(crate) fn from_tensors_unchecked(tensors: Vec < Tensor4 < T > >, center: usize) -> Self` (impl SiteMPO < T >)

Create a SiteMPO from tensors without validation

### `pub fn center(&self) -> usize` (impl SiteMPO < T >)

Get the current orthogonality center position

### `pub fn len(&self) -> usize` (impl SiteMPO < T >)

Get the number of sites

### `pub fn is_empty(&self) -> bool` (impl SiteMPO < T >)

Check if empty

### `pub fn site_tensor(&self, i: usize) -> & Tensor4 < T >` (impl SiteMPO < T >)

Get the site tensor at position i

### `pub fn site_tensor_mut(&mut self, i: usize) -> & mut Tensor4 < T >` (impl SiteMPO < T >)

Get mutable reference to the site tensor at position i

### `pub fn site_tensors(&self) -> & [Tensor4 < T >]` (impl SiteMPO < T >)

Get all site tensors

### `pub fn site_tensors_mut(&mut self) -> & mut [Tensor4 < T >]` (impl SiteMPO < T >)

Get mutable access to site tensors

### `pub fn link_dims(&self) -> Vec < usize >` (impl SiteMPO < T >)

Bond dimensions

### `pub fn site_dims(&self) -> Vec < (usize , usize) >` (impl SiteMPO < T >)

Site dimensions

### `pub fn rank(&self) -> usize` (impl SiteMPO < T >)

Maximum bond dimension

### `pub fn move_center_left(&mut self) -> Result < () >` (impl SiteMPO < T >)

Move the orthogonality center one position to the left

### `pub fn move_center_right(&mut self) -> Result < () >` (impl SiteMPO < T >)

Move the orthogonality center one position to the right

### `pub fn set_center(&mut self, target: usize) -> Result < () >` (impl SiteMPO < T >)

Move the orthogonality center to the specified position

### `pub fn into_mpo(self) -> MPO < T >` (impl SiteMPO < T >)

Convert to basic MPO

### ` fn test_site_mpo_creation()`

### ` fn test_site_mpo_move_center()`

### ` fn test_site_mpo_invalid_center()`

## src/mpo/types.rs

### `pub fn left_dim(&self) -> usize` (trait Tensor4Ops)

Get the left (bond) dimension

### `pub fn site_dim_1(&self) -> usize` (trait Tensor4Ops)

Get the first site (physical) dimension

### `pub fn site_dim_2(&self) -> usize` (trait Tensor4Ops)

Get the second site (physical) dimension

### `pub fn right_dim(&self) -> usize` (trait Tensor4Ops)

Get the right (bond) dimension

### `pub fn get4(&self, l: usize, s1: usize, s2: usize, r: usize) -> & T` (trait Tensor4Ops)

Get element at (left, site1, site2, right)

### `pub fn get4_mut(&mut self, l: usize, s1: usize, s2: usize, r: usize) -> & mut T` (trait Tensor4Ops)

Get mutable element at (left, site1, site2, right)

### `pub fn set4(&mut self, l: usize, s1: usize, s2: usize, r: usize, value: T)` (trait Tensor4Ops)

Set element at (left, site1, site2, right)

### `pub fn slice_site(&self, s1: usize, s2: usize) -> Vec < T >` (trait Tensor4Ops)

Get a slice for fixed site indices: returns (left_dim, right_dim) matrix as flat Vec

### `pub fn as_left_matrix(&self) -> (Vec < T > , usize , usize)` (trait Tensor4Ops)

Reshape this tensor to a matrix (left_dim * site_dim_1 * site_dim_2, right_dim)

### `pub fn as_right_matrix(&self) -> (Vec < T > , usize , usize)` (trait Tensor4Ops)

Reshape this tensor to a matrix (left_dim, site_dim_1 * site_dim_2 * right_dim)

### `pub fn as_center_matrix(&self) -> (Vec < T > , usize , usize)` (trait Tensor4Ops)

Reshape this tensor to a matrix (left_dim * site_dim_1, site_dim_2 * right_dim)

### ` fn left_dim(&self) -> usize` (impl Tensor4 < T >)

### ` fn site_dim_1(&self) -> usize` (impl Tensor4 < T >)

### ` fn site_dim_2(&self) -> usize` (impl Tensor4 < T >)

### ` fn right_dim(&self) -> usize` (impl Tensor4 < T >)

### ` fn get4(&self, l: usize, s1: usize, s2: usize, r: usize) -> & T` (impl Tensor4 < T >)

### ` fn get4_mut(&mut self, l: usize, s1: usize, s2: usize, r: usize) -> & mut T` (impl Tensor4 < T >)

### ` fn set4(&mut self, l: usize, s1: usize, s2: usize, r: usize, value: T)` (impl Tensor4 < T >)

### ` fn slice_site(&self, s1: usize, s2: usize) -> Vec < T >` (impl Tensor4 < T >)

### ` fn as_left_matrix(&self) -> (Vec < T > , usize , usize)` (impl Tensor4 < T >)

### ` fn as_right_matrix(&self) -> (Vec < T > , usize , usize)` (impl Tensor4 < T >)

### ` fn as_center_matrix(&self) -> (Vec < T > , usize , usize)` (impl Tensor4 < T >)

### `pub fn tensor4_zeros(left_dim: usize, site_dim_1: usize, site_dim_2: usize, right_dim: usize) -> Tensor4 < T >`

Create a zero-filled Tensor4

### `pub fn tensor4_from_data(data: Vec < T >, left_dim: usize, site_dim_1: usize, site_dim_2: usize, right_dim: usize) -> Tensor4 < T >`

Create a Tensor4 from flat data (row-major order)

### ` fn test_tensor4_zeros()`

### ` fn test_tensor4_get_set()`

### ` fn test_tensor4_from_data()`

### ` fn test_slice_site()`

### ` fn test_as_left_matrix()`

### ` fn test_as_right_matrix()`

### ` fn test_as_center_matrix()`

## src/mpo/vidal_mpo.rs

### `pub fn from_mpo(_mpo: MPO < T >) -> Result < Self >` (impl VidalMPO < T >)

Create a VidalMPO from an MPO

### `pub(crate) fn from_parts_unchecked(gammas: Vec < Tensor4 < T > >, lambdas: Vec < Vec < f64 > >) -> Self` (impl VidalMPO < T >)

Create a VidalMPO from tensors and lambdas without validation

### `pub fn len(&self) -> usize` (impl VidalMPO < T >)

Get the number of sites

### `pub fn is_empty(&self) -> bool` (impl VidalMPO < T >)

Check if empty

### `pub fn gamma(&self, i: usize) -> & Tensor4 < T >` (impl VidalMPO < T >)

Get the Gamma tensor at position i

### `pub fn gamma_mut(&mut self, i: usize) -> & mut Tensor4 < T >` (impl VidalMPO < T >)

Get mutable reference to the Gamma tensor at position i

### `pub fn gammas(&self) -> & [Tensor4 < T >]` (impl VidalMPO < T >)

Get all Gamma tensors

### `pub fn lambda(&self, i: usize) -> & [f64]` (impl VidalMPO < T >)

Get the Lambda vector at bond i (between sites i and i+1)

### `pub fn lambdas(&self) -> & [Vec < f64 >]` (impl VidalMPO < T >)

Get all Lambda vectors

### `pub fn link_dims(&self) -> Vec < usize >` (impl VidalMPO < T >)

Bond dimensions

### `pub fn site_dims(&self) -> Vec < (usize , usize) >` (impl VidalMPO < T >)

Site dimensions

### `pub fn rank(&self) -> usize` (impl VidalMPO < T >)

Maximum bond dimension

### `pub fn into_mpo(self) -> Result < MPO < T > >` (impl VidalMPO < T >)

Convert to basic MPO

### ` fn test_vidal_mpo_placeholder()`

## src/tensortrain.rs

### `pub fn new(tensors: Vec < Tensor3 < T > >) -> Result < Self >` (impl TensorTrain < T >)

Create a new tensor train from a list of 3D tensors Each tensor should have shape (left_bond, site_dim, right_bond) where the right_bond of tensor i equals the left_bond of tensor i+1.

### `pub(crate) fn from_tensors_unchecked(tensors: Vec < Tensor3 < T > >) -> Self` (impl TensorTrain < T >)

Create a tensor train from tensors without dimension validation (for internal use when dimensions are known to be correct)

### `pub fn zeros(site_dims: & [usize]) -> Self` (impl TensorTrain < T >)

Create a tensor train representing the zero function

### `pub fn constant(site_dims: & [usize], value: T) -> Self` (impl TensorTrain < T >)

Create a tensor train representing a constant function

### `pub fn site_tensors_mut(&mut self) -> & mut [Tensor3 < T >]` (impl TensorTrain < T >)

Get mutable access to the site tensors

### `pub fn scale(&mut self, factor: T)` (impl TensorTrain < T >)

Multiply the tensor train by a scalar

### `pub fn scaled(&self, factor: T) -> Self` (impl TensorTrain < T >)

Create a scaled copy of the tensor train

### `pub fn reverse(&self) -> Self` (impl TensorTrain < T >)

Reverse the tensor train (swap left and right)

### `pub fn fulltensor(&self) -> (Vec < T > , Vec < usize >)` (impl TensorTrain < T >)

Convert the tensor train to a full tensor Returns a flat vector containing all tensor elements in row-major order, along with the shape (site dimensions).

### ` fn len(&self) -> usize` (impl TensorTrain < T >)

### ` fn site_tensor(&self, i: usize) -> & Tensor3 < T >` (impl TensorTrain < T >)

### ` fn site_tensors(&self) -> & [Tensor3 < T >]` (impl TensorTrain < T >)

### ` fn test_tensortrain_zeros()`

### ` fn test_tensortrain_constant()`

### ` fn test_tensortrain_evaluate()`

### ` fn test_tensortrain_scale()`

### ` fn test_tensortrain_reverse()`

### ` fn test_fulltensor()`

### ` fn test_fulltensor_matches_evaluate()`

### ` fn test_log_norm_matches_norm()`

### ` fn test_log_norm_with_varied_values()`

### ` fn test_log_norm_zero_tensor()`

## src/traits.rs

### `pub fn conj(self) -> Self` (trait TTScalar)

Conjugate

### `pub fn abs_sq(self) -> f64` (trait TTScalar)

Absolute value squared

### `pub fn from_f64(val: f64) -> Self` (trait TTScalar)

Create from f64

### ` fn conj(self) -> Self` (impl f64)

### ` fn abs_sq(self) -> f64` (impl f64)

### ` fn from_f64(val: f64) -> Self` (impl f64)

### ` fn conj(self) -> Self` (impl f32)

### ` fn abs_sq(self) -> f64` (impl f32)

### ` fn from_f64(val: f64) -> Self` (impl f32)

### ` fn conj(self) -> Self` (impl num_complex :: Complex64)

### ` fn abs_sq(self) -> f64` (impl num_complex :: Complex64)

### ` fn from_f64(val: f64) -> Self` (impl num_complex :: Complex64)

### ` fn conj(self) -> Self` (impl num_complex :: Complex32)

### ` fn abs_sq(self) -> f64` (impl num_complex :: Complex32)

### ` fn from_f64(val: f64) -> Self` (impl num_complex :: Complex32)

### `pub fn len(&self) -> usize` (trait AbstractTensorTrain)

Number of sites (tensors) in the tensor train

### `pub fn is_empty(&self) -> bool` (trait AbstractTensorTrain default)

Check if the tensor train is empty

### `pub fn site_tensor(&self, i: usize) -> & Tensor3 < T >` (trait AbstractTensorTrain)

Get the site tensor at position i

### `pub fn site_tensors(&self) -> & [Tensor3 < T >]` (trait AbstractTensorTrain)

Get all site tensors

### `pub fn link_dims(&self) -> Vec < usize >` (trait AbstractTensorTrain default)

Bond dimensions along the links between tensors Returns a vector of length L-1 where L is the number of sites

### `pub fn link_dim(&self, i: usize) -> usize` (trait AbstractTensorTrain default)

Bond dimension at the link between tensor i and i+1

### `pub fn site_dims(&self) -> Vec < usize >` (trait AbstractTensorTrain default)

Site dimensions (physical dimensions) for each tensor

### `pub fn site_dim(&self, i: usize) -> usize` (trait AbstractTensorTrain default)

Site dimension at position i

### `pub fn rank(&self) -> usize` (trait AbstractTensorTrain default)

Maximum bond dimension (rank) of the tensor train

### `pub fn evaluate(&self, indices: & [LocalIndex]) -> Result < T >` (trait AbstractTensorTrain default)

Evaluate the tensor train at a given index set

### `pub fn sum(&self) -> T` (trait AbstractTensorTrain default)

Sum over all indices of the tensor train

### `pub fn norm2(&self) -> f64` (trait AbstractTensorTrain default)

Compute the squared Frobenius norm of the tensor train

### `pub fn norm(&self) -> f64` (trait AbstractTensorTrain default)

Compute the Frobenius norm of the tensor train

### `pub fn log_norm(&self) -> f64` (trait AbstractTensorTrain default)

Compute the logarithm of the Frobenius norm of the tensor train This is more numerically stable than `norm().ln()` for tensor trains with very large or very small norms, as it avoids overflow/underflow

## src/types.rs

### `pub fn left_dim(&self) -> usize` (trait Tensor3Ops)

Get the left (bond) dimension

### `pub fn site_dim(&self) -> usize` (trait Tensor3Ops)

Get the site (physical) dimension

### `pub fn right_dim(&self) -> usize` (trait Tensor3Ops)

Get the right (bond) dimension

### `pub fn get3(&self, l: usize, s: usize, r: usize) -> & T` (trait Tensor3Ops)

Get element at (left, site, right)

### `pub fn get3_mut(&mut self, l: usize, s: usize, r: usize) -> & mut T` (trait Tensor3Ops)

Get mutable element at (left, site, right)

### `pub fn set3(&mut self, l: usize, s: usize, r: usize, value: T)` (trait Tensor3Ops)

Set element at (left, site, right)

### `pub fn slice_site(&self, s: usize) -> Vec < T >` (trait Tensor3Ops)

Get a slice for fixed site index: returns (left_dim, right_dim) matrix as flat Vec

### `pub fn as_left_matrix(&self) -> (Vec < T > , usize , usize)` (trait Tensor3Ops)

Reshape this tensor to a matrix (left_dim * site_dim, right_dim)

### `pub fn as_right_matrix(&self) -> (Vec < T > , usize , usize)` (trait Tensor3Ops)

Reshape this tensor to a matrix (left_dim, site_dim * right_dim)

### ` fn left_dim(&self) -> usize` (impl Tensor3 < T >)

### ` fn site_dim(&self) -> usize` (impl Tensor3 < T >)

### ` fn right_dim(&self) -> usize` (impl Tensor3 < T >)

### ` fn get3(&self, l: usize, s: usize, r: usize) -> & T` (impl Tensor3 < T >)

### ` fn get3_mut(&mut self, l: usize, s: usize, r: usize) -> & mut T` (impl Tensor3 < T >)

### ` fn set3(&mut self, l: usize, s: usize, r: usize, value: T)` (impl Tensor3 < T >)

### ` fn slice_site(&self, s: usize) -> Vec < T >` (impl Tensor3 < T >)

### ` fn as_left_matrix(&self) -> (Vec < T > , usize , usize)` (impl Tensor3 < T >)

### ` fn as_right_matrix(&self) -> (Vec < T > , usize , usize)` (impl Tensor3 < T >)

### `pub fn tensor3_zeros(left_dim: usize, site_dim: usize, right_dim: usize) -> Tensor3 < T >`

Create a zero-filled Tensor3

### `pub fn tensor3_from_data(data: Vec < T >, left_dim: usize, site_dim: usize, right_dim: usize) -> Tensor3 < T >`

Create a Tensor3 from flat data (row-major order)

## src/vidal.rs

### ` fn qr_decomp(matrix: & Matrix < T >) -> (Matrix < T > , Matrix < T >)`

Compute QR decomposition

### ` fn lq_decomp(matrix: & Matrix < T >) -> (Matrix < T > , Matrix < T >)`

Compute LQ decomposition

### ` fn tensor3_to_left_matrix(tensor: & Tensor3 < T >) -> Matrix < T >`

Convert Tensor3 to Matrix with left dimensions flattened

### ` fn tensor3_to_right_matrix(tensor: & Tensor3 < T >) -> Matrix < T >`

Convert Tensor3 to Matrix with right dimensions flattened

### `pub fn from_tensor_train(tt: & TensorTrain < T >) -> Result < Self >` (impl VidalTensorTrain < T >)

Create a VidalTensorTrain from a regular TensorTrain

### `pub fn from_tensor_train_with_partition(tt: & TensorTrain < T >, partition: Range < usize >) -> Result < Self >` (impl VidalTensorTrain < T >)

Create a VidalTensorTrain with a specific partition

### `pub fn new(tensors: Vec < Tensor3 < T > >, singular_values: Vec < DiagMatrix >) -> Result < Self >` (impl VidalTensorTrain < T >)

Create a VidalTensorTrain with given tensors and singular values

### `pub fn singular_values(&self, i: usize) -> & DiagMatrix` (impl VidalTensorTrain < T >)

Get the singular values between sites i and i+1

### `pub fn all_singular_values(&self) -> & [DiagMatrix]` (impl VidalTensorTrain < T >)

Get all singular value matrices

### `pub fn partition(&self) -> & Range < usize >` (impl VidalTensorTrain < T >)

Get the partition range

### `pub fn site_tensors_mut(&mut self) -> & mut [Tensor3 < T >]` (impl VidalTensorTrain < T >)

Get mutable access to site tensors

### `pub fn singular_values_mut(&mut self, i: usize) -> & mut DiagMatrix` (impl VidalTensorTrain < T >)

Get mutable access to singular values

### `pub fn to_tensor_train(&self) -> TensorTrain < T >` (impl VidalTensorTrain < T >)

Convert to a regular TensorTrain

### ` fn len(&self) -> usize` (impl VidalTensorTrain < T >)

### ` fn site_tensor(&self, i: usize) -> & Tensor3 < T >` (impl VidalTensorTrain < T >)

### ` fn site_tensors(&self) -> & [Tensor3 < T >]` (impl VidalTensorTrain < T >)

### `pub fn from_vidal(vidal: & VidalTensorTrain < T >) -> Result < Self >` (impl InverseTensorTrain < T >)

Create an InverseTensorTrain from a VidalTensorTrain

### `pub fn from_tensor_train(tt: & TensorTrain < T >) -> Result < Self >` (impl InverseTensorTrain < T >)

Create an InverseTensorTrain from a regular TensorTrain

### `pub fn inverse_singular_values(&self, i: usize) -> & DiagMatrix` (impl InverseTensorTrain < T >)

Get the inverse singular values between sites i and i+1

### `pub fn all_inverse_singular_values(&self) -> & [DiagMatrix]` (impl InverseTensorTrain < T >)

Get all inverse singular value matrices

### `pub fn partition(&self) -> & Range < usize >` (impl InverseTensorTrain < T >)

Get the partition range

### `pub fn site_tensors_mut(&mut self) -> & mut [Tensor3 < T >]` (impl InverseTensorTrain < T >)

Get mutable access to site tensors

### `pub fn set_two_site_tensors(&mut self, i: usize, tensor1: Tensor3 < T >, inv_sv: DiagMatrix, tensor2: Tensor3 < T >) -> Result < () >` (impl InverseTensorTrain < T >)

Set two adjacent site tensors along with their inverse singular values

### `pub fn to_tensor_train(&self) -> TensorTrain < T >` (impl InverseTensorTrain < T >)

Convert to a regular TensorTrain

### ` fn len(&self) -> usize` (impl InverseTensorTrain < T >)

### ` fn site_tensor(&self, i: usize) -> & Tensor3 < T >` (impl InverseTensorTrain < T >)

### ` fn site_tensors(&self) -> & [Tensor3 < T >]` (impl InverseTensorTrain < T >)

### ` fn test_vidal_creation()`

### ` fn test_vidal_to_tensor_train_preserves_sum()`

### ` fn test_inverse_creation()`

### ` fn test_inverse_to_tensor_train_preserves_sum()`

### ` fn test_vidal_singular_values_positive()`

