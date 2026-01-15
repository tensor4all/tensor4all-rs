# tensor4all-tensorbackend

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

### ` fn tensor_to_dtensor(tensor: & mdarray :: Tensor < T , (usize , usize) >) -> DTensor < T , 2 >`

Convert mdarray Tensor<T, (usize, usize)> to DTensor<T, 2>

### `pub fn svd_backend(a: & mut DSlice < T , 2 >) -> Result < SvdResult < T > >`

Compute SVD decomposition using FAER backend.

### `pub fn qr_backend(a: & mut DSlice < T , 2 >) -> (DTensor < T , 2 > , DTensor < T , 2 >)`

Compute QR decomposition using FAER backend.

### `pub fn svd_impl(a: & mut DSlice < Self , 2 >) -> Result < SvdResult < Self > >` (trait SvdBackendImpl)

### `pub fn qr_impl(a: & mut DSlice < Self , 2 >) -> (DTensor < Self , 2 > , DTensor < Self , 2 >)` (trait QrBackendImpl)

### `pub fn svd_backend(a: & mut DSlice < T , 2 >) -> Result < SvdResult < T > >`

Compute SVD decomposition using LAPACK backend.

### `pub fn qr_backend(a: & mut DSlice < T , 2 >) -> (DTensor < T , 2 > , DTensor < T , 2 >)`

Compute QR decomposition using LAPACK backend.

## src/storage.rs

### `pub fn with_capacity(capacity: usize) -> Self` (impl DenseStorage < T >)

### `pub fn from_vec(vec: Vec < T >) -> Self` (impl DenseStorage < T >)

### `pub fn as_slice(&self) -> & [T]` (impl DenseStorage < T >)

### `pub fn as_mut_slice(&mut self) -> & mut [T]` (impl DenseStorage < T >)

### `pub fn into_vec(self) -> Vec < T >` (impl DenseStorage < T >)

### `pub fn len(&self) -> usize` (impl DenseStorage < T >)

### `pub fn is_empty(&self) -> bool` (impl DenseStorage < T >)

### `pub fn capacity(&self) -> usize` (impl DenseStorage < T >)

### `pub fn push(&mut self, val: T)` (impl DenseStorage < T >)

### `pub fn iter(&self) -> std :: slice :: Iter < '_ , T >` (impl DenseStorage < T >)

### `pub fn extend_from_slice(&mut self, other: & [T])` (impl DenseStorage < T >)

### `pub fn get(&self, i: usize) -> T` (impl DenseStorage < T >)

### `pub fn set(&mut self, i: usize, val: T)` (impl DenseStorage < T >)

### `pub fn extend(&mut self, iter: I)` (impl DenseStorage < T >)

### `pub fn permute(&self, dims: & [usize], perm: & [usize]) -> Self` (impl DenseStorage < T >)

Permute the dense storage data according to the given permutation.

### `pub fn contract(&self, dims: & [usize], axes: & [usize], other: & Self, other_dims: & [usize], other_axes: & [usize]) -> Self` (impl DenseStorage < T >)

Contract this dense storage with another dense storage. This method handles non-contiguous contracted axes by permuting the tensors to make the contracted axes contiguous before calling GEMM-based contraction.

### `pub fn random(rng: & mut R, size: usize) -> Self` (impl DenseStorage < f64 >)

Create storage with random values from standard normal distribution.

### `pub fn random(rng: & mut R, size: usize) -> Self` (impl DenseStorage < Complex64 >)

Create storage with random complex values (re, im both from standard normal).

### ` fn contract_via_gemm(a: & [T], dims_a: & [usize], axes_a: & [usize], b: & [T], dims_b: & [usize], axes_b: & [usize]) -> Vec < T >`

Contract two tensors via GEMM (matrix multiplication). This function assumes that contracted axes are already contiguous: - For `a`: contracted axes are at the END (axes_a are the last naxes positions)

### ` fn compute_contraction_permutation(dims: & [usize], axes: & [usize], axes_at_front: bool) -> (Vec < usize > , Vec < usize > , Vec < usize >)`

Compute permutation to make contracted axes contiguous. If `axes_at_front` is true, contracted axes are moved to the front (maintaining original order). If false, contracted axes are moved to the end (maintaining original order).

### `pub fn from_vec(vec: Vec < T >) -> Self` (impl DiagStorage < T >)

### `pub fn as_slice(&self) -> & [T]` (impl DiagStorage < T >)

### `pub fn as_mut_slice(&mut self) -> & mut [T]` (impl DiagStorage < T >)

### `pub fn into_vec(self) -> Vec < T >` (impl DiagStorage < T >)

### `pub fn len(&self) -> usize` (impl DiagStorage < T >)

### `pub fn is_empty(&self) -> bool` (impl DiagStorage < T >)

### `pub fn get(&self, i: usize) -> T` (impl DiagStorage < T >)

### `pub fn set(&mut self, i: usize, val: T)` (impl DiagStorage < T >)

### `pub fn to_dense_vec(&self, dims: & [usize]) -> Vec < T >` (impl DiagStorage < T >)

Convert diagonal storage to a dense vector representation. Creates a dense vector with diagonal elements set and off-diagonal elements as zero.

### `pub fn contract_diag_diag(&self, dims: & [usize], other: & Self, other_dims: & [usize], result_dims: & [usize], make_dense: impl FnOnce (Vec < T >) -> Storage, make_diag: impl FnOnce (Vec < T >) -> Storage) -> Storage` (impl DiagStorage < T >)

Contract this diagonal storage with another diagonal storage of the same type. Returns either a scalar (Dense with one element) or a diagonal storage.

### `pub fn contract_diag_dense(&self, diag_dims: & [usize], axes_diag: & [usize], dense: & DenseStorage < T >, dense_dims: & [usize], axes_dense: & [usize], result_dims: & [usize], make_storage: impl FnOnce (Vec < T >) -> Storage) -> Storage` (impl DiagStorage < T >)

Contract this diagonal storage with a dense storage.

### ` fn contract_diag_dense_impl(diag: & [T], diag_dims: & [usize], axes_diag: & [usize], dense: & [T], dense_dims: & [usize], axes_dense: & [usize], result_dims: & [usize], make_storage: impl FnOnce (Vec < T >) -> Storage) -> Storage`

Generic implementation of Diag × Dense contraction. This function exploits the diagonal structure: for a diagonal tensor, all indices have the same value t (0 <= t < d). When contracting with

### ` fn compute_strides(dims: & [usize]) -> Vec < usize >`

Compute row-major strides for given dimensions.

### ` fn contract_dense_diag_impl(dense: & DenseStorage < T >, dense_dims: & [usize], axes_dense: & [usize], diag: & DiagStorage < T >, diag_dims: & [usize], axes_diag: & [usize], _result_dims: & [usize], make_storage: impl FnOnce (Vec < T >) -> Storage, make_permuted: impl FnOnce (Storage , & [usize] , & [usize]) -> Storage) -> Storage`

Helper for Dense × Diag contraction: compute as Diag × Dense and permute result. This function handles the case where Dense appears first in the contraction. It computes the contraction using `contract_diag_dense_impl` (which assumes Diag first),

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

### `pub fn is_empty(&self) -> bool` (impl Storage)

Check if the storage is empty.

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

### `pub fn try_sub(&self, other: & Storage) -> Result < Storage , String >` (impl Storage)

Try to subtract two storages element-wise. Returns an error if the storages have different types or lengths.

### `pub fn scale(&self, scalar: & crate :: AnyScalar) -> Storage` (impl Storage)

Scale storage by a scalar value. If the scalar is complex but the storage is real, the storage is promoted to complex.

### `pub fn axpby(&self, a: & crate :: AnyScalar, other: & Storage, b: & crate :: AnyScalar) -> Result < Storage , String >` (impl Storage)

Compute linear combination: `a * self + b * other`. Returns an error if the storages have different types or lengths. If any scalar is complex, the result is promoted to complex.

### `pub fn make_mut_storage(arc: & mut Arc < Storage >) -> & mut Storage`

Helper to get a mutable reference to storage, cloning if needed (COW).

### `pub fn mindim(dims: & [usize]) -> usize`

Get the minimum dimension from a slice of dimensions. This is used for DiagTensor where all indices must have the same dimension.

### `pub fn contract_storage(storage_a: & Storage, dims_a: & [usize], axes_a: & [usize], storage_b: & Storage, dims_b: & [usize], axes_b: & [usize], result_dims: & [usize]) -> Storage`

Contract two storage tensors along specified axes. This is an internal helper function that contracts two `Storage` tensors. For Dense tensors, uses mdarray-linalg's contract method.

### ` fn promote_diag_to_c64(diag: & DiagStorage < f64 >) -> DiagStorage < Complex64 >`

Promote Diag<f64> to Diag<Complex64>

### ` fn promote_dense_to_c64(dense: & DenseStorage < f64 >) -> DenseStorage < Complex64 >`

Promote Dense<f64> to Dense<Complex64>

### `pub fn extract_dense_view(storage: & Storage) -> Result < & [Self] , String >` (trait StorageScalar)

Extract a borrowed view of dense storage data (no copy). Returns an error if the storage is not the matching dense type.

### `pub fn extract_dense_cow(storage: & 'a Storage) -> Result < Cow < 'a , [Self] > , String >` (trait StorageScalar default)

Extract dense storage data as `Cow` (borrowed if possible, owned if needed). For dense storage, returns `Cow::Borrowed` (no copy). For other storage types, may need to convert to dense first (copy).

### `pub fn extract_dense(storage: & Storage) -> Result < Vec < Self > , String >` (trait StorageScalar default)

Extract dense storage data as owned `Vec` (always copies). This is a convenience method that calls `extract_dense_cow` and converts to owned.

### `pub fn dense_storage(data: Vec < Self >) -> Arc < Storage >` (trait StorageScalar)

Create `Storage` from owned dense data.

### `pub fn storage_to_dtensor(storage: & Storage, shape: [usize ; 2]) -> Result < DTensor < T , 2 > , String >`

Convert dense storage to a DTensor with rank 2. This function extracts data from dense storage and reshapes it into a `DTensor<T, 2>` with the specified shape `[m, n]`. The data length must match `m * n`.

### ` fn extract_dense_view(storage: & Storage) -> Result < & [Self] , String >` (impl f64)

### ` fn dense_storage(data: Vec < Self >) -> Arc < Storage >` (impl f64)

### ` fn extract_dense_view(storage: & Storage) -> Result < & [Self] , String >` (impl Complex64)

### ` fn dense_storage(data: Vec < Self >) -> Arc < Storage >` (impl Complex64)

### ` fn add(self, rhs: & Storage) -> Self :: Output` (impl & Storage)

### ` fn mul(self, scalar: f64) -> Self :: Output` (impl & Storage)

### ` fn mul(self, scalar: Complex64) -> Self :: Output` (impl & Storage)

### ` fn mul(self, scalar: AnyScalar) -> Self :: Output` (impl & Storage)

### ` fn extract_f64(storage: & Storage) -> Vec < f64 >`

Helper to extract f64 data from storage

### ` fn extract_c64(storage: & Storage) -> Vec < Complex64 >`

Helper to extract Complex64 data from storage

### ` fn test_diag_storage_generic_f64()`

### ` fn test_diag_storage_generic_c64()`

### ` fn test_diag_to_dense_vec_2d()`

### ` fn test_diag_to_dense_vec_3d()`

### ` fn test_contract_diag_dense_2d_all_contracted()`

### ` fn test_contract_diag_dense_2d_one_axis()`

### ` fn test_contract_dense_diag_2d_one_axis()`

### ` fn test_contract_diag_dense_3d()`

### ` fn test_contract_diag_f64_dense_c64()`

### ` fn test_contract_diag_c64_dense_f64()`

### ` fn test_contract_dense_f64_diag_c64()`

### ` fn test_contract_dense_c64_diag_f64()`

### ` fn test_contract_diag_diag_all_contracted()`

### ` fn test_contract_diag_diag_partial()`

