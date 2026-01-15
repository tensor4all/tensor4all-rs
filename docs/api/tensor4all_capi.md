# tensor4all-capi

## src/algorithm.rs

### `pub fn t4a_factorize_algorithm_from_i32(value: i32, out_alg: * mut t4a_factorize_algorithm) -> StatusCode`

Get factorize algorithm from integer value.

### `pub fn t4a_factorize_algorithm_name(alg: t4a_factorize_algorithm) -> * const c_char`

Get the name of a factorize algorithm.

### `pub fn t4a_factorize_algorithm_default() -> t4a_factorize_algorithm`

Get the default factorize algorithm.

### `pub fn t4a_contraction_algorithm_from_i32(value: i32, out_alg: * mut t4a_contraction_algorithm) -> StatusCode`

Get contraction algorithm from integer value.

### `pub fn t4a_contraction_algorithm_name(alg: t4a_contraction_algorithm) -> * const c_char`

Get the name of a contraction algorithm.

### `pub fn t4a_contraction_algorithm_default() -> t4a_contraction_algorithm`

Get the default contraction algorithm.

### `pub fn t4a_compression_algorithm_from_i32(value: i32, out_alg: * mut t4a_compression_algorithm) -> StatusCode`

Get compression algorithm from integer value.

### `pub fn t4a_compression_algorithm_name(alg: t4a_compression_algorithm) -> * const c_char`

Get the name of a compression algorithm.

### `pub fn t4a_compression_algorithm_default() -> t4a_compression_algorithm`

Get the default compression algorithm.

### `pub fn t4a_factorize_algorithm_from_name(name: * const c_char, out_alg: * mut t4a_factorize_algorithm) -> StatusCode`

Get factorize algorithm from string name.

### `pub fn t4a_contraction_algorithm_from_name(name: * const c_char, out_alg: * mut t4a_contraction_algorithm) -> StatusCode`

Get contraction algorithm from string name.

### `pub fn t4a_compression_algorithm_from_name(name: * const c_char, out_alg: * mut t4a_compression_algorithm) -> StatusCode`

Get compression algorithm from string name.

## src/index.rs

### `pub fn t4a_index_new(dim: usize) -> * mut t4a_index`

Create a new index with the given dimension

### `pub fn t4a_index_new_with_tags(dim: usize, tags_csv: * const c_char) -> * mut t4a_index`

Create a new index with the given dimension and tags (comma-separated)

### `pub fn t4a_index_new_with_id(dim: usize, id_hi: u64, id_lo: u64, tags_csv: * const c_char) -> * mut t4a_index`

Create a new index with the given dimension, id, and tags

### `pub fn t4a_index_dim(ptr: * const t4a_index, out_dim: * mut usize) -> StatusCode`

Get the dimension of an index

### `pub fn t4a_index_id_u128(ptr: * const t4a_index, out_hi: * mut u64, out_lo: * mut u64) -> StatusCode`

Get the 128-bit ID of an index as two 64-bit values

### `pub fn t4a_index_get_tags(ptr: * const t4a_index, buf: * mut u8, buf_len: usize, out_len: * mut usize) -> StatusCode`

Get the tags of an index as a comma-separated UTF-8 string If `buf` is null, only writes the required buffer length to `out_len`. Otherwise, writes the tags to `buf` (with null terminator) if it fits.

### `pub fn t4a_index_add_tag(ptr: * mut t4a_index, tag: * const c_char) -> StatusCode`

Add a single tag to an index

### `pub fn t4a_index_set_tags_csv(ptr: * mut t4a_index, tags_csv: * const c_char) -> StatusCode`

Set all tags from a comma-separated string (replaces existing tags)

### `pub fn t4a_index_has_tag(ptr: * const t4a_index, tag: * const c_char) -> i32`

Check if an index has a specific tag

### ` fn test_index_new()`

### ` fn test_index_new_zero_dim()`

### ` fn test_index_with_tags()`

### ` fn test_index_get_tags()`

### ` fn test_index_id()`

### ` fn test_index_clone()`

### ` fn test_index_new_with_id()`

## src/simplett.rs

### `pub(crate) fn new(tt: TensorTrain < f64 >) -> Self` (impl t4a_simplett_f64)

### `pub(crate) fn inner(&self) -> & TensorTrain < f64 >` (impl t4a_simplett_f64)

### `pub(crate) fn inner_mut(&mut self) -> & mut TensorTrain < f64 >` (impl t4a_simplett_f64)

### `pub(crate) fn into_inner(self) -> TensorTrain < f64 >` (impl t4a_simplett_f64)

### ` fn drop(&mut self)` (impl t4a_simplett_f64)

### `pub fn t4a_simplett_f64_release(ptr: * mut t4a_simplett_f64)`

Release a SimpleTT tensor train handle.

### `pub fn t4a_simplett_f64_clone(ptr: * const t4a_simplett_f64) -> * mut t4a_simplett_f64`

Clone a SimpleTT tensor train.

### `pub fn t4a_simplett_f64_constant(site_dims: * const libc :: size_t, n_sites: libc :: size_t, value: libc :: c_double) -> * mut t4a_simplett_f64`

Create a constant tensor train.

### `pub fn t4a_simplett_f64_zeros(site_dims: * const libc :: size_t, n_sites: libc :: size_t) -> * mut t4a_simplett_f64`

Create a zero tensor train.

### `pub fn t4a_simplett_f64_len(ptr: * const t4a_simplett_f64, out_len: * mut libc :: size_t) -> StatusCode`

Get the number of sites.

### `pub fn t4a_simplett_f64_site_dims(ptr: * const t4a_simplett_f64, out_dims: * mut libc :: size_t, buf_len: libc :: size_t) -> StatusCode`

Get the site dimensions.

### `pub fn t4a_simplett_f64_link_dims(ptr: * const t4a_simplett_f64, out_dims: * mut libc :: size_t, buf_len: libc :: size_t) -> StatusCode`

Get the link (bond) dimensions.

### `pub fn t4a_simplett_f64_rank(ptr: * const t4a_simplett_f64, out_rank: * mut libc :: size_t) -> StatusCode`

Get the maximum bond dimension (rank).

### `pub fn t4a_simplett_f64_evaluate(ptr: * const t4a_simplett_f64, indices: * const libc :: size_t, n_indices: libc :: size_t, out_value: * mut libc :: c_double) -> StatusCode`

Evaluate the tensor train at a given multi-index.

### `pub fn t4a_simplett_f64_sum(ptr: * const t4a_simplett_f64, out_value: * mut libc :: c_double) -> StatusCode`

Compute the sum over all indices.

### `pub fn t4a_simplett_f64_norm(ptr: * const t4a_simplett_f64, out_value: * mut libc :: c_double) -> StatusCode`

Compute the Frobenius norm.

### `pub fn t4a_simplett_f64_site_tensor(ptr: * const t4a_simplett_f64, site: libc :: size_t, out_data: * mut libc :: c_double, buf_len: libc :: size_t, out_left_dim: * mut libc :: size_t, out_site_dim: * mut libc :: size_t, out_right_dim: * mut libc :: size_t) -> StatusCode`

Get site tensor data at a specific site. The tensor has shape (left_dim, site_dim, right_dim) in row-major order.

### ` fn test_simplett_constant()`

### ` fn test_simplett_evaluate()`

## src/tensor.rs

### `pub fn t4a_tensor_get_rank(ptr: * const t4a_tensor, out_rank: * mut libc :: size_t) -> StatusCode`

Get the rank (number of indices) of a tensor.

### `pub fn t4a_tensor_get_dims(ptr: * const t4a_tensor, out_dims: * mut libc :: size_t, buf_len: libc :: size_t) -> StatusCode`

Get the dimensions of a tensor.

### `pub fn t4a_tensor_get_indices(ptr: * const t4a_tensor, out_indices: * mut * mut t4a_index, buf_len: libc :: size_t) -> StatusCode`

Get the indices of a tensor as cloned t4a_index handles.

### `pub fn t4a_tensor_get_storage_kind(ptr: * const t4a_tensor, out_kind: * mut t4a_storage_kind) -> StatusCode`

Get the storage kind of a tensor.

### `pub fn t4a_tensor_get_data_f64(ptr: * const t4a_tensor, buf: * mut libc :: c_double, buf_len: libc :: size_t, out_len: * mut libc :: size_t) -> StatusCode`

Get the dense f64 data from a tensor in row-major order.

### `pub fn t4a_tensor_get_data_c64(ptr: * const t4a_tensor, buf_re: * mut libc :: c_double, buf_im: * mut libc :: c_double, buf_len: libc :: size_t, out_len: * mut libc :: size_t) -> StatusCode`

Get the dense complex64 data from a tensor in row-major order.

### `pub fn t4a_tensor_new_dense_f64(rank: libc :: size_t, index_ptrs: * const * const t4a_index, dims: * const libc :: size_t, data: * const libc :: c_double, data_len: libc :: size_t) -> * mut t4a_tensor`

Create a new dense f64 tensor from indices and data.

### `pub fn t4a_tensor_new_dense_c64(rank: libc :: size_t, index_ptrs: * const * const t4a_index, dims: * const libc :: size_t, data_re: * const libc :: c_double, data_im: * const libc :: c_double, data_len: libc :: size_t) -> * mut t4a_tensor`

Create a new dense complex64 tensor from indices and data.

### ` fn test_tensor_lifecycle()`

### ` fn test_tensor_accessors()`

### ` fn test_tensor_c64()`

## src/tensorci.rs

### `pub(crate) fn new(tci: TensorCI2 < f64 >) -> Self` (impl t4a_tci2_f64)

### `pub(crate) fn inner(&self) -> & TensorCI2 < f64 >` (impl t4a_tci2_f64)

### `pub(crate) fn inner_mut(&mut self) -> & mut TensorCI2 < f64 >` (impl t4a_tci2_f64)

### ` fn drop(&mut self)` (impl t4a_tci2_f64)

### `pub fn t4a_tci2_f64_release(ptr: * mut t4a_tci2_f64)`

Release a TensorCI2 handle.

### `pub fn t4a_tci2_f64_new(local_dims: * const libc :: size_t, n_sites: libc :: size_t) -> * mut t4a_tci2_f64`

Create a new TensorCI2 object.

### `pub fn t4a_tci2_f64_len(ptr: * const t4a_tci2_f64, out_len: * mut libc :: size_t) -> StatusCode`

Get the number of sites.

### `pub fn t4a_tci2_f64_rank(ptr: * const t4a_tci2_f64, out_rank: * mut libc :: size_t) -> StatusCode`

Get the current rank (maximum bond dimension).

### `pub fn t4a_tci2_f64_link_dims(ptr: * const t4a_tci2_f64, out_dims: * mut libc :: size_t, buf_len: libc :: size_t) -> StatusCode`

Get the link (bond) dimensions.

### `pub fn t4a_tci2_f64_max_sample_value(ptr: * const t4a_tci2_f64, out_value: * mut libc :: c_double) -> StatusCode`

Get the maximum sample value encountered.

### `pub fn t4a_tci2_f64_max_bond_error(ptr: * const t4a_tci2_f64, out_value: * mut libc :: c_double) -> StatusCode`

Get the maximum bond error from the last sweep.

### `pub fn t4a_tci2_f64_add_global_pivots(ptr: * mut t4a_tci2_f64, pivots: * const libc :: size_t, n_pivots: libc :: size_t, n_sites: libc :: size_t) -> StatusCode`

Add global pivots to the TCI.

### `pub fn t4a_tci2_f64_sweep(ptr: * mut t4a_tci2_f64, eval_fn: EvalCallback, user_data: * mut c_void, abstol: libc :: c_double, max_bonddim: libc :: size_t, n_iters: libc :: size_t, out_error: * mut libc :: c_double) -> StatusCode`

Perform a 2-site sweep. This is the main optimization step. The callback function is called to evaluate the target function at various indices.

### `pub fn t4a_tci2_f64_to_tensor_train(ptr: * const t4a_tci2_f64) -> * mut t4a_simplett_f64`

Convert the TCI to a TensorTrain.

### `pub fn t4a_crossinterpolate2_f64(local_dims: * const libc :: size_t, n_sites: libc :: size_t, initial_pivots: * const libc :: size_t, n_initial_pivots: libc :: size_t, eval_fn: EvalCallback, user_data: * mut c_void, tolerance: libc :: c_double, max_bonddim: libc :: size_t, max_iter: libc :: size_t, out_tci: * mut * mut t4a_tci2_f64, out_final_error: * mut libc :: c_double) -> StatusCode`

Perform cross interpolation of a function. This is the main entry point for TCI. It creates a TCI object, performs optimization sweeps, and returns the result.

### ` fn sum_callback(indices: * const i64, n_indices: libc :: size_t, result: * mut f64, _user_data: * mut c_void) -> i32`

### ` fn test_tci2_new()`

### ` fn test_crossinterpolate2_constant()`

## src/tensortrain.rs

### `pub fn t4a_tt_new(tensors: * const * const t4a_tensor, num_tensors: libc :: size_t) -> * mut t4a_tensortrain`

Create a tensor train from an array of tensors.

### `pub fn t4a_tt_new_empty() -> * mut t4a_tensortrain`

Create an empty tensor train.

### `pub fn t4a_tt_len(ptr: * const t4a_tensortrain, out_len: * mut libc :: size_t) -> StatusCode`

Get the number of sites in the tensor train.

### `pub fn t4a_tt_is_empty(ptr: * const t4a_tensortrain) -> libc :: c_int`

Check if the tensor train is empty.

### `pub fn t4a_tt_tensor(ptr: * const t4a_tensortrain, site: libc :: size_t) -> * mut t4a_tensor`

Get the tensor at a specific site.

### `pub fn t4a_tt_set_tensor(ptr: * mut t4a_tensortrain, site: libc :: size_t, tensor: * const t4a_tensor) -> StatusCode`

Set the tensor at a specific site. This replaces the tensor at the given site and invalidates orthogonality.

### `pub fn t4a_tt_bond_dims(ptr: * const t4a_tensortrain, out_dims: * mut libc :: size_t, buf_len: libc :: size_t) -> StatusCode`

Get the bond dimensions of the tensor train.

### `pub fn t4a_tt_maxbonddim(ptr: * const t4a_tensortrain, out_max: * mut libc :: size_t) -> StatusCode`

Get the maximum bond dimension of the tensor train.

### `pub fn t4a_tt_linkind(ptr: * const t4a_tensortrain, site: libc :: size_t) -> * mut t4a_index`

Get the link index between two adjacent sites.

### `pub fn t4a_tt_isortho(ptr: * const t4a_tensortrain) -> libc :: c_int`

Check if the tensor train has a single orthogonality center.

### `pub fn t4a_tt_orthocenter(ptr: * const t4a_tensortrain, out_center: * mut libc :: size_t) -> StatusCode`

Get the orthogonality center (0-indexed).

### `pub fn t4a_tt_llim(ptr: * const t4a_tensortrain, out_llim: * mut libc :: c_int) -> StatusCode`

Get the left orthogonality limit. Sites 0..llim are guaranteed to be left-orthogonal. Returns -1 if no sites are left-orthogonal.

### `pub fn t4a_tt_rlim(ptr: * const t4a_tensortrain, out_rlim: * mut libc :: c_int) -> StatusCode`

Get the right orthogonality limit. Sites rlim..len are guaranteed to be right-orthogonal. Returns len+1 if no sites are right-orthogonal.

### `pub fn t4a_tt_canonical_form(ptr: * const t4a_tensortrain, out_form: * mut t4a_canonical_form) -> StatusCode`

Get the canonical form used for the tensor train.

### `pub fn t4a_tt_orthogonalize(ptr: * mut t4a_tensortrain, site: libc :: size_t) -> StatusCode`

Orthogonalize the tensor train to have orthogonality center at the given site. Uses QR decomposition (Unitary canonical form) by default.

### `pub fn t4a_tt_orthogonalize_with(ptr: * mut t4a_tensortrain, site: libc :: size_t, form: t4a_canonical_form) -> StatusCode`

Orthogonalize the tensor train with a specific canonical form.

### `pub fn t4a_tt_truncate(ptr: * mut t4a_tensortrain, rtol: libc :: c_double, max_rank: libc :: size_t) -> StatusCode`

Truncate the tensor train bond dimensions.

### `pub fn t4a_tt_norm(ptr: * const t4a_tensortrain, out_norm: * mut libc :: c_double) -> StatusCode`

Compute the norm of the tensor train.

### `pub fn t4a_tt_inner(ptr1: * const t4a_tensortrain, ptr2: * const t4a_tensortrain, out_re: * mut libc :: c_double, out_im: * mut libc :: c_double) -> StatusCode`

Compute the inner product of two tensor trains. Computes <self | other> = sum over all indices of conj(self) * other.

### `pub fn t4a_tt_contract(ptr1: * const t4a_tensortrain, ptr2: * const t4a_tensortrain, method: crate :: types :: t4a_contract_method, max_rank: libc :: size_t, rtol: libc :: c_double, nsweeps: libc :: size_t) -> * mut t4a_tensortrain`

Contract two tensor trains. Both tensor trains must have the same site indices.

### ` fn test_tt_lifecycle()`

### ` fn test_tt_from_tensors()`

## src/types.rs

### `pub(crate) fn new(index: InternalIndex) -> Self` (impl t4a_index)

Create a new t4a_index from an InternalIndex

### `pub(crate) fn inner(&self) -> & InternalIndex` (impl t4a_index)

Get a reference to the inner InternalIndex

### `pub(crate) fn inner_mut(&mut self) -> & mut InternalIndex` (impl t4a_index)

Get a mutable reference to the inner InternalIndex

### ` fn clone(&self) -> Self` (impl t4a_index)

### ` fn drop(&mut self)` (impl t4a_index)

### `pub(crate) fn from_storage(storage: & Storage) -> Self` (impl t4a_storage_kind)

Convert from Rust Storage to t4a_storage_kind

### `pub(crate) fn new(tensor: InternalTensor) -> Self` (impl t4a_tensor)

Create a new t4a_tensor from an InternalTensor

### `pub(crate) fn inner(&self) -> & InternalTensor` (impl t4a_tensor)

Get a reference to the inner InternalTensor

### `pub(crate) fn inner_mut(&mut self) -> & mut InternalTensor` (impl t4a_tensor)

Get a mutable reference to the inner InternalTensor

### ` fn clone(&self) -> Self` (impl t4a_tensor)

### ` fn drop(&mut self)` (impl t4a_tensor)

### `pub(crate) fn new(tt: InternalTensorTrain) -> Self` (impl t4a_tensortrain)

Create a new t4a_tensortrain from an InternalTensorTrain

### `pub(crate) fn inner(&self) -> & InternalTensorTrain` (impl t4a_tensortrain)

Get a reference to the inner InternalTensorTrain

### `pub(crate) fn inner_mut(&mut self) -> & mut InternalTensorTrain` (impl t4a_tensortrain)

Get a mutable reference to the inner InternalTensorTrain

### ` fn clone(&self) -> Self` (impl t4a_tensortrain)

### ` fn drop(&mut self)` (impl t4a_tensortrain)

### ` fn from(form: tensor4all_itensorlike :: CanonicalForm) -> Self` (impl t4a_canonical_form)

### ` fn from(form: t4a_canonical_form) -> Self` (impl tensor4all_itensorlike :: CanonicalForm)

### ` fn from(alg: tensor4all_core :: FactorizeAlg) -> Self` (impl t4a_factorize_algorithm)

### ` fn from(alg: t4a_factorize_algorithm) -> Self` (impl tensor4all_core :: FactorizeAlg)

### ` fn from(method: tensor4all_itensorlike :: ContractMethod) -> Self` (impl t4a_contract_method)

### ` fn from(method: t4a_contract_method) -> Self` (impl tensor4all_itensorlike :: ContractMethod)

