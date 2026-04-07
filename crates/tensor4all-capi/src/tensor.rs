//! C API for TensorDynLen
//!
//! Provides functions for creating, manipulating, and accessing tensors.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;

use num_complex::Complex64;

use tensor4all_core::TensorLike;

use crate::types::{t4a_index, t4a_storage_kind, t4a_tensor, InternalIndex, InternalTensor};
use crate::{
    StatusCode, T4A_BUFFER_TOO_SMALL, T4A_INVALID_ARGUMENT, T4A_NULL_POINTER, T4A_SUCCESS,
};

// Generate lifecycle functions: t4a_tensor_release, t4a_tensor_clone, t4a_tensor_is_assigned
impl_opaque_type_common!(tensor);

fn read_indices_from_ptrs(
    rank: libc::size_t,
    index_ptrs: *const *const t4a_index,
) -> Option<Vec<InternalIndex>> {
    if index_ptrs.is_null() {
        return None;
    }
    let mut indices: Vec<InternalIndex> = Vec::with_capacity(rank);
    for i in 0..rank {
        let idx_ptr = unsafe { *index_ptrs.add(i) };
        if idx_ptr.is_null() {
            return None;
        }
        let idx = unsafe { &*idx_ptr };
        indices.push(idx.inner().clone());
    }
    Some(indices)
}

fn read_dims_from_ptr(rank: libc::size_t, dims: *const libc::size_t) -> Option<Vec<usize>> {
    if dims.is_null() {
        return None;
    }
    Some((0..rank).map(|i| unsafe { *dims.add(i) }).collect())
}

fn indices_dims(indices: &[InternalIndex]) -> Vec<usize> {
    indices.iter().map(|idx| idx.size()).collect()
}

/// Get the rank (number of indices) of a tensor.
///
/// # Safety
/// - `ptr` must be a valid pointer to a t4a_tensor
/// - `out_rank` must be a valid pointer to write the rank
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_get_rank(
    ptr: *const t4a_tensor,
    out_rank: *mut libc::size_t,
) -> StatusCode {
    if ptr.is_null() || out_rank.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tensor = unsafe { &*ptr };
        let rank = tensor.inner().indices.len();
        unsafe { *out_rank = rank };
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

/// Get the dimensions of a tensor.
///
/// # Arguments
/// - `ptr`: Tensor handle
/// - `out_dims`: Buffer to write dimensions (must have length >= rank)
/// - `buf_len`: Length of the buffer
///
/// # Safety
/// - `ptr` must be a valid pointer to a t4a_tensor
/// - `out_dims` must be a valid pointer to a buffer of at least `rank` elements
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_get_dims(
    ptr: *const t4a_tensor,
    out_dims: *mut libc::size_t,
    buf_len: libc::size_t,
) -> StatusCode {
    if ptr.is_null() || out_dims.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tensor = unsafe { &*ptr };
        let dims = tensor.inner().dims();

        if buf_len < dims.len() {
            return T4A_BUFFER_TOO_SMALL;
        }

        unsafe {
            for (i, dim) in dims.iter().copied().enumerate() {
                *out_dims.add(i) = dim;
            }
        }
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

/// Get the indices of a tensor as cloned t4a_index handles.
///
/// # Arguments
/// - `ptr`: Tensor handle
/// - `out_indices`: Buffer to write cloned index handles (must have length >= rank)
/// - `buf_len`: Length of the buffer
///
/// # Safety
/// - `ptr` must be a valid pointer to a t4a_tensor
/// - `out_indices` must be a valid pointer to a buffer of at least `rank` elements
/// - Caller is responsible for releasing the returned index handles
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_get_indices(
    ptr: *const t4a_tensor,
    out_indices: *mut *mut t4a_index,
    buf_len: libc::size_t,
) -> StatusCode {
    if ptr.is_null() || out_indices.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tensor = unsafe { &*ptr };
        let indices = &tensor.inner().indices;

        if buf_len < indices.len() {
            return T4A_BUFFER_TOO_SMALL;
        }

        unsafe {
            for (i, idx) in indices.iter().enumerate() {
                let cloned_idx = Box::new(t4a_index::new(idx.clone()));
                *out_indices.add(i) = Box::into_raw(cloned_idx);
            }
        }
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

/// Get the storage kind of a tensor.
///
/// # Safety
/// - `ptr` must be a valid pointer to a t4a_tensor
/// - `out_kind` must be a valid pointer to write the storage kind
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_get_storage_kind(
    ptr: *const t4a_tensor,
    out_kind: *mut t4a_storage_kind,
) -> StatusCode {
    if ptr.is_null() || out_kind.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tensor = unsafe { &*ptr };
        let kind = t4a_storage_kind::from_storage(tensor.inner().storage().as_ref());
        unsafe { *out_kind = kind };
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

/// Get the dense f64 data from a tensor in column-major order.
///
/// # Arguments
/// - `ptr`: Tensor handle
/// - `buf`: Buffer to write data (if NULL, only out_len is written)
/// - `buf_len`: Length of the buffer
/// - `out_len`: Output: required buffer length
///
/// # Returns
/// - T4A_SUCCESS on success
/// - T4A_BUFFER_TOO_SMALL if buffer is too small (out_len is still written)
/// - T4A_INVALID_ARGUMENT if storage is not DenseF64
///
/// # Safety
/// - `ptr` must be a valid pointer to a t4a_tensor
/// - `buf` can be NULL (to query required length)
/// - `out_len` must be a valid pointer
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_get_data_f64(
    ptr: *const t4a_tensor,
    buf: *mut libc::c_double,
    buf_len: libc::size_t,
    out_len: *mut libc::size_t,
) -> StatusCode {
    if ptr.is_null() || out_len.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tensor = unsafe { &*ptr };

        let data = match tensor.inner().as_slice_f64() {
            Ok(s) => s,
            Err(e) => return crate::err_status(e, T4A_INVALID_ARGUMENT),
        };

        unsafe { *out_len = data.len() };

        if buf.is_null() {
            return T4A_SUCCESS;
        }

        if buf_len < data.len() {
            return T4A_BUFFER_TOO_SMALL;
        }

        unsafe {
            ptr::copy_nonoverlapping(data.as_ptr(), buf, data.len());
        }
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

/// Get the dense complex64 data from a tensor in column-major order.
///
/// # Arguments
/// - `ptr`: Tensor handle
/// - `buf`: Interleaved buffer `[re0, im0, re1, im1, ...]`
/// - `buf_len`: Buffer length in complex elements
/// - `out_len`: Output: required buffer length in complex elements
///
/// # Returns
/// - T4A_SUCCESS on success
/// - T4A_BUFFER_TOO_SMALL if buffer is too small (out_len is still written)
/// - T4A_INVALID_ARGUMENT if storage is not DenseC64
///
/// # Safety
/// - `ptr` must be a valid pointer to a t4a_tensor
/// - `buf` can be NULL (to query required length)
/// - `out_len` must be a valid pointer
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_get_data_c64(
    ptr: *const t4a_tensor,
    buf: *mut libc::c_double,
    buf_len: libc::size_t,
    out_len: *mut libc::size_t,
) -> StatusCode {
    if ptr.is_null() || out_len.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tensor = unsafe { &*ptr };

        let data = match tensor.inner().as_slice_c64() {
            Ok(s) => s,
            Err(e) => return crate::err_status(e, T4A_INVALID_ARGUMENT),
        };

        unsafe { *out_len = data.len() };

        if buf.is_null() {
            return T4A_SUCCESS;
        }

        if buf_len < data.len() {
            return T4A_BUFFER_TOO_SMALL;
        }

        unsafe {
            for (i, z) in data.iter().enumerate() {
                *buf.add(2 * i) = z.re;
                *buf.add(2 * i + 1) = z.im;
            }
        }
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

/// Contract two tensors by matching common indices.
///
/// Finds indices with the same ID and contracts along them.
/// The result tensor has the non-contracted indices from both input tensors.
///
/// # Arguments
/// * `a` - First tensor pointer
/// * `b` - Second tensor pointer
/// * `out` - Output pointer for the result tensor
///
/// # Returns
/// Status code (`T4A_SUCCESS` or error code)
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_contract(
    a: *const t4a_tensor,
    b: *const t4a_tensor,
    out: *mut *mut t4a_tensor,
) -> StatusCode {
    if a.is_null() || b.is_null() || out.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tensor_a = unsafe { &*a };
        let tensor_b = unsafe { &*b };
        let contracted = tensor_a.inner().contract(tensor_b.inner());
        unsafe { *out = Box::into_raw(Box::new(t4a_tensor::new(contracted))) };
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

/// Create a new dense f64 tensor from indices and data.
///
/// # Arguments
/// - `rank`: Number of indices
/// - `index_ptrs`: Array of t4a_index pointers (length = rank)
/// - `dims`: Array of dimensions (length = rank)
/// - `data`: Dense data in column-major order (length = product of dims)
/// - `data_len`: Length of data array
///
/// # Returns
/// - Pointer to new t4a_tensor on success
/// - NULL on error
///
/// # Safety
/// - All pointers must be valid
/// - Caller owns the returned tensor and must call t4a_tensor_release
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_new_dense_f64(
    rank: libc::size_t,
    index_ptrs: *const *const t4a_index,
    dims: *const libc::size_t,
    data: *const libc::c_double,
    data_len: libc::size_t,
) -> *mut t4a_tensor {
    if index_ptrs.is_null() || dims.is_null() || data.is_null() {
        return ptr::null_mut();
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let Some(indices) = read_indices_from_ptrs(rank, index_ptrs) else {
            return ptr::null_mut();
        };

        let Some(input_dims) = read_dims_from_ptr(rank, dims) else {
            return ptr::null_mut();
        };
        let expected_dims = indices_dims(&indices);
        if input_dims != expected_dims {
            return ptr::null_mut();
        }

        // Validate data length
        let expected_len: usize = expected_dims.iter().product();
        if data_len != expected_len {
            return ptr::null_mut();
        }

        // Copy data
        let data_vec: Vec<f64> = unsafe { std::slice::from_raw_parts(data, data_len).to_vec() };

        let Ok(tensor) = InternalTensor::from_dense(indices, data_vec) else {
            return ptr::null_mut();
        };

        Box::into_raw(Box::new(t4a_tensor::new(tensor)))
    }));

    crate::unwrap_catch_ptr(result)
}

/// Create a new dense complex64 tensor from indices and data.
///
/// # Arguments
/// - `rank`: Number of indices
/// - `index_ptrs`: Array of t4a_index pointers (length = rank)
/// - `dims`: Array of dimensions (length = rank)
/// - `data`: Interleaved dense data in column-major order
///   `[re0, im0, re1, im1, ...]` with length `2 * product(dims)`
/// - `data_len`: Number of complex elements in `data`
///
/// # Returns
/// - Pointer to new t4a_tensor on success
/// - NULL on error
///
/// # Safety
/// - All pointers must be valid
/// - Caller owns the returned tensor and must call t4a_tensor_release
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_new_dense_c64(
    rank: libc::size_t,
    index_ptrs: *const *const t4a_index,
    dims: *const libc::size_t,
    data: *const libc::c_double,
    data_len: libc::size_t,
) -> *mut t4a_tensor {
    if index_ptrs.is_null() || dims.is_null() || data.is_null() {
        return ptr::null_mut();
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let Some(indices) = read_indices_from_ptrs(rank, index_ptrs) else {
            return ptr::null_mut();
        };

        let Some(input_dims) = read_dims_from_ptr(rank, dims) else {
            return ptr::null_mut();
        };
        let expected_dims = indices_dims(&indices);
        if input_dims != expected_dims {
            return ptr::null_mut();
        }

        // Validate data length
        let expected_len: usize = expected_dims.iter().product();
        if data_len != expected_len {
            return ptr::null_mut();
        }

        let data_vec: Vec<Complex64> = (0..data_len)
            .map(|i| unsafe { Complex64::new(*data.add(2 * i), *data.add(2 * i + 1)) })
            .collect();

        let Ok(tensor) = InternalTensor::from_dense(indices, data_vec) else {
            return ptr::null_mut();
        };

        Box::into_raw(Box::new(t4a_tensor::new(tensor)))
    }));

    crate::unwrap_catch_ptr(result)
}

/// Create a one-hot tensor with value 1.0 at the specified index positions.
///
/// # Arguments
/// - `rank`: Number of indices
/// - `index_ptrs`: Array of t4a_index pointers (length = rank)
/// - `vals`: Array of 0-indexed positions (length = vals_len)
/// - `vals_len`: Length of vals array (must equal rank)
///
/// # Returns
/// - Pointer to new t4a_tensor on success
/// - NULL on error
///
/// # Safety
/// - All pointers must be valid
/// - Caller owns the returned tensor and must call t4a_tensor_release
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_onehot(
    rank: libc::size_t,
    index_ptrs: *const *const t4a_index,
    vals: *const libc::size_t,
    vals_len: libc::size_t,
) -> *mut t4a_tensor {
    if rank > 0 && (index_ptrs.is_null() || vals.is_null()) {
        return ptr::null_mut();
    }
    if vals_len != rank {
        return ptr::null_mut();
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        // Extract indices
        let mut index_vals: Vec<(InternalIndex, usize)> = Vec::with_capacity(rank);
        for i in 0..rank {
            let idx_ptr = unsafe { *index_ptrs.add(i) };
            if idx_ptr.is_null() {
                return ptr::null_mut();
            }
            let idx = unsafe { &*idx_ptr };
            let val = unsafe { *vals.add(i) };
            index_vals.push((idx.inner().clone(), val));
        }

        match InternalTensor::onehot(&index_vals) {
            Ok(tensor) => Box::into_raw(Box::new(t4a_tensor::new(tensor))),
            Err(e) => crate::err_null(e),
        }
    }));

    crate::unwrap_catch_ptr(result)
}

#[cfg(test)]
mod tests;
