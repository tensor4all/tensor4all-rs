//! C API for TensorDynLen
//!
//! Provides functions for creating, manipulating, and accessing tensors.

use std::panic::catch_unwind;
use std::ptr;

use num_complex::Complex64;

use crate::types::{t4a_index, t4a_storage_kind, t4a_tensor, InternalIndex, InternalTensor};
use crate::{
    StatusCode, T4A_BUFFER_TOO_SMALL, T4A_INTERNAL_ERROR, T4A_INVALID_ARGUMENT, T4A_NULL_POINTER,
    T4A_SUCCESS,
};

// Generate lifecycle functions: t4a_tensor_release, t4a_tensor_clone, t4a_tensor_is_assigned
impl_opaque_type_common!(tensor);

/// Get the rank (number of indices) of a tensor.
///
/// # Safety
/// - `ptr` must be a valid pointer to a t4a_tensor
/// - `out_rank` must be a valid pointer to write the rank
#[no_mangle]
pub extern "C" fn t4a_tensor_get_rank(
    ptr: *const t4a_tensor,
    out_rank: *mut libc::size_t,
) -> StatusCode {
    if ptr.is_null() || out_rank.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(|| {
        let tensor = unsafe { &*ptr };
        let rank = tensor.inner().indices.len();
        unsafe { *out_rank = rank };
        T4A_SUCCESS
    });

    result.unwrap_or(T4A_INTERNAL_ERROR)
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
#[no_mangle]
pub extern "C" fn t4a_tensor_get_dims(
    ptr: *const t4a_tensor,
    out_dims: *mut libc::size_t,
    buf_len: libc::size_t,
) -> StatusCode {
    if ptr.is_null() || out_dims.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(|| {
        let tensor = unsafe { &*ptr };
        let dims = &tensor.inner().dims;

        if buf_len < dims.len() {
            return T4A_BUFFER_TOO_SMALL;
        }

        unsafe {
            for (i, &dim) in dims.iter().enumerate() {
                *out_dims.add(i) = dim;
            }
        }
        T4A_SUCCESS
    });

    result.unwrap_or(T4A_INTERNAL_ERROR)
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
#[no_mangle]
pub extern "C" fn t4a_tensor_get_indices(
    ptr: *const t4a_tensor,
    out_indices: *mut *mut t4a_index,
    buf_len: libc::size_t,
) -> StatusCode {
    if ptr.is_null() || out_indices.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(|| {
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
    });

    result.unwrap_or(T4A_INTERNAL_ERROR)
}

/// Get the storage kind of a tensor.
///
/// # Safety
/// - `ptr` must be a valid pointer to a t4a_tensor
/// - `out_kind` must be a valid pointer to write the storage kind
#[no_mangle]
pub extern "C" fn t4a_tensor_get_storage_kind(
    ptr: *const t4a_tensor,
    out_kind: *mut t4a_storage_kind,
) -> StatusCode {
    if ptr.is_null() || out_kind.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(|| {
        let tensor = unsafe { &*ptr };
        let kind = t4a_storage_kind::from_storage(tensor.inner().storage().as_ref());
        unsafe { *out_kind = kind };
        T4A_SUCCESS
    });

    result.unwrap_or(T4A_INTERNAL_ERROR)
}

/// Get the dense f64 data from a tensor in row-major order.
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
#[no_mangle]
pub extern "C" fn t4a_tensor_get_data_f64(
    ptr: *const t4a_tensor,
    buf: *mut libc::c_double,
    buf_len: libc::size_t,
    out_len: *mut libc::size_t,
) -> StatusCode {
    if ptr.is_null() || out_len.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(|| {
        let tensor = unsafe { &*ptr };

        let data = match tensor.inner().as_slice_f64() {
            Ok(s) => s,
            Err(_) => return T4A_INVALID_ARGUMENT,
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
    });

    result.unwrap_or(T4A_INTERNAL_ERROR)
}

/// Get the dense complex64 data from a tensor in row-major order.
///
/// # Arguments
/// - `ptr`: Tensor handle
/// - `buf_re`: Buffer to write real parts (if NULL, only out_len is written)
/// - `buf_im`: Buffer to write imaginary parts (if NULL, only out_len is written)
/// - `buf_len`: Length of the buffers
/// - `out_len`: Output: required buffer length
///
/// # Returns
/// - T4A_SUCCESS on success
/// - T4A_BUFFER_TOO_SMALL if buffer is too small (out_len is still written)
/// - T4A_INVALID_ARGUMENT if storage is not DenseC64
///
/// # Safety
/// - `ptr` must be a valid pointer to a t4a_tensor
/// - `buf_re` and `buf_im` can be NULL (to query required length)
/// - `out_len` must be a valid pointer
#[no_mangle]
pub extern "C" fn t4a_tensor_get_data_c64(
    ptr: *const t4a_tensor,
    buf_re: *mut libc::c_double,
    buf_im: *mut libc::c_double,
    buf_len: libc::size_t,
    out_len: *mut libc::size_t,
) -> StatusCode {
    if ptr.is_null() || out_len.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(|| {
        let tensor = unsafe { &*ptr };

        let data = match tensor.inner().as_slice_c64() {
            Ok(s) => s,
            Err(_) => return T4A_INVALID_ARGUMENT,
        };

        unsafe { *out_len = data.len() };

        if buf_re.is_null() || buf_im.is_null() {
            return T4A_SUCCESS;
        }

        if buf_len < data.len() {
            return T4A_BUFFER_TOO_SMALL;
        }

        unsafe {
            for (i, z) in data.iter().enumerate() {
                *buf_re.add(i) = z.re;
                *buf_im.add(i) = z.im;
            }
        }
        T4A_SUCCESS
    });

    result.unwrap_or(T4A_INTERNAL_ERROR)
}

/// Create a new dense f64 tensor from indices and data.
///
/// # Arguments
/// - `rank`: Number of indices
/// - `index_ptrs`: Array of t4a_index pointers (length = rank)
/// - `dims`: Array of dimensions (length = rank)
/// - `data`: Dense data in row-major order (length = product of dims)
/// - `data_len`: Length of data array
///
/// # Returns
/// - Pointer to new t4a_tensor on success
/// - NULL on error
///
/// # Safety
/// - All pointers must be valid
/// - Caller owns the returned tensor and must call t4a_tensor_release
#[no_mangle]
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

    let result = catch_unwind(|| {
        // Extract indices
        let mut indices: Vec<InternalIndex> = Vec::with_capacity(rank);
        for i in 0..rank {
            let idx_ptr = unsafe { *index_ptrs.add(i) };
            if idx_ptr.is_null() {
                return ptr::null_mut();
            }
            let idx = unsafe { &*idx_ptr };
            indices.push(idx.inner().clone());
        }

        // Extract dimensions
        let dims_vec: Vec<usize> = (0..rank).map(|i| unsafe { *dims.add(i) }).collect();

        // Validate data length
        let expected_len: usize = dims_vec.iter().product();
        if data_len != expected_len {
            return ptr::null_mut();
        }

        // Copy data
        let data_vec: Vec<f64> = unsafe { std::slice::from_raw_parts(data, data_len).to_vec() };

        // Create tensor using high-level API
        let tensor = InternalTensor::from_dense_f64(indices, data_vec);

        Box::into_raw(Box::new(t4a_tensor::new(tensor)))
    });

    result.unwrap_or(ptr::null_mut())
}

/// Create a new dense complex64 tensor from indices and data.
///
/// # Arguments
/// - `rank`: Number of indices
/// - `index_ptrs`: Array of t4a_index pointers (length = rank)
/// - `dims`: Array of dimensions (length = rank)
/// - `data_re`: Real parts of dense data in row-major order (length = product of dims)
/// - `data_im`: Imaginary parts of dense data in row-major order (length = product of dims)
/// - `data_len`: Length of data arrays
///
/// # Returns
/// - Pointer to new t4a_tensor on success
/// - NULL on error
///
/// # Safety
/// - All pointers must be valid
/// - Caller owns the returned tensor and must call t4a_tensor_release
#[no_mangle]
pub extern "C" fn t4a_tensor_new_dense_c64(
    rank: libc::size_t,
    index_ptrs: *const *const t4a_index,
    dims: *const libc::size_t,
    data_re: *const libc::c_double,
    data_im: *const libc::c_double,
    data_len: libc::size_t,
) -> *mut t4a_tensor {
    if index_ptrs.is_null() || dims.is_null() || data_re.is_null() || data_im.is_null() {
        return ptr::null_mut();
    }

    let result = catch_unwind(|| {
        // Extract indices
        let mut indices: Vec<InternalIndex> = Vec::with_capacity(rank);
        for i in 0..rank {
            let idx_ptr = unsafe { *index_ptrs.add(i) };
            if idx_ptr.is_null() {
                return ptr::null_mut();
            }
            let idx = unsafe { &*idx_ptr };
            indices.push(idx.inner().clone());
        }

        // Extract dimensions
        let dims_vec: Vec<usize> = (0..rank).map(|i| unsafe { *dims.add(i) }).collect();

        // Validate data length
        let expected_len: usize = dims_vec.iter().product();
        if data_len != expected_len {
            return ptr::null_mut();
        }

        // Copy data
        let data_vec: Vec<Complex64> = (0..data_len)
            .map(|i| unsafe { Complex64::new(*data_re.add(i), *data_im.add(i)) })
            .collect();

        // Create tensor using high-level API
        let tensor = InternalTensor::from_dense_c64(indices, data_vec);

        Box::into_raw(Box::new(t4a_tensor::new(tensor)))
    });

    result.unwrap_or(ptr::null_mut())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::*;

    #[test]
    fn test_tensor_lifecycle() {
        // Create indices
        let i = t4a_index_new(2);
        let j = t4a_index_new(3);
        assert!(!i.is_null());
        assert!(!j.is_null());

        // Create tensor
        let index_ptrs = [i as *const _, j as *const _];
        let dims = [2_usize, 3_usize];
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let tensor =
            t4a_tensor_new_dense_f64(2, index_ptrs.as_ptr(), dims.as_ptr(), data.as_ptr(), 6);
        assert!(!tensor.is_null());

        // Test is_assigned
        assert_eq!(t4a_tensor_is_assigned(tensor as *const _), 1);

        // Test clone
        let cloned = t4a_tensor_clone(tensor as *const _);
        assert!(!cloned.is_null());

        // Clean up
        t4a_tensor_release(cloned);
        t4a_tensor_release(tensor);
        t4a_index_release(i);
        t4a_index_release(j);
    }

    #[test]
    fn test_tensor_accessors() {
        // Create indices
        let i = t4a_index_new(2);
        let j = t4a_index_new(3);

        // Create tensor
        let index_ptrs = [i as *const _, j as *const _];
        let dims = [2_usize, 3_usize];
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let tensor =
            t4a_tensor_new_dense_f64(2, index_ptrs.as_ptr(), dims.as_ptr(), data.as_ptr(), 6);

        // Get rank
        let mut rank: usize = 0;
        assert_eq!(
            t4a_tensor_get_rank(tensor as *const _, &mut rank),
            T4A_SUCCESS
        );
        assert_eq!(rank, 2);

        // Get dims
        let mut out_dims = [0_usize; 2];
        assert_eq!(
            t4a_tensor_get_dims(tensor as *const _, out_dims.as_mut_ptr(), 2),
            T4A_SUCCESS
        );
        assert_eq!(out_dims, [2, 3]);

        // Get storage kind
        let mut kind = t4a_storage_kind::DenseC64;
        assert_eq!(
            t4a_tensor_get_storage_kind(tensor as *const _, &mut kind),
            T4A_SUCCESS
        );
        assert_eq!(kind, t4a_storage_kind::DenseF64);

        // Get data
        let mut out_len: usize = 0;
        assert_eq!(
            t4a_tensor_get_data_f64(tensor as *const _, ptr::null_mut(), 0, &mut out_len),
            T4A_SUCCESS
        );
        assert_eq!(out_len, 6);

        let mut out_data = [0.0; 6];
        assert_eq!(
            t4a_tensor_get_data_f64(tensor as *const _, out_data.as_mut_ptr(), 6, &mut out_len),
            T4A_SUCCESS
        );
        assert_eq!(out_data, data);

        // Get indices
        let mut out_indices: [*mut t4a_index; 2] = [ptr::null_mut(); 2];
        assert_eq!(
            t4a_tensor_get_indices(tensor as *const _, out_indices.as_mut_ptr(), 2),
            T4A_SUCCESS
        );

        // Verify indices have correct dimensions
        let mut dim0: usize = 0;
        let mut dim1: usize = 0;
        crate::index::t4a_index_dim(out_indices[0] as *const _, &mut dim0);
        crate::index::t4a_index_dim(out_indices[1] as *const _, &mut dim1);
        assert_eq!(dim0, 2);
        assert_eq!(dim1, 3);

        // Clean up indices
        for idx in &out_indices {
            t4a_index_release(*idx);
        }

        // Clean up
        t4a_tensor_release(tensor);
        t4a_index_release(i);
        t4a_index_release(j);
    }

    #[test]
    fn test_tensor_c64() {
        // Create indices
        let i = t4a_index_new(2);
        let j = t4a_index_new(2);

        // Create tensor
        let index_ptrs = [i as *const _, j as *const _];
        let dims = [2_usize, 2_usize];
        let data_re = [1.0, 2.0, 3.0, 4.0];
        let data_im = [0.5, 1.5, 2.5, 3.5];

        let tensor = t4a_tensor_new_dense_c64(
            2,
            index_ptrs.as_ptr(),
            dims.as_ptr(),
            data_re.as_ptr(),
            data_im.as_ptr(),
            4,
        );
        assert!(!tensor.is_null());

        // Get storage kind
        let mut kind = t4a_storage_kind::DenseF64;
        assert_eq!(
            t4a_tensor_get_storage_kind(tensor as *const _, &mut kind),
            T4A_SUCCESS
        );
        assert_eq!(kind, t4a_storage_kind::DenseC64);

        // Get data
        let mut out_len: usize = 0;
        let mut out_re = [0.0; 4];
        let mut out_im = [0.0; 4];
        assert_eq!(
            t4a_tensor_get_data_c64(
                tensor as *const _,
                out_re.as_mut_ptr(),
                out_im.as_mut_ptr(),
                4,
                &mut out_len
            ),
            T4A_SUCCESS
        );
        assert_eq!(out_len, 4);
        assert_eq!(out_re, data_re);
        assert_eq!(out_im, data_im);

        // Clean up
        t4a_tensor_release(tensor);
        t4a_index_release(i);
        t4a_index_release(j);
    }
}
