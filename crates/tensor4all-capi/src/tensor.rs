//! C API for dense tensor construction, inspection, and contraction.

use std::panic::{catch_unwind, AssertUnwindSafe};

use num_complex::Complex64;

use crate::types::{t4a_index, t4a_scalar_kind, t4a_tensor, InternalIndex, InternalTensor};
use crate::{
    capi_error, clone_opaque, is_assigned_opaque, release_opaque, run_catching, StatusCode,
    T4A_BUFFER_TOO_SMALL, T4A_INVALID_ARGUMENT, T4A_NULL_POINTER, T4A_SUCCESS,
};

/// Release a tensor handle.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_release(obj: *mut t4a_tensor) {
    release_opaque(obj);
}

/// Clone a tensor handle.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_clone(
    src: *const t4a_tensor,
    out: *mut *mut t4a_tensor,
) -> StatusCode {
    clone_opaque(src, out)
}

/// Check whether a tensor handle is assigned.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_is_assigned(obj: *const t4a_tensor) -> i32 {
    is_assigned_opaque(obj)
}

fn read_indices_from_ptrs(
    rank: usize,
    index_ptrs: *const *const t4a_index,
) -> Result<Vec<InternalIndex>, (StatusCode, String)> {
    if rank == 0 {
        return Ok(Vec::new());
    }
    if index_ptrs.is_null() {
        return Err(capi_error(T4A_NULL_POINTER, "index_ptrs is null"));
    }

    let mut indices = Vec::with_capacity(rank);
    for i in 0..rank {
        let idx_ptr = unsafe { *index_ptrs.add(i) };
        if idx_ptr.is_null() {
            return Err(capi_error(
                T4A_NULL_POINTER,
                format!("index_ptrs[{i}] is null"),
            ));
        }
        indices.push(unsafe { (&*idx_ptr).inner().clone() });
    }
    Ok(indices)
}

fn dims_from_indices(indices: &[InternalIndex]) -> Vec<usize> {
    indices.iter().map(|index| index.size()).collect()
}

/// Get the rank of a tensor.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_rank(ptr: *const t4a_tensor, out_rank: *mut usize) -> StatusCode {
    if ptr.is_null() || out_rank.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        *out_rank = (*ptr).inner().indices.len();
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

/// Copy tensor dimensions in index order.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_dims(
    ptr: *const t4a_tensor,
    buf: *mut usize,
    buf_len: usize,
    out_len: *mut usize,
) -> StatusCode {
    if ptr.is_null() || out_len.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        let dims = (*ptr).inner().dims();
        *out_len = dims.len();

        if buf.is_null() {
            return T4A_SUCCESS;
        }
        if buf_len < dims.len() {
            return T4A_BUFFER_TOO_SMALL;
        }

        std::ptr::copy_nonoverlapping(dims.as_ptr(), buf, dims.len());
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

/// Copy cloned index handles describing the tensor axes.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_indices(
    ptr: *const t4a_tensor,
    buf: *mut *mut t4a_index,
    buf_len: usize,
    out_len: *mut usize,
) -> StatusCode {
    if ptr.is_null() || out_len.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        let indices = &(*ptr).inner().indices;
        *out_len = indices.len();

        if buf.is_null() {
            return T4A_SUCCESS;
        }
        if buf_len < indices.len() {
            return T4A_BUFFER_TOO_SMALL;
        }

        for (i, index) in indices.iter().enumerate() {
            *buf.add(i) = Box::into_raw(Box::new(t4a_index::new(index.clone())));
        }
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

/// Get the scalar kind of a tensor.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_scalar_kind(
    ptr: *const t4a_tensor,
    out_kind: *mut t4a_scalar_kind,
) -> StatusCode {
    if ptr.is_null() || out_kind.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        *out_kind = t4a_scalar_kind::from_tensor((*ptr).inner());
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

/// Copy dense `f64` data in column-major order.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_copy_dense_f64(
    ptr: *const t4a_tensor,
    buf: *mut f64,
    buf_len: usize,
    out_len: *mut usize,
) -> StatusCode {
    if ptr.is_null() || out_len.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        let data = match (*ptr).inner().as_slice_f64() {
            Ok(data) => data,
            Err(e) => return crate::err_status(e, T4A_INVALID_ARGUMENT),
        };
        *out_len = data.len();

        if buf.is_null() {
            return T4A_SUCCESS;
        }
        if buf_len < data.len() {
            return T4A_BUFFER_TOO_SMALL;
        }

        std::ptr::copy_nonoverlapping(data.as_ptr(), buf, data.len());
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

/// Copy dense `Complex64` data as interleaved doubles in column-major order.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_copy_dense_c64(
    ptr: *const t4a_tensor,
    buf_interleaved: *mut f64,
    n_complex: usize,
    out_len: *mut usize,
) -> StatusCode {
    if ptr.is_null() || out_len.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        let data = match (*ptr).inner().as_slice_c64() {
            Ok(data) => data,
            Err(e) => return crate::err_status(e, T4A_INVALID_ARGUMENT),
        };
        *out_len = data.len();

        if buf_interleaved.is_null() {
            return T4A_SUCCESS;
        }
        if n_complex < data.len() {
            return T4A_BUFFER_TOO_SMALL;
        }

        for (i, value) in data.iter().enumerate() {
            *buf_interleaved.add(2 * i) = value.re;
            *buf_interleaved.add(2 * i + 1) = value.im;
        }
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

/// Contract two tensors by matching common indices.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_contract(
    a: *const t4a_tensor,
    b: *const t4a_tensor,
    out: *mut *mut t4a_tensor,
) -> StatusCode {
    if a.is_null() || b.is_null() {
        return T4A_NULL_POINTER;
    }

    run_catching(out, || unsafe {
        Ok(t4a_tensor::new((*a).inner().contract((*b).inner())))
    })
}

/// Create a dense real tensor from column-major data.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_new_dense_f64(
    rank: usize,
    index_ptrs: *const *const t4a_index,
    data: *const f64,
    data_len: usize,
    out: *mut *mut t4a_tensor,
) -> StatusCode {
    run_catching(out, || {
        let indices = read_indices_from_ptrs(rank, index_ptrs)?;
        let expected_len: usize = dims_from_indices(&indices).iter().product();
        if expected_len > 0 && data.is_null() {
            return Err(capi_error(T4A_NULL_POINTER, "data is null"));
        }
        if data_len != expected_len {
            return Err(capi_error(
                T4A_INVALID_ARGUMENT,
                format!("data_len mismatch: expected {expected_len}, got {data_len}"),
            ));
        }

        let values = unsafe { std::slice::from_raw_parts(data, data_len) }.to_vec();
        let tensor = InternalTensor::from_dense(indices, values)
            .map_err(|e| capi_error(T4A_INVALID_ARGUMENT, e))?;
        Ok(t4a_tensor::new(tensor))
    })
}

/// Create a dense complex tensor from interleaved column-major data.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_new_dense_c64(
    rank: usize,
    index_ptrs: *const *const t4a_index,
    data_interleaved: *const f64,
    n_complex: usize,
    out: *mut *mut t4a_tensor,
) -> StatusCode {
    run_catching(out, || {
        let indices = read_indices_from_ptrs(rank, index_ptrs)?;
        let expected_len: usize = dims_from_indices(&indices).iter().product();
        if expected_len > 0 && data_interleaved.is_null() {
            return Err(capi_error(T4A_NULL_POINTER, "data_interleaved is null"));
        }
        if n_complex != expected_len {
            return Err(capi_error(
                T4A_INVALID_ARGUMENT,
                format!("n_complex mismatch: expected {expected_len}, got {n_complex}"),
            ));
        }

        let raw = unsafe { std::slice::from_raw_parts(data_interleaved, 2 * n_complex) };
        let values = (0..n_complex)
            .map(|i| Complex64::new(raw[2 * i], raw[2 * i + 1]))
            .collect::<Vec<_>>();
        let tensor = InternalTensor::from_dense(indices, values)
            .map_err(|e| capi_error(T4A_INVALID_ARGUMENT, e))?;
        Ok(t4a_tensor::new(tensor))
    })
}

#[cfg(test)]
mod tests;
