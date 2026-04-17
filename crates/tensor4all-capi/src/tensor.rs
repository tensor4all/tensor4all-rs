//! C API for dense tensor construction, inspection, and contraction.

use std::panic::{catch_unwind, AssertUnwindSafe};

use num_complex::Complex64;
use tensor4all_core::{qr_with, svd_with, QrOptions, SvdOptions, TruncationParams};

use crate::types::{t4a_index, t4a_scalar_kind, t4a_tensor, InternalIndex, InternalTensor};
use crate::{
    capi_error, clone_opaque, is_assigned_opaque, panic_message, release_opaque, run_catching,
    set_last_error, CapiResult, StatusCode, T4A_BUFFER_TOO_SMALL, T4A_INTERNAL_ERROR,
    T4A_INVALID_ARGUMENT, T4A_NULL_POINTER, T4A_SUCCESS,
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

fn run_status<F>(f: F) -> StatusCode
where
    F: FnOnce() -> CapiResult<()>,
{
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => T4A_SUCCESS,
        Ok(Err((code, msg))) => {
            set_last_error(&msg);
            code
        }
        Err(panic) => {
            let msg = panic_message(&*panic);
            set_last_error(&msg);
            T4A_INTERNAL_ERROR
        }
    }
}

fn require_tensor<'a>(ptr: *const t4a_tensor) -> Result<&'a t4a_tensor, (StatusCode, String)> {
    if ptr.is_null() {
        return Err(capi_error(T4A_NULL_POINTER, "tensor is null"));
    }
    Ok(unsafe { &*ptr })
}

fn build_svd_options(rtol: f64, cutoff: f64, maxdim: usize) -> SvdOptions {
    let mut truncation = TruncationParams::new();
    if cutoff > 0.0 {
        truncation = truncation.with_cutoff(cutoff);
    } else if rtol > 0.0 {
        truncation = truncation.with_rtol(rtol);
    }
    if maxdim > 0 {
        truncation = truncation.with_max_rank(maxdim);
    }
    SvdOptions { truncation }
}

fn build_qr_options(rtol: f64) -> QrOptions {
    // `0.0` is the C-API sentinel for exact QR without truncation.
    QrOptions::with_rtol(rtol)
}

fn box_tensor_handle(tensor: InternalTensor) -> *mut t4a_tensor {
    Box::into_raw(Box::new(t4a_tensor::new(tensor)))
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

/// Compute the SVD of a tensor split by the requested left indices.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_svd(
    tensor: *const t4a_tensor,
    left_inds: *const *const t4a_index,
    n_left: usize,
    rtol: f64,
    cutoff: f64,
    maxdim: usize,
    out_u: *mut *mut t4a_tensor,
    out_s: *mut *mut t4a_tensor,
    out_v: *mut *mut t4a_tensor,
) -> StatusCode {
    run_status(|| {
        let tensor = require_tensor(tensor)?;
        if out_u.is_null() || out_s.is_null() || out_v.is_null() {
            return Err(capi_error(
                T4A_NULL_POINTER,
                "out_u, out_s, or out_v is null",
            ));
        }

        let left_inds = read_indices_from_ptrs(n_left, left_inds)?;
        let options = build_svd_options(rtol, cutoff, maxdim);
        let (u, s, v) = match t4a_scalar_kind::from_tensor(tensor.inner()) {
            t4a_scalar_kind::F64 => svd_with::<f64>(tensor.inner(), &left_inds, &options),
            t4a_scalar_kind::C64 => svd_with::<Complex64>(tensor.inner(), &left_inds, &options),
        }
        .map_err(|err| capi_error(T4A_INVALID_ARGUMENT, err))?;

        unsafe {
            *out_u = box_tensor_handle(u);
            *out_s = box_tensor_handle(s);
            *out_v = box_tensor_handle(v);
        }
        Ok(())
    })
}

/// Compute the QR decomposition of a tensor split by the requested left indices.
///
/// The tensor is unfolded into a matrix by treating `left_inds[..n_left]` as row
/// indices and all remaining tensor indices as columns, then factorized as
/// `tensor = Q * R`.
///
/// # Arguments
/// * `tensor` - Input tensor handle to factorize.
/// * `left_inds` - Pointers to the indices that should appear on the left side
///   of the factorization.
/// * `n_left` - Number of entries in `left_inds`.
/// * `rtol` - Relative truncation tolerance for QR row-norm truncation. Pass
///   `0.0` to disable truncation and keep the exact QR rank. Pass any other
///   finite, non-negative value to request truncation for this call.
/// * `out_q` - Output slot that receives a newly allocated `Q` tensor handle on
///   success.
/// * `out_r` - Output slot that receives a newly allocated `R` tensor handle on
///   success.
///
/// # Returns
/// Returns `T4A_SUCCESS` on success and writes owned tensor handles to
/// `out_q` and `out_r`. The caller must release both handles with
/// [`t4a_tensor_release`].
///
/// # Errors
/// Returns:
/// - `T4A_NULL_POINTER` if `tensor`, any required index pointer, `out_q`, or
///   `out_r` is null.
/// - `T4A_INVALID_ARGUMENT` if the index split is invalid, QR factorization
///   fails, or `rtol` is negative or non-finite (for example `NaN` or
///   `+/-inf`).
/// - `T4A_INTERNAL_ERROR` if the Rust implementation panics while processing
///   the request.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_qr(
    tensor: *const t4a_tensor,
    left_inds: *const *const t4a_index,
    n_left: usize,
    rtol: f64,
    out_q: *mut *mut t4a_tensor,
    out_r: *mut *mut t4a_tensor,
) -> StatusCode {
    run_status(|| {
        let tensor = require_tensor(tensor)?;
        if out_q.is_null() || out_r.is_null() {
            return Err(capi_error(T4A_NULL_POINTER, "out_q or out_r is null"));
        }

        let left_inds = read_indices_from_ptrs(n_left, left_inds)?;
        let options = build_qr_options(rtol);
        let (q, r) = match t4a_scalar_kind::from_tensor(tensor.inner()) {
            t4a_scalar_kind::F64 => qr_with::<f64>(tensor.inner(), &left_inds, &options),
            t4a_scalar_kind::C64 => qr_with::<Complex64>(tensor.inner(), &left_inds, &options),
        }
        .map_err(|err| capi_error(T4A_INVALID_ARGUMENT, err))?;

        unsafe {
            *out_q = box_tensor_handle(q);
            *out_r = box_tensor_handle(r);
        }
        Ok(())
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
