//! Test crate for C callback function pointer experiments
//!
//! This crate tests the trampoline pattern for passing closures from Julia to Rust
//! via C function pointers.

use std::ffi::c_void;

/// Callback function type for evaluating a function at given indices.
///
/// # Arguments
/// * `indices` - Pointer to array of indices
/// * `n_indices` - Number of indices
/// * `result` - Pointer to store the result
/// * `user_data` - Opaque pointer to user data (Julia Ref object)
///
/// # Returns
/// * 0 on success
/// * non-zero on error
pub type EvalCallback = extern "C" fn(
    indices: *const i64,
    n_indices: usize,
    result: *mut f64,
    user_data: *mut c_void,
) -> i32;

/// Simple test: call the callback once with fixed indices [1, 2, 3]
///
/// # Safety
/// - `callback` must be a valid function pointer
/// - `user_data` must be valid for the callback's use
#[no_mangle]
pub unsafe extern "C" fn t4a_callback_test_simple(
    callback: EvalCallback,
    user_data: *mut c_void,
    result: *mut f64,
) -> i32 {
    let indices = [1i64, 2, 3];
    let mut local_result = 0.0;

    let status = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        callback(indices.as_ptr(), indices.len(), &mut local_result, user_data)
    }));

    match status {
        Ok(0) => {
            *result = local_result;
            0
        }
        Ok(err) => err,
        Err(_) => -1, // Panic occurred
    }
}

/// Test: call the callback multiple times and sum the results
///
/// Calls the callback with indices [i, i+1, i+2] for i in 0..n_calls
///
/// # Safety
/// - `callback` must be a valid function pointer
/// - `user_data` must be valid for the callback's use
#[no_mangle]
pub unsafe extern "C" fn t4a_callback_test_multiple(
    callback: EvalCallback,
    user_data: *mut c_void,
    n_calls: usize,
    result: *mut f64,
) -> i32 {
    let mut sum = 0.0;

    for i in 0..n_calls {
        let indices = [i as i64, (i + 1) as i64, (i + 2) as i64];
        let mut local_result = 0.0;

        let status = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            callback(indices.as_ptr(), indices.len(), &mut local_result, user_data)
        }));

        match status {
            Ok(0) => sum += local_result,
            Ok(err) => return err,
            Err(_) => return -1,
        }
    }

    *result = sum;
    0
}

/// Test: call the callback with variable-length indices
///
/// # Safety
/// - `callback` must be a valid function pointer
/// - `user_data` must be valid for the callback's use
/// - `indices` must point to `n_indices` valid i64 values
#[no_mangle]
pub unsafe extern "C" fn t4a_callback_test_with_indices(
    callback: EvalCallback,
    user_data: *mut c_void,
    indices: *const i64,
    n_indices: usize,
    result: *mut f64,
) -> i32 {
    let mut local_result = 0.0;

    let status = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        callback(indices, n_indices, &mut local_result, user_data)
    }));

    match status {
        Ok(0) => {
            *result = local_result;
            0
        }
        Ok(err) => err,
        Err(_) => -1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test callback that sums the indices
    extern "C" fn sum_callback(
        indices: *const i64,
        n_indices: usize,
        result: *mut f64,
        _user_data: *mut c_void,
    ) -> i32 {
        unsafe {
            let slice = std::slice::from_raw_parts(indices, n_indices);
            *result = slice.iter().sum::<i64>() as f64;
        }
        0
    }

    // Test callback that multiplies by a factor stored in user_data
    extern "C" fn multiply_callback(
        indices: *const i64,
        n_indices: usize,
        result: *mut f64,
        user_data: *mut c_void,
    ) -> i32 {
        unsafe {
            let slice = std::slice::from_raw_parts(indices, n_indices);
            let factor = *(user_data as *const f64);
            *result = slice.iter().sum::<i64>() as f64 * factor;
        }
        0
    }

    #[test]
    fn test_simple_callback() {
        let mut result = 0.0;
        unsafe {
            let status = t4a_callback_test_simple(
                sum_callback,
                std::ptr::null_mut(),
                &mut result,
            );
            assert_eq!(status, 0);
            assert_eq!(result, 6.0); // 1 + 2 + 3
        }
    }

    #[test]
    fn test_callback_with_user_data() {
        let factor = 2.0f64;
        let mut result = 0.0;
        unsafe {
            let status = t4a_callback_test_simple(
                multiply_callback,
                &factor as *const f64 as *mut c_void,
                &mut result,
            );
            assert_eq!(status, 0);
            assert_eq!(result, 12.0); // (1 + 2 + 3) * 2
        }
    }

    #[test]
    fn test_multiple_callbacks() {
        let mut result = 0.0;
        unsafe {
            let status = t4a_callback_test_multiple(
                sum_callback,
                std::ptr::null_mut(),
                3,
                &mut result,
            );
            assert_eq!(status, 0);
            // i=0: 0+1+2=3, i=1: 1+2+3=6, i=2: 2+3+4=9 => sum=18
            assert_eq!(result, 18.0);
        }
    }

    #[test]
    fn test_with_custom_indices() {
        let indices = [10i64, 20, 30, 40];
        let mut result = 0.0;
        unsafe {
            let status = t4a_callback_test_with_indices(
                sum_callback,
                std::ptr::null_mut(),
                indices.as_ptr(),
                indices.len(),
                &mut result,
            );
            assert_eq!(status, 0);
            assert_eq!(result, 100.0);
        }
    }
}
