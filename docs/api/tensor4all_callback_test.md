# tensor4all-callback-test

## src/lib.rs

### `pub fn t4a_callback_test_simple(callback: EvalCallback, user_data: * mut c_void, result: * mut f64) -> i32`

Simple test: call the callback once with fixed indices [1, 2, 3]

### `pub fn t4a_callback_test_multiple(callback: EvalCallback, user_data: * mut c_void, n_calls: usize, result: * mut f64) -> i32`

Test: call the callback multiple times and sum the results Calls the callback with indices [i, i+1, i+2] for i in 0..n_calls

### `pub fn t4a_callback_test_with_indices(callback: EvalCallback, user_data: * mut c_void, indices: * const i64, n_indices: usize, result: * mut f64) -> i32`

Test: call the callback with variable-length indices

### ` fn sum_callback(indices: * const i64, n_indices: usize, result: * mut f64, _user_data: * mut c_void) -> i32`

### ` fn multiply_callback(indices: * const i64, n_indices: usize, result: * mut f64, user_data: * mut c_void) -> i32`

### ` fn test_simple_callback()`

### ` fn test_callback_with_user_data()`

### ` fn test_multiple_callbacks()`

### ` fn test_with_custom_indices()`

