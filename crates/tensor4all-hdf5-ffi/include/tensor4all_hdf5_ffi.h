/**
 * @file tensor4all_hdf5_ffi.h
 * @brief C API for HDF5 runtime library loading
 *
 * This header provides functions for initializing the HDF5 library at runtime
 * using dlopen/LoadLibrary. This allows applications to use HDF5 libraries
 * provided by Julia (HDF5_jll) or Python (h5py) at runtime.
 *
 * @example
 * @code
 * #include "tensor4all_hdf5_ffi.h"
 *
 * int main() {
 *     // Initialize with system HDF5
 *     int status = hdf5_ffi_init("/usr/lib/libhdf5.so");
 *     if (status != HDF5_FFI_SUCCESS) {
 *         printf("Error: %s\n", hdf5_ffi_status_message(status));
 *         return 1;
 *     }
 *
 *     // Check initialization
 *     if (hdf5_ffi_is_initialized()) {
 *         printf("HDF5 initialized successfully\n");
 *     }
 *
 *     // Get library path
 *     size_t len;
 *     hdf5_ffi_library_path(NULL, 0, &len);
 *     char* path = malloc(len + 1);
 *     hdf5_ffi_library_path(path, len + 1, &len);
 *     printf("Using HDF5 from: %s\n", path);
 *     free(path);
 *
 *     return 0;
 * }
 * @endcode
 */

#ifndef TENSOR4ALL_HDF5_FFI_H
#define TENSOR4ALL_HDF5_FFI_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Status Codes
 * ============================================================================ */

/** Operation succeeded. */
#define HDF5_FFI_SUCCESS 0

/** Null pointer error. */
#define HDF5_FFI_NULL_POINTER (-1)

/** Invalid argument. */
#define HDF5_FFI_INVALID_ARGUMENT (-2)

/** Library already initialized with a different path. */
#define HDF5_FFI_ALREADY_INITIALIZED (-3)

/** Failed to load the library. */
#define HDF5_FFI_LIBRARY_LOAD_ERROR (-4)

/** Library not initialized. */
#define HDF5_FFI_NOT_INITIALIZED (-5)

/** Internal error (panic or other unexpected error). */
#define HDF5_FFI_INTERNAL_ERROR (-6)

/** Buffer too small. */
#define HDF5_FFI_BUFFER_TOO_SMALL (-7)

/* ============================================================================
 * Initialization Functions
 * ============================================================================ */

/**
 * @brief Initialize HDF5 by loading the library from the given path.
 *
 * Must be called before any HDF5 operations.
 *
 * @param library_path Path to the HDF5 shared library (null-terminated string).
 *        Examples:
 *        - Linux: "/usr/lib/libhdf5.so", "/path/to/libhdf5.so.200"
 *        - macOS: "/usr/local/lib/libhdf5.dylib"
 *        - Windows: "C:\\path\\to\\hdf5.dll"
 *
 * @return Status code:
 *         - HDF5_FFI_SUCCESS (0) if initialization succeeds
 *         - HDF5_FFI_NULL_POINTER if library_path is NULL
 *         - HDF5_FFI_ALREADY_INITIALIZED if already initialized with different path
 *         - HDF5_FFI_LIBRARY_LOAD_ERROR if the library cannot be loaded
 *         - HDF5_FFI_INTERNAL_ERROR on unexpected error
 *
 * @note This function is thread-safe. Multiple calls with the same path are
 *       idempotent. If called concurrently, only one thread will perform
 *       the initialization.
 */
int hdf5_ffi_init(const char* library_path);

/**
 * @brief Check if HDF5 has been initialized.
 *
 * @return 1 if HDF5 has been initialized, 0 otherwise.
 *
 * @note This function is thread-safe.
 */
int hdf5_ffi_is_initialized(void);

/**
 * @brief Get the path used for HDF5 initialization.
 *
 * This function follows the "query-then-fill" pattern:
 * 1. Call with buf = NULL to get the required buffer length in out_len
 * 2. Allocate a buffer of size out_len + 1 (for null terminator)
 * 3. Call again with the allocated buffer
 *
 * @param buf Buffer to write the path (can be NULL to query length only)
 * @param buf_len Length of the buffer
 * @param out_len Output: required buffer length (not including null terminator)
 *
 * @return Status code:
 *         - HDF5_FFI_SUCCESS if the path was written (or only length queried)
 *         - HDF5_FFI_NULL_POINTER if out_len is NULL
 *         - HDF5_FFI_NOT_INITIALIZED if HDF5 has not been initialized
 *         - HDF5_FFI_BUFFER_TOO_SMALL if buffer is too small (length still written)
 *         - HDF5_FFI_INTERNAL_ERROR on unexpected error
 *
 * @note This function is thread-safe.
 */
int hdf5_ffi_library_path(char* buf, size_t buf_len, size_t* out_len);

/**
 * @brief Get a human-readable error message for a status code.
 *
 * @param status The status code to describe
 *
 * @return A pointer to a static string describing the error. The string is
 *         valid for the lifetime of the program and must not be freed.
 *
 * @note This function is thread-safe.
 */
const char* hdf5_ffi_status_message(int status);

/**
 * @brief Get the version of this library.
 *
 * @return A pointer to a static string containing the version (e.g., "0.1.0").
 *         The string is valid for the lifetime of the program and must not
 *         be freed.
 */
const char* hdf5_ffi_version(void);

#ifdef __cplusplus
}
#endif

#endif /* TENSOR4ALL_HDF5_FFI_H */
