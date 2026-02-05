#!/usr/bin/env python3
"""
Integration test for tensor4all-hdf5-ffi with Python via runtime-loading (dlopen)

This test verifies that the FFI library can be loaded from Python and
the HDF5 library can be initialized via runtime loading.

Environment variables:
    TENSOR4ALL_LIB - Path to the tensor4all_hdf5_ffi shared library
    HDF5_LIB       - Path to the HDF5 shared library
"""

import ctypes
import os
import sys


def main():
    # Get library paths from environment
    tensor4all_lib = os.environ.get("TENSOR4ALL_LIB")
    hdf5_lib = os.environ.get("HDF5_LIB")

    if not tensor4all_lib:
        print("ERROR: TENSOR4ALL_LIB environment variable not set", file=sys.stderr)
        sys.exit(1)

    if not hdf5_lib:
        print("ERROR: HDF5_LIB environment variable not set", file=sys.stderr)
        sys.exit(1)

    print("Testing tensor4all-hdf5-ffi from Python")
    print(f"  TENSOR4ALL_LIB: {tensor4all_lib}")
    print(f"  HDF5_LIB: {hdf5_lib}")
    print()

    # Load the tensor4all library
    print("Loading tensor4all-hdf5-ffi library...")
    lib = ctypes.CDLL(tensor4all_lib)
    print("  Library loaded successfully")

    # Define function signatures
    # Note: C API uses hdf5_ffi_ prefix

    lib.hdf5_ffi_version.argtypes = []
    lib.hdf5_ffi_version.restype = ctypes.c_char_p

    lib.hdf5_ffi_init.argtypes = [ctypes.c_char_p]
    lib.hdf5_ffi_init.restype = ctypes.c_int

    lib.hdf5_ffi_is_initialized.argtypes = []
    lib.hdf5_ffi_is_initialized.restype = ctypes.c_int

    lib.hdf5_ffi_library_path.argtypes = [
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.hdf5_ffi_library_path.restype = ctypes.c_int

    lib.hdf5_ffi_status_message.argtypes = [ctypes.c_int]
    lib.hdf5_ffi_status_message.restype = ctypes.c_char_p

    # Test version function (should work without initialization)
    print("\nTesting hdf5_ffi_version...")
    version = lib.hdf5_ffi_version()
    print(f"  Library version: {version.decode('utf-8')}")

    # Test is_initialized before init
    print("\nTesting hdf5_ffi_is_initialized (before init)...")
    is_init = lib.hdf5_ffi_is_initialized()
    print(f"  Is initialized: {is_init}")
    assert is_init == 0, f"Expected not initialized (0), got {is_init}"

    # Initialize HDF5 with runtime loading
    print("\nTesting hdf5_ffi_init with HDF5 library...")
    status = lib.hdf5_ffi_init(hdf5_lib.encode("utf-8"))
    status_msg = lib.hdf5_ffi_status_message(status).decode("utf-8")
    print(f"  Init status: {status} ({status_msg})")
    assert status == 0, f"hdf5_ffi_init failed with status {status}: {status_msg}"

    # Test is_initialized after init
    print("\nTesting hdf5_ffi_is_initialized (after init)...")
    is_init = lib.hdf5_ffi_is_initialized()
    print(f"  Is initialized: {is_init}")
    assert is_init == 1, f"Expected initialized (1), got {is_init}"

    # Test library_path
    print("\nTesting hdf5_ffi_library_path...")
    path_buf = ctypes.create_string_buffer(1024)
    path_len = ctypes.c_size_t()
    status = lib.hdf5_ffi_library_path(path_buf, 1024, ctypes.byref(path_len))
    if status == 0 and path_len.value > 0:
        path = path_buf.value.decode("utf-8")
        print(f"  Library path: {path}")
        assert path == hdf5_lib, f"Library path mismatch: expected {hdf5_lib}, got {path}"
    else:
        status_msg = lib.hdf5_ffi_status_message(status).decode("utf-8")
        print(f"  Library path: (not available, status={status}: {status_msg})")

    print("\n" + "=" * 50)
    print("All Python integration tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
