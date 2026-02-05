#!/usr/bin/env julia
#
# Integration test for tensor4all-hdf5-ffi with Julia via runtime-loading (dlopen)
#
# This test verifies that the FFI library can be loaded from Julia and
# the HDF5 library can be initialized via runtime loading.
#
# Environment variables:
#   TENSOR4ALL_LIB - Path to the tensor4all_hdf5_ffi shared library
#   HDF5_LIB       - Path to the HDF5 shared library

using Libdl

# Get library paths from environment
tensor4all_lib = get(ENV, "TENSOR4ALL_LIB", nothing)
hdf5_lib = get(ENV, "HDF5_LIB", nothing)

if tensor4all_lib === nothing
    error("TENSOR4ALL_LIB environment variable not set")
end

if hdf5_lib === nothing
    error("HDF5_LIB environment variable not set")
end

println("Testing tensor4all-hdf5-ffi from Julia")
println("  TENSOR4ALL_LIB: $tensor4all_lib")
println("  HDF5_LIB: $hdf5_lib")

# Diagnostic: check if HDF5 library file exists and is accessible
println("\nDiagnostics:")
if isfile(hdf5_lib)
    println("  HDF5 file exists: yes")
    println("  HDF5 file size: $(filesize(hdf5_lib)) bytes")
else
    println("  HDF5 file exists: no")
    # Try to find what HDF5 files are available
    hdf5_dir = dirname(hdf5_lib)
    if isdir(hdf5_dir)
        println("  Files in $(hdf5_dir) matching libhdf5*:")
        for f in readdir(hdf5_dir)
            if startswith(f, "libhdf5")
                println("    $f")
            end
        end
    end
end

# Diagnostic: try to dlopen HDF5 directly from Julia
println("\n  Testing direct dlopen of HDF5 from Julia...")
try
    hdf5_handle = Libdl.dlopen(hdf5_lib)
    println("  Direct HDF5 dlopen: success")
    Libdl.dlclose(hdf5_handle)
catch e
    println("  Direct HDF5 dlopen: FAILED - $e")
end

println()

# Load the tensor4all library
println("Loading tensor4all-hdf5-ffi library...")
lib = Libdl.dlopen(tensor4all_lib)
println("  Library loaded successfully")

# Get function pointers (note: C API uses hdf5_ffi_ prefix)
hdf5_ffi_version = Libdl.dlsym(lib, :hdf5_ffi_version)
hdf5_ffi_init = Libdl.dlsym(lib, :hdf5_ffi_init)
hdf5_ffi_is_initialized = Libdl.dlsym(lib, :hdf5_ffi_is_initialized)
hdf5_ffi_library_path = Libdl.dlsym(lib, :hdf5_ffi_library_path)
hdf5_ffi_status_message = Libdl.dlsym(lib, :hdf5_ffi_status_message)

# Helper function to get status message
function get_status_message(status::Cint)
    msg_ptr = ccall(hdf5_ffi_status_message, Cstring, (Cint,), status)
    return unsafe_string(msg_ptr)
end

# Test version function (should work without initialization)
println("\nTesting hdf5_ffi_version...")
version_ptr = ccall(hdf5_ffi_version, Cstring, ())
version = unsafe_string(version_ptr)
println("  Library version: $version")

# Test is_initialized before init
println("\nTesting hdf5_ffi_is_initialized (before init)...")
is_init = ccall(hdf5_ffi_is_initialized, Cint, ())
println("  Is initialized: $is_init")
@assert is_init == 0 "Expected not initialized (0), got $is_init"

# Initialize HDF5 with runtime loading
println("\nTesting hdf5_ffi_init with HDF5 library...")
# Use standard Julia ccall pattern - Julia handles string conversion automatically
status = ccall(hdf5_ffi_init, Cint, (Cstring,), hdf5_lib)
status_msg = get_status_message(status)
println("  Init status: $status ($status_msg)")
@assert status == 0 "hdf5_ffi_init failed with status $status: $status_msg"

# Test is_initialized after init
println("\nTesting hdf5_ffi_is_initialized (after init)...")
is_init = ccall(hdf5_ffi_is_initialized, Cint, ())
println("  Is initialized: $is_init")
@assert is_init == 1 "Expected initialized (1), got $is_init"

# Test library_path
println("\nTesting hdf5_ffi_library_path...")
path_buf = Vector{UInt8}(undef, 1024)
path_len = Ref{Csize_t}(0)
status = ccall(hdf5_ffi_library_path, Cint, (Ptr{UInt8}, Csize_t, Ptr{Csize_t}),
               path_buf, length(path_buf), path_len)
if status == 0 && path_len[] > 0
    path = String(path_buf[1:path_len[]])
    println("  Library path: $path")
    @assert path == hdf5_lib "Library path mismatch: expected $hdf5_lib, got $path"
else
    status_msg = get_status_message(status)
    println("  Library path: (not available, status=$status: $status_msg)")
end

# Clean up
Libdl.dlclose(lib)

println("\n" * "="^50)
println("All Julia integration tests passed!")
println("="^50)
