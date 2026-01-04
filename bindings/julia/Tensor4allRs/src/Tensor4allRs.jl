module Tensor4allRs

using Libdl

# Library handle
const _lib = Ref{Ptr{Cvoid}}(C_NULL)

# Status codes (must match Rust constants)
const T4A_SUCCESS = 0
const T4A_NULL_POINTER = -1
const T4A_INVALID_ARGUMENT = -2
const T4A_TAG_OVERFLOW = -3
const T4A_TAG_TOO_LONG = -4
const T4A_BUFFER_TOO_SMALL = -5
const T4A_INTERNAL_ERROR = -6

"""
    init_library(libpath::String)

Initialize the Tensor4allRs library by loading the shared library.

# Arguments
- `libpath`: Path to the `libtensor4all_capi` shared library

# Example
```julia
Tensor4allRs.init_library("/path/to/libtensor4all_capi.dylib")
```
"""
function init_library(libpath::String)
    if _lib[] != C_NULL
        # Already initialized
        return
    end
    _lib[] = Libdl.dlopen(libpath)
    if _lib[] == C_NULL
        error("Failed to load library: $libpath")
    end
end

"""
    get_lib()

Get the library handle. Throws an error if the library has not been initialized.
"""
function get_lib()
    if _lib[] == C_NULL
        error("Library not initialized. Call Tensor4allRs.init_library(path) first.")
    end
    return _lib[]
end

# Include submodules
include("tensortrain.jl")

export TensorTrainF64, TensorTrainC64
export zeros_tt, constant_tt
export site_dims, link_dims, rank, evaluate, sum_tt, norm_tt, log_norm_tt
export scale!, scaled, fulltensor

end # module
