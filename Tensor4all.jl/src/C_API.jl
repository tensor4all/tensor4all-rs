"""
    C_API

Low-level C API bindings for tensor4all Rust library.

This module provides direct ccall wrappers for the tensor4all-capi functions.
Users should use the high-level API in the parent module instead.
"""
module C_API

using Libdl

# Status codes (must match Rust side)
const T4A_SUCCESS = Cint(0)
const T4A_NULL_POINTER = Cint(-1)
const T4A_INVALID_ARGUMENT = Cint(-2)
const T4A_TAG_OVERFLOW = Cint(-3)
const T4A_TAG_TOO_LONG = Cint(-4)
const T4A_BUFFER_TOO_SMALL = Cint(-5)
const T4A_INTERNAL_ERROR = Cint(-6)

# Library handle
const _lib_handle = Ref{Ptr{Cvoid}}(C_NULL)
const _lib_path = Ref{String}("")

"""
    libpath() -> String

Get the path to the tensor4all-capi shared library.
"""
function libpath()
    if isempty(_lib_path[])
        # Look in deps directory
        deps_dir = joinpath(@__DIR__, "..", "deps")
        lib_name = "libtensor4all_capi." * Libdl.dlext
        path = joinpath(deps_dir, lib_name)
        if !isfile(path)
            error("""
                tensor4all-capi library not found at: $path
                Please run `Pkg.build("Tensor4all")` to build the library.
                """)
        end
        _lib_path[] = path
    end
    return _lib_path[]
end

"""
    libhandle() -> Ptr{Cvoid}

Get the handle to the loaded library.
"""
function libhandle()
    if _lib_handle[] == C_NULL
        _lib_handle[] = Libdl.dlopen(libpath())
    end
    return _lib_handle[]
end

"""
    check_status(status::Cint)

Check a status code and throw an error if it indicates failure.
"""
function check_status(status::Cint)
    status == T4A_SUCCESS && return nothing

    msg = if status == T4A_NULL_POINTER
        "Null pointer error"
    elseif status == T4A_INVALID_ARGUMENT
        "Invalid argument"
    elseif status == T4A_TAG_OVERFLOW
        "Too many tags (max 4)"
    elseif status == T4A_TAG_TOO_LONG
        "Tag too long (max 16 chars)"
    elseif status == T4A_BUFFER_TOO_SMALL
        "Buffer too small"
    elseif status == T4A_INTERNAL_ERROR
        "Internal error"
    else
        "Unknown error (code: $status)"
    end

    error("Tensor4all C API error: $msg")
end

# ============================================================================
# Index lifecycle functions
# ============================================================================

"""
    t4a_index_new(dim::Integer) -> Ptr{Cvoid}

Create a new index with the given dimension.
"""
function t4a_index_new(dim::Integer)
    return ccall(
        (:t4a_index_new, libpath()),
        Ptr{Cvoid},
        (Csize_t,),
        Csize_t(dim)
    )
end

"""
    t4a_index_new_with_tags(dim::Integer, tags::AbstractString) -> Ptr{Cvoid}

Create a new index with the given dimension and comma-separated tags.
"""
function t4a_index_new_with_tags(dim::Integer, tags::AbstractString)
    return ccall(
        (:t4a_index_new_with_tags, libpath()),
        Ptr{Cvoid},
        (Csize_t, Cstring),
        Csize_t(dim),
        tags
    )
end

"""
    t4a_index_new_with_id(dim::Integer, id_hi::UInt64, id_lo::UInt64, tags::AbstractString) -> Ptr{Cvoid}

Create a new index with specified dimension, ID, and tags.
"""
function t4a_index_new_with_id(dim::Integer, id_hi::UInt64, id_lo::UInt64, tags::AbstractString)
    return ccall(
        (:t4a_index_new_with_id, libpath()),
        Ptr{Cvoid},
        (Csize_t, UInt64, UInt64, Cstring),
        Csize_t(dim),
        id_hi,
        id_lo,
        isempty(tags) ? C_NULL : tags
    )
end

"""
    t4a_index_release(ptr::Ptr{Cvoid})

Release an index (called by finalizer).
"""
function t4a_index_release(ptr::Ptr{Cvoid})
    ptr == C_NULL && return
    ccall(
        (:t4a_index_release, libpath()),
        Cvoid,
        (Ptr{Cvoid},),
        ptr
    )
end

"""
    t4a_index_clone(ptr::Ptr{Cvoid}) -> Ptr{Cvoid}

Clone an index.
"""
function t4a_index_clone(ptr::Ptr{Cvoid})
    return ccall(
        (:t4a_index_clone, libpath()),
        Ptr{Cvoid},
        (Ptr{Cvoid},),
        ptr
    )
end

"""
    t4a_index_is_assigned(ptr::Ptr{Cvoid}) -> Bool

Check if an index pointer is valid.
"""
function t4a_index_is_assigned(ptr::Ptr{Cvoid})
    result = ccall(
        (:t4a_index_is_assigned, libpath()),
        Cint,
        (Ptr{Cvoid},),
        ptr
    )
    return result == 1
end

# ============================================================================
# Index accessors
# ============================================================================

"""
    t4a_index_dim(ptr::Ptr{Cvoid}, out_dim::Ref{Csize_t}) -> Cint

Get the dimension of an index.
"""
function t4a_index_dim(ptr::Ptr{Cvoid}, out_dim::Ref{Csize_t})
    return ccall(
        (:t4a_index_dim, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}),
        ptr,
        out_dim
    )
end

"""
    t4a_index_id_u128(ptr::Ptr{Cvoid}, out_hi::Ref{UInt64}, out_lo::Ref{UInt64}) -> Cint

Get the 128-bit ID of an index as two 64-bit values.
"""
function t4a_index_id_u128(ptr::Ptr{Cvoid}, out_hi::Ref{UInt64}, out_lo::Ref{UInt64})
    return ccall(
        (:t4a_index_id_u128, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{UInt64}, Ptr{UInt64}),
        ptr,
        out_hi,
        out_lo
    )
end

"""
    t4a_index_get_tags(ptr::Ptr{Cvoid}, buf, buf_len::Integer, out_len::Ref{Csize_t}) -> Cint

Get tags as a comma-separated string.
"""
function t4a_index_get_tags(ptr::Ptr{Cvoid}, buf, buf_len::Integer, out_len::Ref{Csize_t})
    return ccall(
        (:t4a_index_get_tags, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{UInt8}, Csize_t, Ptr{Csize_t}),
        ptr,
        buf,
        Csize_t(buf_len),
        out_len
    )
end

# ============================================================================
# Index modifiers
# ============================================================================

"""
    t4a_index_add_tag(ptr::Ptr{Cvoid}, tag::AbstractString) -> Cint

Add a tag to an index.
"""
function t4a_index_add_tag(ptr::Ptr{Cvoid}, tag::AbstractString)
    return ccall(
        (:t4a_index_add_tag, libpath()),
        Cint,
        (Ptr{Cvoid}, Cstring),
        ptr,
        tag
    )
end

"""
    t4a_index_set_tags_csv(ptr::Ptr{Cvoid}, tags::AbstractString) -> Cint

Set all tags from a comma-separated string.
"""
function t4a_index_set_tags_csv(ptr::Ptr{Cvoid}, tags::AbstractString)
    return ccall(
        (:t4a_index_set_tags_csv, libpath()),
        Cint,
        (Ptr{Cvoid}, Cstring),
        ptr,
        tags
    )
end

"""
    t4a_index_has_tag(ptr::Ptr{Cvoid}, tag::AbstractString) -> Cint

Check if an index has a specific tag.
Returns 1 if yes, 0 if no, negative on error.
"""
function t4a_index_has_tag(ptr::Ptr{Cvoid}, tag::AbstractString)
    return ccall(
        (:t4a_index_has_tag, libpath()),
        Cint,
        (Ptr{Cvoid}, Cstring),
        ptr,
        tag
    )
end

# ============================================================================
# Storage kind enum
# ============================================================================

# Storage kind enum (must match Rust side)
const STORAGE_DENSE_F64 = Cint(0)
const STORAGE_DENSE_C64 = Cint(1)
const STORAGE_DIAG_F64 = Cint(2)
const STORAGE_DIAG_C64 = Cint(3)

# ============================================================================
# Tensor lifecycle functions
# ============================================================================

"""
    t4a_tensor_release(ptr::Ptr{Cvoid})

Release a tensor (called by finalizer).
"""
function t4a_tensor_release(ptr::Ptr{Cvoid})
    ptr == C_NULL && return
    ccall(
        (:t4a_tensor_release, libpath()),
        Cvoid,
        (Ptr{Cvoid},),
        ptr
    )
end

"""
    t4a_tensor_clone(ptr::Ptr{Cvoid}) -> Ptr{Cvoid}

Clone a tensor.
"""
function t4a_tensor_clone(ptr::Ptr{Cvoid})
    return ccall(
        (:t4a_tensor_clone, libpath()),
        Ptr{Cvoid},
        (Ptr{Cvoid},),
        ptr
    )
end

"""
    t4a_tensor_is_assigned(ptr::Ptr{Cvoid}) -> Bool

Check if a tensor pointer is valid.
"""
function t4a_tensor_is_assigned(ptr::Ptr{Cvoid})
    result = ccall(
        (:t4a_tensor_is_assigned, libpath()),
        Cint,
        (Ptr{Cvoid},),
        ptr
    )
    return result == 1
end

# ============================================================================
# Tensor accessors
# ============================================================================

"""
    t4a_tensor_get_rank(ptr::Ptr{Cvoid}, out_rank::Ref{Csize_t}) -> Cint

Get the rank (number of indices) of a tensor.
"""
function t4a_tensor_get_rank(ptr::Ptr{Cvoid}, out_rank::Ref{Csize_t})
    return ccall(
        (:t4a_tensor_get_rank, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}),
        ptr,
        out_rank
    )
end

"""
    t4a_tensor_get_dims(ptr::Ptr{Cvoid}, out_dims::Vector{Csize_t}, buf_len::Integer) -> Cint

Get the dimensions of a tensor.
"""
function t4a_tensor_get_dims(ptr::Ptr{Cvoid}, out_dims::Vector{Csize_t}, buf_len::Integer)
    return ccall(
        (:t4a_tensor_get_dims, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}, Csize_t),
        ptr,
        out_dims,
        Csize_t(buf_len)
    )
end

"""
    t4a_tensor_get_indices(ptr::Ptr{Cvoid}, out_indices::Vector{Ptr{Cvoid}}, buf_len::Integer) -> Cint

Get the indices of a tensor as cloned t4a_index handles.
Caller is responsible for releasing the returned indices.
"""
function t4a_tensor_get_indices(ptr::Ptr{Cvoid}, out_indices::Vector{Ptr{Cvoid}}, buf_len::Integer)
    return ccall(
        (:t4a_tensor_get_indices, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Ptr{Cvoid}}, Csize_t),
        ptr,
        out_indices,
        Csize_t(buf_len)
    )
end

"""
    t4a_tensor_get_storage_kind(ptr::Ptr{Cvoid}, out_kind::Ref{Cint}) -> Cint

Get the storage kind of a tensor.
"""
function t4a_tensor_get_storage_kind(ptr::Ptr{Cvoid}, out_kind::Ref{Cint})
    return ccall(
        (:t4a_tensor_get_storage_kind, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Cint}),
        ptr,
        out_kind
    )
end

"""
    t4a_tensor_get_data_f64(ptr::Ptr{Cvoid}, buf::Union{Ptr{Cdouble},Nothing}, buf_len::Integer, out_len::Ref{Csize_t}) -> Cint

Get dense f64 data from a tensor in row-major order.
If buf is C_NULL, only out_len is written (to query required length).
"""
function t4a_tensor_get_data_f64(ptr::Ptr{Cvoid}, buf, buf_len::Integer, out_len::Ref{Csize_t})
    return ccall(
        (:t4a_tensor_get_data_f64, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Cdouble}, Csize_t, Ptr{Csize_t}),
        ptr,
        buf === nothing ? C_NULL : buf,
        Csize_t(buf_len),
        out_len
    )
end

"""
    t4a_tensor_get_data_c64(ptr::Ptr{Cvoid}, buf_re, buf_im, buf_len::Integer, out_len::Ref{Csize_t}) -> Cint

Get dense complex64 data from a tensor in row-major order.
If buf_re or buf_im is C_NULL, only out_len is written (to query required length).
"""
function t4a_tensor_get_data_c64(ptr::Ptr{Cvoid}, buf_re, buf_im, buf_len::Integer, out_len::Ref{Csize_t})
    return ccall(
        (:t4a_tensor_get_data_c64, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Csize_t, Ptr{Csize_t}),
        ptr,
        buf_re === nothing ? C_NULL : buf_re,
        buf_im === nothing ? C_NULL : buf_im,
        Csize_t(buf_len),
        out_len
    )
end

# ============================================================================
# Tensor constructors
# ============================================================================

"""
    t4a_tensor_new_dense_f64(rank::Integer, index_ptrs::Vector{Ptr{Cvoid}}, dims::Vector{Csize_t}, data::Vector{Cdouble}) -> Ptr{Cvoid}

Create a new dense f64 tensor from indices and data in row-major order.
"""
function t4a_tensor_new_dense_f64(rank::Integer, index_ptrs::Vector{Ptr{Cvoid}}, dims::Vector{Csize_t}, data::Vector{Cdouble})
    return ccall(
        (:t4a_tensor_new_dense_f64, libpath()),
        Ptr{Cvoid},
        (Csize_t, Ptr{Ptr{Cvoid}}, Ptr{Csize_t}, Ptr{Cdouble}, Csize_t),
        Csize_t(rank),
        index_ptrs,
        dims,
        data,
        Csize_t(length(data))
    )
end

"""
    t4a_tensor_new_dense_c64(rank::Integer, index_ptrs::Vector{Ptr{Cvoid}}, dims::Vector{Csize_t}, data_re::Vector{Cdouble}, data_im::Vector{Cdouble}) -> Ptr{Cvoid}

Create a new dense complex64 tensor from indices and real/imag data in row-major order.
"""
function t4a_tensor_new_dense_c64(rank::Integer, index_ptrs::Vector{Ptr{Cvoid}}, dims::Vector{Csize_t}, data_re::Vector{Cdouble}, data_im::Vector{Cdouble})
    @assert length(data_re) == length(data_im) "Real and imaginary data must have same length"
    return ccall(
        (:t4a_tensor_new_dense_c64, libpath()),
        Ptr{Cvoid},
        (Csize_t, Ptr{Ptr{Cvoid}}, Ptr{Csize_t}, Ptr{Cdouble}, Ptr{Cdouble}, Csize_t),
        Csize_t(rank),
        index_ptrs,
        dims,
        data_re,
        data_im,
        Csize_t(length(data_re))
    )
end

end # module C_API
