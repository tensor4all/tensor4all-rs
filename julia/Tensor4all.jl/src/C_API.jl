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

# ============================================================================
# TensorTrain lifecycle functions
# ============================================================================

"""
    t4a_tt_new_empty() -> Ptr{Cvoid}

Create an empty tensor train.
"""
function t4a_tt_new_empty()
    return ccall(
        (:t4a_tt_new_empty, libpath()),
        Ptr{Cvoid},
        ()
    )
end

"""
    t4a_tt_new(tensor_ptrs::Vector{Ptr{Cvoid}}, num_tensors::Integer) -> Ptr{Cvoid}

Create a tensor train from an array of tensors.
"""
function t4a_tt_new(tensor_ptrs::Vector{Ptr{Cvoid}}, num_tensors::Integer)
    return ccall(
        (:t4a_tt_new, libpath()),
        Ptr{Cvoid},
        (Ptr{Ptr{Cvoid}}, Csize_t),
        tensor_ptrs,
        Csize_t(num_tensors)
    )
end

"""
    t4a_tensortrain_release(ptr::Ptr{Cvoid})

Release a tensor train (called by finalizer).
"""
function t4a_tensortrain_release(ptr::Ptr{Cvoid})
    ptr == C_NULL && return
    ccall(
        (:t4a_tensortrain_release, libpath()),
        Cvoid,
        (Ptr{Cvoid},),
        ptr
    )
end

"""
    t4a_tensortrain_clone(ptr::Ptr{Cvoid}) -> Ptr{Cvoid}

Clone a tensor train.
"""
function t4a_tensortrain_clone(ptr::Ptr{Cvoid})
    return ccall(
        (:t4a_tensortrain_clone, libpath()),
        Ptr{Cvoid},
        (Ptr{Cvoid},),
        ptr
    )
end

# ============================================================================
# TensorTrain accessors
# ============================================================================

"""
    t4a_tt_len(ptr::Ptr{Cvoid}, out_len::Ref{Csize_t}) -> Cint

Get the number of sites in the tensor train.
"""
function t4a_tt_len(ptr::Ptr{Cvoid}, out_len::Ref{Csize_t})
    return ccall(
        (:t4a_tt_len, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}),
        ptr,
        out_len
    )
end

"""
    t4a_tt_is_empty(ptr::Ptr{Cvoid}) -> Cint

Check if the tensor train is empty.
Returns 1 if empty, 0 if not, negative on error.
"""
function t4a_tt_is_empty(ptr::Ptr{Cvoid})
    return ccall(
        (:t4a_tt_is_empty, libpath()),
        Cint,
        (Ptr{Cvoid},),
        ptr
    )
end

"""
    t4a_tt_tensor(ptr::Ptr{Cvoid}, site::Integer) -> Ptr{Cvoid}

Get the tensor at a specific site (0-indexed).
Returns a new tensor handle that the caller owns.
"""
function t4a_tt_tensor(ptr::Ptr{Cvoid}, site::Integer)
    return ccall(
        (:t4a_tt_tensor, libpath()),
        Ptr{Cvoid},
        (Ptr{Cvoid}, Csize_t),
        ptr,
        Csize_t(site)
    )
end

"""
    t4a_tt_set_tensor(ptr::Ptr{Cvoid}, site::Integer, tensor::Ptr{Cvoid}) -> Cint

Set the tensor at a specific site (0-indexed).
The tensor is cloned into the tensor train. This invalidates orthogonality.
"""
function t4a_tt_set_tensor(ptr::Ptr{Cvoid}, site::Integer, tensor::Ptr{Cvoid})
    return ccall(
        (:t4a_tt_set_tensor, libpath()),
        Cint,
        (Ptr{Cvoid}, Csize_t, Ptr{Cvoid}),
        ptr,
        Csize_t(site),
        tensor
    )
end

"""
    t4a_tt_bond_dims(ptr::Ptr{Cvoid}, out_dims::Vector{Csize_t}, buf_len::Integer) -> Cint

Get the bond dimensions of the tensor train.
"""
function t4a_tt_bond_dims(ptr::Ptr{Cvoid}, out_dims::Vector{Csize_t}, buf_len::Integer)
    return ccall(
        (:t4a_tt_bond_dims, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}, Csize_t),
        ptr,
        out_dims,
        Csize_t(buf_len)
    )
end

"""
    t4a_tt_maxbonddim(ptr::Ptr{Cvoid}, out_max::Ref{Csize_t}) -> Cint

Get the maximum bond dimension of the tensor train.
"""
function t4a_tt_maxbonddim(ptr::Ptr{Cvoid}, out_max::Ref{Csize_t})
    return ccall(
        (:t4a_tt_maxbonddim, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}),
        ptr,
        out_max
    )
end

"""
    t4a_tt_linkind(ptr::Ptr{Cvoid}, site::Integer) -> Ptr{Cvoid}

Get the link index between sites `site` and `site+1` (0-indexed).
Returns a new index handle that the caller owns, or NULL if no link exists.
"""
function t4a_tt_linkind(ptr::Ptr{Cvoid}, site::Integer)
    return ccall(
        (:t4a_tt_linkind, libpath()),
        Ptr{Cvoid},
        (Ptr{Cvoid}, Csize_t),
        ptr,
        Csize_t(site)
    )
end

# ============================================================================
# TensorTrain orthogonality
# ============================================================================

"""
    t4a_tt_isortho(ptr::Ptr{Cvoid}) -> Cint

Check if the tensor train has a single orthogonality center.
Returns 1 if yes, 0 if no, negative on error.
"""
function t4a_tt_isortho(ptr::Ptr{Cvoid})
    return ccall(
        (:t4a_tt_isortho, libpath()),
        Cint,
        (Ptr{Cvoid},),
        ptr
    )
end

"""
    t4a_tt_orthocenter(ptr::Ptr{Cvoid}, out_center::Ref{Csize_t}) -> Cint

Get the orthogonality center (0-indexed).
"""
function t4a_tt_orthocenter(ptr::Ptr{Cvoid}, out_center::Ref{Csize_t})
    return ccall(
        (:t4a_tt_orthocenter, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}),
        ptr,
        out_center
    )
end

"""
    t4a_tt_llim(ptr::Ptr{Cvoid}, out_llim::Ref{Cint}) -> Cint

Get the left orthogonality limit.
"""
function t4a_tt_llim(ptr::Ptr{Cvoid}, out_llim::Ref{Cint})
    return ccall(
        (:t4a_tt_llim, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Cint}),
        ptr,
        out_llim
    )
end

"""
    t4a_tt_rlim(ptr::Ptr{Cvoid}, out_rlim::Ref{Cint}) -> Cint

Get the right orthogonality limit.
"""
function t4a_tt_rlim(ptr::Ptr{Cvoid}, out_rlim::Ref{Cint})
    return ccall(
        (:t4a_tt_rlim, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Cint}),
        ptr,
        out_rlim
    )
end

"""
    t4a_tt_canonical_form(ptr::Ptr{Cvoid}, out_form::Ref{Cint}) -> Cint

Get the canonical form of the tensor train.
"""
function t4a_tt_canonical_form(ptr::Ptr{Cvoid}, out_form::Ref{Cint})
    return ccall(
        (:t4a_tt_canonical_form, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Cint}),
        ptr,
        out_form
    )
end

# ============================================================================
# TensorTrain operations
# ============================================================================

"""
    t4a_tt_orthogonalize(ptr::Ptr{Cvoid}, site::Integer) -> Cint

Orthogonalize the tensor train to have orthogonality center at the given site.
Uses QR decomposition (Unitary canonical form).
"""
function t4a_tt_orthogonalize(ptr::Ptr{Cvoid}, site::Integer)
    return ccall(
        (:t4a_tt_orthogonalize, libpath()),
        Cint,
        (Ptr{Cvoid}, Csize_t),
        ptr,
        Csize_t(site)
    )
end

"""
    t4a_tt_orthogonalize_with(ptr::Ptr{Cvoid}, site::Integer, form::Cint) -> Cint

Orthogonalize the tensor train with a specific canonical form.
form: 0=Unitary, 1=LU, 2=CI
"""
function t4a_tt_orthogonalize_with(ptr::Ptr{Cvoid}, site::Integer, form::Integer)
    return ccall(
        (:t4a_tt_orthogonalize_with, libpath()),
        Cint,
        (Ptr{Cvoid}, Csize_t, Cint),
        ptr,
        Csize_t(site),
        Cint(form)
    )
end

"""
    t4a_tt_truncate(ptr::Ptr{Cvoid}, rtol::Float64, max_rank::Integer) -> Cint

Truncate the tensor train bond dimensions.
rtol: relative tolerance (use 0.0 for default)
max_rank: maximum bond dimension (use 0 for no limit)
"""
function t4a_tt_truncate(ptr::Ptr{Cvoid}, rtol::Float64, max_rank::Integer)
    return ccall(
        (:t4a_tt_truncate, libpath()),
        Cint,
        (Ptr{Cvoid}, Cdouble, Csize_t),
        ptr,
        rtol,
        Csize_t(max_rank)
    )
end

"""
    t4a_tt_norm(ptr::Ptr{Cvoid}, out_norm::Ref{Cdouble}) -> Cint

Compute the norm of the tensor train.
"""
function t4a_tt_norm(ptr::Ptr{Cvoid}, out_norm::Ref{Cdouble})
    return ccall(
        (:t4a_tt_norm, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Cdouble}),
        ptr,
        out_norm
    )
end

"""
    t4a_tt_inner(ptr1::Ptr{Cvoid}, ptr2::Ptr{Cvoid}, out_re::Ref{Cdouble}, out_im::Ref{Cdouble}) -> Cint

Compute the inner product of two tensor trains.
"""
function t4a_tt_inner(ptr1::Ptr{Cvoid}, ptr2::Ptr{Cvoid}, out_re::Ref{Cdouble}, out_im::Ref{Cdouble})
    return ccall(
        (:t4a_tt_inner, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}),
        ptr1,
        ptr2,
        out_re,
        out_im
    )
end

"""
    t4a_tt_contract(ptr1::Ptr{Cvoid}, ptr2::Ptr{Cvoid}, method::Integer, max_rank::Integer, rtol::Float64, nhalfsweeps::Integer) -> Ptr{Cvoid}

Contract two tensor trains.
method: 0=Zipup, 1=Fit
max_rank: maximum bond dimension (use 0 for no limit)
rtol: relative tolerance (use 0.0 for default)
nhalfsweeps: number of half-sweeps for Fit method (must be a multiple of 2)
"""
function t4a_tt_contract(ptr1::Ptr{Cvoid}, ptr2::Ptr{Cvoid}, method::Integer, max_rank::Integer, rtol::Float64, nhalfsweeps::Integer)
    return ccall(
        (:t4a_tt_contract, libpath()),
        Ptr{Cvoid},
        (Ptr{Cvoid}, Ptr{Cvoid}, Cint, Csize_t, Cdouble, Csize_t),
        ptr1,
        ptr2,
        Cint(method),
        Csize_t(max_rank),
        rtol,
        Csize_t(nhalfsweeps)
    )
end

# ============================================================================
# SimpleTT (simple tensor train) lifecycle functions
# ============================================================================

"""
    t4a_simplett_f64_release(ptr::Ptr{Cvoid})

Release a SimpleTT tensor train.
"""
function t4a_simplett_f64_release(ptr::Ptr{Cvoid})
    ptr == C_NULL && return
    ccall(
        (:t4a_simplett_f64_release, libpath()),
        Cvoid,
        (Ptr{Cvoid},),
        ptr
    )
end

"""
    t4a_simplett_f64_clone(ptr::Ptr{Cvoid}) -> Ptr{Cvoid}

Clone a SimpleTT tensor train.
"""
function t4a_simplett_f64_clone(ptr::Ptr{Cvoid})
    return ccall(
        (:t4a_simplett_f64_clone, libpath()),
        Ptr{Cvoid},
        (Ptr{Cvoid},),
        ptr
    )
end

# ============================================================================
# SimpleTT constructors
# ============================================================================

"""
    t4a_simplett_f64_constant(site_dims::Vector{Csize_t}, value::Float64) -> Ptr{Cvoid}

Create a constant SimpleTT tensor train.
"""
function t4a_simplett_f64_constant(site_dims::Vector{Csize_t}, value::Float64)
    return ccall(
        (:t4a_simplett_f64_constant, libpath()),
        Ptr{Cvoid},
        (Ptr{Csize_t}, Csize_t, Cdouble),
        site_dims,
        Csize_t(length(site_dims)),
        value
    )
end

"""
    t4a_simplett_f64_zeros(site_dims::Vector{Csize_t}) -> Ptr{Cvoid}

Create a zero SimpleTT tensor train.
"""
function t4a_simplett_f64_zeros(site_dims::Vector{Csize_t})
    return ccall(
        (:t4a_simplett_f64_zeros, libpath()),
        Ptr{Cvoid},
        (Ptr{Csize_t}, Csize_t),
        site_dims,
        Csize_t(length(site_dims))
    )
end

# ============================================================================
# SimpleTT accessors
# ============================================================================

"""
    t4a_simplett_f64_len(ptr::Ptr{Cvoid}, out_len::Ref{Csize_t}) -> Cint

Get the number of sites.
"""
function t4a_simplett_f64_len(ptr::Ptr{Cvoid}, out_len::Ref{Csize_t})
    return ccall(
        (:t4a_simplett_f64_len, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}),
        ptr,
        out_len
    )
end

"""
    t4a_simplett_f64_site_dims(ptr::Ptr{Cvoid}, out_dims::Vector{Csize_t}) -> Cint

Get the site dimensions.
"""
function t4a_simplett_f64_site_dims(ptr::Ptr{Cvoid}, out_dims::Vector{Csize_t})
    return ccall(
        (:t4a_simplett_f64_site_dims, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}, Csize_t),
        ptr,
        out_dims,
        Csize_t(length(out_dims))
    )
end

"""
    t4a_simplett_f64_link_dims(ptr::Ptr{Cvoid}, out_dims::Vector{Csize_t}) -> Cint

Get the link (bond) dimensions.
"""
function t4a_simplett_f64_link_dims(ptr::Ptr{Cvoid}, out_dims::Vector{Csize_t})
    return ccall(
        (:t4a_simplett_f64_link_dims, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}, Csize_t),
        ptr,
        out_dims,
        Csize_t(length(out_dims))
    )
end

"""
    t4a_simplett_f64_rank(ptr::Ptr{Cvoid}, out_rank::Ref{Csize_t}) -> Cint

Get the maximum bond dimension (rank).
"""
function t4a_simplett_f64_rank(ptr::Ptr{Cvoid}, out_rank::Ref{Csize_t})
    return ccall(
        (:t4a_simplett_f64_rank, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}),
        ptr,
        out_rank
    )
end

"""
    t4a_simplett_f64_evaluate(ptr::Ptr{Cvoid}, indices::Vector{Csize_t}, out_value::Ref{Cdouble}) -> Cint

Evaluate the tensor train at a given multi-index.
"""
function t4a_simplett_f64_evaluate(ptr::Ptr{Cvoid}, indices::Vector{Csize_t}, out_value::Ref{Cdouble})
    return ccall(
        (:t4a_simplett_f64_evaluate, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}, Csize_t, Ptr{Cdouble}),
        ptr,
        indices,
        Csize_t(length(indices)),
        out_value
    )
end

"""
    t4a_simplett_f64_sum(ptr::Ptr{Cvoid}, out_value::Ref{Cdouble}) -> Cint

Compute the sum over all indices.
"""
function t4a_simplett_f64_sum(ptr::Ptr{Cvoid}, out_value::Ref{Cdouble})
    return ccall(
        (:t4a_simplett_f64_sum, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Cdouble}),
        ptr,
        out_value
    )
end

"""
    t4a_simplett_f64_norm(ptr::Ptr{Cvoid}, out_value::Ref{Cdouble}) -> Cint

Compute the Frobenius norm.
"""
function t4a_simplett_f64_norm(ptr::Ptr{Cvoid}, out_value::Ref{Cdouble})
    return ccall(
        (:t4a_simplett_f64_norm, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Cdouble}),
        ptr,
        out_value
    )
end

"""
    t4a_simplett_f64_site_tensor(ptr::Ptr{Cvoid}, site::Integer, out_data::Vector{Cdouble},
                                  out_left_dim::Ref{Csize_t}, out_site_dim::Ref{Csize_t},
                                  out_right_dim::Ref{Csize_t}) -> Cint

Get site tensor data at a specific site.
"""
function t4a_simplett_f64_site_tensor(
    ptr::Ptr{Cvoid},
    site::Integer,
    out_data::Vector{Cdouble},
    out_left_dim::Ref{Csize_t},
    out_site_dim::Ref{Csize_t},
    out_right_dim::Ref{Csize_t}
)
    return ccall(
        (:t4a_simplett_f64_site_tensor, libpath()),
        Cint,
        (Ptr{Cvoid}, Csize_t, Ptr{Cdouble}, Csize_t, Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Csize_t}),
        ptr,
        Csize_t(site),
        out_data,
        Csize_t(length(out_data)),
        out_left_dim,
        out_site_dim,
        out_right_dim
    )
end

# ============================================================================
# TensorCI2 lifecycle functions
# ============================================================================

"""
    t4a_tci2_f64_release(ptr::Ptr{Cvoid})

Release a TensorCI2 object.
"""
function t4a_tci2_f64_release(ptr::Ptr{Cvoid})
    ptr == C_NULL && return
    ccall(
        (:t4a_tci2_f64_release, libpath()),
        Cvoid,
        (Ptr{Cvoid},),
        ptr
    )
end

"""
    t4a_tci2_f64_new(local_dims::Vector{Csize_t}) -> Ptr{Cvoid}

Create a new TensorCI2 object.
"""
function t4a_tci2_f64_new(local_dims::Vector{Csize_t})
    return ccall(
        (:t4a_tci2_f64_new, libpath()),
        Ptr{Cvoid},
        (Ptr{Csize_t}, Csize_t),
        local_dims,
        Csize_t(length(local_dims))
    )
end

# ============================================================================
# TensorCI2 accessors
# ============================================================================

"""
    t4a_tci2_f64_len(ptr::Ptr{Cvoid}, out_len::Ref{Csize_t}) -> Cint

Get the number of sites.
"""
function t4a_tci2_f64_len(ptr::Ptr{Cvoid}, out_len::Ref{Csize_t})
    return ccall(
        (:t4a_tci2_f64_len, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}),
        ptr,
        out_len
    )
end

"""
    t4a_tci2_f64_rank(ptr::Ptr{Cvoid}, out_rank::Ref{Csize_t}) -> Cint

Get the current rank (maximum bond dimension).
"""
function t4a_tci2_f64_rank(ptr::Ptr{Cvoid}, out_rank::Ref{Csize_t})
    return ccall(
        (:t4a_tci2_f64_rank, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}),
        ptr,
        out_rank
    )
end

"""
    t4a_tci2_f64_link_dims(ptr::Ptr{Cvoid}, out_dims::Vector{Csize_t}) -> Cint

Get the link (bond) dimensions.
"""
function t4a_tci2_f64_link_dims(ptr::Ptr{Cvoid}, out_dims::Vector{Csize_t})
    return ccall(
        (:t4a_tci2_f64_link_dims, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}, Csize_t),
        ptr,
        out_dims,
        Csize_t(length(out_dims))
    )
end

"""
    t4a_tci2_f64_max_sample_value(ptr::Ptr{Cvoid}, out_value::Ref{Cdouble}) -> Cint

Get the maximum sample value encountered.
"""
function t4a_tci2_f64_max_sample_value(ptr::Ptr{Cvoid}, out_value::Ref{Cdouble})
    return ccall(
        (:t4a_tci2_f64_max_sample_value, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Cdouble}),
        ptr,
        out_value
    )
end

"""
    t4a_tci2_f64_max_bond_error(ptr::Ptr{Cvoid}, out_value::Ref{Cdouble}) -> Cint

Get the maximum bond error from the last sweep.
"""
function t4a_tci2_f64_max_bond_error(ptr::Ptr{Cvoid}, out_value::Ref{Cdouble})
    return ccall(
        (:t4a_tci2_f64_max_bond_error, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Cdouble}),
        ptr,
        out_value
    )
end

# ============================================================================
# TensorCI2 pivot operations
# ============================================================================

"""
    t4a_tci2_f64_add_global_pivots(ptr::Ptr{Cvoid}, pivots::Vector{Csize_t}, n_pivots::Integer, n_sites::Integer) -> Cint

Add global pivots to the TCI. Pivots are stored as a flat array.
"""
function t4a_tci2_f64_add_global_pivots(ptr::Ptr{Cvoid}, pivots::Vector{Csize_t}, n_pivots::Integer, n_sites::Integer)
    return ccall(
        (:t4a_tci2_f64_add_global_pivots, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}, Csize_t, Csize_t),
        ptr,
        pivots,
        Csize_t(n_pivots),
        Csize_t(n_sites)
    )
end

# ============================================================================
# TensorCI2 conversion
# ============================================================================

"""
    t4a_tci2_f64_to_tensor_train(ptr::Ptr{Cvoid}) -> Ptr{Cvoid}

Convert a TCI to a SimpleTT TensorTrain.
"""
function t4a_tci2_f64_to_tensor_train(ptr::Ptr{Cvoid})
    return ccall(
        (:t4a_tci2_f64_to_tensor_train, libpath()),
        Ptr{Cvoid},
        (Ptr{Cvoid},),
        ptr
    )
end

# ============================================================================
# TensorCI2 high-level crossinterpolate2
# ============================================================================

# Callback type for evaluation function
# signature: (indices_ptr, n_indices, result_ptr, user_data) -> status
const EvalCallback = Ptr{Cvoid}

"""
    t4a_crossinterpolate2_f64(local_dims, initial_pivots, n_initial_pivots,
                               eval_fn, user_data, tolerance, max_bonddim, max_iter,
                               out_tci, out_final_error) -> Cint

Perform cross interpolation of a function.
"""
function t4a_crossinterpolate2_f64(
    local_dims::Vector{Csize_t},
    initial_pivots::Vector{Csize_t},  # flat array
    n_initial_pivots::Integer,
    eval_fn::Ptr{Cvoid},
    user_data::Ptr{Cvoid},
    tolerance::Float64,
    max_bonddim::Integer,
    max_iter::Integer,
    out_tci::Ref{Ptr{Cvoid}},
    out_final_error::Ref{Cdouble}
)
    return ccall(
        (:t4a_crossinterpolate2_f64, libpath()),
        Cint,
        (Ptr{Csize_t}, Csize_t, Ptr{Csize_t}, Csize_t, Ptr{Cvoid}, Ptr{Cvoid},
         Cdouble, Csize_t, Csize_t, Ptr{Ptr{Cvoid}}, Ptr{Cdouble}),
        local_dims,
        Csize_t(length(local_dims)),
        isempty(initial_pivots) ? C_NULL : initial_pivots,
        Csize_t(n_initial_pivots),
        eval_fn,
        user_data,
        tolerance,
        Csize_t(max_bonddim),
        Csize_t(max_iter),
        out_tci,
        out_final_error
    )
end

end # module C_API
