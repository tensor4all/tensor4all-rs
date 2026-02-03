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
    t4a_index_new_with_id(dim::Integer, id::UInt64, tags::AbstractString) -> Ptr{Cvoid}

Create a new index with specified dimension, ID, and tags.
"""
function t4a_index_new_with_id(dim::Integer, id::UInt64, tags::AbstractString)
    return ccall(
        (:t4a_index_new_with_id, libpath()),
        Ptr{Cvoid},
        (Csize_t, UInt64, Cstring),
        Csize_t(dim),
        id,
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
    t4a_index_id(ptr::Ptr{Cvoid}, out_id::Ref{UInt64}) -> Cint

Get the 64-bit ID of an index (compatible with ITensors.jl's IDType = UInt64).
"""
function t4a_index_id(ptr::Ptr{Cvoid}, out_id::Ref{UInt64})
    return ccall(
        (:t4a_index_id, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{UInt64}),
        ptr,
        out_id
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

"""
    t4a_tensor_onehot(rank::Integer, index_ptrs::Vector{Ptr{Cvoid}}, vals::Vector{Csize_t}) -> Ptr{Cvoid}

Create a one-hot tensor with value 1.0 at the specified positions (0-indexed).
"""
function t4a_tensor_onehot(rank::Integer, index_ptrs::Vector{Ptr{Cvoid}}, vals::Vector{Csize_t})
    return ccall(
        (:t4a_tensor_onehot, libpath()),
        Ptr{Cvoid},
        (Csize_t, Ptr{Ptr{Cvoid}}, Ptr{Csize_t}, Csize_t),
        Csize_t(rank),
        index_ptrs,
        vals,
        Csize_t(rank)
    )
end

# ============================================================================
# TreeTN lifecycle functions
# ============================================================================

"""
    t4a_treetn_release(ptr::Ptr{Cvoid})

Release a tree tensor network (called by finalizer).
"""
function t4a_treetn_release(ptr::Ptr{Cvoid})
    ptr == C_NULL && return
    ccall(
        (:t4a_treetn_release, libpath()),
        Cvoid,
        (Ptr{Cvoid},),
        ptr
    )
end

"""
    t4a_treetn_clone(ptr::Ptr{Cvoid}) -> Ptr{Cvoid}

Clone a tree tensor network.
"""
function t4a_treetn_clone(ptr::Ptr{Cvoid})
    return ccall(
        (:t4a_treetn_clone, libpath()),
        Ptr{Cvoid},
        (Ptr{Cvoid},),
        ptr
    )
end

"""
    t4a_treetn_is_assigned(ptr::Ptr{Cvoid}) -> Bool

Check if a tree tensor network pointer is valid.
"""
function t4a_treetn_is_assigned(ptr::Ptr{Cvoid})
    result = ccall(
        (:t4a_treetn_is_assigned, libpath()),
        Cint,
        (Ptr{Cvoid},),
        ptr
    )
    return result == 1
end

# ============================================================================
# TreeTN constructors
# ============================================================================

"""
    t4a_treetn_new(tensors::Ptr{Ptr{Cvoid}}, n_tensors::Integer, out::Ref{Ptr{Cvoid}}) -> Cint

Create a tree tensor network from an array of tensors.
Node names are assigned as 0, 1, ..., n_tensors-1.
"""
function t4a_treetn_new(tensors, n_tensors::Integer, out)
    return ccall(
        (:t4a_treetn_new, libpath()),
        Cint,
        (Ptr{Ptr{Cvoid}}, Csize_t, Ptr{Ptr{Cvoid}}),
        tensors,
        Csize_t(n_tensors),
        out
    )
end

# ============================================================================
# TreeTN accessors
# ============================================================================

"""
    t4a_treetn_num_vertices(ptr::Ptr{Cvoid}, out::Ref{Csize_t}) -> Cint

Get the number of vertices (nodes) in the tree tensor network.
"""
function t4a_treetn_num_vertices(ptr::Ptr{Cvoid}, out::Ref{Csize_t})
    return ccall(
        (:t4a_treetn_num_vertices, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}),
        ptr,
        out
    )
end

"""
    t4a_treetn_num_edges(ptr::Ptr{Cvoid}, out::Ref{Csize_t}) -> Cint

Get the number of edges (bonds) in the tree tensor network.
"""
function t4a_treetn_num_edges(ptr::Ptr{Cvoid}, out::Ref{Csize_t})
    return ccall(
        (:t4a_treetn_num_edges, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}),
        ptr,
        out
    )
end

"""
    t4a_treetn_tensor(ptr::Ptr{Cvoid}, vertex::Integer, out::Ref{Ptr{Cvoid}}) -> Cint

Get the tensor at a specific vertex (0-indexed).
"""
function t4a_treetn_tensor(ptr::Ptr{Cvoid}, vertex::Integer, out)
    return ccall(
        (:t4a_treetn_tensor, libpath()),
        Cint,
        (Ptr{Cvoid}, Csize_t, Ptr{Ptr{Cvoid}}),
        ptr,
        Csize_t(vertex),
        out
    )
end

"""
    t4a_treetn_set_tensor(ptr::Ptr{Cvoid}, vertex::Integer, tensor::Ptr{Cvoid}) -> Cint

Set the tensor at a specific vertex (0-indexed).
"""
function t4a_treetn_set_tensor(ptr::Ptr{Cvoid}, vertex::Integer, tensor::Ptr{Cvoid})
    return ccall(
        (:t4a_treetn_set_tensor, libpath()),
        Cint,
        (Ptr{Cvoid}, Csize_t, Ptr{Cvoid}),
        ptr,
        Csize_t(vertex),
        tensor
    )
end

"""
    t4a_treetn_neighbors(ptr::Ptr{Cvoid}, vertex::Integer, out_buf, buf_size::Integer, n_out::Ref{Csize_t}) -> Cint

Get the neighbors of a vertex.
"""
function t4a_treetn_neighbors(ptr::Ptr{Cvoid}, vertex::Integer, out_buf, buf_size::Integer, n_out::Ref{Csize_t})
    return ccall(
        (:t4a_treetn_neighbors, libpath()),
        Cint,
        (Ptr{Cvoid}, Csize_t, Ptr{Csize_t}, Csize_t, Ptr{Csize_t}),
        ptr,
        Csize_t(vertex),
        out_buf,
        Csize_t(buf_size),
        n_out
    )
end

"""
    t4a_treetn_linkind(ptr::Ptr{Cvoid}, v1::Integer, v2::Integer, out::Ref{Ptr{Cvoid}}) -> Cint

Get the link (bond) index on the edge between two vertices.
"""
function t4a_treetn_linkind(ptr::Ptr{Cvoid}, v1::Integer, v2::Integer, out)
    return ccall(
        (:t4a_treetn_linkind, libpath()),
        Cint,
        (Ptr{Cvoid}, Csize_t, Csize_t, Ptr{Ptr{Cvoid}}),
        ptr,
        Csize_t(v1),
        Csize_t(v2),
        out
    )
end

"""
    t4a_treetn_siteinds(ptr::Ptr{Cvoid}, vertex::Integer, out_buf, buf_size::Integer, n_out::Ref{Csize_t}) -> Cint

Get the site (physical) indices at a vertex.
"""
function t4a_treetn_siteinds(ptr::Ptr{Cvoid}, vertex::Integer, out_buf, buf_size::Integer, n_out::Ref{Csize_t})
    return ccall(
        (:t4a_treetn_siteinds, libpath()),
        Cint,
        (Ptr{Cvoid}, Csize_t, Ptr{Ptr{Cvoid}}, Csize_t, Ptr{Csize_t}),
        ptr,
        Csize_t(vertex),
        out_buf,
        Csize_t(buf_size),
        n_out
    )
end

"""
    t4a_treetn_bond_dim(ptr::Ptr{Cvoid}, v1::Integer, v2::Integer, out::Ref{Csize_t}) -> Cint

Get the bond dimension on the edge between two vertices.
"""
function t4a_treetn_bond_dim(ptr::Ptr{Cvoid}, v1::Integer, v2::Integer, out::Ref{Csize_t})
    return ccall(
        (:t4a_treetn_bond_dim, libpath()),
        Cint,
        (Ptr{Cvoid}, Csize_t, Csize_t, Ptr{Csize_t}),
        ptr,
        Csize_t(v1),
        Csize_t(v2),
        out
    )
end

# ============================================================================
# TreeTN MPS convenience functions
# ============================================================================

"""
    t4a_treetn_linkind_at(ptr::Ptr{Cvoid}, i::Integer, out::Ref{Ptr{Cvoid}}) -> Cint

Get the link index between vertex i and i+1 (MPS convention, 0-indexed).
"""
function t4a_treetn_linkind_at(ptr::Ptr{Cvoid}, i::Integer, out)
    return ccall(
        (:t4a_treetn_linkind_at, libpath()),
        Cint,
        (Ptr{Cvoid}, Csize_t, Ptr{Ptr{Cvoid}}),
        ptr,
        Csize_t(i),
        out
    )
end

"""
    t4a_treetn_bond_dim_at(ptr::Ptr{Cvoid}, i::Integer, out::Ref{Csize_t}) -> Cint

Get the bond dimension between vertex i and i+1 (MPS convention, 0-indexed).
"""
function t4a_treetn_bond_dim_at(ptr::Ptr{Cvoid}, i::Integer, out::Ref{Csize_t})
    return ccall(
        (:t4a_treetn_bond_dim_at, libpath()),
        Cint,
        (Ptr{Cvoid}, Csize_t, Ptr{Csize_t}),
        ptr,
        Csize_t(i),
        out
    )
end

"""
    t4a_treetn_bond_dims(ptr::Ptr{Cvoid}, out, n::Integer) -> Cint

Get all bond dimensions for an MPS-like TreeTN (vertices 0, 1, ..., n-1).
Writes n-1 bond dimensions.
"""
function t4a_treetn_bond_dims(ptr::Ptr{Cvoid}, out, n::Integer)
    return ccall(
        (:t4a_treetn_bond_dims, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}, Csize_t),
        ptr,
        out,
        Csize_t(n)
    )
end

"""
    t4a_treetn_maxbonddim(ptr::Ptr{Cvoid}, out::Ref{Csize_t}) -> Cint

Get the maximum bond dimension of an MPS-like TreeTN.
"""
function t4a_treetn_maxbonddim(ptr::Ptr{Cvoid}, out::Ref{Csize_t})
    return ccall(
        (:t4a_treetn_maxbonddim, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}),
        ptr,
        out
    )
end

# ============================================================================
# TreeTN orthogonalization
# ============================================================================

"""
    t4a_treetn_orthogonalize(ptr::Ptr{Cvoid}, vertex::Integer) -> Cint

Orthogonalize the tree tensor network to a single vertex.
Uses QR decomposition (Unitary canonical form) by default.
"""
function t4a_treetn_orthogonalize(ptr::Ptr{Cvoid}, vertex::Integer)
    return ccall(
        (:t4a_treetn_orthogonalize, libpath()),
        Cint,
        (Ptr{Cvoid}, Csize_t),
        ptr,
        Csize_t(vertex)
    )
end

"""
    t4a_treetn_orthogonalize_with(ptr::Ptr{Cvoid}, vertex::Integer, form::Integer) -> Cint

Orthogonalize the tree tensor network to a vertex with a specific canonical form.
form: 0=Unitary, 1=LU, 2=CI
"""
function t4a_treetn_orthogonalize_with(ptr::Ptr{Cvoid}, vertex::Integer, form::Integer)
    return ccall(
        (:t4a_treetn_orthogonalize_with, libpath()),
        Cint,
        (Ptr{Cvoid}, Csize_t, Cint),
        ptr,
        Csize_t(vertex),
        Cint(form)
    )
end

"""
    t4a_treetn_ortho_center(ptr::Ptr{Cvoid}, out_vertices, buf_size::Integer, n_out::Ref{Csize_t}) -> Cint

Get the canonical (orthogonality) region of the tree tensor network.
"""
function t4a_treetn_ortho_center(ptr::Ptr{Cvoid}, out_vertices, buf_size::Integer, n_out::Ref{Csize_t})
    return ccall(
        (:t4a_treetn_ortho_center, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}, Csize_t, Ptr{Csize_t}),
        ptr,
        out_vertices,
        Csize_t(buf_size),
        n_out
    )
end

"""
    t4a_treetn_canonical_form(ptr::Ptr{Cvoid}, out::Ref{Cint}) -> Cint

Get the canonical form used for the tree tensor network.
"""
function t4a_treetn_canonical_form(ptr::Ptr{Cvoid}, out)
    return ccall(
        (:t4a_treetn_canonical_form, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Cint}),
        ptr,
        out
    )
end

# ============================================================================
# TreeTN operations
# ============================================================================

"""
    t4a_treetn_truncate(ptr::Ptr{Cvoid}, rtol::Float64, cutoff::Float64, maxdim::Integer) -> Cint

Truncate the tree tensor network bond dimensions.
rtol: relative tolerance (0.0 for not set)
cutoff: ITensorMPS.jl cutoff (0.0 for not set). Converted to rtol = sqrt(cutoff).
maxdim: maximum bond dimension (0 for no limit)
"""
function t4a_treetn_truncate(ptr::Ptr{Cvoid}, rtol::Float64, cutoff::Float64, maxdim::Integer)
    return ccall(
        (:t4a_treetn_truncate, libpath()),
        Cint,
        (Ptr{Cvoid}, Cdouble, Cdouble, Csize_t),
        ptr,
        rtol,
        cutoff,
        Csize_t(maxdim)
    )
end

"""
    t4a_treetn_inner(a::Ptr{Cvoid}, b::Ptr{Cvoid}, out_re::Ref{Cdouble}, out_im::Ref{Cdouble}) -> Cint

Compute the inner product of two tree tensor networks.
"""
function t4a_treetn_inner(a::Ptr{Cvoid}, b::Ptr{Cvoid}, out_re::Ref{Cdouble}, out_im::Ref{Cdouble})
    return ccall(
        (:t4a_treetn_inner, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}),
        a,
        b,
        out_re,
        out_im
    )
end

"""
    t4a_treetn_norm(ptr::Ptr{Cvoid}, out::Ref{Cdouble}) -> Cint

Compute the norm of the tree tensor network.
"""
function t4a_treetn_norm(ptr::Ptr{Cvoid}, out::Ref{Cdouble})
    return ccall(
        (:t4a_treetn_norm, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Cdouble}),
        ptr,
        out
    )
end

"""
    t4a_treetn_lognorm(ptr::Ptr{Cvoid}, out::Ref{Cdouble}) -> Cint

Compute the log-norm of the tree tensor network.
"""
function t4a_treetn_lognorm(ptr::Ptr{Cvoid}, out::Ref{Cdouble})
    return ccall(
        (:t4a_treetn_lognorm, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Cdouble}),
        ptr,
        out
    )
end

"""
    t4a_treetn_add(a::Ptr{Cvoid}, b::Ptr{Cvoid}, out::Ref{Ptr{Cvoid}}) -> Cint

Add two tree tensor networks using direct-sum construction.
"""
function t4a_treetn_add(a::Ptr{Cvoid}, b::Ptr{Cvoid}, out)
    return ccall(
        (:t4a_treetn_add, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Ptr{Cvoid}}),
        a,
        b,
        out
    )
end

"""
    t4a_treetn_contract(a, b, method, rtol, cutoff, maxdim, out) -> Cint

Contract two tree tensor networks.
method: 0=Zipup, 1=Fit, 2=Naive
"""
function t4a_treetn_contract(a::Ptr{Cvoid}, b::Ptr{Cvoid}, method::Integer,
                             rtol::Float64, cutoff::Float64, maxdim::Integer, out)
    return ccall(
        (:t4a_treetn_contract, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Cvoid}, Cint, Cdouble, Cdouble, Csize_t, Ptr{Ptr{Cvoid}}),
        a,
        b,
        Cint(method),
        rtol,
        cutoff,
        Csize_t(maxdim),
        out
    )
end

"""
    t4a_treetn_to_dense(ptr::Ptr{Cvoid}, out::Ref{Ptr{Cvoid}}) -> Cint

Convert tree tensor network to a dense tensor by contracting all link indices.
"""
function t4a_treetn_to_dense(ptr::Ptr{Cvoid}, out)
    return ccall(
        (:t4a_treetn_to_dense, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Ptr{Cvoid}}),
        ptr,
        out
    )
end

"""
    t4a_treetn_linsolve(operator, rhs, init, a0, a1, nsweeps, rtol, cutoff, maxdim, out) -> Cint

Solve (a0 + a1 * A) * x = b for x using DMRG-like sweeps.
"""
function t4a_treetn_linsolve(operator::Ptr{Cvoid}, rhs::Ptr{Cvoid}, init::Ptr{Cvoid},
                             a0::Float64, a1::Float64, nsweeps::Integer,
                             rtol::Float64, cutoff::Float64, maxdim::Integer, out)
    return ccall(
        (:t4a_treetn_linsolve, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid},
         Cdouble, Cdouble, Csize_t,
         Cdouble, Cdouble, Csize_t, Ptr{Ptr{Cvoid}}),
        operator, rhs, init,
        a0, a1, Csize_t(nsweeps),
        rtol, cutoff, Csize_t(maxdim),
        out
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

# ============================================================================
# HDF5 save/load functions
# ============================================================================

"""
    t4a_hdf5_save_itensor(filepath::AbstractString, name::AbstractString, tensor::Ptr{Cvoid}) -> Cint

Save a tensor as an ITensors.jl-compatible ITensor in an HDF5 file.
"""
function t4a_hdf5_save_itensor(filepath::AbstractString, name::AbstractString, tensor::Ptr{Cvoid})
    return ccall(
        (:t4a_hdf5_save_itensor, libpath()),
        Cint,
        (Cstring, Cstring, Ptr{Cvoid}),
        filepath,
        name,
        tensor
    )
end

"""
    t4a_hdf5_load_itensor(filepath::AbstractString, name::AbstractString, out::Ref{Ptr{Cvoid}}) -> Cint

Load a tensor from an ITensors.jl-compatible ITensor in an HDF5 file.
"""
function t4a_hdf5_load_itensor(filepath::AbstractString, name::AbstractString, out::Ref{Ptr{Cvoid}})
    return ccall(
        (:t4a_hdf5_load_itensor, libpath()),
        Cint,
        (Cstring, Cstring, Ptr{Ptr{Cvoid}}),
        filepath,
        name,
        out
    )
end

"""
    t4a_hdf5_save_mps(filepath::AbstractString, name::AbstractString, ttn::Ptr{Cvoid}) -> Cint

Save a tree tensor network (MPS) as an ITensorMPS.jl-compatible MPS in an HDF5 file.
"""
function t4a_hdf5_save_mps(filepath::AbstractString, name::AbstractString, ttn::Ptr{Cvoid})
    return ccall(
        (:t4a_hdf5_save_mps, libpath()),
        Cint,
        (Cstring, Cstring, Ptr{Cvoid}),
        filepath,
        name,
        ttn
    )
end

"""
    t4a_hdf5_load_mps(filepath::AbstractString, name::AbstractString, out::Ref{Ptr{Cvoid}}) -> Cint

Load a tree tensor network (MPS) from an ITensorMPS.jl-compatible MPS in an HDF5 file.
Returns a `t4a_treetn` handle.
"""
function t4a_hdf5_load_mps(filepath::AbstractString, name::AbstractString, out::Ref{Ptr{Cvoid}})
    return ccall(
        (:t4a_hdf5_load_mps, libpath()),
        Cint,
        (Cstring, Cstring, Ptr{Ptr{Cvoid}}),
        filepath,
        name,
        out
    )
end

# ============================================================================
# QuanticsGrids: DiscretizedGrid functions
# ============================================================================

# Unfolding scheme enum (must match Rust side)
const UNFOLDING_FUSED = Cint(0)
const UNFOLDING_INTERLEAVED = Cint(1)

"""
    t4a_qgrid_disc_new(ndims, rs_arr, lower_arr, upper_arr, unfolding, out) -> Cint
"""
function t4a_qgrid_disc_new(ndims::Integer, rs_arr, lower_arr, upper_arr, unfolding::Integer, out)
    return ccall(
        (:t4a_qgrid_disc_new, libpath()),
        Cint,
        (Csize_t, Ptr{Csize_t}, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Ptr{Cvoid}}),
        Csize_t(ndims),
        rs_arr,
        lower_arr,
        upper_arr,
        Cint(unfolding),
        out
    )
end

function t4a_qgrid_disc_release(ptr::Ptr{Cvoid})
    ptr == C_NULL && return
    ccall(
        (:t4a_qgrid_disc_release, libpath()),
        Cvoid,
        (Ptr{Cvoid},),
        ptr
    )
end

function t4a_qgrid_disc_clone(ptr::Ptr{Cvoid})
    return ccall(
        (:t4a_qgrid_disc_clone, libpath()),
        Ptr{Cvoid},
        (Ptr{Cvoid},),
        ptr
    )
end

function t4a_qgrid_disc_ndims(ptr::Ptr{Cvoid}, out::Ref{Csize_t})
    return ccall(
        (:t4a_qgrid_disc_ndims, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}),
        ptr,
        out
    )
end

function t4a_qgrid_disc_rs(ptr::Ptr{Cvoid}, out_arr, buf_size::Integer)
    return ccall(
        (:t4a_qgrid_disc_rs, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}, Csize_t),
        ptr,
        out_arr,
        Csize_t(buf_size)
    )
end

function t4a_qgrid_disc_local_dims(ptr::Ptr{Cvoid}, out_arr, buf_size::Integer, n_out::Ref{Csize_t})
    return ccall(
        (:t4a_qgrid_disc_local_dims, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}, Csize_t, Ptr{Csize_t}),
        ptr,
        out_arr,
        Csize_t(buf_size),
        n_out
    )
end

function t4a_qgrid_disc_lower_bound(ptr::Ptr{Cvoid}, out_arr, buf_size::Integer)
    return ccall(
        (:t4a_qgrid_disc_lower_bound, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Cdouble}, Csize_t),
        ptr,
        out_arr,
        Csize_t(buf_size)
    )
end

function t4a_qgrid_disc_upper_bound(ptr::Ptr{Cvoid}, out_arr, buf_size::Integer)
    return ccall(
        (:t4a_qgrid_disc_upper_bound, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Cdouble}, Csize_t),
        ptr,
        out_arr,
        Csize_t(buf_size)
    )
end

function t4a_qgrid_disc_grid_step(ptr::Ptr{Cvoid}, out_arr, buf_size::Integer)
    return ccall(
        (:t4a_qgrid_disc_grid_step, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Cdouble}, Csize_t),
        ptr,
        out_arr,
        Csize_t(buf_size)
    )
end

function t4a_qgrid_disc_origcoord_to_quantics(ptr::Ptr{Cvoid}, coord_arr, ndims::Integer,
                                                out_arr, buf_size::Integer, n_out::Ref{Csize_t})
    return ccall(
        (:t4a_qgrid_disc_origcoord_to_quantics, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Cdouble}, Csize_t, Ptr{Int64}, Csize_t, Ptr{Csize_t}),
        ptr,
        coord_arr,
        Csize_t(ndims),
        out_arr,
        Csize_t(buf_size),
        n_out
    )
end

function t4a_qgrid_disc_quantics_to_origcoord(ptr::Ptr{Cvoid}, quantics_arr, n_quantics::Integer,
                                                out_arr, buf_size::Integer)
    return ccall(
        (:t4a_qgrid_disc_quantics_to_origcoord, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Int64}, Csize_t, Ptr{Cdouble}, Csize_t),
        ptr,
        quantics_arr,
        Csize_t(n_quantics),
        out_arr,
        Csize_t(buf_size)
    )
end

function t4a_qgrid_disc_origcoord_to_grididx(ptr::Ptr{Cvoid}, coord_arr, ndims::Integer,
                                               out_arr, buf_size::Integer)
    return ccall(
        (:t4a_qgrid_disc_origcoord_to_grididx, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Cdouble}, Csize_t, Ptr{Int64}, Csize_t),
        ptr,
        coord_arr,
        Csize_t(ndims),
        out_arr,
        Csize_t(buf_size)
    )
end

function t4a_qgrid_disc_grididx_to_origcoord(ptr::Ptr{Cvoid}, grididx_arr, ndims::Integer,
                                               out_arr, buf_size::Integer)
    return ccall(
        (:t4a_qgrid_disc_grididx_to_origcoord, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Int64}, Csize_t, Ptr{Cdouble}, Csize_t),
        ptr,
        grididx_arr,
        Csize_t(ndims),
        out_arr,
        Csize_t(buf_size)
    )
end

function t4a_qgrid_disc_grididx_to_quantics(ptr::Ptr{Cvoid}, grididx_arr, ndims::Integer,
                                              out_arr, buf_size::Integer, n_out::Ref{Csize_t})
    return ccall(
        (:t4a_qgrid_disc_grididx_to_quantics, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Int64}, Csize_t, Ptr{Int64}, Csize_t, Ptr{Csize_t}),
        ptr,
        grididx_arr,
        Csize_t(ndims),
        out_arr,
        Csize_t(buf_size),
        n_out
    )
end

function t4a_qgrid_disc_quantics_to_grididx(ptr::Ptr{Cvoid}, quantics_arr, n_quantics::Integer,
                                              out_arr, buf_size::Integer, n_out::Ref{Csize_t})
    return ccall(
        (:t4a_qgrid_disc_quantics_to_grididx, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Int64}, Csize_t, Ptr{Int64}, Csize_t, Ptr{Csize_t}),
        ptr,
        quantics_arr,
        Csize_t(n_quantics),
        out_arr,
        Csize_t(buf_size),
        n_out
    )
end

# ============================================================================
# QuanticsGrids: InherentDiscreteGrid functions
# ============================================================================

function t4a_qgrid_int_new(ndims::Integer, rs_arr, origin_arr, unfolding::Integer, out)
    return ccall(
        (:t4a_qgrid_int_new, libpath()),
        Cint,
        (Csize_t, Ptr{Csize_t}, Ptr{Int64}, Cint, Ptr{Ptr{Cvoid}}),
        Csize_t(ndims),
        rs_arr,
        origin_arr,
        Cint(unfolding),
        out
    )
end

function t4a_qgrid_int_release(ptr::Ptr{Cvoid})
    ptr == C_NULL && return
    ccall(
        (:t4a_qgrid_int_release, libpath()),
        Cvoid,
        (Ptr{Cvoid},),
        ptr
    )
end

function t4a_qgrid_int_clone(ptr::Ptr{Cvoid})
    return ccall(
        (:t4a_qgrid_int_clone, libpath()),
        Ptr{Cvoid},
        (Ptr{Cvoid},),
        ptr
    )
end

function t4a_qgrid_int_ndims(ptr::Ptr{Cvoid}, out::Ref{Csize_t})
    return ccall(
        (:t4a_qgrid_int_ndims, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}),
        ptr,
        out
    )
end

function t4a_qgrid_int_rs(ptr::Ptr{Cvoid}, out_arr, buf_size::Integer)
    return ccall(
        (:t4a_qgrid_int_rs, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}, Csize_t),
        ptr,
        out_arr,
        Csize_t(buf_size)
    )
end

function t4a_qgrid_int_local_dims(ptr::Ptr{Cvoid}, out_arr, buf_size::Integer, n_out::Ref{Csize_t})
    return ccall(
        (:t4a_qgrid_int_local_dims, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}, Csize_t, Ptr{Csize_t}),
        ptr,
        out_arr,
        Csize_t(buf_size),
        n_out
    )
end

function t4a_qgrid_int_origin(ptr::Ptr{Cvoid}, out_arr, buf_size::Integer)
    return ccall(
        (:t4a_qgrid_int_origin, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Int64}, Csize_t),
        ptr,
        out_arr,
        Csize_t(buf_size)
    )
end

function t4a_qgrid_int_origcoord_to_quantics(ptr::Ptr{Cvoid}, coord_arr, ndims::Integer,
                                               out_arr, buf_size::Integer, n_out::Ref{Csize_t})
    return ccall(
        (:t4a_qgrid_int_origcoord_to_quantics, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Int64}, Csize_t, Ptr{Int64}, Csize_t, Ptr{Csize_t}),
        ptr,
        coord_arr,
        Csize_t(ndims),
        out_arr,
        Csize_t(buf_size),
        n_out
    )
end

function t4a_qgrid_int_quantics_to_origcoord(ptr::Ptr{Cvoid}, quantics_arr, n_quantics::Integer,
                                               out_arr, buf_size::Integer)
    return ccall(
        (:t4a_qgrid_int_quantics_to_origcoord, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Int64}, Csize_t, Ptr{Int64}, Csize_t),
        ptr,
        quantics_arr,
        Csize_t(n_quantics),
        out_arr,
        Csize_t(buf_size)
    )
end

function t4a_qgrid_int_origcoord_to_grididx(ptr::Ptr{Cvoid}, coord_arr, ndims::Integer,
                                              out_arr, buf_size::Integer)
    return ccall(
        (:t4a_qgrid_int_origcoord_to_grididx, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Int64}, Csize_t, Ptr{Int64}, Csize_t),
        ptr,
        coord_arr,
        Csize_t(ndims),
        out_arr,
        Csize_t(buf_size)
    )
end

function t4a_qgrid_int_grididx_to_origcoord(ptr::Ptr{Cvoid}, grididx_arr, ndims::Integer,
                                              out_arr, buf_size::Integer)
    return ccall(
        (:t4a_qgrid_int_grididx_to_origcoord, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Int64}, Csize_t, Ptr{Int64}, Csize_t),
        ptr,
        grididx_arr,
        Csize_t(ndims),
        out_arr,
        Csize_t(buf_size)
    )
end

function t4a_qgrid_int_grididx_to_quantics(ptr::Ptr{Cvoid}, grididx_arr, ndims::Integer,
                                             out_arr, buf_size::Integer, n_out::Ref{Csize_t})
    return ccall(
        (:t4a_qgrid_int_grididx_to_quantics, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Int64}, Csize_t, Ptr{Int64}, Csize_t, Ptr{Csize_t}),
        ptr,
        grididx_arr,
        Csize_t(ndims),
        out_arr,
        Csize_t(buf_size),
        n_out
    )
end

function t4a_qgrid_int_quantics_to_grididx(ptr::Ptr{Cvoid}, quantics_arr, n_quantics::Integer,
                                             out_arr, buf_size::Integer, n_out::Ref{Csize_t})
    return ccall(
        (:t4a_qgrid_int_quantics_to_grididx, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Int64}, Csize_t, Ptr{Int64}, Csize_t, Ptr{Csize_t}),
        ptr,
        quantics_arr,
        Csize_t(n_quantics),
        out_arr,
        Csize_t(buf_size),
        n_out
    )
end

# ============================================================================
# QuanticsTCI: QTCI lifecycle functions
# ============================================================================

function t4a_qtci_f64_release(ptr::Ptr{Cvoid})
    ptr == C_NULL && return
    ccall(
        (:t4a_qtci_f64_release, libpath()),
        Cvoid,
        (Ptr{Cvoid},),
        ptr
    )
end

# ============================================================================
# QuanticsTCI: High-level interpolation functions
# ============================================================================

"""
    t4a_quanticscrossinterpolate_f64(grid, eval_fn, user_data, tolerance, max_bonddim, max_iter, out_qtci) -> Cint

Continuous domain interpolation using a DiscretizedGrid.
"""
function t4a_quanticscrossinterpolate_f64(
    grid::Ptr{Cvoid},
    eval_fn::Ptr{Cvoid},
    user_data::Ptr{Cvoid},
    tolerance::Cdouble,
    max_bonddim::Csize_t,
    max_iter::Csize_t,
    out_qtci::Ref{Ptr{Cvoid}},
)
    return ccall(
        (:t4a_quanticscrossinterpolate_f64, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Cdouble, Csize_t, Csize_t, Ptr{Ptr{Cvoid}}),
        grid, eval_fn, user_data, tolerance, max_bonddim, max_iter, out_qtci
    )
end

"""
    t4a_quanticscrossinterpolate_discrete_f64(sizes, ndims, eval_fn, user_data, tolerance, max_bonddim, max_iter, unfoldingscheme, out_qtci) -> Cint

Discrete domain interpolation with integer indices.
"""
function t4a_quanticscrossinterpolate_discrete_f64(
    sizes::Vector{Csize_t},
    ndims::Csize_t,
    eval_fn::Ptr{Cvoid},
    user_data::Ptr{Cvoid},
    tolerance::Cdouble,
    max_bonddim::Csize_t,
    max_iter::Csize_t,
    unfoldingscheme::Cint,
    out_qtci::Ref{Ptr{Cvoid}},
)
    return ccall(
        (:t4a_quanticscrossinterpolate_discrete_f64, libpath()),
        Cint,
        (Ptr{Csize_t}, Csize_t, Ptr{Cvoid}, Ptr{Cvoid}, Cdouble, Csize_t, Csize_t, Cint, Ptr{Ptr{Cvoid}}),
        sizes, ndims, eval_fn, user_data, tolerance, max_bonddim, max_iter, unfoldingscheme, out_qtci
    )
end

# ============================================================================
# QuanticsTCI: Accessors
# ============================================================================

function t4a_qtci_f64_rank(ptr::Ptr{Cvoid}, out_rank::Ref{Csize_t})
    return ccall(
        (:t4a_qtci_f64_rank, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}),
        ptr, out_rank
    )
end

function t4a_qtci_f64_link_dims(ptr::Ptr{Cvoid}, out_dims::Vector{Csize_t}, buf_len::Csize_t)
    return ccall(
        (:t4a_qtci_f64_link_dims, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}, Csize_t),
        ptr, out_dims, buf_len
    )
end

# ============================================================================
# QuanticsTCI: Operations
# ============================================================================

function t4a_qtci_f64_evaluate(ptr::Ptr{Cvoid}, indices::Vector{Int64}, n_indices::Csize_t, out_value::Ref{Cdouble})
    return ccall(
        (:t4a_qtci_f64_evaluate, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Int64}, Csize_t, Ptr{Cdouble}),
        ptr, indices, n_indices, out_value
    )
end

function t4a_qtci_f64_sum(ptr::Ptr{Cvoid}, out_value::Ref{Cdouble})
    return ccall(
        (:t4a_qtci_f64_sum, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Cdouble}),
        ptr, out_value
    )
end

function t4a_qtci_f64_integral(ptr::Ptr{Cvoid}, out_value::Ref{Cdouble})
    return ccall(
        (:t4a_qtci_f64_integral, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Cdouble}),
        ptr, out_value
    )
end

function t4a_qtci_f64_to_tensor_train(ptr::Ptr{Cvoid})
    return ccall(
        (:t4a_qtci_f64_to_tensor_train, libpath()),
        Ptr{Cvoid},
        (Ptr{Cvoid},),
        ptr
    )
end

# ============================================================================
# LinearOperator (linop) lifecycle functions
# ============================================================================

"""
    t4a_linop_release(ptr::Ptr{Cvoid})

Release a linear operator (called by finalizer).
"""
function t4a_linop_release(ptr::Ptr{Cvoid})
    ptr == C_NULL && return
    ccall(
        (:t4a_linop_release, libpath()),
        Cvoid,
        (Ptr{Cvoid},),
        ptr
    )
end

"""
    t4a_linop_clone(ptr::Ptr{Cvoid}) -> Ptr{Cvoid}

Clone a linear operator.
"""
function t4a_linop_clone(ptr::Ptr{Cvoid})
    return ccall(
        (:t4a_linop_clone, libpath()),
        Ptr{Cvoid},
        (Ptr{Cvoid},),
        ptr
    )
end

"""
    t4a_linop_is_assigned(ptr::Ptr{Cvoid}) -> Bool

Check if a linear operator pointer is valid.
"""
function t4a_linop_is_assigned(ptr::Ptr{Cvoid})
    result = ccall(
        (:t4a_linop_is_assigned, libpath()),
        Cint,
        (Ptr{Cvoid},),
        ptr
    )
    return result == 1
end

# ============================================================================
# QuanticsTransform operator construction functions
# ============================================================================

"""
    t4a_qtransform_shift(r, offset, bc, out) -> Cint

Create a shift operator: f(x) = g(x + offset) mod 2^r.
"""
function t4a_qtransform_shift(r::Csize_t, offset::Int64, bc::Cint, out)
    return ccall(
        (:t4a_qtransform_shift, libpath()),
        Cint,
        (Csize_t, Int64, Cint, Ptr{Ptr{Cvoid}}),
        r, offset, bc, out
    )
end

"""
    t4a_qtransform_flip(r, bc, out) -> Cint

Create a flip operator: f(x) = g(2^r - x).
"""
function t4a_qtransform_flip(r::Csize_t, bc::Cint, out)
    return ccall(
        (:t4a_qtransform_flip, libpath()),
        Cint,
        (Csize_t, Cint, Ptr{Ptr{Cvoid}}),
        r, bc, out
    )
end

"""
    t4a_qtransform_phase_rotation(r, theta, out) -> Cint

Create a phase rotation operator: f(x) = exp(i*theta*x) * g(x).
"""
function t4a_qtransform_phase_rotation(r::Csize_t, theta::Cdouble, out)
    return ccall(
        (:t4a_qtransform_phase_rotation, libpath()),
        Cint,
        (Csize_t, Cdouble, Ptr{Ptr{Cvoid}}),
        r, theta, out
    )
end

"""
    t4a_qtransform_cumsum(r, out) -> Cint

Create a cumulative sum operator: y_i = sum_{j<i} x_j.
"""
function t4a_qtransform_cumsum(r::Csize_t, out)
    return ccall(
        (:t4a_qtransform_cumsum, libpath()),
        Cint,
        (Csize_t, Ptr{Ptr{Cvoid}}),
        r, out
    )
end

"""
    t4a_qtransform_fourier(r, forward, maxbonddim, tolerance, out) -> Cint

Create a Fourier transform operator.
forward: 1 for forward, 0 for inverse.
maxbonddim: 0 = default (12). tolerance: 0.0 = default (1e-14).
"""
function t4a_qtransform_fourier(r::Csize_t, forward::Cint, maxbonddim::Csize_t, tolerance::Cdouble, out)
    return ccall(
        (:t4a_qtransform_fourier, libpath()),
        Cint,
        (Csize_t, Cint, Csize_t, Cdouble, Ptr{Ptr{Cvoid}}),
        r, forward, maxbonddim, tolerance, out
    )
end

"""
    t4a_linop_apply(op, state, method, rtol, maxdim, out) -> Cint

Apply a linear operator to a TreeTN state.
method: 0=Naive, 1=Zipup, 2=Fit.
"""
function t4a_linop_apply(op::Ptr{Cvoid}, state::Ptr{Cvoid}, method::Cint,
                         rtol::Cdouble, maxdim::Csize_t, out)
    return ccall(
        (:t4a_linop_apply, libpath()),
        Cint,
        (Ptr{Cvoid}, Ptr{Cvoid}, Cint, Cdouble, Csize_t, Ptr{Ptr{Cvoid}}),
        op, state, method, rtol, maxdim, out
    )
end

end # module C_API
