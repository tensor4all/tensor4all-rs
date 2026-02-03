"""
    Tensor4all

Julia wrapper for the tensor4all Rust library.

Provides tensor network types compatible with ITensors.jl, backed by efficient
Rust implementations.

# Basic Usage

```julia
using Tensor4all

# Create an index with dimension 5
i = Index(5)

# Create an index with tags
j = Index(3; tags="Site,n=1")

# Access properties
dim(i)   # dimension
id(i)    # unique ID (UInt64)
tags(i)  # tags as comma-separated string
```

# ITensors.jl Integration

When ITensors.jl is loaded, bidirectional conversion is available:

```julia
using Tensor4all
using ITensors

# Tensor4all.Index → ITensors.Index
t4a_idx = Tensor4all.Index(4; tags="Site")
it_idx = ITensors.Index(t4a_idx)

# ITensors.Index → Tensor4all.Index
it_idx2 = ITensors.Index(3, "Link")
t4a_idx2 = Tensor4all.Index(it_idx2)
```
"""
module Tensor4all

using LinearAlgebra

include("C_API.jl")

# Library management for submodules
using Libdl
get_lib() = C_API.libhandle()

# Include submodules (respecting crate hierarchy)
# 1. Algorithm (tensor4all-core-common)
include("Algorithm.jl")

# Re-export public API
# Core types (tensor4all-core-common, tensor4all-core-tensor)
export Index, dim, tags, id, hastag
export Tensor, rank, dims, indices, storage_kind, data
export StorageKind, DenseF64, DenseC64, DiagF64, DiagC64

# Re-export Algorithm submodule and utilities
using .Algorithm: get_default_svd_rtol, resolve_truncation_tolerance
export Algorithm
export get_default_svd_rtol, resolve_truncation_tolerance

"""
    Index

A tensor index with dimension, ID, and tags.

Wraps a Rust `DynIndex` (= `Index<DynId, TagSet>`) which corresponds to
ITensors.jl's `Index{Int}` (no quantum number symmetry).

# Constructors

- `Index(dim::Integer)` - Create index with dimension
- `Index(dim::Integer; tags::AbstractString)` - Create with tags
- `Index(dim::Integer, id::UInt64; tags::AbstractString)` - Create with specific ID

# Properties

- `dim(i::Index)` - Get the dimension
- `id(i::Index)` - Get the unique ID as UInt64
- `tags(i::Index)` - Get tags as comma-separated string
- `hastag(i::Index, tag::AbstractString)` - Check if index has a tag
"""
mutable struct Index
    ptr::Ptr{Cvoid}

    function Index(ptr::Ptr{Cvoid})
        if ptr == C_NULL
            error("Failed to create Index (null pointer from C API)")
        end
        idx = new(ptr)
        finalizer(idx) do x
            C_API.t4a_index_release(x.ptr)
        end
        return idx
    end
end

# Constructors
function Index(dim::Integer; tags::AbstractString="")
    dim > 0 || throw(ArgumentError("Index dimension must be positive, got $dim"))
    if isempty(tags)
        ptr = C_API.t4a_index_new(dim)
    else
        ptr = C_API.t4a_index_new_with_tags(dim, tags)
    end
    return Index(ptr)
end

function Index(dim::Integer, id::UInt64; tags::AbstractString="")
    dim > 0 || throw(ArgumentError("Index dimension must be positive, got $dim"))
    ptr = C_API.t4a_index_new_with_id(dim, id, tags)
    return Index(ptr)
end

# Accessors
"""
    dim(i::Index) -> Int

Get the dimension of an index.
"""
function dim(i::Index)
    d = Ref{Csize_t}(0)
    status = C_API.t4a_index_dim(i.ptr, d)
    C_API.check_status(status)
    return Int(d[])
end

"""
    id(i::Index) -> UInt64

Get the unique ID of an index.
"""
function id(i::Index)
    out_id = Ref{UInt64}(0)
    status = C_API.t4a_index_id(i.ptr, out_id)
    C_API.check_status(status)
    return out_id[]
end

"""
    tags(i::Index) -> String

Get the tags of an index as a comma-separated string.
"""
function tags(i::Index)
    # First query required length
    len = Ref{Csize_t}(0)
    status = C_API.t4a_index_get_tags(i.ptr, C_NULL, 0, len)
    C_API.check_status(status)

    if len[] <= 1
        return ""
    end

    # Allocate buffer and get tags
    buf = Vector{UInt8}(undef, len[])
    status = C_API.t4a_index_get_tags(i.ptr, buf, len[], len)
    C_API.check_status(status)

    # Convert to string (excluding null terminator)
    return String(buf[1:end-1])
end

"""
    hastag(i::Index, tag::AbstractString) -> Bool

Check if an index has a specific tag.
"""
function hastag(i::Index, tag::AbstractString)
    result = C_API.t4a_index_has_tag(i.ptr, tag)
    result < 0 && error("Error checking tag")
    return result == 1
end

# Show method
function Base.show(io::IO, i::Index)
    d = dim(i)
    t = tags(i)
    id_val = id(i)
    id_hex = string(id_val, base=16)
    id_short = length(id_hex) >= 8 ? id_hex[end-7:end] : id_hex  # Last 8 hex digits
    if isempty(t)
        print(io, "(dim=$d|id=...$id_short)")
    else
        print(io, "(dim=$d|id=...$id_short|\"$t\")")
    end
end

function Base.show(io::IO, ::MIME"text/plain", i::Index)
    println(io, "Tensor4all.Index")
    println(io, "  dim: ", dim(i))
    println(io, "  id:  ", string(id(i), base=16))
    t = tags(i)
    if !isempty(t)
        println(io, "  tags: ", t)
    end
end

# Clone (shallow copy - same as deepcopy for this type)
function Base.copy(i::Index)
    ptr = C_API.t4a_index_clone(i.ptr)
    return Index(ptr)
end

# Deep copy
function Base.deepcopy(i::Index)
    ptr = C_API.t4a_index_clone(i.ptr)
    return Index(ptr)
end

# Equality based on ID + tags (matching ITensors.jl semantics with plev=0)
function Base.:(==)(i1::Index, i2::Index)
    return id(i1) == id(i2) && tags(i1) == tags(i2)
end

function Base.hash(i::Index, h::UInt)
    return hash(tags(i), hash(id(i), h))
end

# ============================================================================
# Index Utilities
# ============================================================================

"""
    sim(i::Index) -> Index

Create a new index with the same dimension and tags but a new unique ID.
This is useful for creating "similar" indices that won't contract with the original.
"""
function sim(i::Index)
    return Index(dim(i); tags=tags(i))
end

"""
    hascommoninds(inds1, inds2) -> Bool

Check if two collections of indices have any common indices (by ID).

# Example
```julia
i, j, k = Index(2), Index(3), Index(4)
hascommoninds([i, j], [j, k])  # true
hascommoninds([i], [k])        # false
```
"""
function hascommoninds(inds1, inds2)
    ids1 = Set(id(i) for i in inds1)
    for i in inds2
        if id(i) in ids1
            return true
        end
    end
    return false
end

"""
    commoninds(inds1, inds2) -> Vector{Index}

Return the indices that appear in both collections (by ID).
Returns indices from inds1.

# Example
```julia
i, j, k = Index(2), Index(3), Index(4)
commoninds([i, j], [j, k])  # [j]
```
"""
function commoninds(inds1, inds2)
    ids2 = Set(id(i) for i in inds2)
    return [i for i in inds1 if id(i) in ids2]
end

# Alias for ITensors compatibility
const common_inds = commoninds

"""
    commonind(inds1, inds2) -> Union{Index, Nothing}

Return the first common index, or nothing if none exists.
"""
function commonind(inds1, inds2)
    result = commoninds(inds1, inds2)
    return isempty(result) ? nothing : first(result)
end

"""
    uniqueinds(inds1, inds2) -> Vector{Index}

Return the indices in inds1 that do not appear in inds2 (by ID).

# Example
```julia
i, j, k = Index(2), Index(3), Index(4)
uniqueinds([i, j], [j, k])  # [i]
```
"""
function uniqueinds(inds1, inds2)
    ids2 = Set(id(i) for i in inds2)
    return [i for i in inds1 if !(id(i) in ids2)]
end

"""
    uniqueind(inds1, inds2) -> Union{Index, Nothing}

Return the first unique index in inds1 (not in inds2), or nothing if none exists.
"""
function uniqueind(inds1, inds2)
    result = uniqueinds(inds1, inds2)
    return isempty(result) ? nothing : first(result)
end

"""
    noncommoninds(inds1, inds2) -> Vector{Index}

Return all indices that appear in only one of the collections.
Equivalent to union of uniqueinds(inds1, inds2) and uniqueinds(inds2, inds1).
"""
function noncommoninds(inds1, inds2)
    return vcat(uniqueinds(inds1, inds2), uniqueinds(inds2, inds1))
end

"""
    replaceinds(inds, old_inds, new_inds) -> Vector{Index}

Replace indices in `inds` according to the mapping old_inds → new_inds.
"""
function replaceinds(inds, old_inds, new_inds)
    length(old_inds) == length(new_inds) || error("old_inds and new_inds must have same length")
    id_map = Dict(id(o) => n for (o, n) in zip(old_inds, new_inds))
    return [get(id_map, id(i), i) for i in inds]
end

"""
    replaceind(inds, old_ind, new_ind) -> Vector{Index}

Replace a single index in `inds`.
"""
function replaceind(inds, old_ind::Index, new_ind::Index)
    return replaceinds(inds, [old_ind], [new_ind])
end

export sim, hascommoninds, commoninds, common_inds, commonind
export uniqueinds, uniqueind, noncommoninds, replaceinds, replaceind

# ============================================================================
# Curried/Predicate Index Functions
# ============================================================================

# Curried version of hascommoninds for use with findfirst/findall
# Note: This is implemented as a struct to avoid ambiguity with the 2-arg version

"""
    HasCommonIndsPredicate

A predicate type for checking if an object has common indices with a given set.
Used internally by `hascommoninds(is)` curried form.
"""
struct HasCommonIndsPredicate
    target_ids::Set{UInt64}
end

function (p::HasCommonIndsPredicate)(x)
    for idx in indices(x)
        if id(idx) in p.target_ids
            return true
        end
    end
    return false
end

"""
    hascommoninds(is::Vector{Index}) -> HasCommonIndsPredicate

Return a predicate that checks if its argument has common indices with `is`.
Useful for `findfirst`, `findall`, etc.

# Example
```julia
sites = [Index(2), Index(3), Index(4)]
tt = random_tt(sites; linkdims=2)
findfirst(hascommoninds(sites[2:2]), tt)  # Returns 2
```
"""
function hascommoninds(is::Vector{Index})
    return HasCommonIndsPredicate(Set(id(i) for i in is))
end

hascommoninds(i::Index) = hascommoninds([i])

"""
    hasind(i::Index) -> Function

Return a function that checks if its argument has the index `i`.

# Example
```julia
sites = [Index(2), Index(3)]
tt = random_tt(sites; linkdims=2)
findfirst(hasind(sites[1]), tt)  # Returns 1
```
"""
hasind(i::Index) = x -> any(idx -> id(idx) == id(i), indices(x))

"""
    hasinds(is) -> Function

Return a function that checks if its argument has all the indices in `is`.

# Example
```julia
i, j = Index(2), Index(3)
t = Tensor([i, j], rand(2, 3))
hasinds([i, j])(t)  # true
hasinds([i])(t)     # true
```
"""
function hasinds(is)
    return function(x)
        x_inds = indices(x)
        x_ids = Set(id(idx) for idx in x_inds)
        for i in is
            if !(id(i) in x_ids)
                return false
            end
        end
        return true
    end
end

hasinds(i::Index) = hasind(i)

export hasind, hasinds

# ============================================================================
# Tensor Type
# ============================================================================

"""
    StorageKind

Enum representing the storage type of a tensor.
"""
@enum StorageKind begin
    DenseF64 = 0
    DenseC64 = 1
    DiagF64 = 2
    DiagC64 = 3
end

"""
    Tensor

A tensor with dynamically-typed indices and storage.

Wraps a Rust `TensorDynLen` which corresponds to
ITensors.jl's `ITensor`.

# Constructors

- `Tensor(indices::Vector{Index}, data::Array{Float64})` - Create dense f64 tensor
- `Tensor(indices::Vector{Index}, data::Array{ComplexF64})` - Create dense complex tensor

# Properties

- `rank(t::Tensor)` - Get the number of indices
- `dims(t::Tensor)` - Get dimensions as a tuple
- `indices(t::Tensor)` - Get indices as a vector
- `storage_kind(t::Tensor)` - Get the storage kind (DenseF64, DenseC64, etc.)
- `data(t::Tensor)` - Get the tensor data as an Array
"""
mutable struct Tensor
    ptr::Ptr{Cvoid}

    function Tensor(ptr::Ptr{Cvoid})
        if ptr == C_NULL
            error("Failed to create Tensor (null pointer from C API)")
        end
        t = new(ptr)
        finalizer(t) do x
            C_API.t4a_tensor_release(x.ptr)
        end
        return t
    end
end

# ============================================================================
# Memory Order Conversion Helpers
# ============================================================================

"""
    _is_column_major_contiguous(A::AbstractArray) -> Bool

Check if an array is column-major contiguous (standard Julia layout).
This is critical before passing data to FFI.
"""
function _is_column_major_contiguous(A::AbstractArray)
    expected_strides = cumprod((1, size(A)...)[1:end-1])
    return strides(A) == expected_strides
end

"""
    _ensure_contiguous(A::AbstractArray{T,N}) where {T,N} -> Array{T,N}

Ensure array is contiguous, copying if necessary.
CRITICAL: Always call before passing arrays to FFI.
"""
function _ensure_contiguous(A::AbstractArray{T,N}) where {T,N}
    if _is_column_major_contiguous(A)
        return A isa Array ? A : Array{T,N}(A)
    end
    # Materialize to contiguous Array
    return Array{T,N}(A)
end

"""
    _column_to_row_major(arr::Array, dims::Tuple) -> Vector

Convert column-major array to row-major flat vector for Rust.
"""
function _column_to_row_major(arr::AbstractArray)
    arr = _ensure_contiguous(arr)
    ndims(arr) == 0 && return vec(arr)
    # Reverse dimensions to get row-major layout
    perm = reverse(1:ndims(arr))
    permuted = permutedims(arr, perm)
    permuted = _ensure_contiguous(permuted)
    return vec(permuted)
end

"""
    _row_to_column_major(data::Vector, dims::Tuple) -> Array

Convert row-major flat vector from Rust to column-major Julia array.
"""
function _row_to_column_major(data::Vector{T}, dims::Tuple) where T
    isempty(dims) && return reshape(data, 1)
    # Data is row-major, so reverse dims for reshape
    arr = reshape(data, reverse(dims)...)
    # Reverse back to get column-major
    perm = reverse(1:length(dims))
    return permutedims(arr, perm)
end

# ============================================================================
# Tensor Constructors
# ============================================================================

"""
    Tensor(indices::Vector{Index}, data::AbstractArray{Float64})

Create a dense f64 tensor from indices and column-major data.
"""
function Tensor(inds::Vector{Index}, data::AbstractArray{Float64})
    isempty(inds) && error("Tensor must have at least one index")

    expected_dims = Tuple(dim(idx) for idx in inds)
    size(data) == expected_dims || error("Data size $(size(data)) doesn't match index dims $expected_dims")

    # Convert to row-major for Rust
    row_major_data = _column_to_row_major(data)

    # Prepare C-API call
    r = length(inds)
    index_ptrs = [idx.ptr for idx in inds]
    dims_vec = Csize_t[dim(idx) for idx in inds]

    ptr = C_API.t4a_tensor_new_dense_f64(r, index_ptrs, dims_vec, Cdouble.(row_major_data))
    return Tensor(ptr)
end

"""
    Tensor(indices::Vector{Index}, data::AbstractArray{ComplexF64})

Create a dense complex64 tensor from indices and column-major data.
"""
function Tensor(inds::Vector{Index}, data::AbstractArray{ComplexF64})
    isempty(inds) && error("Tensor must have at least one index")

    expected_dims = Tuple(dim(idx) for idx in inds)
    size(data) == expected_dims || error("Data size $(size(data)) doesn't match index dims $expected_dims")

    # Convert to row-major for Rust
    row_major_data = _column_to_row_major(data)

    # Prepare C-API call
    r = length(inds)
    index_ptrs = [idx.ptr for idx in inds]
    dims_vec = Csize_t[dim(idx) for idx in inds]
    data_re = Cdouble[real(z) for z in row_major_data]
    data_im = Cdouble[imag(z) for z in row_major_data]

    ptr = C_API.t4a_tensor_new_dense_c64(r, index_ptrs, dims_vec, data_re, data_im)
    return Tensor(ptr)
end

"""
    onehot(ivs::Pair{Index,<:Integer}...)

Create a one-hot tensor with value 1.0 at the specified index positions.
Compatible with ITensors.jl's `onehot(i => 1, j => 2)`.

Uses 1-based indexing (converted to 0-based internally for Rust).

# Examples
```julia
i = Index(3)
j = Index(4)
A = onehot(i => 1)           # A[i=>1] == 1.0
B = onehot(i => 2, j => 3)   # B[i=>2,j=>3] == 1.0
```
"""
function onehot(ivs::Pair{Index,<:Integer}...)
    if isempty(ivs)
        ptr = C_API.t4a_tensor_onehot(0, Ptr{Cvoid}[], Csize_t[])
        return Tensor(ptr)
    end
    for (k, iv) in enumerate(ivs)
        v = iv.second
        d = dim(iv.first)
        (1 <= v <= d) || throw(ArgumentError("onehot: value $v at position $k is out of range 1:$d"))
    end
    r = length(ivs)
    index_ptrs = Ptr{Cvoid}[iv.first.ptr for iv in ivs]
    # Convert 1-based Julia to 0-based Rust
    vals = Csize_t[iv.second - 1 for iv in ivs]
    ptr = C_API.t4a_tensor_onehot(r, index_ptrs, vals)
    return Tensor(ptr)
end

export onehot

# ============================================================================
# Tensor Accessors
# ============================================================================

"""
    rank(t::Tensor) -> Int

Get the number of indices (rank) of a tensor.
"""
function rank(t::Tensor)
    r = Ref{Csize_t}(0)
    status = C_API.t4a_tensor_get_rank(t.ptr, r)
    C_API.check_status(status)
    return Int(r[])
end

"""
    dims(t::Tensor) -> NTuple{N, Int}

Get the dimensions of a tensor as a tuple.
"""
function dims(t::Tensor)
    r = rank(t)
    out_dims = Vector{Csize_t}(undef, r)
    status = C_API.t4a_tensor_get_dims(t.ptr, out_dims, r)
    C_API.check_status(status)
    return Tuple(Int.(out_dims))
end

"""
    indices(t::Tensor) -> Vector{Index}

Get the indices of a tensor.
"""
function indices(t::Tensor)
    r = rank(t)
    out_ptrs = Vector{Ptr{Cvoid}}(undef, r)
    status = C_API.t4a_tensor_get_indices(t.ptr, out_ptrs, r)
    C_API.check_status(status)

    # Wrap each returned pointer as an Index
    return [Index(ptr) for ptr in out_ptrs]
end

"""
    storage_kind(t::Tensor) -> StorageKind

Get the storage kind of a tensor.
"""
function storage_kind(t::Tensor)
    kind = Ref{Cint}(0)
    status = C_API.t4a_tensor_get_storage_kind(t.ptr, kind)
    C_API.check_status(status)
    return StorageKind(kind[])
end

"""
    data(t::Tensor) -> Array

Get the tensor data as a column-major Julia array.
"""
function data(t::Tensor)
    kind = storage_kind(t)
    d = dims(t)

    if kind == DenseF64
        # Query length
        out_len = Ref{Csize_t}(0)
        status = C_API.t4a_tensor_get_data_f64(t.ptr, nothing, 0, out_len)
        C_API.check_status(status)

        # Get data
        buf = Vector{Cdouble}(undef, out_len[])
        status = C_API.t4a_tensor_get_data_f64(t.ptr, buf, out_len[], out_len)
        C_API.check_status(status)

        # Convert from row-major to column-major
        return _row_to_column_major(buf, d)

    elseif kind == DenseC64
        # Query length
        out_len = Ref{Csize_t}(0)
        status = C_API.t4a_tensor_get_data_c64(t.ptr, nothing, nothing, 0, out_len)
        C_API.check_status(status)

        # Get data
        buf_re = Vector{Cdouble}(undef, out_len[])
        buf_im = Vector{Cdouble}(undef, out_len[])
        status = C_API.t4a_tensor_get_data_c64(t.ptr, buf_re, buf_im, out_len[], out_len)
        C_API.check_status(status)

        # Combine and convert
        buf = [ComplexF64(r, i) for (r, i) in zip(buf_re, buf_im)]
        return _row_to_column_major(buf, d)

    else
        error("Unsupported storage kind for data extraction: $kind")
    end
end

"""
    Array(t::Tensor, inds::Vector{Index}) -> Array

Get the tensor data as a Julia array with indices in the specified order.

This is similar to ITensors.jl's `Array(T, i, j, k, ...)` syntax.
The returned array has dimensions ordered according to the provided indices.

# Arguments
- `t`: The tensor to extract data from
- `inds`: Vector of indices specifying the desired dimension order

# Example
```julia
i = Index(2)
j = Index(3)
T = Tensor([i, j], rand(2, 3))

# Get data with original order
A1 = Array(T, [i, j])  # shape (2, 3)

# Get data with permuted order
A2 = Array(T, [j, i])  # shape (3, 2), transposed
```
"""
function Base.Array(t::Tensor, inds::Vector{Index})
    t_inds = indices(t)

    # Check that inds contains exactly the same indices as t
    if length(inds) != length(t_inds)
        error("Number of indices ($(length(inds))) doesn't match tensor rank ($(length(t_inds)))")
    end

    # Find permutation: perm[i] = position of inds[i] in t_inds
    perm = Int[]
    for idx in inds
        pos = findfirst(x -> x == idx, t_inds)
        if pos === nothing
            error("Index not found in tensor: $idx")
        end
        push!(perm, pos)
    end

    # Check for duplicates
    if length(unique(perm)) != length(perm)
        error("Duplicate indices in requested order")
    end

    # Get data in tensor's native order
    arr = data(t)

    # If perm is identity, no need to permute
    if perm == collect(1:length(perm))
        return arr
    end

    # Permute dimensions
    return permutedims(arr, perm)
end

# Convenience: Array(t, i, j, k, ...) varargs form
function Base.Array(t::Tensor, inds::Index...)
    return Array(t, collect(inds))
end

"""
    Tensor(inds::Vector{Index}, arr::AbstractArray, source_inds::Vector{Index})

Create a tensor from an array, specifying both the target indices and source indices.

The `source_inds` specifies which index corresponds to which dimension of `arr`.
The data is permuted so that the tensor's internal order matches `inds`.

# Example
```julia
i = Index(2)
j = Index(3)
A = rand(3, 2)  # Note: j dimension first

# Create tensor with indices [i, j] from array with dimensions [j, i]
T = Tensor([i, j], A, [j, i])
```
"""
function Tensor(inds::Vector{Index}, arr::AbstractArray, source_inds::Vector{Index})
    # Find permutation from source_inds to inds
    if length(inds) != length(source_inds)
        error("Target and source index counts must match")
    end
    if length(inds) != ndims(arr)
        error("Number of indices doesn't match array dimensions")
    end

    # Find permutation: perm[i] = position of inds[i] in source_inds
    perm = Int[]
    for idx in inds
        pos = findfirst(x -> x == idx, source_inds)
        if pos === nothing
            error("Index not found in source indices: $idx")
        end
        push!(perm, pos)
    end

    # Permute array to match target index order
    if perm != collect(1:length(perm))
        arr = permutedims(arr, perm)
    end

    return Tensor(inds, arr)
end

# ============================================================================
# Tensor Display
# ============================================================================

function Base.show(io::IO, t::Tensor)
    r = rank(t)
    d = dims(t)
    k = storage_kind(t)
    print(io, "Tensor(rank=$r, dims=$d, storage=$k)")
end

function Base.show(io::IO, ::MIME"text/plain", t::Tensor)
    println(io, "Tensor4all.Tensor")
    println(io, "  rank: ", rank(t))
    println(io, "  dims: ", dims(t))
    println(io, "  storage: ", storage_kind(t))
    println(io, "  indices:")
    for (i, idx) in enumerate(indices(t))
        println(io, "    [$i] ", idx)
    end
end

# Clone (shallow copy - same as deepcopy for this type)
function Base.copy(t::Tensor)
    ptr = C_API.t4a_tensor_clone(t.ptr)
    return Tensor(ptr)
end

# Deep copy
function Base.deepcopy(t::Tensor)
    ptr = C_API.t4a_tensor_clone(t.ptr)
    return Tensor(ptr)
end

# ============================================================================
# HDF5 Save/Load for Tensor (ITensors.jl compatible)
# ============================================================================

"""
    save_itensor(filepath::AbstractString, name::AbstractString, t::Tensor)

Save a tensor to an HDF5 file in ITensors.jl-compatible format.

# Arguments
- `filepath`: Path to the HDF5 file (will be created/overwritten)
- `name`: Name of the HDF5 group to write the tensor to
- `t`: Tensor to save
"""
function save_itensor(filepath::AbstractString, name::AbstractString, t::Tensor)
    status = C_API.t4a_hdf5_save_itensor(filepath, name, t.ptr)
    C_API.check_status(status)
    return nothing
end

"""
    load_itensor(filepath::AbstractString, name::AbstractString) -> Tensor

Load a tensor from an HDF5 file in ITensors.jl-compatible format.

# Arguments
- `filepath`: Path to the HDF5 file
- `name`: Name of the HDF5 group containing the tensor

# Returns
A `Tensor` loaded from the file.
"""
function load_itensor(filepath::AbstractString, name::AbstractString)
    out = Ref{Ptr{Cvoid}}(C_NULL)
    status = C_API.t4a_hdf5_load_itensor(filepath, name, out)
    C_API.check_status(status)
    return Tensor(out[])
end

export save_itensor, load_itensor

# ============================================================================
# SimpleTT Submodule (Simple Tensor Train)
# ============================================================================
# SimpleTT is a simple tensor train (TT/MPS) library with statically determined shapes
# (site dimensions are fixed at construction time).
# Use: using Tensor4all.SimpleTT
include("SimpleTT.jl")

# ============================================================================
# TensorCI Submodule (Tensor Cross Interpolation)
# ============================================================================
# TensorCI provides tensor cross interpolation algorithms.
# Use: using Tensor4all.TensorCI
include("TensorCI.jl")

# ============================================================================
# TreeTN Submodule (Tree Tensor Network: MPS, MPO, TTN)
# ============================================================================
# Tree tensor network functionality is in a separate submodule.
# Use: using Tensor4all.TreeTN
include("TreeTN.jl")

# ============================================================================
# QuanticsGrids Submodule
# ============================================================================
# Quantics grid types for coordinate conversions in QTT methods.
# Use: using Tensor4all.QuanticsGrids
include("QuanticsGrids.jl")

# ============================================================================
# QuanticsTCI Submodule (Quantics Tensor Cross Interpolation)
# ============================================================================
# Quantics TCI combines TCI with quantics grid representations.
# Use: using Tensor4all.QuanticsTCI
include("QuanticsTCI.jl")

# ============================================================================
# QuanticsTransform Submodule
# ============================================================================
# Quantics transformation operators (shift, flip, phase rotation, cumsum, Fourier).
# Use: using Tensor4all.QuanticsTransform
include("QuanticsTransform.jl")

end # module
