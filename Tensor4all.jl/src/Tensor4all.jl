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
id(i)    # unique ID (UInt128)
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

include("C_API.jl")

# Re-export public API
export Index, dim, tags, id, hastag
export Tensor, rank, dims, indices, storage_kind, data
export StorageKind, DenseF64, DenseC64, DiagF64, DiagC64

"""
    Index

A tensor index with dimension, ID, and tags.

Wraps a Rust `DefaultIndex<DynId, NoSymmSpace>` which corresponds to
ITensors.jl's `Index{Int}` (no quantum number symmetry).

# Constructors

- `Index(dim::Integer)` - Create index with dimension
- `Index(dim::Integer; tags::AbstractString)` - Create with tags
- `Index(dim::Integer, id::UInt128; tags::AbstractString)` - Create with specific ID

# Properties

- `dim(i::Index)` - Get the dimension
- `id(i::Index)` - Get the unique ID as UInt128
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

function Index(dim::Integer, id::UInt128; tags::AbstractString="")
    dim > 0 || throw(ArgumentError("Index dimension must be positive, got $dim"))
    id_hi = UInt64(id >> 64)
    id_lo = UInt64(id & 0xFFFFFFFFFFFFFFFF)
    ptr = C_API.t4a_index_new_with_id(dim, id_hi, id_lo, tags)
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
    id(i::Index) -> UInt128

Get the unique ID of an index.
"""
function id(i::Index)
    hi = Ref{UInt64}(0)
    lo = Ref{UInt64}(0)
    status = C_API.t4a_index_id_u128(i.ptr, hi, lo)
    C_API.check_status(status)
    return (UInt128(hi[]) << 64) | UInt128(lo[])
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
    id_short = string(id_val, base=16)[end-7:end]  # Last 8 hex digits
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

# Clone
function Base.copy(i::Index)
    ptr = C_API.t4a_index_clone(i.ptr)
    return Index(ptr)
end

# Equality based on ID (same as Rust side)
function Base.:(==)(i1::Index, i2::Index)
    return id(i1) == id(i2)
end

function Base.hash(i::Index, h::UInt)
    return hash(id(i), h)
end

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

Wraps a Rust `TensorDynLen<DynId, NoSymmSpace>` which corresponds to
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

# Clone
function Base.copy(t::Tensor)
    ptr = C_API.t4a_tensor_clone(t.ptr)
    return Tensor(ptr)
end

end # module
