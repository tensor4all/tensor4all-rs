"""
    TensorTrain

Submodule for tensor train (Matrix Product State) operations.

# Basic Usage

```julia
using Tensor4all.TensorTrain

# Create a constant tensor train
tt = constant(Float64, [2, 3, 2], 1.0)

# Operations
s = sum(tt)
n = norm(tt)

# Arithmetic
tt2 = constant(Float64, [2, 3, 2], 2.0)
tt3 = tt + tt2
```
"""
module TensorTrain

using Libdl

# Import parent module's library management
import ..get_lib, ..T4A_SUCCESS

# ============================================================================
# TensorTrainF64
# ============================================================================

"""
    TensorTrainF64

A tensor train (Matrix Product State) with Float64 elements.

Wraps the Rust `TensorTrain<f64>` type via the C API.
"""
mutable struct TensorTrainF64
    ptr::Ptr{Cvoid}

    function TensorTrainF64(ptr::Ptr{Cvoid})
        ptr == C_NULL && error("Failed to create TensorTrainF64 (null pointer)")
        tt = new(ptr)
        finalizer(_release_tt_f64, tt)
        return tt
    end
end

function _release_tt_f64(tt::TensorTrainF64)
    if tt.ptr != C_NULL
        lib = get_lib()
        ccall(Libdl.dlsym(lib, :t4a_tt_f64_release), Cvoid, (Ptr{Cvoid},), tt.ptr)
        tt.ptr = C_NULL
    end
end

# ============================================================================
# TensorTrainC64 (Complex{Float64})
# ============================================================================

"""
    TensorTrainC64

A tensor train (Matrix Product State) with Complex{Float64} elements.

Wraps the Rust `TensorTrain<Complex64>` type via the C API.
"""
mutable struct TensorTrainC64
    ptr::Ptr{Cvoid}

    function TensorTrainC64(ptr::Ptr{Cvoid})
        ptr == C_NULL && error("Failed to create TensorTrainC64 (null pointer)")
        tt = new(ptr)
        finalizer(_release_tt_c64, tt)
        return tt
    end
end

function _release_tt_c64(tt::TensorTrainC64)
    if tt.ptr != C_NULL
        lib = get_lib()
        ccall(Libdl.dlsym(lib, :t4a_tt_c64_release), Cvoid, (Ptr{Cvoid},), tt.ptr)
        tt.ptr = C_NULL
    end
end

# ============================================================================
# Constructors - Float64
# ============================================================================

"""
    zeros(::Type{Float64}, site_dims::Vector{Int})

Create a tensor train representing the zero function.
"""
function zeros(::Type{Float64}, site_dims::Vector{Int})
    lib = get_lib()
    dims = convert(Vector{Csize_t}, site_dims)
    ptr = ccall(
        Libdl.dlsym(lib, :t4a_tt_f64_new_zeros),
        Ptr{Cvoid},
        (Ptr{Csize_t}, Csize_t),
        dims,
        length(dims),
    )
    return TensorTrainF64(ptr)
end

"""
    constant(::Type{Float64}, site_dims::Vector{Int}, value::Float64)

Create a tensor train representing a constant function.
"""
function constant(::Type{Float64}, site_dims::Vector{Int}, value::Float64)
    lib = get_lib()
    dims = convert(Vector{Csize_t}, site_dims)
    ptr = ccall(
        Libdl.dlsym(lib, :t4a_tt_f64_new_constant),
        Ptr{Cvoid},
        (Ptr{Csize_t}, Csize_t, Cdouble),
        dims,
        length(dims),
        value,
    )
    return TensorTrainF64(ptr)
end

# ============================================================================
# Constructors - ComplexF64
# ============================================================================

"""
    zeros(::Type{ComplexF64}, site_dims::Vector{Int})

Create a complex tensor train representing the zero function.
"""
function zeros(::Type{ComplexF64}, site_dims::Vector{Int})
    lib = get_lib()
    dims = convert(Vector{Csize_t}, site_dims)
    ptr = ccall(
        Libdl.dlsym(lib, :t4a_tt_c64_new_zeros),
        Ptr{Cvoid},
        (Ptr{Csize_t}, Csize_t),
        dims,
        length(dims),
    )
    return TensorTrainC64(ptr)
end

"""
    constant(::Type{ComplexF64}, site_dims::Vector{Int}, value::ComplexF64)

Create a complex tensor train representing a constant function.
"""
function constant(::Type{ComplexF64}, site_dims::Vector{Int}, value::ComplexF64)
    lib = get_lib()
    dims = convert(Vector{Csize_t}, site_dims)
    ptr = ccall(
        Libdl.dlsym(lib, :t4a_tt_c64_new_constant),
        Ptr{Cvoid},
        (Ptr{Csize_t}, Csize_t, Cdouble, Cdouble),
        dims,
        length(dims),
        real(value),
        imag(value),
    )
    return TensorTrainC64(ptr)
end

# ============================================================================
# Properties - Float64
# ============================================================================

function Base.length(tt::TensorTrainF64)
    lib = get_lib()
    out_len = Ref{Csize_t}(0)
    status = ccall(
        Libdl.dlsym(lib, :t4a_tt_f64_len),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}),
        tt.ptr,
        out_len,
    )
    status == T4A_SUCCESS || error("Failed to get length: status = $status")
    return Int(out_len[])
end

"""
    site_dims(tt::TensorTrainF64)

Get the site (physical) dimensions of the tensor train.
"""
function site_dims(tt::TensorTrainF64)
    lib = get_lib()
    n = length(tt)
    out_dims = Vector{Csize_t}(undef, n)
    status = ccall(
        Libdl.dlsym(lib, :t4a_tt_f64_site_dims),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}, Csize_t),
        tt.ptr,
        out_dims,
        n,
    )
    status == T4A_SUCCESS || error("Failed to get site_dims: status = $status")
    return convert(Vector{Int}, out_dims)
end

"""
    link_dims(tt::TensorTrainF64)

Get the link (bond) dimensions of the tensor train.
"""
function link_dims(tt::TensorTrainF64)
    lib = get_lib()
    n = length(tt)
    if n <= 1
        return Int[]
    end
    out_dims = Vector{Csize_t}(undef, n - 1)
    status = ccall(
        Libdl.dlsym(lib, :t4a_tt_f64_link_dims),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}, Csize_t),
        tt.ptr,
        out_dims,
        n - 1,
    )
    status == T4A_SUCCESS || error("Failed to get link_dims: status = $status")
    return convert(Vector{Int}, out_dims)
end

"""
    rank(tt::TensorTrainF64)

Get the maximum bond dimension (rank) of the tensor train.
"""
function rank(tt::TensorTrainF64)
    lib = get_lib()
    out_rank = Ref{Csize_t}(0)
    status = ccall(
        Libdl.dlsym(lib, :t4a_tt_f64_rank),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}),
        tt.ptr,
        out_rank,
    )
    status == T4A_SUCCESS || error("Failed to get rank: status = $status")
    return Int(out_rank[])
end

# ============================================================================
# Properties - ComplexF64
# ============================================================================

function Base.length(tt::TensorTrainC64)
    lib = get_lib()
    out_len = Ref{Csize_t}(0)
    status = ccall(
        Libdl.dlsym(lib, :t4a_tt_c64_len),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}),
        tt.ptr,
        out_len,
    )
    status == T4A_SUCCESS || error("Failed to get length: status = $status")
    return Int(out_len[])
end

function site_dims(tt::TensorTrainC64)
    lib = get_lib()
    n = length(tt)
    out_dims = Vector{Csize_t}(undef, n)
    status = ccall(
        Libdl.dlsym(lib, :t4a_tt_c64_site_dims),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}, Csize_t),
        tt.ptr,
        out_dims,
        n,
    )
    status == T4A_SUCCESS || error("Failed to get site_dims: status = $status")
    return convert(Vector{Int}, out_dims)
end

function link_dims(tt::TensorTrainC64)
    lib = get_lib()
    n = length(tt)
    if n <= 1
        return Int[]
    end
    out_dims = Vector{Csize_t}(undef, n - 1)
    status = ccall(
        Libdl.dlsym(lib, :t4a_tt_c64_link_dims),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}, Csize_t),
        tt.ptr,
        out_dims,
        n - 1,
    )
    status == T4A_SUCCESS || error("Failed to get link_dims: status = $status")
    return convert(Vector{Int}, out_dims)
end

function rank(tt::TensorTrainC64)
    lib = get_lib()
    out_rank = Ref{Csize_t}(0)
    status = ccall(
        Libdl.dlsym(lib, :t4a_tt_c64_rank),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}),
        tt.ptr,
        out_rank,
    )
    status == T4A_SUCCESS || error("Failed to get rank: status = $status")
    return Int(out_rank[])
end

# ============================================================================
# Evaluation - Float64
# ============================================================================

"""
    evaluate(tt::TensorTrainF64, indices::Vector{Int})

Evaluate the tensor train at a given index set.

# Arguments
- `indices`: Vector of indices (1-based in Julia, converted to 0-based for Rust)
"""
function evaluate(tt::TensorTrainF64, indices::Vector{Int})
    lib = get_lib()
    idx = convert(Vector{Csize_t}, indices .- 1)
    out_value = Ref{Cdouble}(0.0)
    status = ccall(
        Libdl.dlsym(lib, :t4a_tt_f64_evaluate),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}, Csize_t, Ptr{Cdouble}),
        tt.ptr,
        idx,
        length(idx),
        out_value,
    )
    status == T4A_SUCCESS || error("Failed to evaluate: status = $status")
    return out_value[]
end

function evaluate(tt::TensorTrainC64, indices::Vector{Int})
    lib = get_lib()
    idx = convert(Vector{Csize_t}, indices .- 1)
    out_re = Ref{Cdouble}(0.0)
    out_im = Ref{Cdouble}(0.0)
    status = ccall(
        Libdl.dlsym(lib, :t4a_tt_c64_evaluate),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}, Csize_t, Ptr{Cdouble}, Ptr{Cdouble}),
        tt.ptr,
        idx,
        length(idx),
        out_re,
        out_im,
    )
    status == T4A_SUCCESS || error("Failed to evaluate: status = $status")
    return ComplexF64(out_re[], out_im[])
end

# ============================================================================
# Reductions - Float64
# ============================================================================

"""
    sum(tt::TensorTrainF64)

Compute the sum of all elements in the tensor train.
"""
function Base.sum(tt::TensorTrainF64)
    lib = get_lib()
    out_sum = Ref{Cdouble}(0.0)
    status = ccall(
        Libdl.dlsym(lib, :t4a_tt_f64_sum),
        Cint,
        (Ptr{Cvoid}, Ptr{Cdouble}),
        tt.ptr,
        out_sum,
    )
    status == T4A_SUCCESS || error("Failed to compute sum: status = $status")
    return out_sum[]
end

function Base.sum(tt::TensorTrainC64)
    lib = get_lib()
    out_re = Ref{Cdouble}(0.0)
    out_im = Ref{Cdouble}(0.0)
    status = ccall(
        Libdl.dlsym(lib, :t4a_tt_c64_sum),
        Cint,
        (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}),
        tt.ptr,
        out_re,
        out_im,
    )
    status == T4A_SUCCESS || error("Failed to compute sum: status = $status")
    return ComplexF64(out_re[], out_im[])
end

"""
    norm(tt::TensorTrainF64)

Compute the Frobenius norm of the tensor train.
"""
function norm(tt::TensorTrainF64)
    lib = get_lib()
    out_norm = Ref{Cdouble}(0.0)
    status = ccall(
        Libdl.dlsym(lib, :t4a_tt_f64_norm),
        Cint,
        (Ptr{Cvoid}, Ptr{Cdouble}),
        tt.ptr,
        out_norm,
    )
    status == T4A_SUCCESS || error("Failed to compute norm: status = $status")
    return out_norm[]
end

function norm(tt::TensorTrainC64)
    lib = get_lib()
    out_norm = Ref{Cdouble}(0.0)
    status = ccall(
        Libdl.dlsym(lib, :t4a_tt_c64_norm),
        Cint,
        (Ptr{Cvoid}, Ptr{Cdouble}),
        tt.ptr,
        out_norm,
    )
    status == T4A_SUCCESS || error("Failed to compute norm: status = $status")
    return out_norm[]
end

"""
    log_norm(tt)

Compute the logarithm of the Frobenius norm.

This is more numerically stable than `log(norm(tt))` for very large or small norms.
"""
function log_norm(tt::TensorTrainF64)
    lib = get_lib()
    out_log_norm = Ref{Cdouble}(0.0)
    status = ccall(
        Libdl.dlsym(lib, :t4a_tt_f64_log_norm),
        Cint,
        (Ptr{Cvoid}, Ptr{Cdouble}),
        tt.ptr,
        out_log_norm,
    )
    status == T4A_SUCCESS || error("Failed to compute log_norm: status = $status")
    return out_log_norm[]
end

function log_norm(tt::TensorTrainC64)
    lib = get_lib()
    out_log_norm = Ref{Cdouble}(0.0)
    status = ccall(
        Libdl.dlsym(lib, :t4a_tt_c64_log_norm),
        Cint,
        (Ptr{Cvoid}, Ptr{Cdouble}),
        tt.ptr,
        out_log_norm,
    )
    status == T4A_SUCCESS || error("Failed to compute log_norm: status = $status")
    return out_log_norm[]
end

# ============================================================================
# Scaling
# ============================================================================

"""
    scale!(tt::TensorTrainF64, factor::Float64)

Scale the tensor train by a factor in place.
"""
function scale!(tt::TensorTrainF64, factor::Float64)
    lib = get_lib()
    status = ccall(
        Libdl.dlsym(lib, :t4a_tt_f64_scale_inplace),
        Cint,
        (Ptr{Cvoid}, Cdouble),
        tt.ptr,
        factor,
    )
    status == T4A_SUCCESS || error("Failed to scale: status = $status")
    return tt
end

function scale!(tt::TensorTrainC64, factor::Number)
    f = ComplexF64(factor)
    lib = get_lib()
    status = ccall(
        Libdl.dlsym(lib, :t4a_tt_c64_scale_inplace),
        Cint,
        (Ptr{Cvoid}, Cdouble, Cdouble),
        tt.ptr,
        real(f),
        imag(f),
    )
    status == T4A_SUCCESS || error("Failed to scale: status = $status")
    return tt
end

"""
    scaled(tt, factor)

Create a new tensor train scaled by a factor. The original is unchanged.
"""
function scaled(tt::TensorTrainF64, factor::Float64)
    lib = get_lib()
    ptr = ccall(
        Libdl.dlsym(lib, :t4a_tt_f64_scaled),
        Ptr{Cvoid},
        (Ptr{Cvoid}, Cdouble),
        tt.ptr,
        factor,
    )
    return TensorTrainF64(ptr)
end

function scaled(tt::TensorTrainC64, factor::Number)
    f = ComplexF64(factor)
    lib = get_lib()
    ptr = ccall(
        Libdl.dlsym(lib, :t4a_tt_c64_scaled),
        Ptr{Cvoid},
        (Ptr{Cvoid}, Cdouble, Cdouble),
        tt.ptr,
        real(f),
        imag(f),
    )
    return TensorTrainC64(ptr)
end

# ============================================================================
# Full tensor conversion
# ============================================================================

"""
    _row_to_column_major(data::Vector{T}, dims::Tuple) where T

Convert row-major data to column-major (Julia convention).
"""
function _row_to_column_major(data::Vector{T}, dims::Tuple) where {T}
    if isempty(dims)
        return data
    end
    arr = reshape(data, reverse(dims)...)
    perm = reverse(1:length(dims))
    return permutedims(arr, perm)
end

"""
    fulltensor(tt)

Convert the tensor train to a full tensor (dense array).

Warning: This can be very large for high-dimensional tensors!
"""
function fulltensor(tt::TensorTrainF64)
    lib = get_lib()

    out_len = Ref{Csize_t}(0)
    status = ccall(
        Libdl.dlsym(lib, :t4a_tt_f64_fulltensor),
        Cint,
        (Ptr{Cvoid}, Ptr{Cdouble}, Csize_t, Ptr{Csize_t}),
        tt.ptr,
        C_NULL,
        0,
        out_len,
    )
    status == T4A_SUCCESS || error("Failed to query fulltensor length: status = $status")

    data = Vector{Cdouble}(undef, out_len[])
    status = ccall(
        Libdl.dlsym(lib, :t4a_tt_f64_fulltensor),
        Cint,
        (Ptr{Cvoid}, Ptr{Cdouble}, Csize_t, Ptr{Csize_t}),
        tt.ptr,
        data,
        out_len[],
        out_len,
    )
    status == T4A_SUCCESS || error("Failed to get fulltensor: status = $status")

    dims = site_dims(tt)
    return _row_to_column_major(data, Tuple(dims))
end

function fulltensor(tt::TensorTrainC64)
    lib = get_lib()

    out_len = Ref{Csize_t}(0)
    status = ccall(
        Libdl.dlsym(lib, :t4a_tt_c64_fulltensor),
        Cint,
        (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Csize_t, Ptr{Csize_t}),
        tt.ptr,
        C_NULL,
        C_NULL,
        0,
        out_len,
    )
    status == T4A_SUCCESS || error("Failed to query fulltensor length: status = $status")

    data_re = Vector{Cdouble}(undef, out_len[])
    data_im = Vector{Cdouble}(undef, out_len[])
    status = ccall(
        Libdl.dlsym(lib, :t4a_tt_c64_fulltensor),
        Cint,
        (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Csize_t, Ptr{Csize_t}),
        tt.ptr,
        data_re,
        data_im,
        out_len[],
        out_len,
    )
    status == T4A_SUCCESS || error("Failed to get fulltensor: status = $status")

    data = ComplexF64.(data_re, data_im)
    dims = site_dims(tt)
    return _row_to_column_major(data, Tuple(dims))
end

# ============================================================================
# Copy
# ============================================================================

function Base.copy(tt::TensorTrainF64)
    lib = get_lib()
    ptr = ccall(
        Libdl.dlsym(lib, :t4a_tt_f64_clone),
        Ptr{Cvoid},
        (Ptr{Cvoid},),
        tt.ptr,
    )
    return TensorTrainF64(ptr)
end

function Base.copy(tt::TensorTrainC64)
    lib = get_lib()
    ptr = ccall(
        Libdl.dlsym(lib, :t4a_tt_c64_clone),
        Ptr{Cvoid},
        (Ptr{Cvoid},),
        tt.ptr,
    )
    return TensorTrainC64(ptr)
end

# ============================================================================
# Arithmetic Operations
# ============================================================================

"""
    add(a, b)

Add two tensor trains element-wise.
"""
function add(a::TensorTrainF64, b::TensorTrainF64)
    lib = get_lib()
    ptr = ccall(
        Libdl.dlsym(lib, :t4a_tt_f64_add),
        Ptr{Cvoid},
        (Ptr{Cvoid}, Ptr{Cvoid}),
        a.ptr,
        b.ptr,
    )
    return TensorTrainF64(ptr)
end

function add(a::TensorTrainC64, b::TensorTrainC64)
    lib = get_lib()
    ptr = ccall(
        Libdl.dlsym(lib, :t4a_tt_c64_add),
        Ptr{Cvoid},
        (Ptr{Cvoid}, Ptr{Cvoid}),
        a.ptr,
        b.ptr,
    )
    return TensorTrainC64(ptr)
end

"""
    sub(a, b)

Subtract two tensor trains element-wise (a - b).
"""
function sub(a::TensorTrainF64, b::TensorTrainF64)
    lib = get_lib()
    ptr = ccall(
        Libdl.dlsym(lib, :t4a_tt_f64_sub),
        Ptr{Cvoid},
        (Ptr{Cvoid}, Ptr{Cvoid}),
        a.ptr,
        b.ptr,
    )
    return TensorTrainF64(ptr)
end

function sub(a::TensorTrainC64, b::TensorTrainC64)
    lib = get_lib()
    ptr = ccall(
        Libdl.dlsym(lib, :t4a_tt_c64_sub),
        Ptr{Cvoid},
        (Ptr{Cvoid}, Ptr{Cvoid}),
        a.ptr,
        b.ptr,
    )
    return TensorTrainC64(ptr)
end

"""
    negate(tt)

Negate a tensor train (multiply by -1).
"""
function negate(tt::TensorTrainF64)
    lib = get_lib()
    ptr = ccall(
        Libdl.dlsym(lib, :t4a_tt_f64_negate),
        Ptr{Cvoid},
        (Ptr{Cvoid},),
        tt.ptr,
    )
    return TensorTrainF64(ptr)
end

function negate(tt::TensorTrainC64)
    lib = get_lib()
    ptr = ccall(
        Libdl.dlsym(lib, :t4a_tt_c64_negate),
        Ptr{Cvoid},
        (Ptr{Cvoid},),
        tt.ptr,
    )
    return TensorTrainC64(ptr)
end

"""
    reverse(tt)

Reverse a tensor train (swap left and right).
"""
function Base.reverse(tt::TensorTrainF64)
    lib = get_lib()
    ptr = ccall(
        Libdl.dlsym(lib, :t4a_tt_f64_reverse),
        Ptr{Cvoid},
        (Ptr{Cvoid},),
        tt.ptr,
    )
    return TensorTrainF64(ptr)
end

function Base.reverse(tt::TensorTrainC64)
    lib = get_lib()
    ptr = ccall(
        Libdl.dlsym(lib, :t4a_tt_c64_reverse),
        Ptr{Cvoid},
        (Ptr{Cvoid},),
        tt.ptr,
    )
    return TensorTrainC64(ptr)
end

"""
    hadamard(a, b)

Compute the Hadamard (element-wise) product of two tensor trains.
"""
function hadamard(a::TensorTrainF64, b::TensorTrainF64)
    lib = get_lib()
    ptr = ccall(
        Libdl.dlsym(lib, :t4a_tt_f64_hadamard),
        Ptr{Cvoid},
        (Ptr{Cvoid}, Ptr{Cvoid}),
        a.ptr,
        b.ptr,
    )
    return TensorTrainF64(ptr)
end

function hadamard(a::TensorTrainC64, b::TensorTrainC64)
    lib = get_lib()
    ptr = ccall(
        Libdl.dlsym(lib, :t4a_tt_c64_hadamard),
        Ptr{Cvoid},
        (Ptr{Cvoid}, Ptr{Cvoid}),
        a.ptr,
        b.ptr,
    )
    return TensorTrainC64(ptr)
end

"""
    dot(a, b)

Compute the inner product (dot product) of two tensor trains.
"""
function dot(a::TensorTrainF64, b::TensorTrainF64)
    lib = get_lib()
    out_dot = Ref{Cdouble}(0.0)
    status = ccall(
        Libdl.dlsym(lib, :t4a_tt_f64_dot),
        Cint,
        (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cdouble}),
        a.ptr,
        b.ptr,
        out_dot,
    )
    status == T4A_SUCCESS || error("Failed to compute dot product: status = $status")
    return out_dot[]
end

function dot(a::TensorTrainC64, b::TensorTrainC64)
    lib = get_lib()
    out_re = Ref{Cdouble}(0.0)
    out_im = Ref{Cdouble}(0.0)
    status = ccall(
        Libdl.dlsym(lib, :t4a_tt_c64_dot),
        Cint,
        (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}),
        a.ptr,
        b.ptr,
        out_re,
        out_im,
    )
    status == T4A_SUCCESS || error("Failed to compute dot product: status = $status")
    return ComplexF64(out_re[], out_im[])
end

# ============================================================================
# Compression
# ============================================================================

"""
    compress!(tt; tolerance=1e-12, max_bond_dim=0)

Compress a tensor train in-place.
"""
function compress!(tt::TensorTrainF64; tolerance::Float64 = 1e-12, max_bond_dim::Int = 0)
    lib = get_lib()
    status = ccall(
        Libdl.dlsym(lib, :t4a_tt_f64_compress),
        Cint,
        (Ptr{Cvoid}, Cdouble, Csize_t),
        tt.ptr,
        tolerance,
        max_bond_dim,
    )
    status == T4A_SUCCESS || error("Failed to compress: status = $status")
    return tt
end

function compress!(tt::TensorTrainC64; tolerance::Float64 = 1e-12, max_bond_dim::Int = 0)
    lib = get_lib()
    status = ccall(
        Libdl.dlsym(lib, :t4a_tt_c64_compress),
        Cint,
        (Ptr{Cvoid}, Cdouble, Csize_t),
        tt.ptr,
        tolerance,
        max_bond_dim,
    )
    status == T4A_SUCCESS || error("Failed to compress: status = $status")
    return tt
end

"""
    compressed(tt; tolerance=1e-12, max_bond_dim=0)

Create a compressed copy of a tensor train.
"""
function compressed(tt::TensorTrainF64; tolerance::Float64 = 1e-12, max_bond_dim::Int = 0)
    lib = get_lib()
    ptr = ccall(
        Libdl.dlsym(lib, :t4a_tt_f64_compressed),
        Ptr{Cvoid},
        (Ptr{Cvoid}, Cdouble, Csize_t),
        tt.ptr,
        tolerance,
        max_bond_dim,
    )
    return TensorTrainF64(ptr)
end

function compressed(tt::TensorTrainC64; tolerance::Float64 = 1e-12, max_bond_dim::Int = 0)
    lib = get_lib()
    ptr = ccall(
        Libdl.dlsym(lib, :t4a_tt_c64_compressed),
        Ptr{Cvoid},
        (Ptr{Cvoid}, Cdouble, Csize_t),
        tt.ptr,
        tolerance,
        max_bond_dim,
    )
    return TensorTrainC64(ptr)
end

# ============================================================================
# Operator overloads
# ============================================================================

Base.:+(a::TensorTrainF64, b::TensorTrainF64) = add(a, b)
Base.:-(a::TensorTrainF64, b::TensorTrainF64) = sub(a, b)
Base.:-(tt::TensorTrainF64) = negate(tt)
Base.:*(a::TensorTrainF64, b::TensorTrainF64) = hadamard(a, b)
Base.:*(tt::TensorTrainF64, factor::Real) = scaled(tt, Float64(factor))
Base.:*(factor::Real, tt::TensorTrainF64) = scaled(tt, Float64(factor))

Base.:+(a::TensorTrainC64, b::TensorTrainC64) = add(a, b)
Base.:-(a::TensorTrainC64, b::TensorTrainC64) = sub(a, b)
Base.:-(tt::TensorTrainC64) = negate(tt)
Base.:*(a::TensorTrainC64, b::TensorTrainC64) = hadamard(a, b)
Base.:*(tt::TensorTrainC64, factor::Number) = scaled(tt, factor)
Base.:*(factor::Number, tt::TensorTrainC64) = scaled(tt, factor)

# ============================================================================
# Exports
# ============================================================================

export TensorTrainF64, TensorTrainC64
export zeros, constant
export site_dims, link_dims, rank, evaluate
export norm, log_norm
export scale!, scaled, fulltensor
export add, sub, negate, hadamard, dot
export compress!, compressed

end # module TensorTrain
