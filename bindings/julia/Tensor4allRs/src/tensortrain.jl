# TensorTrain bindings for Julia

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

"""
    zeros_tt(::Type{Float64}, site_dims::Vector{Int})

Create a tensor train representing the zero function.

# Arguments
- `site_dims`: Vector of site dimensions

# Returns
A new `TensorTrainF64` with all elements equal to zero.
"""
function zeros_tt(::Type{Float64}, site_dims::Vector{Int})
    lib = get_lib()
    dims = convert(Vector{Csize_t}, site_dims)
    ptr = ccall(
        Libdl.dlsym(lib, :t4a_tt_f64_new_zeros),
        Ptr{Cvoid},
        (Ptr{Csize_t}, Csize_t),
        dims,
        length(dims),
    )
    ptr == C_NULL && error("Failed to create TensorTrainF64")
    return TensorTrainF64(ptr)
end

"""
    constant_tt(::Type{Float64}, site_dims::Vector{Int}, value::Float64)

Create a tensor train representing a constant function.

# Arguments
- `site_dims`: Vector of site dimensions
- `value`: The constant value

# Returns
A new `TensorTrainF64` with all elements equal to `value`.
"""
function constant_tt(::Type{Float64}, site_dims::Vector{Int}, value::Float64)
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
    ptr == C_NULL && error("Failed to create TensorTrainF64")
    return TensorTrainF64(ptr)
end

"""
    Base.length(tt::TensorTrainF64)

Get the number of sites in the tensor train.
"""
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

"""
    evaluate(tt::TensorTrainF64, indices::Vector{Int})

Evaluate the tensor train at a given index set.

# Arguments
- `indices`: Vector of indices (1-based in Julia, converted to 0-based for Rust)

# Returns
The value of the tensor train at the given indices.
"""
function evaluate(tt::TensorTrainF64, indices::Vector{Int})
    lib = get_lib()
    # Convert to 0-based indices for Rust
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

"""
    sum_tt(tt::TensorTrainF64)

Compute the sum of all elements in the tensor train.
"""
function sum_tt(tt::TensorTrainF64)
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

"""
    norm_tt(tt::TensorTrainF64)

Compute the Frobenius norm of the tensor train.
"""
function norm_tt(tt::TensorTrainF64)
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

"""
    log_norm_tt(tt::TensorTrainF64)

Compute the logarithm of the Frobenius norm.

This is more numerically stable than `log(norm_tt(tt))` for very large or small norms.
"""
function log_norm_tt(tt::TensorTrainF64)
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

"""
    scaled(tt::TensorTrainF64, factor::Float64)

Create a new tensor train scaled by a factor.

The original tensor train is unchanged.
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
    ptr == C_NULL && error("Failed to create scaled TensorTrainF64")
    return TensorTrainF64(ptr)
end

"""
    fulltensor(tt::TensorTrainF64)

Convert the tensor train to a full tensor (dense array).

Returns the data reshaped to the site dimensions in column-major order (Julia convention).

Warning: This can be very large for high-dimensional tensors!
"""
function fulltensor(tt::TensorTrainF64)
    lib = get_lib()

    # Query length
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

    # Get data (row-major from Rust)
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

    # Convert row-major to column-major
    dims = site_dims(tt)
    return _row_to_column_major(data, Tuple(dims))
end

"""
    _row_to_column_major(data::Vector{T}, dims::Tuple) where T

Convert row-major data to column-major (Julia convention).
"""
function _row_to_column_major(data::Vector{T}, dims::Tuple) where {T}
    if isempty(dims)
        return data
    end
    # Data is row-major, so reverse dims for reshape
    arr = reshape(data, reverse(dims)...)
    # Reverse back to get column-major
    perm = reverse(1:length(dims))
    return permutedims(arr, perm)
end

"""
    Base.copy(tt::TensorTrainF64)

Create a copy (clone) of the tensor train.
"""
function Base.copy(tt::TensorTrainF64)
    lib = get_lib()
    ptr = ccall(
        Libdl.dlsym(lib, :t4a_tt_f64_clone),
        Ptr{Cvoid},
        (Ptr{Cvoid},),
        tt.ptr,
    )
    ptr == C_NULL && error("Failed to clone TensorTrainF64")
    return TensorTrainF64(ptr)
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

"""
    zeros_tt(::Type{ComplexF64}, site_dims::Vector{Int})

Create a complex tensor train representing the zero function.
"""
function zeros_tt(::Type{ComplexF64}, site_dims::Vector{Int})
    lib = get_lib()
    dims = convert(Vector{Csize_t}, site_dims)
    ptr = ccall(
        Libdl.dlsym(lib, :t4a_tt_c64_new_zeros),
        Ptr{Cvoid},
        (Ptr{Csize_t}, Csize_t),
        dims,
        length(dims),
    )
    ptr == C_NULL && error("Failed to create TensorTrainC64")
    return TensorTrainC64(ptr)
end

"""
    constant_tt(::Type{ComplexF64}, site_dims::Vector{Int}, value::ComplexF64)

Create a complex tensor train representing a constant function.
"""
function constant_tt(::Type{ComplexF64}, site_dims::Vector{Int}, value::ComplexF64)
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
    ptr == C_NULL && error("Failed to create TensorTrainC64")
    return TensorTrainC64(ptr)
end

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

function sum_tt(tt::TensorTrainC64)
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

function norm_tt(tt::TensorTrainC64)
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

function log_norm_tt(tt::TensorTrainC64)
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
    ptr == C_NULL && error("Failed to create scaled TensorTrainC64")
    return TensorTrainC64(ptr)
end

function fulltensor(tt::TensorTrainC64)
    lib = get_lib()

    # Query length
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

    # Get data
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

    # Combine to complex and convert layout
    data = ComplexF64.(data_re, data_im)
    dims = site_dims(tt)
    return _row_to_column_major(data, Tuple(dims))
end

function Base.copy(tt::TensorTrainC64)
    lib = get_lib()
    ptr = ccall(
        Libdl.dlsym(lib, :t4a_tt_c64_clone),
        Ptr{Cvoid},
        (Ptr{Cvoid},),
        tt.ptr,
    )
    ptr == C_NULL && error("Failed to clone TensorTrainC64")
    return TensorTrainC64(ptr)
end
