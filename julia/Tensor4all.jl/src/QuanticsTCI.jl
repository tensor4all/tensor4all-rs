"""
    QuanticsTCI

High-level Julia wrappers for Quantics Tensor Cross Interpolation.

This module combines TCI with quantics grid representations to efficiently
interpolate functions on continuous or discrete domains.
"""
module QuanticsTCI

using ..C_API
using ..QuanticsGrids: DiscretizedGrid, InherentDiscreteGrid
using ..SimpleTT: SimpleTensorTrain

export QuanticsTensorCI2
export quanticscrossinterpolate, quanticscrossinterpolate_discrete
export rank, link_dims, evaluate, sum, integral, to_tensor_train

# ============================================================================
# Callback infrastructure for passing Julia functions to Rust
# ============================================================================

# For continuous domain (f64 coordinates)
function _trampoline_f64(
    coords_ptr::Ptr{Float64},
    ndims::Csize_t,
    result_ptr::Ptr{Float64},
    user_data::Ptr{Cvoid},
)::Cint
    try
        f_ref = unsafe_pointer_to_objref(user_data)::Ref{Any}
        f = f_ref[]
        coords = unsafe_wrap(Array, coords_ptr, Int(ndims))
        val = Float64(f(coords...))
        unsafe_store!(result_ptr, val)
        return Cint(0)
    catch e
        @error "Error in QTCI callback" exception = (e, catch_backtrace())
        return Cint(-1)
    end
end

# For discrete domain (i64 indices, 1-indexed)
function _trampoline_i64(
    indices_ptr::Ptr{Int64},
    ndims::Csize_t,
    result_ptr::Ptr{Float64},
    user_data::Ptr{Cvoid},
)::Cint
    try
        f_ref = unsafe_pointer_to_objref(user_data)::Ref{Any}
        f = f_ref[]
        indices = unsafe_wrap(Array, indices_ptr, Int(ndims))
        val = Float64(f(indices...))
        unsafe_store!(result_ptr, val)
        return Cint(0)
    catch e
        @error "Error in QTCI callback" exception = (e, catch_backtrace())
        return Cint(-1)
    end
end

# Create C function pointers lazily to avoid precompilation issues
const _TRAMPOLINE_F64_PTR = Ref{Ptr{Cvoid}}(C_NULL)
const _TRAMPOLINE_I64_PTR = Ref{Ptr{Cvoid}}(C_NULL)

function _get_trampoline_f64_ptr()
    if _TRAMPOLINE_F64_PTR[] == C_NULL
        _TRAMPOLINE_F64_PTR[] = @cfunction(
            _trampoline_f64,
            Cint,
            (Ptr{Float64}, Csize_t, Ptr{Float64}, Ptr{Cvoid})
        )
    end
    return _TRAMPOLINE_F64_PTR[]
end

function _get_trampoline_i64_ptr()
    if _TRAMPOLINE_I64_PTR[] == C_NULL
        _TRAMPOLINE_I64_PTR[] = @cfunction(
            _trampoline_i64,
            Cint,
            (Ptr{Int64}, Csize_t, Ptr{Float64}, Ptr{Cvoid})
        )
    end
    return _TRAMPOLINE_I64_PTR[]
end

# ============================================================================
# QuanticsTensorCI2 type
# ============================================================================

"""
    QuanticsTensorCI2

A Quantics TCI (Tensor Cross Interpolation) object.

This wraps the Rust QuanticsTCI and provides access to the interpolated
quantics tensor train representation of a function.

# Methods

- `rank(qtci)` - Get the maximum bond dimension
- `link_dims(qtci)` - Get the link (bond) dimensions
- `evaluate(qtci, indices)` - Evaluate at grid indices
- `sum(qtci)` - Compute the factorized sum over all grid points
- `integral(qtci)` - Compute the integral over the continuous domain
- `to_tensor_train(qtci)` - Convert to SimpleTensorTrain

# Callable Interface

The object can be called directly with indices:
```julia
qtci(1, 2)  # Equivalent to evaluate(qtci, [1, 2])
```
"""
mutable struct QuanticsTensorCI2
    ptr::Ptr{Cvoid}

    function QuanticsTensorCI2(ptr::Ptr{Cvoid})
        ptr == C_NULL && error("Failed to create QuanticsTensorCI2: null pointer")
        qtci = new(ptr)
        finalizer(qtci) do obj
            C_API.t4a_qtci_f64_release(obj.ptr)
        end
        return qtci
    end
end

"""
    rank(qtci::QuanticsTensorCI2) -> Int

Get the maximum bond dimension (rank).
"""
function rank(qtci::QuanticsTensorCI2)
    out_rank = Ref{Csize_t}(0)
    status = C_API.t4a_qtci_f64_rank(qtci.ptr, out_rank)
    C_API.check_status(status)
    return Int(out_rank[])
end

"""
    link_dims(qtci::QuanticsTensorCI2) -> Vector{Int}

Get the link (bond) dimensions.
"""
function link_dims(qtci::QuanticsTensorCI2)
    r = rank(qtci)
    r == 0 && return Int[]
    # Use a generous buffer since we don't know the exact number of sites
    buf = Vector{Csize_t}(undef, 1024)
    status = C_API.t4a_qtci_f64_link_dims(qtci.ptr, buf, Csize_t(length(buf)))
    C_API.check_status(status)
    # Find actual length by looking for trailing zeros
    result = Int.(buf)
    last_nonzero = findlast(x -> x > 0, result)
    return isnothing(last_nonzero) ? Int[] : result[1:last_nonzero]
end

"""
    evaluate(qtci::QuanticsTensorCI2, indices::Vector{<:Integer}) -> Float64

Evaluate the QTCI at the given grid indices (1-indexed).
"""
function evaluate(qtci::QuanticsTensorCI2, indices::Vector{<:Integer})
    idx = Int64.(indices)
    out_value = Ref{Cdouble}(0.0)
    status = C_API.t4a_qtci_f64_evaluate(qtci.ptr, idx, Csize_t(length(idx)), out_value)
    C_API.check_status(status)
    return out_value[]
end

# Callable interface: qtci(i, j, ...)
function (qtci::QuanticsTensorCI2)(indices::Integer...)
    evaluate(qtci, collect(Int64, indices))
end

"""
    sum(qtci::QuanticsTensorCI2) -> Float64

Compute the factorized sum over all grid points.
"""
function sum(qtci::QuanticsTensorCI2)
    out_value = Ref{Cdouble}(0.0)
    status = C_API.t4a_qtci_f64_sum(qtci.ptr, out_value)
    C_API.check_status(status)
    return out_value[]
end

"""
    integral(qtci::QuanticsTensorCI2) -> Float64

Compute the integral over the continuous domain.

This is the sum multiplied by the grid step sizes.
Only meaningful for QTCI constructed with a DiscretizedGrid.
For discrete grids, this returns the plain sum.
"""
function integral(qtci::QuanticsTensorCI2)
    out_value = Ref{Cdouble}(0.0)
    status = C_API.t4a_qtci_f64_integral(qtci.ptr, out_value)
    C_API.check_status(status)
    return out_value[]
end

"""
    to_tensor_train(qtci::QuanticsTensorCI2) -> SimpleTensorTrain

Convert the QTCI to a SimpleTensorTrain.
"""
function to_tensor_train(qtci::QuanticsTensorCI2)
    ptr = C_API.t4a_qtci_f64_to_tensor_train(qtci.ptr)
    ptr == C_NULL && error("Failed to convert QTCI to TensorTrain")
    return SimpleTensorTrain(ptr)
end

# Display
function Base.show(io::IO, qtci::QuanticsTensorCI2)
    r = rank(qtci)
    print(io, "QuanticsTensorCI2(rank=$r)")
end

# ============================================================================
# High-level interpolation functions
# ============================================================================

"""
    quanticscrossinterpolate(grid::DiscretizedGrid, f; kwargs...) -> QuanticsTensorCI2

Perform quantics cross interpolation on a continuous domain.

# Arguments
- `grid`: DiscretizedGrid describing the domain
- `f`: Function that takes Float64 coordinates and returns Float64

# Keyword Arguments
- `tolerance`: Convergence tolerance (default: 1e-8)
- `max_bonddim`: Maximum bond dimension, 0 = unlimited (default: 0)
- `max_iter`: Maximum iterations (default: 200)

# Example
```julia
using Tensor4all.QuanticsGrids
using Tensor4all.QuanticsTCI

grid = DiscretizedGrid(1, 10, [0.0], [1.0])
qtci = quanticscrossinterpolate(grid, x -> sin(x))
integral(qtci)  # Should be close to 1 - cos(1) ~ 0.4597
```
"""
function quanticscrossinterpolate(
    grid::DiscretizedGrid,
    f;
    tolerance::Float64 = 1e-8,
    max_bonddim::Int = 0,
    max_iter::Int = 200,
)
    f_ref = Ref{Any}(f)
    out_qtci = Ref{Ptr{Cvoid}}(C_NULL)

    GC.@preserve f_ref begin
        user_data = pointer_from_objref(f_ref)
        trampoline_ptr = _get_trampoline_f64_ptr()

        status = C_API.t4a_quanticscrossinterpolate_f64(
            grid.ptr,
            trampoline_ptr,
            user_data,
            tolerance,
            Csize_t(max_bonddim),
            Csize_t(max_iter),
            out_qtci,
        )

        C_API.check_status(status)
    end

    return QuanticsTensorCI2(out_qtci[])
end

"""
    quanticscrossinterpolate_discrete(sizes, f; kwargs...) -> QuanticsTensorCI2

Perform quantics cross interpolation on a discrete integer domain.

# Arguments
- `sizes`: Grid sizes per dimension (must be powers of 2)
- `f`: Function that takes 1-indexed integer indices and returns Float64

# Keyword Arguments
- `tolerance`: Convergence tolerance (default: 1e-8)
- `max_bonddim`: Maximum bond dimension, 0 = unlimited (default: 0)
- `max_iter`: Maximum iterations (default: 200)
- `unfoldingscheme`: :interleaved or :fused (default: :interleaved)

# Example
```julia
using Tensor4all.QuanticsTCI

qtci = quanticscrossinterpolate_discrete([8, 8], (i, j) -> Float64(i + j))
qtci(3, 4)  # Should be close to 7.0
```
"""
function quanticscrossinterpolate_discrete(
    sizes::Vector{<:Integer},
    f;
    tolerance::Float64 = 1e-8,
    max_bonddim::Int = 0,
    max_iter::Int = 200,
    unfoldingscheme::Symbol = :interleaved,
)
    scheme = unfoldingscheme == :fused ? Cint(0) : Cint(1)

    f_ref = Ref{Any}(f)
    out_qtci = Ref{Ptr{Cvoid}}(C_NULL)
    sizes_c = Csize_t.(sizes)

    GC.@preserve f_ref begin
        user_data = pointer_from_objref(f_ref)
        trampoline_ptr = _get_trampoline_i64_ptr()

        status = C_API.t4a_quanticscrossinterpolate_discrete_f64(
            sizes_c,
            Csize_t(length(sizes)),
            trampoline_ptr,
            user_data,
            tolerance,
            Csize_t(max_bonddim),
            Csize_t(max_iter),
            scheme,
            out_qtci,
        )

        C_API.check_status(status)
    end

    return QuanticsTensorCI2(out_qtci[])
end

end # module QuanticsTCI
