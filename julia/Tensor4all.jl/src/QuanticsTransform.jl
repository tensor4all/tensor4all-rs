"""
    QuanticsTransform

Quantics transformation operators for tree tensor networks (MPS).

Provides operators for shift, flip, phase rotation, cumulative sum, and
Fourier transform in quantics representation, wrapping the Rust
`tensor4all-quanticstransform` crate via C API.

# Usage
```julia
using Tensor4all.QuanticsTransform

op = shift_operator(4, 1)
result = apply(op, mps)
```
"""
module QuanticsTransform

using ..C_API
using ..TreeTN: TreeTensorNetwork

export LinearOperator
export shift_operator, flip_operator, phase_rotation_operator, cumsum_operator, fourier_operator
export apply
export BoundaryCondition, Periodic, Open

# ============================================================================
# Boundary condition enum
# ============================================================================

"""
    BoundaryCondition

Boundary condition for quantics operators.

- `Periodic` (0): Periodic boundary conditions
- `Open` (1): Open boundary conditions
"""
@enum BoundaryCondition begin
    Periodic = 0
    Open = 1
end

# ============================================================================
# LinearOperator type
# ============================================================================

"""
    LinearOperator

A linear operator that can be applied to a tree tensor network (MPS).

Wraps a Rust `LinearOperator` from the quantics transform crate.
Created via operator construction functions like `shift_operator`, `flip_operator`, etc.
"""
mutable struct LinearOperator
    ptr::Ptr{Cvoid}

    function LinearOperator(ptr::Ptr{Cvoid})
        ptr == C_NULL && error("Failed to create LinearOperator: null pointer")
        op = new(ptr)
        finalizer(op) do obj
            C_API.t4a_linop_release(obj.ptr)
        end
        return op
    end
end

# ============================================================================
# Operator construction functions
# ============================================================================

"""
    shift_operator(r::Integer, offset::Integer; bc=Periodic) -> LinearOperator

Create a shift operator: f(x) = g(x + offset) mod 2^r.

# Arguments
- `r`: Number of quantics bits
- `offset`: Shift offset (can be negative)
- `bc`: Boundary condition (`Periodic` or `Open`)
"""
function shift_operator(r::Integer, offset::Integer; bc::BoundaryCondition=Periodic)
    out = Ref{Ptr{Cvoid}}(C_NULL)
    status = C_API.t4a_qtransform_shift(Csize_t(r), Int64(offset), Cint(Int(bc)), out)
    C_API.check_status(status)
    return LinearOperator(out[])
end

"""
    flip_operator(r::Integer; bc=Periodic) -> LinearOperator

Create a flip operator: f(x) = g(2^r - x).

# Arguments
- `r`: Number of quantics bits
- `bc`: Boundary condition (`Periodic` or `Open`)
"""
function flip_operator(r::Integer; bc::BoundaryCondition=Periodic)
    out = Ref{Ptr{Cvoid}}(C_NULL)
    status = C_API.t4a_qtransform_flip(Csize_t(r), Cint(Int(bc)), out)
    C_API.check_status(status)
    return LinearOperator(out[])
end

"""
    phase_rotation_operator(r::Integer, theta::Real) -> LinearOperator

Create a phase rotation operator: f(x) = exp(i*theta*x) * g(x).

# Arguments
- `r`: Number of quantics bits
- `theta`: Phase angle
"""
function phase_rotation_operator(r::Integer, theta::Real)
    out = Ref{Ptr{Cvoid}}(C_NULL)
    status = C_API.t4a_qtransform_phase_rotation(Csize_t(r), Cdouble(theta), out)
    C_API.check_status(status)
    return LinearOperator(out[])
end

"""
    cumsum_operator(r::Integer) -> LinearOperator

Create a cumulative sum operator: y_i = sum_{j<i} x_j.

# Arguments
- `r`: Number of quantics bits
"""
function cumsum_operator(r::Integer)
    out = Ref{Ptr{Cvoid}}(C_NULL)
    status = C_API.t4a_qtransform_cumsum(Csize_t(r), out)
    C_API.check_status(status)
    return LinearOperator(out[])
end

"""
    fourier_operator(r::Integer; forward=true, maxbonddim=0, tolerance=0.0) -> LinearOperator

Create a Quantics Fourier Transform operator.

# Arguments
- `r`: Number of quantics bits
- `forward`: `true` for forward FT, `false` for inverse
- `maxbonddim`: Maximum bond dimension (0 = default of 12)
- `tolerance`: Tolerance for FT construction (0.0 = default of 1e-14)
"""
function fourier_operator(r::Integer; forward::Bool=true, maxbonddim::Integer=0, tolerance::Real=0.0)
    out = Ref{Ptr{Cvoid}}(C_NULL)
    status = C_API.t4a_qtransform_fourier(
        Csize_t(r), Cint(forward ? 1 : 0),
        Csize_t(maxbonddim), Cdouble(tolerance), out
    )
    C_API.check_status(status)
    return LinearOperator(out[])
end

# ============================================================================
# Operator application
# ============================================================================

"""
    apply(op::LinearOperator, state::TreeTensorNetwork; method=:naive, rtol=0.0, maxdim=0) -> TreeTensorNetwork

Apply a linear operator to a tree tensor network (MPS).

# Arguments
- `op`: Linear operator
- `state`: Input MPS/TreeTN
- `method`: Contraction method - `:naive`, `:zipup`, or `:fit`
- `rtol`: Relative tolerance (0.0 = default)
- `maxdim`: Maximum bond dimension (0 = unlimited)
"""
function apply(op::LinearOperator, state::TreeTensorNetwork;
               method::Symbol=:naive, rtol::Real=0.0, maxdim::Integer=0)
    method_int = if method == :naive
        Cint(0)
    elseif method == :zipup
        Cint(1)
    elseif method == :fit
        Cint(2)
    else
        error("Unknown method: $method. Use :naive, :zipup, or :fit")
    end

    out = Ref{Ptr{Cvoid}}(C_NULL)
    status = C_API.t4a_linop_apply(
        op.ptr, state.handle, method_int,
        Cdouble(rtol), Csize_t(maxdim), out
    )
    C_API.check_status(status)

    # Get the number of vertices from the result
    n_out = Ref{Csize_t}(0)
    status = C_API.t4a_treetn_num_vertices(out[], n_out)
    C_API.check_status(status)
    n = Int(n_out[])

    # Result TreeTN has usize node names (0, 1, ..., n-1)
    node_names = collect(0:n-1)
    node_map = Dict{Int, Int}(name => name for name in node_names)
    return TreeTensorNetwork{Int}(out[], node_map, node_names)
end

# ============================================================================
# Display
# ============================================================================

function Base.show(io::IO, op::LinearOperator)
    print(io, "LinearOperator()")
end

# ============================================================================
# Copy / Clone
# ============================================================================

function Base.copy(op::LinearOperator)
    ptr = C_API.t4a_linop_clone(op.ptr)
    return LinearOperator(ptr)
end

function Base.deepcopy(op::LinearOperator)
    ptr = C_API.t4a_linop_clone(op.ptr)
    return LinearOperator(ptr)
end

end # module QuanticsTransform
