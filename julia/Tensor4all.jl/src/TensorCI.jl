"""
    TensorCI

High-level Julia wrappers for TensorCI2 from tensor4all-tensorci.

This provides tensor cross interpolation algorithms for approximating
high-dimensional functions as tensor trains.
"""
module TensorCI

using ..C_API
using ..SimpleTT: SimpleTensorTrain

export TensorCI2, crossinterpolate2, crossinterpolate2_tci

# ============================================================================
# Callback infrastructure for passing Julia functions to Rust
# ============================================================================

"""
    _trampoline(indices_ptr::Ptr{Int64}, n_indices::Csize_t,
                result_ptr::Ptr{Float64}, user_data::Ptr{Cvoid})::Cint

Internal trampoline function for C callbacks. Converts C-style call to Julia closure.
"""
function _trampoline(
    indices_ptr::Ptr{Int64},
    n_indices::Csize_t,
    result_ptr::Ptr{Float64},
    user_data::Ptr{Cvoid}
)::Cint
    try
        # Recover the Julia function from user_data
        f_ref = unsafe_pointer_to_objref(user_data)::Ref{Any}
        f = f_ref[]

        # Convert indices from C array
        indices = unsafe_wrap(Array, indices_ptr, Int(n_indices))

        # Call the function (returns Float64)
        val = Float64(f(indices...))

        # Store result
        unsafe_store!(result_ptr, val)
        return Cint(0)
    catch e
        # Log error and return error status
        @error "Error in TCI callback" exception=(e, catch_backtrace())
        return Cint(-1)
    end
end

# Create the C function pointer lazily to avoid precompilation issues
const _TRAMPOLINE_PTR_REF = Ref{Ptr{Cvoid}}(C_NULL)

function _get_trampoline_ptr()
    if _TRAMPOLINE_PTR_REF[] == C_NULL
        _TRAMPOLINE_PTR_REF[] = @cfunction(
            _trampoline,
            Cint,
            (Ptr{Int64}, Csize_t, Ptr{Float64}, Ptr{Cvoid})
        )
    end
    return _TRAMPOLINE_PTR_REF[]
end

# ============================================================================
# TensorCI2 type
# ============================================================================

"""
    TensorCI2{T<:Real}

A TCI (Tensor Cross Interpolation) object for 2-site algorithm.

This wraps the Rust TensorCI2 and provides access to the interpolated
tensor train representation of a function.
"""
mutable struct TensorCI2{T<:Real}
    ptr::Ptr{Cvoid}
    local_dims::Vector{Int}

    function TensorCI2{Float64}(ptr::Ptr{Cvoid}, local_dims::Vector{Int})
        ptr == C_NULL && error("Failed to create TensorCI2: null pointer")
        tci = new{Float64}(ptr, local_dims)
        finalizer(tci) do obj
            C_API.t4a_tci2_f64_release(obj.ptr)
        end
        return tci
    end
end

"""
    TensorCI2(local_dims::Vector{<:Integer})

Create a new empty TensorCI2 object with the given local dimensions.

# Example
```julia
tci = TensorCI2([2, 3, 4])  # 3 sites with dimensions 2, 3, 4
```
"""
function TensorCI2(local_dims::Vector{<:Integer})
    dims = Csize_t.(local_dims)
    ptr = C_API.t4a_tci2_f64_new(dims)
    return TensorCI2{Float64}(ptr, Int.(local_dims))
end

"""
    length(tci::TensorCI2) -> Int

Get the number of sites.
"""
function Base.length(tci::TensorCI2{Float64})
    out_len = Ref{Csize_t}(0)
    status = C_API.t4a_tci2_f64_len(tci.ptr, out_len)
    C_API.check_status(status)
    return Int(out_len[])
end

"""
    rank(tci::TensorCI2) -> Int

Get the current maximum bond dimension (rank).
"""
function rank(tci::TensorCI2{Float64})
    out_rank = Ref{Csize_t}(0)
    status = C_API.t4a_tci2_f64_rank(tci.ptr, out_rank)
    C_API.check_status(status)
    return Int(out_rank[])
end

"""
    link_dims(tci::TensorCI2) -> Vector{Int}

Get the link (bond) dimensions.
"""
function link_dims(tci::TensorCI2{Float64})
    n = length(tci)
    n <= 1 && return Int[]
    dims = Vector{Csize_t}(undef, n - 1)
    status = C_API.t4a_tci2_f64_link_dims(tci.ptr, dims)
    C_API.check_status(status)
    return Int.(dims)
end

"""
    max_sample_value(tci::TensorCI2) -> Float64

Get the maximum sample value encountered during interpolation.
"""
function max_sample_value(tci::TensorCI2{Float64})
    out_value = Ref{Cdouble}(0.0)
    status = C_API.t4a_tci2_f64_max_sample_value(tci.ptr, out_value)
    C_API.check_status(status)
    return out_value[]
end

"""
    max_bond_error(tci::TensorCI2) -> Float64

Get the maximum bond error from the last sweep.
"""
function max_bond_error(tci::TensorCI2{Float64})
    out_value = Ref{Cdouble}(0.0)
    status = C_API.t4a_tci2_f64_max_bond_error(tci.ptr, out_value)
    C_API.check_status(status)
    return out_value[]
end

"""
    add_global_pivots!(tci::TensorCI2, pivots::Vector{Vector{Int}})

Add global pivots to the TCI. Each pivot is a vector of indices (0-based).
"""
function add_global_pivots!(tci::TensorCI2{Float64}, pivots::Vector{Vector{Int}})
    isempty(pivots) && return

    n_sites = length(tci)
    n_pivots = length(pivots)

    # Flatten pivots into a single array
    flat_pivots = Csize_t[]
    for pivot in pivots
        length(pivot) == n_sites || error("Pivot length must match number of sites")
        append!(flat_pivots, Csize_t.(pivot))
    end

    status = C_API.t4a_tci2_f64_add_global_pivots(tci.ptr, flat_pivots, n_pivots, n_sites)
    C_API.check_status(status)
end

"""
    to_tensor_train(tci::TensorCI2) -> SimpleTensorTrain

Convert the TCI to a SimpleTensorTrain.
"""
function to_tensor_train(tci::TensorCI2{Float64})
    ptr = C_API.t4a_tci2_f64_to_tensor_train(tci.ptr)
    ptr == C_NULL && error("Failed to convert TCI to TensorTrain")
    return SimpleTensorTrain(ptr)
end

"""
    show(io::IO, tci::TensorCI2)

Display TCI information.
"""
function Base.show(io::IO, tci::TensorCI2{T}) where T
    n = length(tci)
    r = rank(tci)
    print(io, "TensorCI2{$T}(sites=$n, rank=$r)")
end

function Base.show(io::IO, ::MIME"text/plain", tci::TensorCI2{T}) where T
    println(io, "TensorCI2{$T}")
    println(io, "  Sites: $(length(tci))")
    println(io, "  Local dims: $(tci.local_dims)")
    println(io, "  Link dims: $(link_dims(tci))")
    println(io, "  Max rank: $(rank(tci))")
    println(io, "  Max bond error: $(max_bond_error(tci))")
end

# ============================================================================
# High-level crossinterpolate2 function
# ============================================================================

"""
    crossinterpolate2(f, local_dims; kwargs...) -> (SimpleTensorTrain, Float64)

Perform cross interpolation of a function `f` to obtain a tensor train approximation.

# Arguments
- `f`: A function that takes `n_sites` integer arguments (0-based indices) and returns a Float64
- `local_dims`: Vector of local dimensions for each site

# Keyword Arguments
- `initial_pivots`: Initial pivots (default: `[[0, 0, ...]]`)
- `tolerance`: Relative tolerance for convergence (default: `1e-8`)
- `max_bonddim`: Maximum bond dimension (default: unlimited)
- `max_iter`: Maximum number of iterations (default: `20`)

# Returns
- `tt`: The resulting SimpleTensorTrain
- `final_error`: The final error estimate

# Example
```julia
# Interpolate a simple function
f(i, j, k) = 1.0 + 0.1 * i + 0.01 * j + 0.001 * k
tt, err = crossinterpolate2(f, [10, 10, 10]; tolerance=1e-10)

# Evaluate the TT
tt(0, 5, 3)  # Should be close to f(0, 5, 3)
```
"""
function crossinterpolate2_tci(
    f,
    local_dims::Vector{<:Integer};
    initial_pivots::Vector{Vector{Int}} = [zeros(Int, length(local_dims))],
    tolerance::Float64 = 1e-8,
    max_bonddim::Int = 0,  # 0 means unlimited
    max_iter::Int = 20
)
    n_sites = length(local_dims)
    dims = Csize_t.(local_dims)

    # Flatten initial pivots
    flat_pivots = Csize_t[]
    for pivot in initial_pivots
        length(pivot) == n_sites || error("Initial pivot length must match number of sites")
        append!(flat_pivots, Csize_t.(pivot))
    end
    n_initial_pivots = length(initial_pivots)

    # Wrap the function in a Ref so we can pass it as user_data
    f_ref = Ref{Any}(f)

    # Output variables
    out_tci = Ref{Ptr{Cvoid}}(C_NULL)
    out_final_error = Ref{Cdouble}(0.0)

    # Call C API with GC protection
    GC.@preserve f_ref begin
        user_data = pointer_from_objref(f_ref)
        trampoline_ptr = _get_trampoline_ptr()

        status = C_API.t4a_crossinterpolate2_f64(
            dims,
            flat_pivots,
            n_initial_pivots,
            trampoline_ptr,
            user_data,
            tolerance,
            max_bonddim,
            max_iter,
            out_tci,
            out_final_error
        )

        C_API.check_status(status)
    end

    # Wrap result
    tci = TensorCI2{Float64}(out_tci[], Int.(local_dims))
    return tci, out_final_error[]
end

"""
    crossinterpolate2(f, local_dims; kwargs...) -> (SimpleTensorTrain, Float64)

High-level wrapper that returns a `SimpleTensorTrain`. For advanced use, see
[`crossinterpolate2_tci`](@ref), which also returns the underlying `TensorCI2`.
"""
function crossinterpolate2(
    f,
    local_dims::Vector{<:Integer};
    initial_pivots::Vector{Vector{Int}} = [zeros(Int, length(local_dims))],
    tolerance::Float64 = 1e-8,
    max_bonddim::Int = 0,  # 0 means unlimited
    max_iter::Int = 20
)
    tci, err = crossinterpolate2_tci(
        f,
        local_dims;
        initial_pivots=initial_pivots,
        tolerance=tolerance,
        max_bonddim=max_bonddim,
        max_iter=max_iter,
    )
    return to_tensor_train(tci), err
end

end # module TensorCI
