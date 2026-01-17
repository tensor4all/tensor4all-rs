"""
    SimpleTT

High-level Julia wrappers for SimpleTT tensor trains from tensor4all-simplett.

SimpleTT is a simple tensor train (TT/MPS) library with statically determined shapes
(site dimensions are fixed at construction time).
"""
module SimpleTT

using LinearAlgebra
using ..C_API

export SimpleTensorTrain

"""
    SimpleTensorTrain{T<:Real}

A simple tensor train (TT/MPS) with statically determined shapes.

SimpleTensorTrain is a simple tensor train library where site dimensions are fixed
at construction time.

Currently only supports `Float64` values.
"""
mutable struct SimpleTensorTrain{T<:Real}
    ptr::Ptr{Cvoid}

    function SimpleTensorTrain{Float64}(ptr::Ptr{Cvoid})
        ptr == C_NULL && error("Failed to create SimpleTensorTrain: null pointer")
        tt = new{Float64}(ptr)
        finalizer(tt) do obj
            C_API.t4a_simplett_f64_release(obj.ptr)
        end
        return tt
    end
end

# Convenience constructor
SimpleTensorTrain(ptr::Ptr{Cvoid}) = SimpleTensorTrain{Float64}(ptr)

"""
    SimpleTensorTrain(site_dims::Vector{<:Integer}, value::Real)

Create a constant tensor train with the given site dimensions and value.

# Example
```julia
tt = SimpleTensorTrain([2, 3, 4], 1.0)  # All elements equal to 1.0
```
"""
function SimpleTensorTrain(site_dims::Vector{<:Integer}, value::Real)
    dims = Csize_t.(site_dims)
    ptr = C_API.t4a_simplett_f64_constant(dims, Float64(value))
    return SimpleTensorTrain{Float64}(ptr)
end

"""
    zeros(::Type{SimpleTensorTrain}, site_dims::Vector{<:Integer})

Create a zero tensor train with the given site dimensions.

# Example
```julia
tt = zeros(SimpleTensorTrain, [2, 3, 4])
```
"""
function Base.zeros(::Type{SimpleTensorTrain}, site_dims::Vector{<:Integer})
    dims = Csize_t.(site_dims)
    ptr = C_API.t4a_simplett_f64_zeros(dims)
    return SimpleTensorTrain{Float64}(ptr)
end

"""
    copy(tt::SimpleTensorTrain)

Create a deep copy of the tensor train.
"""
function Base.copy(tt::SimpleTensorTrain{Float64})
    new_ptr = C_API.t4a_simplett_f64_clone(tt.ptr)
    return SimpleTensorTrain{Float64}(new_ptr)
end

"""
    length(tt::SimpleTensorTrain) -> Int

Get the number of sites in the tensor train.
"""
function Base.length(tt::SimpleTensorTrain{Float64})
    out_len = Ref{Csize_t}(0)
    status = C_API.t4a_simplett_f64_len(tt.ptr, out_len)
    C_API.check_status(status)
    return Int(out_len[])
end

"""
    site_dims(tt::SimpleTensorTrain) -> Vector{Int}

Get the site (physical) dimensions.
"""
function site_dims(tt::SimpleTensorTrain{Float64})
    n = length(tt)
    dims = Vector{Csize_t}(undef, n)
    status = C_API.t4a_simplett_f64_site_dims(tt.ptr, dims)
    C_API.check_status(status)
    return Int.(dims)
end

"""
    link_dims(tt::SimpleTensorTrain) -> Vector{Int}

Get the link (bond) dimensions. Returns n-1 values for n sites.
"""
function link_dims(tt::SimpleTensorTrain{Float64})
    n = length(tt)
    n <= 1 && return Int[]
    dims = Vector{Csize_t}(undef, n - 1)
    status = C_API.t4a_simplett_f64_link_dims(tt.ptr, dims)
    C_API.check_status(status)
    return Int.(dims)
end

"""
    rank(tt::SimpleTensorTrain) -> Int

Get the maximum bond dimension (rank).
"""
function rank(tt::SimpleTensorTrain{Float64})
    out_rank = Ref{Csize_t}(0)
    status = C_API.t4a_simplett_f64_rank(tt.ptr, out_rank)
    C_API.check_status(status)
    return Int(out_rank[])
end

"""
    evaluate(tt::SimpleTensorTrain, indices::Vector{<:Integer}) -> Float64

Evaluate the tensor train at a given multi-index (0-based indexing).

# Example
```julia
tt = SimpleTensorTrain([2, 3, 4], 2.0)
val = evaluate(tt, [0, 1, 2])  # Returns 2.0
```
"""
function evaluate(tt::SimpleTensorTrain{Float64}, indices::Vector{<:Integer})
    idx = Csize_t.(indices)
    out_value = Ref{Cdouble}(0.0)
    status = C_API.t4a_simplett_f64_evaluate(tt.ptr, idx, out_value)
    C_API.check_status(status)
    return out_value[]
end

# Callable interface
(tt::SimpleTensorTrain{Float64})(indices::Vector{<:Integer}) = evaluate(tt, indices)
(tt::SimpleTensorTrain{Float64})(indices::Integer...) = evaluate(tt, collect(indices))

"""
    sum(tt::SimpleTensorTrain) -> Float64

Compute the sum over all tensor train elements.
"""
function Base.sum(tt::SimpleTensorTrain{Float64})
    out_value = Ref{Cdouble}(0.0)
    status = C_API.t4a_simplett_f64_sum(tt.ptr, out_value)
    C_API.check_status(status)
    return out_value[]
end

"""
    norm(tt::SimpleTensorTrain) -> Float64

Compute the Frobenius norm of the tensor train.
"""
function LinearAlgebra.norm(tt::SimpleTensorTrain{Float64})
    out_value = Ref{Cdouble}(0.0)
    status = C_API.t4a_simplett_f64_norm(tt.ptr, out_value)
    C_API.check_status(status)
    return out_value[]
end

"""
    site_tensor(tt::SimpleTensorTrain, site::Integer) -> Array{Float64, 3}

Get the site tensor at a specific site (0-based indexing).
Returns array with shape (left_dim, site_dim, right_dim).
"""
function site_tensor(tt::SimpleTensorTrain{Float64}, site::Integer)
    # First get dimensions to allocate buffer
    n = length(tt)
    0 <= site < n || error("Site index out of bounds: $site (n=$n)")

    sdims = site_dims(tt)
    ldims = link_dims(tt)

    left_dim = site == 0 ? 1 : ldims[site]
    site_dim = sdims[site + 1]  # Julia 1-based
    right_dim = site == n - 1 ? 1 : ldims[site + 1]

    total_size = left_dim * site_dim * right_dim
    data = Vector{Cdouble}(undef, total_size)

    out_left = Ref{Csize_t}(0)
    out_site = Ref{Csize_t}(0)
    out_right = Ref{Csize_t}(0)

    status = C_API.t4a_simplett_f64_site_tensor(tt.ptr, site, data, out_left, out_site, out_right)
    C_API.check_status(status)

    # Reshape to 3D array (row-major from Rust, need to permute)
    return reshape(data, (Int(out_right[]), Int(out_site[]), Int(out_left[]))) |>
           x -> permutedims(x, (3, 2, 1))
end

"""
    show(io::IO, tt::SimpleTensorTrain)

Display tensor train information.
"""
function Base.show(io::IO, tt::SimpleTensorTrain{T}) where T
    n = length(tt)
    r = rank(tt)
    print(io, "SimpleTensorTrain{$T}(sites=$n, rank=$r)")
end

function Base.show(io::IO, ::MIME"text/plain", tt::SimpleTensorTrain{T}) where T
    n = length(tt)
    println(io, "SimpleTensorTrain{$T}")
    println(io, "  Sites: $n")
    println(io, "  Site dims: $(site_dims(tt))")
    println(io, "  Link dims: $(link_dims(tt))")
    println(io, "  Max rank: $(rank(tt))")
end

end # module SimpleTT
