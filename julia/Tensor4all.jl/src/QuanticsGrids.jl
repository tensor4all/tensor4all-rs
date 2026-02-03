"""
    QuanticsGrids

Quantics grid types for efficient conversion between quantics indices,
grid indices, and original coordinates.

Provides two grid types:
- `DiscretizedGrid`: continuous domain with floating-point coordinates
- `InherentDiscreteGrid`: integer coordinate domain

# Example

```julia
using Tensor4all.QuanticsGrids

# Create a 1D grid with 8 bits (256 grid points) over [0, 1)
grid = DiscretizedGrid(1, 8, [0.0], [1.0])

# Convert coordinates
quantics = origcoord_to_quantics(grid, [0.5])
coord = quantics_to_origcoord(grid, quantics)
```
"""
module QuanticsGrids

using ..Tensor4all: C_API

export DiscretizedGrid, InherentDiscreteGrid
export origcoord_to_quantics, quantics_to_origcoord
export origcoord_to_grididx, grididx_to_origcoord
export grididx_to_quantics, quantics_to_grididx

# ============================================================================
# Unfolding scheme helper
# ============================================================================

function _unfolding_to_cint(unfolding::Symbol)
    if unfolding == :fused
        return C_API.UNFOLDING_FUSED
    elseif unfolding == :interleaved
        return C_API.UNFOLDING_INTERLEAVED
    else
        error("Unknown unfolding scheme: $unfolding. Use :fused or :interleaved.")
    end
end

# ============================================================================
# DiscretizedGrid
# ============================================================================

"""
    DiscretizedGrid

A discretized grid with continuous domain and floating-point coordinates.

# Constructors

- `DiscretizedGrid(ndims, R, lower, upper; unfolding=:fused)` - uniform R
- `DiscretizedGrid(ndims, rs, lower, upper; unfolding=:fused)` - per-dimension R

# Properties

- `ndims(g)` - Number of dimensions
- `rs(g)` - Resolution (bits) per dimension
- `local_dimensions(g)` - Local dimensions of tensor sites
- `lower_bound(g)` - Lower bounds per dimension
- `upper_bound(g)` - Upper bounds per dimension
- `grid_step(g)` - Grid spacing per dimension
"""
mutable struct DiscretizedGrid
    ptr::Ptr{Cvoid}
    _ndims::Int
    _rs::Vector{Int}

    function DiscretizedGrid(ptr::Ptr{Cvoid}, ndims::Int, rs::Vector{Int})
        if ptr == C_NULL
            error("Failed to create DiscretizedGrid (null pointer from C API)")
        end
        grid = new(ptr, ndims, rs)
        finalizer(grid) do x
            C_API.t4a_qgrid_disc_release(x.ptr)
        end
        return grid
    end
end

"""
    DiscretizedGrid(ndims::Int, R::Int, lower, upper; unfolding=:fused)

Create a DiscretizedGrid with uniform resolution R for all dimensions.
"""
function DiscretizedGrid(ndims::Int, R::Int, lower::AbstractVector{<:Real},
                         upper::AbstractVector{<:Real}; unfolding::Symbol=:fused)
    rs_vec = fill(R, ndims)
    return DiscretizedGrid(ndims, rs_vec, lower, upper; unfolding=unfolding)
end

"""
    DiscretizedGrid(ndims::Int, rs::Vector{Int}, lower, upper; unfolding=:fused)

Create a DiscretizedGrid with per-dimension resolutions.
"""
function DiscretizedGrid(ndims::Int, rs::AbstractVector{<:Integer},
                         lower::AbstractVector{<:Real}, upper::AbstractVector{<:Real};
                         unfolding::Symbol=:fused)
    length(rs) == ndims || error("rs must have length $ndims, got $(length(rs))")
    length(lower) == ndims || error("lower must have length $ndims, got $(length(lower))")
    length(upper) == ndims || error("upper must have length $ndims, got $(length(upper))")

    rs_c = Csize_t.(rs)
    lower_c = Cdouble.(lower)
    upper_c = Cdouble.(upper)
    unfolding_c = _unfolding_to_cint(unfolding)

    out = Ref{Ptr{Cvoid}}(C_NULL)
    status = C_API.t4a_qgrid_disc_new(ndims, rs_c, lower_c, upper_c, unfolding_c, out)
    C_API.check_status(status)

    return DiscretizedGrid(out[], ndims, Int.(rs))
end

# Properties

"""
    ndims(g::DiscretizedGrid) -> Int

Number of dimensions.
"""
Base.ndims(g::DiscretizedGrid) = g._ndims

"""
    rs(g::DiscretizedGrid) -> Vector{Int}

Resolution (bits) per dimension.
"""
rs(g::DiscretizedGrid) = g._rs

"""
    local_dimensions(g::DiscretizedGrid) -> Vector{Int}

Local dimensions of all tensor sites.
"""
function local_dimensions(g::DiscretizedGrid)
    max_sites = sum(g._rs) + g._ndims  # upper bound
    out = Vector{Csize_t}(undef, max_sites)
    n_out = Ref{Csize_t}(0)
    status = C_API.t4a_qgrid_disc_local_dims(g.ptr, out, max_sites, n_out)
    C_API.check_status(status)
    return Int.(out[1:n_out[]])
end

"""
    lower_bound(g::DiscretizedGrid) -> Vector{Float64}

Lower bounds per dimension.
"""
function lower_bound(g::DiscretizedGrid)
    out = Vector{Cdouble}(undef, g._ndims)
    status = C_API.t4a_qgrid_disc_lower_bound(g.ptr, out, g._ndims)
    C_API.check_status(status)
    return out
end

"""
    upper_bound(g::DiscretizedGrid) -> Vector{Float64}

Upper bounds per dimension.
"""
function upper_bound(g::DiscretizedGrid)
    out = Vector{Cdouble}(undef, g._ndims)
    status = C_API.t4a_qgrid_disc_upper_bound(g.ptr, out, g._ndims)
    C_API.check_status(status)
    return out
end

"""
    grid_step(g::DiscretizedGrid) -> Vector{Float64}

Grid spacing per dimension.
"""
function grid_step(g::DiscretizedGrid)
    out = Vector{Cdouble}(undef, g._ndims)
    status = C_API.t4a_qgrid_disc_grid_step(g.ptr, out, g._ndims)
    C_API.check_status(status)
    return out
end

# Coordinate conversions

"""
    origcoord_to_quantics(g::DiscretizedGrid, coord) -> Vector{Int64}

Convert original (continuous) coordinates to quantics indices.
"""
function origcoord_to_quantics(g::DiscretizedGrid, coord::AbstractVector{<:Real})
    coord_c = Cdouble.(coord)
    max_sites = sum(g._rs) + g._ndims
    out = Vector{Int64}(undef, max_sites)
    n_out = Ref{Csize_t}(0)
    status = C_API.t4a_qgrid_disc_origcoord_to_quantics(
        g.ptr, coord_c, length(coord), out, max_sites, n_out)
    C_API.check_status(status)
    return out[1:n_out[]]
end

"""
    quantics_to_origcoord(g::DiscretizedGrid, quantics) -> Vector{Float64}

Convert quantics indices to original (continuous) coordinates.
"""
function quantics_to_origcoord(g::DiscretizedGrid, quantics::AbstractVector{<:Integer})
    quantics_c = Int64.(quantics)
    out = Vector{Cdouble}(undef, g._ndims)
    status = C_API.t4a_qgrid_disc_quantics_to_origcoord(
        g.ptr, quantics_c, length(quantics), out, g._ndims)
    C_API.check_status(status)
    return out
end

"""
    origcoord_to_grididx(g::DiscretizedGrid, coord) -> Vector{Int64}

Convert original coordinates to grid indices (1-indexed).
"""
function origcoord_to_grididx(g::DiscretizedGrid, coord::AbstractVector{<:Real})
    coord_c = Cdouble.(coord)
    out = Vector{Int64}(undef, g._ndims)
    status = C_API.t4a_qgrid_disc_origcoord_to_grididx(
        g.ptr, coord_c, length(coord), out, g._ndims)
    C_API.check_status(status)
    return out
end

"""
    grididx_to_origcoord(g::DiscretizedGrid, grididx) -> Vector{Float64}

Convert grid indices (1-indexed) to original coordinates.
"""
function grididx_to_origcoord(g::DiscretizedGrid, grididx::AbstractVector{<:Integer})
    grididx_c = Int64.(grididx)
    out = Vector{Cdouble}(undef, g._ndims)
    status = C_API.t4a_qgrid_disc_grididx_to_origcoord(
        g.ptr, grididx_c, length(grididx), out, g._ndims)
    C_API.check_status(status)
    return out
end

"""
    grididx_to_quantics(g::DiscretizedGrid, grididx) -> Vector{Int64}

Convert grid indices to quantics indices.
"""
function grididx_to_quantics(g::DiscretizedGrid, grididx::AbstractVector{<:Integer})
    grididx_c = Int64.(grididx)
    max_sites = sum(g._rs) + g._ndims
    out = Vector{Int64}(undef, max_sites)
    n_out = Ref{Csize_t}(0)
    status = C_API.t4a_qgrid_disc_grididx_to_quantics(
        g.ptr, grididx_c, length(grididx), out, max_sites, n_out)
    C_API.check_status(status)
    return out[1:n_out[]]
end

"""
    quantics_to_grididx(g::DiscretizedGrid, quantics) -> Vector{Int64}

Convert quantics indices to grid indices.
"""
function quantics_to_grididx(g::DiscretizedGrid, quantics::AbstractVector{<:Integer})
    quantics_c = Int64.(quantics)
    out = Vector{Int64}(undef, g._ndims)
    n_out = Ref{Csize_t}(0)
    status = C_API.t4a_qgrid_disc_quantics_to_grididx(
        g.ptr, quantics_c, length(quantics), out, g._ndims, n_out)
    C_API.check_status(status)
    return out[1:n_out[]]
end

# Display
function Base.show(io::IO, g::DiscretizedGrid)
    print(io, "DiscretizedGrid(ndims=$(g._ndims), rs=$(g._rs))")
end

# ============================================================================
# InherentDiscreteGrid
# ============================================================================

"""
    InherentDiscreteGrid

A discrete grid for quantics tensor train representations with integer coordinates.

# Constructors

- `InherentDiscreteGrid(ndims, R; origin=nothing, unfolding=:fused)` - uniform R
- `InherentDiscreteGrid(ndims, rs; origin=nothing, unfolding=:fused)` - per-dimension R
"""
mutable struct InherentDiscreteGrid
    ptr::Ptr{Cvoid}
    _ndims::Int
    _rs::Vector{Int}

    function InherentDiscreteGrid(ptr::Ptr{Cvoid}, ndims::Int, rs::Vector{Int})
        if ptr == C_NULL
            error("Failed to create InherentDiscreteGrid (null pointer from C API)")
        end
        grid = new(ptr, ndims, rs)
        finalizer(grid) do x
            C_API.t4a_qgrid_int_release(x.ptr)
        end
        return grid
    end
end

"""
    InherentDiscreteGrid(ndims::Int, R::Int; origin=nothing, unfolding=:fused)

Create an InherentDiscreteGrid with uniform resolution R.
"""
function InherentDiscreteGrid(ndims::Int, R::Int;
                               origin::Union{Nothing,AbstractVector{<:Integer}}=nothing,
                               unfolding::Symbol=:fused)
    rs_vec = fill(R, ndims)
    return InherentDiscreteGrid(ndims, rs_vec; origin=origin, unfolding=unfolding)
end

"""
    InherentDiscreteGrid(ndims::Int, rs::Vector{Int}; origin=nothing, unfolding=:fused)

Create an InherentDiscreteGrid with per-dimension resolutions.
"""
function InherentDiscreteGrid(ndims::Int, rs::AbstractVector{<:Integer};
                               origin::Union{Nothing,AbstractVector{<:Integer}}=nothing,
                               unfolding::Symbol=:fused)
    length(rs) == ndims || error("rs must have length $ndims, got $(length(rs))")

    rs_c = Csize_t.(rs)
    unfolding_c = _unfolding_to_cint(unfolding)

    origin_ptr = if origin !== nothing
        length(origin) == ndims || error("origin must have length $ndims, got $(length(origin))")
        Int64.(origin)
    else
        C_NULL
    end

    out = Ref{Ptr{Cvoid}}(C_NULL)
    status = C_API.t4a_qgrid_int_new(ndims, rs_c, origin_ptr, unfolding_c, out)
    C_API.check_status(status)

    return InherentDiscreteGrid(out[], ndims, Int.(rs))
end

# Properties

Base.ndims(g::InherentDiscreteGrid) = g._ndims
rs(g::InherentDiscreteGrid) = g._rs

function local_dimensions(g::InherentDiscreteGrid)
    max_sites = sum(g._rs) + g._ndims
    out = Vector{Csize_t}(undef, max_sites)
    n_out = Ref{Csize_t}(0)
    status = C_API.t4a_qgrid_int_local_dims(g.ptr, out, max_sites, n_out)
    C_API.check_status(status)
    return Int.(out[1:n_out[]])
end

"""
    origin(g::InherentDiscreteGrid) -> Vector{Int64}

Origin per dimension.
"""
function origin(g::InherentDiscreteGrid)
    out = Vector{Int64}(undef, g._ndims)
    status = C_API.t4a_qgrid_int_origin(g.ptr, out, g._ndims)
    C_API.check_status(status)
    return out
end

# Coordinate conversions

function origcoord_to_quantics(g::InherentDiscreteGrid, coord::AbstractVector{<:Integer})
    coord_c = Int64.(coord)
    max_sites = sum(g._rs) + g._ndims
    out = Vector{Int64}(undef, max_sites)
    n_out = Ref{Csize_t}(0)
    status = C_API.t4a_qgrid_int_origcoord_to_quantics(
        g.ptr, coord_c, length(coord), out, max_sites, n_out)
    C_API.check_status(status)
    return out[1:n_out[]]
end

function quantics_to_origcoord(g::InherentDiscreteGrid, quantics::AbstractVector{<:Integer})
    quantics_c = Int64.(quantics)
    out = Vector{Int64}(undef, g._ndims)
    status = C_API.t4a_qgrid_int_quantics_to_origcoord(
        g.ptr, quantics_c, length(quantics), out, g._ndims)
    C_API.check_status(status)
    return out
end

function origcoord_to_grididx(g::InherentDiscreteGrid, coord::AbstractVector{<:Integer})
    coord_c = Int64.(coord)
    out = Vector{Int64}(undef, g._ndims)
    status = C_API.t4a_qgrid_int_origcoord_to_grididx(
        g.ptr, coord_c, length(coord), out, g._ndims)
    C_API.check_status(status)
    return out
end

function grididx_to_origcoord(g::InherentDiscreteGrid, grididx::AbstractVector{<:Integer})
    grididx_c = Int64.(grididx)
    out = Vector{Int64}(undef, g._ndims)
    status = C_API.t4a_qgrid_int_grididx_to_origcoord(
        g.ptr, grididx_c, length(grididx), out, g._ndims)
    C_API.check_status(status)
    return out
end

function grididx_to_quantics(g::InherentDiscreteGrid, grididx::AbstractVector{<:Integer})
    grididx_c = Int64.(grididx)
    max_sites = sum(g._rs) + g._ndims
    out = Vector{Int64}(undef, max_sites)
    n_out = Ref{Csize_t}(0)
    status = C_API.t4a_qgrid_int_grididx_to_quantics(
        g.ptr, grididx_c, length(grididx), out, max_sites, n_out)
    C_API.check_status(status)
    return out[1:n_out[]]
end

function quantics_to_grididx(g::InherentDiscreteGrid, quantics::AbstractVector{<:Integer})
    quantics_c = Int64.(quantics)
    out = Vector{Int64}(undef, g._ndims)
    n_out = Ref{Csize_t}(0)
    status = C_API.t4a_qgrid_int_quantics_to_grididx(
        g.ptr, quantics_c, length(quantics), out, g._ndims, n_out)
    C_API.check_status(status)
    return out[1:n_out[]]
end

# Display
function Base.show(io::IO, g::InherentDiscreteGrid)
    print(io, "InherentDiscreteGrid(ndims=$(g._ndims), rs=$(g._rs))")
end

end # module QuanticsGrids
