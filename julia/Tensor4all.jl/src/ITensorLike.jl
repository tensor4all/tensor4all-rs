"""
    ITensorLike

Tensor train (MPS) functionality inspired by ITensorMPS.jl.

This submodule provides TensorTrain and related operations.

# Usage
```julia
using Tensor4all
using Tensor4all.ITensorLike

# Create a random tensor train
sites = [Index(2) for _ in 1:5]
tt = random_tt(sites; linkdims=4)

# Operations
orthogonalize!(tt, 3)
truncate!(tt; maxdim=10)
```
"""
module ITensorLike

using LinearAlgebra

# Import from parent module
import ..Tensor4all: Index, Tensor, dim, id, tags, indices, rank, dims, data
import ..Tensor4all: hascommoninds, commoninds, uniqueinds, HasCommonIndsPredicate
import ..Tensor4all: C_API

# ============================================================================
# CanonicalForm Enum
# ============================================================================

"""
    CanonicalForm

Enum representing the method used for canonicalization.

- `Unitary`: QR-based canonicalization (tensors are isometric)
- `LU`: LU-based canonicalization (one factor has unit diagonal)
- `CI`: Cross Interpolation based canonicalization
"""
@enum CanonicalForm begin
    Unitary = 0
    LU = 1
    CI = 2
end

export CanonicalForm, Unitary, LU, CI

# ============================================================================
# TensorTrain Type
# ============================================================================

"""
    TensorTrain

A tensor train (MPS) with orthogonality tracking.

Wraps a Rust `TensorTrain` which corresponds to
ITensorMPS.jl's `MPS`.

# Constructors

- `TensorTrain(tensors::Vector{Tensor})` - Create from tensors

# Properties

- `length(tt)` - Number of sites
- `tensors(tt)` - Get all tensors
- `bond_dims(tt)` - Get bond dimensions
- `maxbonddim(tt)` - Get maximum bond dimension
- `isortho(tt)` - Check if orthogonalized
- `orthocenter(tt)` - Get orthogonality center (1-indexed)

# Operations

- `orthogonalize!(tt, site)` - Orthogonalize to site (1-indexed)
- `truncate!(tt; rtol=0.0, maxdim=0)` - Truncate bond dimensions
- `norm(tt)` - Compute norm
"""
mutable struct TensorTrain
    ptr::Ptr{Cvoid}

    function TensorTrain(ptr::Ptr{Cvoid})
        if ptr == C_NULL
            error("Failed to create TensorTrain (null pointer from C API)")
        end
        tt = new(ptr)
        finalizer(tt) do x
            C_API.t4a_tensortrain_release(x.ptr)
        end
        return tt
    end
end

export TensorTrain

# ============================================================================
# TensorTrain Constructors
# ============================================================================

"""
    TensorTrain(tensors::Vector{Tensor})

Create a tensor train from a vector of tensors.
Adjacent tensors must share exactly one common index (the link index).
"""
function TensorTrain(ts::Vector{Tensor})
    isempty(ts) && return TensorTrain(C_API.t4a_tt_new_empty())

    tensor_ptrs = Ptr{Cvoid}[t.ptr for t in ts]
    ptr = C_API.t4a_tt_new(tensor_ptrs, length(ts))
    ptr == C_NULL && error("Failed to create TensorTrain: invalid tensor structure")
    return TensorTrain(ptr)
end

# ============================================================================
# TensorTrain Accessors
# ============================================================================

"""
    length(tt::TensorTrain) -> Int

Get the number of sites in the tensor train.
"""
function Base.length(tt::TensorTrain)
    len = Ref{Csize_t}(0)
    status = C_API.t4a_tt_len(tt.ptr, len)
    C_API.check_status(status)
    return Int(len[])
end

"""
    isempty(tt::TensorTrain) -> Bool

Check if the tensor train is empty.
"""
function Base.isempty(tt::TensorTrain)
    result = C_API.t4a_tt_is_empty(tt.ptr)
    result < 0 && error("Error checking if TensorTrain is empty")
    return result == 1
end

"""
    tensors(tt::TensorTrain) -> Vector{Tensor}

Get all tensors in the tensor train.
"""
function tensors(tt::TensorTrain)
    n = length(tt)
    result = Vector{Tensor}(undef, n)
    for i in 1:n
        ptr = C_API.t4a_tt_tensor(tt.ptr, i - 1)  # 0-indexed in C API
        ptr == C_NULL && error("Failed to get tensor at site $i")
        result[i] = Tensor(ptr)
    end
    return result
end

export tensors

"""
    getindex(tt::TensorTrain, i::Integer) -> Tensor

Get the tensor at site `i` (1-indexed).
"""
function Base.getindex(tt::TensorTrain, i::Integer)
    n = length(tt)
    (1 <= i <= n) || throw(BoundsError(tt, i))
    ptr = C_API.t4a_tt_tensor(tt.ptr, i - 1)  # 0-indexed in C API
    ptr == C_NULL && error("Failed to get tensor at site $i")
    return Tensor(ptr)
end

"""
    setindex!(tt::TensorTrain, tensor::Tensor, i::Integer)

Set the tensor at site `i` (1-indexed).
This replaces the tensor at the given site and invalidates orthogonality.

Note: The tensor must have compatible indices with the existing tensor train
structure (shared link indices with neighbors).
"""
function Base.setindex!(tt::TensorTrain, tensor::Tensor, i::Integer)
    n = length(tt)
    (1 <= i <= n) || throw(BoundsError(tt, i))
    status = C_API.t4a_tt_set_tensor(tt.ptr, i - 1, tensor.ptr)
    C_API.check_status(status)
    return tt
end

"""
    bond_dims(tt::TensorTrain) -> Vector{Int}

Get the bond dimensions of the tensor train.
Returns a vector of length `length(tt) - 1`.
"""
function bond_dims(tt::TensorTrain)
    n = length(tt)
    n <= 1 && return Int[]

    out_dims = Vector{Csize_t}(undef, n - 1)
    status = C_API.t4a_tt_bond_dims(tt.ptr, out_dims, n - 1)
    C_API.check_status(status)
    return Int.(out_dims)
end

export bond_dims

"""
    maxbonddim(tt::TensorTrain) -> Int

Get the maximum bond dimension of the tensor train.
"""
function maxbonddim(tt::TensorTrain)
    max_dim = Ref{Csize_t}(0)
    status = C_API.t4a_tt_maxbonddim(tt.ptr, max_dim)
    C_API.check_status(status)
    return Int(max_dim[])
end

export maxbonddim

"""
    linkind(tt::TensorTrain, i::Integer) -> Union{Index, Nothing}

Get the link index between sites `i` and `i+1` (1-indexed).
Returns `nothing` if no link exists.
"""
function linkind(tt::TensorTrain, i::Integer)
    n = length(tt)
    (1 <= i < n) || return nothing
    ptr = C_API.t4a_tt_linkind(tt.ptr, i - 1)  # 0-indexed in C API
    ptr == C_NULL && return nothing
    return Index(ptr)
end

export linkind

"""
    linkinds(tt::TensorTrain) -> Vector{Index}

Get all link indices of the tensor train.
"""
function linkinds(tt::TensorTrain)
    n = length(tt)
    n <= 1 && return Index[]

    result = Vector{Index}(undef, n - 1)
    for i in 1:(n-1)
        idx = linkind(tt, i)
        idx === nothing && error("Missing link index at site $i")
        result[i] = idx
    end
    return result
end

export linkinds

# ============================================================================
# TensorTrain Orthogonality
# ============================================================================

"""
    isortho(tt::TensorTrain) -> Bool

Check if the tensor train has a single orthogonality center.
"""
function isortho(tt::TensorTrain)
    result = C_API.t4a_tt_isortho(tt.ptr)
    result < 0 && error("Error checking orthogonality")
    return result == 1
end

export isortho

"""
    orthocenter(tt::TensorTrain) -> Union{Int, Nothing}

Get the orthogonality center (1-indexed).
Returns `nothing` if no single center exists.
"""
function orthocenter(tt::TensorTrain)
    center = Ref{Csize_t}(0)
    status = C_API.t4a_tt_orthocenter(tt.ptr, center)
    status == C_API.T4A_INVALID_ARGUMENT && return nothing
    C_API.check_status(status)
    return Int(center[]) + 1  # Convert to 1-indexed
end

export orthocenter

"""
    llim(tt::TensorTrain) -> Int

Get the left orthogonality limit (1-indexed).
Sites 1:llim are guaranteed to be left-orthogonal.
Returns 0 if no sites are left-orthogonal.
"""
function llim(tt::TensorTrain)
    out_llim = Ref{Cint}(0)
    status = C_API.t4a_tt_llim(tt.ptr, out_llim)
    C_API.check_status(status)
    # Rust returns -1 for no left ortho, we return 0 (1-indexed adjustment)
    return Int(out_llim[]) + 1
end

export llim

"""
    rlim(tt::TensorTrain) -> Int

Get the right orthogonality limit (1-indexed).
Sites rlim:end are guaranteed to be right-orthogonal.
Returns `length(tt) + 1` if no sites are right-orthogonal.
"""
function rlim(tt::TensorTrain)
    out_rlim = Ref{Cint}(0)
    status = C_API.t4a_tt_rlim(tt.ptr, out_rlim)
    C_API.check_status(status)
    # Rust uses 0-indexed, convert to 1-indexed
    return Int(out_rlim[]) + 1
end

export rlim

"""
    canonical_form(tt::TensorTrain) -> Union{CanonicalForm, Nothing}

Get the canonical form used for the tensor train.
Returns `nothing` if no canonical form is set.
"""
function canonical_form(tt::TensorTrain)
    form = Ref{Cint}(0)
    status = C_API.t4a_tt_canonical_form(tt.ptr, form)
    status == C_API.T4A_INVALID_ARGUMENT && return nothing
    C_API.check_status(status)
    return CanonicalForm(form[])
end

export canonical_form

# ============================================================================
# TensorTrain Operations
# ============================================================================

"""
    orthogonalize!(tt::TensorTrain, site::Integer; form::CanonicalForm=Unitary)

Orthogonalize the tensor train to have orthogonality center at `site` (1-indexed).

# Arguments
- `site`: Target site for orthogonality center (1-indexed)
- `form`: Canonical form to use (default: `Unitary`)
"""
function orthogonalize!(tt::TensorTrain, site::Integer; form::CanonicalForm=Unitary)
    n = length(tt)
    (1 <= site <= n) || throw(ArgumentError("Site $site out of bounds [1, $n]"))

    status = C_API.t4a_tt_orthogonalize_with(tt.ptr, site - 1, Int(form))  # 0-indexed
    C_API.check_status(status)
    return tt
end

export orthogonalize!

"""
    truncate!(tt::TensorTrain; rtol=0.0, cutoff=0.0, maxdim=0)

Truncate the tensor train bond dimensions in-place.

# Arguments
- `rtol`: Relative tolerance for truncation (use 0.0 for not set)
- `cutoff`: ITensorMPS.jl cutoff (use 0.0 for not set). Converted to rtol = √cutoff.
  If both `rtol` and `cutoff` are positive, `cutoff` takes precedence.
- `maxdim`: Maximum bond dimension (use 0 for no limit)
"""
function truncate!(tt::TensorTrain; rtol::Real=0.0, cutoff::Real=0.0, maxdim::Integer=0)
    status = C_API.t4a_tt_truncate(tt.ptr, Float64(rtol), Float64(cutoff), maxdim)
    C_API.check_status(status)
    return tt
end

export truncate!

"""
    truncate(tt::TensorTrain; rtol=0.0, cutoff=0.0, maxdim=0) -> TensorTrain

Return a truncated copy of the tensor train (immutable version).

# Arguments
- `rtol`: Relative tolerance for truncation (use 0.0 for not set)
- `cutoff`: ITensorMPS.jl cutoff (use 0.0 for not set). Converted to rtol = √cutoff.
  If both `rtol` and `cutoff` are positive, `cutoff` takes precedence.
- `maxdim`: Maximum bond dimension (use 0 for no limit)

# Returns
A new truncated tensor train.
"""
function truncate(tt::TensorTrain; rtol::Real=0.0, cutoff::Real=0.0, maxdim::Integer=0)
    tt_copy = copy(tt)
    truncate!(tt_copy; rtol=rtol, cutoff=cutoff, maxdim=maxdim)
    return tt_copy
end

export truncate

"""
    norm(tt::TensorTrain) -> Float64

Compute the norm of the tensor train.
"""
function LinearAlgebra.norm(tt::TensorTrain)
    out_norm = Ref{Cdouble}(0.0)
    status = C_API.t4a_tt_norm(tt.ptr, out_norm)
    C_API.check_status(status)
    return out_norm[]
end

"""
    inner(tt1::TensorTrain, tt2::TensorTrain) -> ComplexF64

Compute the inner product <tt1|tt2>.
"""
function inner(tt1::TensorTrain, tt2::TensorTrain)
    out_re = Ref{Cdouble}(0.0)
    out_im = Ref{Cdouble}(0.0)
    status = C_API.t4a_tt_inner(tt1.ptr, tt2.ptr, out_re, out_im)
    C_API.check_status(status)
    return ComplexF64(out_re[], out_im[])
end

export inner

# ============================================================================
# TensorTrain Contract
# ============================================================================

# Internal: resolve full/half sweep parameters.
#
# ITensorMPS.jl convention:
# - nsweeps: number of full sweeps
# - nhalfsweeps: number of half-sweeps (forward or backward)
# A full sweep = 2 half-sweeps.
#
# Policy here: explicit `nhalfsweeps` takes precedence over derived `nsweeps`.
function _resolve_nhalfsweeps(nsweeps::Integer, nhalfsweeps::Integer)
    return nhalfsweeps > 0 ? nhalfsweeps : (nsweeps > 0 ? nsweeps * 2 : 0)
end

# Internal: convert method specification to C API integer
# Supports Symbol (:zipup, :fit, :naive), String ("zipup", "fit", "naive"), or Integer (0, 1, 2)
function _contract_method_to_int(method::Symbol)
    method === :zipup && return 0
    method === :fit && return 1
    method === :naive && return 2
    throw(ArgumentError("Unknown contract method: $method. Use :zipup, :fit, or :naive"))
end

function _contract_method_to_int(method::AbstractString)
    m = lowercase(method)
    m == "zipup" && return 0
    m == "fit" && return 1
    m == "naive" && return 2
    throw(ArgumentError("Unknown contract method: $method. Use \"zipup\", \"fit\", or \"naive\""))
end

_contract_method_to_int(method::Integer) = Int(method)

"""
    contract(tt1::TensorTrain, tt2::TensorTrain; kwargs...) -> TensorTrain

Contract two tensor trains with the same site indices.

# Arguments
- `tt1`: First tensor train
- `tt2`: Second tensor train
- `method`: Contraction method - `:zipup` (default), `:fit`, or `:naive`. Also accepts strings.
  - `:zipup`: Fast, one-pass contraction (default)
  - `:fit`: Variational optimization with sweeps
  - `:naive`: Contract to full tensor and decompose back (O(exp(n)) memory, for debugging)
- `maxdim`: Maximum bond dimension (0 for no limit)
- `rtol`: Relative tolerance (0.0 = not set)
- `cutoff`: ITensorMPS.jl cutoff (0.0 = not set). Converted to rtol = √cutoff.
  If both `rtol` and `cutoff` are positive, `cutoff` takes precedence.
- `nsweeps`: Number of full sweeps for fit method (0 = use Rust default)
- `nhalfsweeps`: Number of half-sweeps for fit method (0 = use Rust default).
  If `nhalfsweeps > 0`, it takes precedence over `nsweeps`.

# Returns
A new tensor train representing the contraction result.

# Example
```julia
# Contract using zipup (default)
result = contract(tt1, tt2)

# Contract with max bond dimension
result = contract(tt1, tt2; maxdim=50)

# Contract using fit with symbol
result = contract(tt1, tt2; method=:fit, nsweeps=5)

# Contract using naive (for debugging)
result = contract(tt1, tt2; method=:naive)
```
"""
function contract(tt1::TensorTrain, tt2::TensorTrain;
                  method::Union{Symbol, AbstractString, Integer}=:zipup,
                  maxdim::Integer=0,
                  rtol::Real=0.0,
                  cutoff::Real=0.0,
                  nsweeps::Integer=0,
                  nhalfsweeps::Integer=0)
    method_int = _contract_method_to_int(method)
    actual_nhalfsweeps = _resolve_nhalfsweeps(nsweeps, nhalfsweeps)
    ptr = C_API.t4a_tt_contract(tt1.ptr, tt2.ptr, method_int, maxdim,
                                Float64(rtol), Float64(cutoff), actual_nhalfsweeps)
    ptr == C_NULL && error("Tensor train contraction failed")
    return TensorTrain(ptr)
end

export contract

# ============================================================================
# TensorTrain Addition
# ============================================================================

"""
    add(tt1::TensorTrain, tt2::TensorTrain) -> TensorTrain

Add two tensor trains using direct-sum construction.
The result has bond dimensions that are the sum of the inputs' bond dimensions.
"""
function add(tt1::TensorTrain, tt2::TensorTrain)
    ptr = C_API.t4a_tt_add(tt1.ptr, tt2.ptr)
    ptr == C_NULL && error("Tensor train addition failed")
    return TensorTrain(ptr)
end

Base.:+(tt1::TensorTrain, tt2::TensorTrain) = add(tt1, tt2)

export add

# ============================================================================
# TensorTrain Dense Conversion
# ============================================================================

"""
    to_dense(tt::TensorTrain) -> Tensor

Convert the tensor train to a dense tensor by contracting all link indices.
"""
function to_dense(tt::TensorTrain)
    ptr = C_API.t4a_tt_to_dense(tt.ptr)
    ptr == C_NULL && error("to_dense failed")
    return Tensor(ptr)
end

export to_dense

# ============================================================================
# TensorTrain Linear Solver
# ============================================================================

"""
    linsolve(operator::TensorTrain, rhs::TensorTrain, init::TensorTrain; kwargs...) -> TensorTrain

Solve `(a0 + a1 * A) * x = b` for `x` using DMRG-like sweeps with local GMRES.

# Arguments
- `operator`: The operator A (MPO as TensorTrain)
- `rhs`: The right-hand side b (MPS as TensorTrain)
- `init`: Initial guess for x (MPS as TensorTrain)

# Keyword Arguments
- `nsweeps::Integer=5`: Number of full sweeps (each sweep = 2 half-sweeps)
- `nhalfsweeps::Integer=0`: Number of half-sweeps (overrides nsweeps if > 0)
- `maxdim::Integer=0`: Maximum bond dimension (0 = no limit)
- `rtol::Real=0.0`: Relative tolerance (0.0 = not set)
- `cutoff::Real=0.0`: ITensorMPS.jl cutoff (0.0 = not set). Converted to rtol = √cutoff.
- `krylov_tol::Real=0.0`: GMRES tolerance (0.0 = use default 1e-10)
- `krylov_maxiter::Integer=0`: Max GMRES iterations (0 = use default 100)
- `krylov_dim::Integer=0`: Krylov subspace dimension (0 = use default 30)
- `a0::Real=0.0`: Coefficient a₀ in (a₀ + a₁*A)*x = b
- `a1::Real=1.0`: Coefficient a₁ in (a₀ + a₁*A)*x = b
- `convergence_tol::Real=-1.0`: Early termination tolerance (negative = disabled)

# Returns
The solution `x` as a TensorTrain.
"""
function linsolve(operator::TensorTrain, rhs::TensorTrain, init::TensorTrain;
                  nsweeps::Integer=5,
                  nhalfsweeps::Integer=0,
                  maxdim::Integer=0,
                  rtol::Real=0.0,
                  cutoff::Real=0.0,
                  krylov_tol::Real=0.0,
                  krylov_maxiter::Integer=0,
                  krylov_dim::Integer=0,
                  a0::Real=0.0,
                  a1::Real=1.0,
                  convergence_tol::Real=-1.0)
    actual_nhalfsweeps = _resolve_nhalfsweeps(nsweeps, nhalfsweeps)
    ptr = C_API.t4a_tt_linsolve(operator.ptr, rhs.ptr, init.ptr,
                                actual_nhalfsweeps, maxdim,
                                Float64(rtol), Float64(cutoff),
                                Float64(krylov_tol), krylov_maxiter, krylov_dim,
                                Float64(a0), Float64(a1), Float64(convergence_tol))
    ptr == C_NULL && error("linsolve failed")
    return TensorTrain(ptr)
end

export linsolve

# ============================================================================
# Type Aliases (ITensorMPS.jl compatibility)
# ============================================================================

"""
    MPS

Type alias for `TensorTrain` (Matrix Product State).
"""
const MPS = TensorTrain

"""
    MPO

Type alias for `TensorTrain` (Matrix Product Operator).
"""
const MPO = TensorTrain

export MPS, MPO

# ============================================================================
# TensorTrain Display
# ============================================================================

function Base.show(io::IO, tt::TensorTrain)
    n = length(tt)
    max_bd = n > 1 ? maxbonddim(tt) : 0
    print(io, "TensorTrain(sites=$n, maxbonddim=$max_bd)")
end

function Base.show(io::IO, ::MIME"text/plain", tt::TensorTrain)
    println(io, "Tensor4all.ITensorLike.TensorTrain")
    println(io, "  sites: ", length(tt))
    bd = bond_dims(tt)
    if !isempty(bd)
        println(io, "  bond_dims: ", bd)
        println(io, "  maxbonddim: ", maximum(bd))
    end
    if isortho(tt)
        println(io, "  orthocenter: ", orthocenter(tt))
        cf = canonical_form(tt)
        cf !== nothing && println(io, "  canonical_form: ", cf)
    end
end

# Clone (shallow copy - same as deepcopy for this type)
function Base.copy(tt::TensorTrain)
    ptr = C_API.t4a_tensortrain_clone(tt.ptr)
    return TensorTrain(ptr)
end

# Deep copy
function Base.deepcopy(tt::TensorTrain)
    ptr = C_API.t4a_tensortrain_clone(tt.ptr)
    return TensorTrain(ptr)
end

# ============================================================================
# TensorTrain Iterator
# ============================================================================

"""
    iterate(tt::TensorTrain)
    iterate(tt::TensorTrain, state)

Iterate over tensors in the tensor train.

# Example
```julia
for tensor in tt
    println(rank(tensor))
end
```
"""
function Base.iterate(tt::TensorTrain, state=1)
    n = length(tt)
    if state > n
        return nothing
    end
    return (tt[state], state + 1)
end

Base.eltype(::Type{TensorTrain}) = Tensor
Base.firstindex(tt::TensorTrain) = 1
Base.lastindex(tt::TensorTrain) = length(tt)
Base.eachindex(tt::TensorTrain) = 1:length(tt)
Base.keys(tt::TensorTrain) = 1:length(tt)

# Support for findfirst/findall with predicates
# Note: We implement these manually instead of delegating to Vector because
# HasCommonIndsPredicate is not a Function type.
function Base.findfirst(pred::HasCommonIndsPredicate, tt::TensorTrain)
    for j in 1:length(tt)
        if pred(tt[j])
            return j
        end
    end
    return nothing
end

function Base.findall(pred::HasCommonIndsPredicate, tt::TensorTrain)
    result = Int[]
    for j in 1:length(tt)
        if pred(tt[j])
            push!(result, j)
        end
    end
    return result
end

function Base.findfirst(f::Function, tt::TensorTrain)
    for j in 1:length(tt)
        if f(tt[j])
            return j
        end
    end
    return nothing
end

function Base.findall(f::Function, tt::TensorTrain)
    result = Int[]
    for j in 1:length(tt)
        if f(tt[j])
            push!(result, j)
        end
    end
    return result
end

# ============================================================================
# TensorTrain Site Search
# ============================================================================

"""
    findsite(tt::TensorTrain, is) -> Union{Int, Nothing}
    findsite(tt::TensorTrain, i::Index) -> Union{Int, Nothing}

Return the first site of the TensorTrain that has at least one index
in common with the index or collection of indices `is`.

Returns `nothing` if no site has common indices.

# Example
```julia
sites = [Index(2), Index(3), Index(4)]
tt = random_tt(sites; linkdims=2)
findsite(tt, sites[2])  # Returns 2
```
"""
function findsite(tt::TensorTrain, is)
    pred = hascommoninds(is isa Index ? [is] : collect(is))
    return findfirst(pred, tt)
end

findsite(tt::TensorTrain, i::Index) = findsite(tt, [i])

export findsite

"""
    findsites(tt::TensorTrain, is) -> Vector{Int}
    findsites(tt::TensorTrain, i::Index) -> Vector{Int}

Return all sites of the TensorTrain that have at least one index
in common with the index or collection of indices `is`.

# Example
```julia
sites = [Index(2), Index(3), Index(4)]
tt = random_tt(sites; linkdims=2)
findsites(tt, sites[2])  # Returns [2]
findsites(tt, [sites[1], sites[3]])  # Returns [1, 3]
```
"""
function findsites(tt::TensorTrain, is)
    pred = hascommoninds(is isa Index ? [is] : collect(is))
    return findall(pred, tt)
end

findsites(tt::TensorTrain, i::Index) = findsites(tt, [i])

export findsites

"""
    siteinds(tt::TensorTrain, j::Integer) -> Vector{Index}

Return the site indices at site `j` (1-indexed).
Site indices are those unique to the tensor at site `j`
(not shared with neighboring tensors).

# Example
```julia
sites = [Index(2; tags="s1"), Index(3; tags="s2")]
tt = random_tt(sites; linkdims=2)
siteinds(tt, 1)  # Returns the site index at site 1
```
"""
function siteinds(tt::TensorTrain, j::Integer)
    n = length(tt)
    (1 <= j <= n) || throw(BoundsError(tt, j))

    t = tt[j]
    t_inds = indices(t)

    if n == 1
        # Single site: all indices are site indices
        return collect(t_inds)
    elseif j == 1
        # First site: site indices are those not shared with next tensor
        return uniqueinds(t_inds, indices(tt[j + 1]))
    elseif j == n
        # Last site: site indices are those not shared with previous tensor
        return uniqueinds(t_inds, indices(tt[j - 1]))
    else
        # Middle site: site indices are those not shared with neighbors
        return uniqueinds(t_inds, vcat(indices(tt[j - 1]), indices(tt[j + 1])))
    end
end

"""
    siteinds(tt::TensorTrain) -> Vector{Vector{Index}}

Return all site indices for all sites.

# Example
```julia
sites = [Index(2), Index(3), Index(4)]
tt = random_tt(sites; linkdims=2)
all_sites = siteinds(tt)  # Returns [[s1], [s2], [s3]]
```
"""
function siteinds(tt::TensorTrain)
    return [siteinds(tt, j) for j in 1:length(tt)]
end

export siteinds

"""
    siteind(tt::TensorTrain, j::Integer) -> Union{Index, Nothing}

Return the first site index at site `j`, or `nothing` if there are no site indices.
Convenience function when expecting exactly one site index per site.
"""
function siteind(tt::TensorTrain, j::Integer)
    inds = siteinds(tt, j)
    return isempty(inds) ? nothing : first(inds)
end

export siteind

# ============================================================================
# TensorTrain from Array list
# ============================================================================

"""
    TensorTrain(arrays::Vector{<:AbstractArray}, site_inds::Vector{<:Vector{Index}})

Create a tensor train from a list of arrays and their corresponding site indices.
Link indices are created automatically between adjacent tensors.

# Arguments
- `arrays`: Vector of arrays, where each array corresponds to a tensor in the train
- `site_inds`: Vector of vectors of site indices for each tensor

# Example
```julia
# Create a 2-site tensor train
s1 = Index(2)
s2 = Index(3)
arr1 = rand(2, 4)  # site_dim × bond_dim
arr2 = rand(4, 3)  # bond_dim × site_dim
tt = TensorTrain([arr1, arr2], [[s1], [s2]])
```
"""
function TensorTrain(arrays::Vector{<:AbstractArray{T}}, site_inds::Vector{<:Vector{Index}}) where T<:Number
    N = length(arrays)
    N == 0 && return TensorTrain(Tensor[])
    length(site_inds) == N || throw(ArgumentError("Length of arrays and site_inds must match"))

    # Infer link dimensions from array shapes
    # For site i, array shape is (left_bond, site_dims..., right_bond)
    # First tensor: (site_dims..., right_bond)
    # Last tensor: (left_bond, site_dims...)
    # Middle tensors: (left_bond, site_dims..., right_bond)

    tensors_list = Tensor[]
    link_indices = Index[]

    for i in 1:N
        arr = arrays[i]
        s_inds = site_inds[i]
        site_dim_prod = prod(dim(s) for s in s_inds)
        arr_dims = size(arr)
        total_dim = prod(arr_dims)

        # Determine link dimensions
        if i == 1
            # First tensor: dims = (site_dims..., right_bond)
            right_bond_dim = total_dim ÷ site_dim_prod
            if N > 1
                l_right = Index(right_bond_dim; tags="Link,l=$i")
                push!(link_indices, l_right)
                all_inds = [s_inds..., l_right]
            else
                all_inds = s_inds
            end
        elseif i == N
            # Last tensor: dims = (left_bond, site_dims...)
            l_left = link_indices[i-1]
            all_inds = [l_left, s_inds...]
        else
            # Middle tensor: dims = (left_bond, site_dims..., right_bond)
            l_left = link_indices[i-1]
            left_bond_dim = dim(l_left)
            right_bond_dim = total_dim ÷ (left_bond_dim * site_dim_prod)
            l_right = Index(right_bond_dim; tags="Link,l=$i")
            push!(link_indices, l_right)
            all_inds = [l_left, s_inds..., l_right]
        end

        # Create tensor
        t = Tensor(all_inds, arr)
        push!(tensors_list, t)
    end

    return TensorTrain(tensors_list)
end

"""
    TensorTrain(arrays::Vector{<:AbstractArray}, site_inds::Vector{Index})

Convenience constructor when each tensor has exactly one site index.
"""
function TensorTrain(arrays::Vector{<:AbstractArray{T}}, site_inds::Vector{Index}) where T<:Number
    return TensorTrain(arrays, [[s] for s in site_inds])
end

# ============================================================================
# Random TensorTrain
# ============================================================================

"""
    random_tt([::Type{T},] site_inds::Vector{Index}; linkdims::Union{Integer,Vector{<:Integer}}=1)

Construct a random tensor train with the given site indices and link dimensions.

# Arguments
- `T`: Element type (default: Float64)
- `site_inds`: Vector of site indices
- `linkdims`: Link dimension(s) - either a single integer (uniform) or a vector of integers

# Example
```julia
sites = [Index(2) for _ in 1:5]
tt = random_tt(sites; linkdims=4)
tt_complex = random_tt(ComplexF64, sites; linkdims=4)
```
"""
function random_tt(::Type{T}, site_inds::Vector{Index}; linkdims::Union{Integer,Vector{<:Integer}}=1) where T<:Number
    N = length(site_inds)
    N == 0 && return TensorTrain(Tensor[])

    # Normalize linkdims to a vector
    if linkdims isa Integer
        _linkdims = fill(linkdims, N - 1)
    else
        length(linkdims) == N - 1 || throw(ArgumentError("linkdims must have length $(N-1), got $(length(linkdims))"))
        _linkdims = collect(linkdims)
    end

    tensors_list = Tensor[]
    links = Index[]

    # Create link indices
    for i in 1:(N-1)
        push!(links, Index(_linkdims[i]; tags="Link,l=$i"))
    end

    for i in 1:N
        s = site_inds[i]

        if N == 1
            # Single site: just site index
            inds = [s]
            data = randn(T, dim(s))
        elseif i == 1
            # First site: site index, right link
            inds = [s, links[1]]
            data = randn(T, dim(s), dim(links[1]))
        elseif i == N
            # Last site: left link, site index
            inds = [links[N-1], s]
            data = randn(T, dim(links[N-1]), dim(s))
        else
            # Middle site: left link, site index, right link
            inds = [links[i-1], s, links[i]]
            data = randn(T, dim(links[i-1]), dim(s), dim(links[i]))
        end

        push!(tensors_list, Tensor(inds, data))
    end

    return TensorTrain(tensors_list)
end

function random_tt(site_inds::Vector{Index}; linkdims::Union{Integer,Vector{<:Integer}}=1)
    return random_tt(Float64, site_inds; linkdims=linkdims)
end

export random_tt

# ============================================================================
# HDF5 Save/Load for TensorTrain (ITensorMPS.jl compatible)
# ============================================================================

"""
    save_mps(filepath::AbstractString, name::AbstractString, tt::TensorTrain)

Save a tensor train to an HDF5 file in ITensorMPS.jl-compatible MPS format.

# Arguments
- `filepath`: Path to the HDF5 file (will be created/overwritten)
- `name`: Name of the HDF5 group to write the MPS to
- `tt`: Tensor train to save
"""
function save_mps(filepath::AbstractString, name::AbstractString, tt::TensorTrain)
    status = C_API.t4a_hdf5_save_mps(filepath, name, tt.ptr)
    C_API.check_status(status)
    return nothing
end

"""
    load_mps(filepath::AbstractString, name::AbstractString) -> TensorTrain

Load a tensor train from an HDF5 file in ITensorMPS.jl-compatible MPS format.

# Arguments
- `filepath`: Path to the HDF5 file
- `name`: Name of the HDF5 group containing the MPS

# Returns
A `TensorTrain` loaded from the file.
"""
function load_mps(filepath::AbstractString, name::AbstractString)
    out = Ref{Ptr{Cvoid}}(C_NULL)
    status = C_API.t4a_hdf5_load_mps(filepath, name, out)
    C_API.check_status(status)
    return TensorTrain(out[])
end

export save_mps, load_mps

end # module ITensorLike
