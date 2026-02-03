"""
    TreeTN

Tree Tensor Network (TTN) functionality, including MPS and MPO.

This submodule provides TreeTensorNetwork and related operations,
wrapping the Rust `tensor4all-treetn` crate via C API.

# Usage
```julia
using Tensor4all
using Tensor4all.TreeTN

# Create MPS from tensors
sites = [Index(2) for _ in 1:5]
mps = random_mps(sites; linkdims=4)

# Operations
orthogonalize!(mps, 3)
truncate!(mps; maxdim=10)
```
"""
module TreeTN

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
# TreeTensorNetwork Type
# ============================================================================

"""
    TreeTensorNetwork{V}

A tree tensor network with vertices of type `V`.

Wraps a Rust `DefaultTreeTN<usize>` which supports arbitrary tree topologies.
For MPS/MPO, vertices are integers (1-indexed in Julia, 0-indexed in C).

# Type Parameters
- `V`: Vertex name type. `Int` for MPS/MPO (1-indexed Julia convention).

# Constructors
- `TreeTensorNetwork{V}(tensors::Vector{Pair{V, Tensor}})` - Create from named tensors
- `MPS(tensors::Vector{Tensor})` - Create MPS with vertices 1, 2, ..., n
- `MPO(tensors::Vector{Tensor})` - Create MPO with vertices 1, 2, ..., n
"""
mutable struct TreeTensorNetwork{V}
    handle::Ptr{Cvoid}
    node_map::Dict{V, Int}     # V -> 0-based C index
    node_names::Vector{V}      # 0-based C index -> V (index is offset+1 for Julia)

    function TreeTensorNetwork{V}(handle::Ptr{Cvoid}, node_map::Dict{V, Int}, node_names::Vector{V}) where V
        if handle == C_NULL
            error("Failed to create TreeTensorNetwork (null pointer from C API)")
        end
        ttn = new{V}(handle, node_map, node_names)
        finalizer(ttn) do x
            if x.handle != C_NULL
                C_API.t4a_treetn_release(x.handle)
                x.handle = C_NULL
            end
        end
        ttn
    end
end

const TTN = TreeTensorNetwork

export TreeTensorNetwork, TTN

# ============================================================================
# Node name conversion helpers
# ============================================================================

"""Convert Julia vertex name to 0-based C index."""
function _to_c_vertex(ttn::TreeTensorNetwork{V}, v::V) where V
    haskey(ttn.node_map, v) || throw(ArgumentError("Vertex $v not found in TTN"))
    return ttn.node_map[v]
end

"""Convert 0-based C index to Julia vertex name."""
function _from_c_vertex(ttn::TreeTensorNetwork{V}, idx::Integer) where V
    return ttn.node_names[idx + 1]  # 0-based to 1-based Julia array index
end

# ============================================================================
# MPS/MPO type aliases and constructors
# ============================================================================

"""
    MPS

Type alias for `TreeTensorNetwork{Int}` (Matrix Product State).
Vertices are 1-indexed (Julia convention).
"""
const MPS = TreeTensorNetwork{Int}

"""
    MPO

Type alias for `TreeTensorNetwork{Int}` (Matrix Product Operator).
Vertices are 1-indexed (Julia convention).
"""
const MPO = TreeTensorNetwork{Int}

export MPS, MPO

"""
    MPS(tensors::Vector{Tensor})

Create an MPS from a vector of tensors. Vertices are named 1, 2, ..., n (1-indexed).
Adjacent tensors must share exactly one common index (the link index).
"""
function MPS(tensors::Vector{Tensor})
    isempty(tensors) && error("Cannot create MPS from empty tensor list")
    n = length(tensors)

    # Build node map: Julia 1-indexed -> C 0-indexed
    node_map = Dict{Int, Int}(i => i - 1 for i in 1:n)
    node_names = collect(1:n)

    # Create via C API
    tensor_ptrs = Ptr{Cvoid}[t.ptr for t in tensors]
    out = Ref{Ptr{Cvoid}}(C_NULL)
    status = C_API.t4a_treetn_new(tensor_ptrs, n, out)
    C_API.check_status(status)

    return TreeTensorNetwork{Int}(out[], node_map, node_names)
end

# Note: MPO(tensors::Vector{Tensor}) is not defined separately because
# MPO === MPS === TreeTensorNetwork{Int}, so MPS(tensors) works for both.
# Defining a separate function would overwrite the MPS constructor.

# ============================================================================
# General TreeTN constructor
# ============================================================================

"""
    TreeTensorNetwork{V}(pairs::Vector{Pair{V, Tensor}}) where V

Create a TreeTensorNetwork from vertex-name => tensor pairs.
Tensors are connected by matching index IDs (einsum rule).
C API nodes are named 0, 1, ..., n-1 in the order of the input pairs.
"""
function TreeTensorNetwork{V}(pairs::Vector{Pair{V, Tensor}}) where V
    isempty(pairs) && error("Cannot create TreeTensorNetwork from empty list")
    n = length(pairs)

    node_names = V[p.first for p in pairs]
    node_map = Dict{V, Int}(name => i - 1 for (i, name) in enumerate(node_names))

    tensor_ptrs = Ptr{Cvoid}[p.second.ptr for p in pairs]
    out = Ref{Ptr{Cvoid}}(C_NULL)
    status = C_API.t4a_treetn_new(tensor_ptrs, n, out)
    C_API.check_status(status)

    return TreeTensorNetwork{V}(out[], node_map, node_names)
end

# ============================================================================
# Accessors
# ============================================================================

"""
    nv(ttn::TreeTensorNetwork) -> Int

Get the number of vertices (nodes) in the tree tensor network.
"""
function nv(ttn::TreeTensorNetwork)
    out = Ref{Csize_t}(0)
    status = C_API.t4a_treetn_num_vertices(ttn.handle, out)
    C_API.check_status(status)
    return Int(out[])
end

export nv

"""
    ne(ttn::TreeTensorNetwork) -> Int

Get the number of edges (bonds) in the tree tensor network.
"""
function ne(ttn::TreeTensorNetwork)
    out = Ref{Csize_t}(0)
    status = C_API.t4a_treetn_num_edges(ttn.handle, out)
    C_API.check_status(status)
    return Int(out[])
end

export ne

"""
    length(ttn::TreeTensorNetwork) -> Int

Get the number of vertices. Alias for `nv`.
"""
Base.length(ttn::TreeTensorNetwork) = nv(ttn)

"""
    vertices(ttn::TreeTensorNetwork) -> Vector{V}

Get all vertex names.
"""
function vertices(ttn::TreeTensorNetwork{V}) where V
    return copy(ttn.node_names)
end

export vertices

"""
    neighbors(ttn::TreeTensorNetwork{V}, v::V) -> Vector{V}

Get the neighbors of vertex `v`.
"""
function neighbors(ttn::TreeTensorNetwork{V}, v::V) where V
    c_v = _to_c_vertex(ttn, v)
    buf = Vector{Csize_t}(undef, nv(ttn))
    n_out = Ref{Csize_t}(0)
    status = C_API.t4a_treetn_neighbors(ttn.handle, c_v, buf, length(buf), n_out)
    C_API.check_status(status)
    return V[_from_c_vertex(ttn, buf[i]) for i in 1:n_out[]]
end

export neighbors

"""
    getindex(ttn::TreeTensorNetwork{V}, v::V) -> Tensor

Get the tensor at vertex `v`.
"""
function Base.getindex(ttn::TreeTensorNetwork{V}, v::V) where V
    c_v = _to_c_vertex(ttn, v)
    out = Ref{Ptr{Cvoid}}(C_NULL)
    status = C_API.t4a_treetn_tensor(ttn.handle, c_v, out)
    C_API.check_status(status)
    return Tensor(out[])
end

"""
    setindex!(ttn::TreeTensorNetwork{V}, tensor::Tensor, v::V)

Set the tensor at vertex `v`.
"""
function Base.setindex!(ttn::TreeTensorNetwork{V}, tensor::Tensor, v::V) where V
    c_v = _to_c_vertex(ttn, v)
    status = C_API.t4a_treetn_set_tensor(ttn.handle, c_v, tensor.ptr)
    C_API.check_status(status)
    return ttn
end

# ============================================================================
# Index accessors
# ============================================================================

"""
    siteinds(ttn::TreeTensorNetwork{V}, v::V) -> Vector{Index}

Get the site (physical) indices at vertex `v`.
"""
function siteinds(ttn::TreeTensorNetwork{V}, v::V) where V
    c_v = _to_c_vertex(ttn, v)
    # First call with a reasonable buffer size
    buf_size = 16
    idx_buf = Vector{Ptr{Cvoid}}(undef, buf_size)
    n_out = Ref{Csize_t}(0)
    status = C_API.t4a_treetn_siteinds(ttn.handle, c_v, idx_buf, buf_size, n_out)
    if status == C_API.T4A_BUFFER_TOO_SMALL
        buf_size = Int(n_out[])
        idx_buf = Vector{Ptr{Cvoid}}(undef, buf_size)
        status = C_API.t4a_treetn_siteinds(ttn.handle, c_v, idx_buf, buf_size, n_out)
    end
    C_API.check_status(status)
    return Index[Index(idx_buf[i]) for i in 1:n_out[]]
end

export siteinds

"""
    siteind(ttn::TreeTensorNetwork{Int}, j::Integer) -> Union{Index, Nothing}

Get the first site index at MPS site `j` (1-indexed).
"""
function siteind(ttn::TreeTensorNetwork{Int}, j::Integer)
    inds = siteinds(ttn, j)
    return isempty(inds) ? nothing : first(inds)
end

export siteind

"""
    linkind(ttn::TreeTensorNetwork{V}, v1::V, v2::V) -> Index

Get the link (bond) index on the edge between vertices `v1` and `v2`.
"""
function linkind(ttn::TreeTensorNetwork{V}, v1::V, v2::V) where V
    c_v1 = _to_c_vertex(ttn, v1)
    c_v2 = _to_c_vertex(ttn, v2)
    out = Ref{Ptr{Cvoid}}(C_NULL)
    status = C_API.t4a_treetn_linkind(ttn.handle, c_v1, c_v2, out)
    C_API.check_status(status)
    return Index(out[])
end

"""
    linkind(mps::TreeTensorNetwork{Int}, j::Integer) -> Index

Get the link index between MPS sites `j` and `j+1` (1-indexed).
"""
function linkind(mps::TreeTensorNetwork{Int}, j::Integer)
    # Julia 1-indexed j: link between j and j+1
    # C 0-indexed: link between j-1 and j
    out = Ref{Ptr{Cvoid}}(C_NULL)
    status = C_API.t4a_treetn_linkind_at(mps.handle, j - 1, out)
    C_API.check_status(status)
    return Index(out[])
end

export linkind

"""
    linkinds(ttn::TreeTensorNetwork{Int}) -> Vector{Index}

Get all link indices of an MPS-like TTN.
"""
function linkinds(ttn::TreeTensorNetwork{Int})
    n = nv(ttn)
    n <= 1 && return Index[]
    return Index[linkind(ttn, j) for j in 1:(n-1)]
end

export linkinds

"""
    linkdim(ttn::TreeTensorNetwork{V}, v1::V, v2::V) -> Int

Get the bond dimension on the edge between vertices `v1` and `v2`.
"""
function linkdim(ttn::TreeTensorNetwork{V}, v1::V, v2::V) where V
    c_v1 = _to_c_vertex(ttn, v1)
    c_v2 = _to_c_vertex(ttn, v2)
    out = Ref{Csize_t}(0)
    status = C_API.t4a_treetn_bond_dim(ttn.handle, c_v1, c_v2, out)
    C_API.check_status(status)
    return Int(out[])
end

"""
    linkdim(mps::TreeTensorNetwork{Int}, j::Integer) -> Int

Get the bond dimension between MPS sites `j` and `j+1` (1-indexed).
"""
function linkdim(mps::TreeTensorNetwork{Int}, j::Integer)
    out = Ref{Csize_t}(0)
    status = C_API.t4a_treetn_bond_dim_at(mps.handle, j - 1, out)
    C_API.check_status(status)
    return Int(out[])
end

export linkdim

"""
    linkdims(mps::TreeTensorNetwork{Int}) -> Vector{Int}

Get all bond dimensions for an MPS-like TTN.
"""
function linkdims(mps::TreeTensorNetwork{Int})
    n = nv(mps)
    n <= 1 && return Int[]
    out = Vector{Csize_t}(undef, n - 1)
    status = C_API.t4a_treetn_bond_dims(mps.handle, out, n - 1)
    C_API.check_status(status)
    return Int.(out)
end

export linkdims

"""
    maxbonddim(mps::TreeTensorNetwork{Int}) -> Int

Get the maximum bond dimension of an MPS-like TTN.
"""
function maxbonddim(mps::TreeTensorNetwork{Int})
    out = Ref{Csize_t}(0)
    status = C_API.t4a_treetn_maxbonddim(mps.handle, out)
    C_API.check_status(status)
    return Int(out[])
end

export maxbonddim

# ============================================================================
# Orthogonalization
# ============================================================================

"""
    orthogonalize!(ttn::TreeTensorNetwork{V}, v::V; form::CanonicalForm=Unitary)

Orthogonalize the tree tensor network in place to vertex `v`.

# Arguments
- `v`: Target vertex for orthogonality center
- `form`: Canonical form to use (default: `Unitary`)
"""
function orthogonalize!(ttn::TreeTensorNetwork{V}, v::V; form::CanonicalForm=Unitary) where V
    c_v = _to_c_vertex(ttn, v)
    status = C_API.t4a_treetn_orthogonalize_with(ttn.handle, c_v, Int(form))
    C_API.check_status(status)
    return ttn
end

export orthogonalize!

"""
    orthogonalize(ttn::TreeTensorNetwork{V}, v::V; form::CanonicalForm=Unitary)

Return an orthogonalized copy of the tree tensor network.
"""
function orthogonalize(ttn::TreeTensorNetwork{V}, v::V; form::CanonicalForm=Unitary) where V
    ttn_copy = copy(ttn)
    orthogonalize!(ttn_copy, v; form=form)
    return ttn_copy
end

export orthogonalize

"""
    canonical_form(ttn::TreeTensorNetwork) -> Union{CanonicalForm, Nothing}

Get the canonical form used for the tree tensor network.
Returns `nothing` if no canonical form is set.
"""
function canonical_form(ttn::TreeTensorNetwork)
    out = Ref{Cint}(0)
    status = C_API.t4a_treetn_canonical_form(ttn.handle, out)
    status == C_API.T4A_INVALID_ARGUMENT && return nothing
    C_API.check_status(status)
    return CanonicalForm(out[])
end

export canonical_form

# ============================================================================
# Truncation
# ============================================================================

"""
    truncate!(ttn::TreeTensorNetwork; rtol=0.0, cutoff=0.0, maxdim=0)

Truncate the tree tensor network bond dimensions in-place.

# Arguments
- `rtol`: Relative tolerance for truncation (0.0 = not set)
- `cutoff`: ITensorMPS.jl cutoff (0.0 = not set). Converted to rtol = sqrt(cutoff).
- `maxdim`: Maximum bond dimension (0 = no limit)
"""
function truncate!(ttn::TreeTensorNetwork; rtol::Real=0.0, cutoff::Real=0.0, maxdim::Integer=0)
    status = C_API.t4a_treetn_truncate(ttn.handle, Float64(rtol), Float64(cutoff), maxdim)
    C_API.check_status(status)
    return ttn
end

export truncate!

"""
    truncate(ttn::TreeTensorNetwork; rtol=0.0, cutoff=0.0, maxdim=0) -> TreeTensorNetwork

Return a truncated copy of the tree tensor network.
"""
function truncate(ttn::TreeTensorNetwork; rtol::Real=0.0, cutoff::Real=0.0, maxdim::Integer=0)
    ttn_copy = copy(ttn)
    truncate!(ttn_copy; rtol=rtol, cutoff=cutoff, maxdim=maxdim)
    return ttn_copy
end

export truncate

# ============================================================================
# Scalar operations
# ============================================================================

"""
    inner(a::TreeTensorNetwork, b::TreeTensorNetwork) -> ComplexF64

Compute the inner product <a|b>.
"""
function inner(a::TreeTensorNetwork, b::TreeTensorNetwork)
    out_re = Ref{Cdouble}(0.0)
    out_im = Ref{Cdouble}(0.0)
    status = C_API.t4a_treetn_inner(a.handle, b.handle, out_re, out_im)
    C_API.check_status(status)
    return ComplexF64(out_re[], out_im[])
end

export inner

"""
    norm(ttn::TreeTensorNetwork) -> Float64

Compute the norm of the tree tensor network.
"""
function LinearAlgebra.norm(ttn::TreeTensorNetwork)
    out = Ref{Cdouble}(0.0)
    status = C_API.t4a_treetn_norm(ttn.handle, out)
    C_API.check_status(status)
    return out[]
end

"""
    lognorm(ttn::TreeTensorNetwork) -> Float64

Compute the log-norm of the tree tensor network.
"""
function lognorm(ttn::TreeTensorNetwork)
    out = Ref{Cdouble}(0.0)
    status = C_API.t4a_treetn_lognorm(ttn.handle, out)
    C_API.check_status(status)
    return out[]
end

export lognorm

# ============================================================================
# Contraction and Addition
# ============================================================================

# Internal: convert method specification to C API integer
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
    contract(a::TreeTensorNetwork, b::TreeTensorNetwork; kwargs...) -> TreeTensorNetwork

Contract two tree tensor networks.

# Arguments
- `method`: Contraction method - `:zipup` (default), `:fit`, or `:naive`
- `maxdim`: Maximum bond dimension (0 for no limit)
- `rtol`: Relative tolerance (0.0 = not set)
- `cutoff`: ITensorMPS.jl cutoff (0.0 = not set)
"""
function contract(a::TreeTensorNetwork{V}, b::TreeTensorNetwork{V};
                  method::Union{Symbol, AbstractString, Integer}=:zipup,
                  maxdim::Integer=0,
                  rtol::Real=0.0,
                  cutoff::Real=0.0) where V
    method_int = _contract_method_to_int(method)
    out = Ref{Ptr{Cvoid}}(C_NULL)
    status = C_API.t4a_treetn_contract(a.handle, b.handle, method_int,
                                        Float64(rtol), Float64(cutoff), maxdim, out)
    C_API.check_status(status)
    # Result has same node names as input
    return TreeTensorNetwork{V}(out[], copy(a.node_map), copy(a.node_names))
end

export contract

"""
    +(a::TreeTensorNetwork, b::TreeTensorNetwork) -> TreeTensorNetwork

Add two tree tensor networks using direct-sum construction.
"""
function Base.:+(a::TreeTensorNetwork{V}, b::TreeTensorNetwork{V}) where V
    out = Ref{Ptr{Cvoid}}(C_NULL)
    status = C_API.t4a_treetn_add(a.handle, b.handle, out)
    C_API.check_status(status)
    return TreeTensorNetwork{V}(out[], copy(a.node_map), copy(a.node_names))
end

# ============================================================================
# Dense conversion
# ============================================================================

"""
    to_dense(ttn::TreeTensorNetwork) -> Tensor

Convert the tree tensor network to a dense tensor by contracting all link indices.
"""
function to_dense(ttn::TreeTensorNetwork)
    out = Ref{Ptr{Cvoid}}(C_NULL)
    status = C_API.t4a_treetn_to_dense(ttn.handle, out)
    C_API.check_status(status)
    return Tensor(out[])
end

export to_dense

# ============================================================================
# Linear solver
# ============================================================================

"""
    linsolve(A::TreeTensorNetwork, b::TreeTensorNetwork, x0::TreeTensorNetwork; kwargs...) -> TreeTensorNetwork

Solve `(a0 + a1 * A) * x = b` for `x` using DMRG-like sweeps.

# Arguments
- `A`: The operator (MPO/TTN operator)
- `b`: The right-hand side (MPS/TTN state)
- `x0`: Initial guess (MPS/TTN state)

# Keyword Arguments
- `a0::Real=0.0`: Coefficient a0
- `a1::Real=1.0`: Coefficient a1
- `nsweeps::Integer=10`: Number of full sweeps
- `rtol::Real=0.0`: Relative tolerance
- `cutoff::Real=0.0`: ITensorMPS.jl cutoff
- `maxdim::Integer=0`: Maximum bond dimension
"""
function linsolve(A::TreeTensorNetwork{V}, b::TreeTensorNetwork{V}, x0::TreeTensorNetwork{V};
                  a0::Real=0.0, a1::Real=1.0, nsweeps::Integer=10,
                  rtol::Real=0.0, cutoff::Real=0.0, maxdim::Integer=0) where V
    out = Ref{Ptr{Cvoid}}(C_NULL)
    status = C_API.t4a_treetn_linsolve(A.handle, b.handle, x0.handle,
                                        Float64(a0), Float64(a1), nsweeps,
                                        Float64(rtol), Float64(cutoff), maxdim, out)
    C_API.check_status(status)
    return TreeTensorNetwork{V}(out[], copy(x0.node_map), copy(x0.node_names))
end

export linsolve

# ============================================================================
# Copy / Clone
# ============================================================================

function Base.copy(ttn::TreeTensorNetwork{V}) where V
    ptr = C_API.t4a_treetn_clone(ttn.handle)
    return TreeTensorNetwork{V}(ptr, copy(ttn.node_map), copy(ttn.node_names))
end

function Base.deepcopy(ttn::TreeTensorNetwork{V}) where V
    ptr = C_API.t4a_treetn_clone(ttn.handle)
    return TreeTensorNetwork{V}(ptr, copy(ttn.node_map), copy(ttn.node_names))
end

# ============================================================================
# Display
# ============================================================================

function Base.show(io::IO, ttn::TreeTensorNetwork{V}) where V
    n = nv(ttn)
    print(io, "TreeTensorNetwork{$V}(nv=$n, ne=$(ne(ttn)))")
end

function Base.show(io::IO, ::MIME"text/plain", ttn::TreeTensorNetwork{V}) where V
    println(io, "Tensor4all.TreeTN.TreeTensorNetwork{$V}")
    println(io, "  vertices: ", nv(ttn))
    println(io, "  edges: ", ne(ttn))
    println(io, "  vertex_names: ", ttn.node_names)
    cf = canonical_form(ttn)
    cf !== nothing && println(io, "  canonical_form: ", cf)
end

# ============================================================================
# Iterator
# ============================================================================

function Base.iterate(ttn::TreeTensorNetwork{V}, state=1) where V
    if state > nv(ttn)
        return nothing
    end
    v = ttn.node_names[state]
    return (ttn[v], state + 1)
end

Base.eltype(::Type{TreeTensorNetwork{V}}) where V = Tensor
Base.firstindex(ttn::TreeTensorNetwork{Int}) = 1
Base.lastindex(ttn::TreeTensorNetwork{Int}) = nv(ttn)
Base.eachindex(ttn::TreeTensorNetwork{Int}) = 1:nv(ttn)
Base.keys(ttn::TreeTensorNetwork{V}) where V = ttn.node_names

# Support for findfirst/findall with predicates
function Base.findfirst(pred::HasCommonIndsPredicate, ttn::TreeTensorNetwork{V}) where V
    for v in ttn.node_names
        if pred(ttn[v])
            return v
        end
    end
    return nothing
end

function Base.findall(pred::HasCommonIndsPredicate, ttn::TreeTensorNetwork{V}) where V
    result = V[]
    for v in ttn.node_names
        if pred(ttn[v])
            push!(result, v)
        end
    end
    return result
end

function Base.findfirst(f::Function, ttn::TreeTensorNetwork{V}) where V
    for v in ttn.node_names
        if f(ttn[v])
            return v
        end
    end
    return nothing
end

function Base.findall(f::Function, ttn::TreeTensorNetwork{V}) where V
    result = V[]
    for v in ttn.node_names
        if f(ttn[v])
            push!(result, v)
        end
    end
    return result
end

# ============================================================================
# Site search
# ============================================================================

"""
    findsite(ttn::TreeTensorNetwork, is)

Return the first vertex that has common indices with `is`.
"""
function findsite(ttn::TreeTensorNetwork, is)
    pred = hascommoninds(is isa Index ? [is] : collect(is))
    return findfirst(pred, ttn)
end

findsite(ttn::TreeTensorNetwork, i::Index) = findsite(ttn, [i])

export findsite

"""
    findsites(ttn::TreeTensorNetwork, is)

Return all vertices that have common indices with `is`.
"""
function findsites(ttn::TreeTensorNetwork, is)
    pred = hascommoninds(is isa Index ? [is] : collect(is))
    return findall(pred, ttn)
end

findsites(ttn::TreeTensorNetwork, i::Index) = findsites(ttn, [i])

export findsites

# ============================================================================
# MPS/MPO from arrays
# ============================================================================

"""
    MPS(arrays::Vector{<:AbstractArray{T}}, site_inds::Vector{<:Vector{Index}}) where T<:Number

Create an MPS from arrays and site indices. Link indices are created automatically.
"""
function MPS(arrays::Vector{<:AbstractArray{T}}, site_inds::Vector{<:Vector{Index}}) where T<:Number
    N = length(arrays)
    N == 0 && error("Cannot create MPS from empty arrays")
    length(site_inds) == N || throw(ArgumentError("Length of arrays and site_inds must match"))

    tensors_list = Tensor[]
    link_indices = Index[]

    for i in 1:N
        arr = arrays[i]
        s_inds = site_inds[i]
        site_dim_prod = prod(dim(s) for s in s_inds)
        total_dim = prod(size(arr))

        if i == 1
            right_bond_dim = total_dim ÷ site_dim_prod
            if N > 1
                l_right = Index(right_bond_dim; tags="Link,l=$i")
                push!(link_indices, l_right)
                all_inds = [s_inds..., l_right]
            else
                all_inds = s_inds
            end
        elseif i == N
            l_left = link_indices[i-1]
            all_inds = [l_left, s_inds...]
        else
            l_left = link_indices[i-1]
            left_bond_dim = dim(l_left)
            right_bond_dim = total_dim ÷ (left_bond_dim * site_dim_prod)
            l_right = Index(right_bond_dim; tags="Link,l=$i")
            push!(link_indices, l_right)
            all_inds = [l_left, s_inds..., l_right]
        end

        push!(tensors_list, Tensor(all_inds, arr))
    end

    return MPS(tensors_list)
end

"""
    MPS(arrays::Vector{<:AbstractArray{T}}, site_inds::Vector{Index}) where T<:Number

Convenience constructor when each tensor has exactly one site index.
"""
function MPS(arrays::Vector{<:AbstractArray{T}}, site_inds::Vector{Index}) where T<:Number
    return MPS(arrays, [[s] for s in site_inds])
end

# ============================================================================
# Random MPS
# ============================================================================

"""
    random_mps([::Type{T},] site_inds::Vector{Index}; linkdims::Union{Integer,Vector{<:Integer}}=1)

Construct a random MPS with the given site indices and link dimensions.
"""
function random_mps(::Type{T}, site_inds::Vector{Index}; linkdims::Union{Integer,Vector{<:Integer}}=1) where T<:Number
    N = length(site_inds)
    N == 0 && error("Cannot create random MPS from empty site indices")

    if linkdims isa Integer
        _linkdims = fill(linkdims, N - 1)
    else
        length(linkdims) == N - 1 || throw(ArgumentError("linkdims must have length $(N-1)"))
        _linkdims = collect(linkdims)
    end

    tensors_list = Tensor[]
    links = Index[]

    for i in 1:(N-1)
        push!(links, Index(_linkdims[i]; tags="Link,l=$i"))
    end

    for i in 1:N
        s = site_inds[i]
        if N == 1
            inds = [s]
            d = randn(T, dim(s))
        elseif i == 1
            inds = [s, links[1]]
            d = randn(T, dim(s), dim(links[1]))
        elseif i == N
            inds = [links[N-1], s]
            d = randn(T, dim(links[N-1]), dim(s))
        else
            inds = [links[i-1], s, links[i]]
            d = randn(T, dim(links[i-1]), dim(s), dim(links[i]))
        end
        push!(tensors_list, Tensor(inds, d))
    end

    return MPS(tensors_list)
end

function random_mps(site_inds::Vector{Index}; linkdims::Union{Integer,Vector{<:Integer}}=1)
    return random_mps(Float64, site_inds; linkdims=linkdims)
end

export random_mps

# Keep random_tt as alias for backward compatibility
const random_tt = random_mps
export random_tt

# ============================================================================
# HDF5 Save/Load
# ============================================================================

"""
    save_mps(filepath::AbstractString, name::AbstractString, mps::TreeTensorNetwork{Int})

Save an MPS to an HDF5 file in ITensorMPS.jl-compatible format.
"""
function save_mps(filepath::AbstractString, name::AbstractString, mps::TreeTensorNetwork{Int})
    status = C_API.t4a_hdf5_save_mps(filepath, name, mps.handle)
    C_API.check_status(status)
    return nothing
end

"""
    load_mps(filepath::AbstractString, name::AbstractString) -> MPS

Load an MPS from an HDF5 file in ITensorMPS.jl-compatible format.

Returns a `TreeTensorNetwork{Int}` (MPS) with vertices named 1, 2, ..., n.
"""
function load_mps(filepath::AbstractString, name::AbstractString)
    out = Ref{Ptr{Cvoid}}(C_NULL)
    status = C_API.t4a_hdf5_load_mps(filepath, name, out)
    C_API.check_status(status)

    # The C API now returns a t4a_treetn handle directly.
    # Query num_vertices to build the node map.
    n_out = Ref{Csize_t}(0)
    status = C_API.t4a_treetn_num_vertices(out[], n_out)
    C_API.check_status(status)
    n = Int(n_out[])

    # Build node map: Julia 1-indexed -> C 0-indexed
    node_map = Dict{Int, Int}(i => i - 1 for i in 1:n)
    node_names = collect(1:n)

    return TreeTensorNetwork{Int}(out[], node_map, node_names)
end

export save_mps, load_mps

end # module TreeTN
