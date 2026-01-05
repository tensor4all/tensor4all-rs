"""
    MPO

Submodule for MPO (Matrix Product Operator) operations (corresponds to tensor4all-mpocontraction).

# Basic Usage

```julia
using Tensor4all.MPO

# Create an identity MPO
mpo = identity(Float64, [2, 3])

# Contract two MPOs
result = contract(mpo_a, mpo_b; algorithm=:zipup, tolerance=1e-12)
```
"""
module MPO

using Libdl

# Import parent module's library management
import ..get_lib, ..C_API, ..Algorithm

# Use C_API's status code
const T4A_SUCCESS = C_API.T4A_SUCCESS

# ============================================================================
# Contraction Algorithm Enum (from Algorithm module)
# ============================================================================

# Contraction algorithm values (must match Rust t4a_contraction_algorithm)
const CONTRACTION_NAIVE = Cint(0)
const CONTRACTION_ZIPUP = Cint(1)
const CONTRACTION_FIT = Cint(2)

"""
    algorithm_to_cint(alg::Symbol) -> Cint

Convert Julia symbol to C API contraction algorithm integer.
"""
function algorithm_to_cint(alg::Symbol)
    if alg == :naive
        return CONTRACTION_NAIVE
    elseif alg == :zipup
        return CONTRACTION_ZIPUP
    elseif alg == :fit
        return CONTRACTION_FIT
    else
        error("Unknown contraction algorithm: $alg. Use :naive, :zipup, or :fit")
    end
end

# ============================================================================
# MPOF64
# ============================================================================

"""
    MPOF64

A Matrix Product Operator with Float64 elements.

Wraps the Rust `MPO<f64>` type via the C API.
"""
mutable struct MPOF64
    ptr::Ptr{Cvoid}

    function MPOF64(ptr::Ptr{Cvoid})
        ptr == C_NULL && error("Failed to create MPOF64 (null pointer)")
        mpo = new(ptr)
        finalizer(_release_mpo_f64, mpo)
        return mpo
    end
end

function _release_mpo_f64(mpo::MPOF64)
    if mpo.ptr != C_NULL
        lib = get_lib()
        ccall(Libdl.dlsym(lib, :t4a_mpo_f64_release), Cvoid, (Ptr{Cvoid},), mpo.ptr)
        mpo.ptr = C_NULL
    end
end

# ============================================================================
# MPOC64 (Complex{Float64})
# ============================================================================

"""
    MPOC64

A Matrix Product Operator with Complex{Float64} elements.

Wraps the Rust `MPO<Complex64>` type via the C API.
"""
mutable struct MPOC64
    ptr::Ptr{Cvoid}

    function MPOC64(ptr::Ptr{Cvoid})
        ptr == C_NULL && error("Failed to create MPOC64 (null pointer)")
        mpo = new(ptr)
        finalizer(_release_mpo_c64, mpo)
        return mpo
    end
end

function _release_mpo_c64(mpo::MPOC64)
    if mpo.ptr != C_NULL
        lib = get_lib()
        ccall(Libdl.dlsym(lib, :t4a_mpo_c64_release), Cvoid, (Ptr{Cvoid},), mpo.ptr)
        mpo.ptr = C_NULL
    end
end

# ============================================================================
# Constructors - Float64
# ============================================================================

"""
    zeros(::Type{Float64}, site_dims::Vector{Tuple{Int,Int}})

Create an MPO representing the zero operator.
"""
function zeros(::Type{Float64}, site_dims::Vector{Tuple{Int,Int}})
    lib = get_lib()
    dims1 = convert(Vector{Csize_t}, [d[1] for d in site_dims])
    dims2 = convert(Vector{Csize_t}, [d[2] for d in site_dims])
    ptr = ccall(
        Libdl.dlsym(lib, :t4a_mpo_f64_new_zeros),
        Ptr{Cvoid},
        (Ptr{Csize_t}, Ptr{Csize_t}, Csize_t),
        dims1,
        dims2,
        length(site_dims),
    )
    return MPOF64(ptr)
end

"""
    constant(::Type{Float64}, site_dims::Vector{Tuple{Int,Int}}, value::Float64)

Create an MPO representing a constant operator.
"""
function constant(::Type{Float64}, site_dims::Vector{Tuple{Int,Int}}, value::Float64)
    lib = get_lib()
    dims1 = convert(Vector{Csize_t}, [d[1] for d in site_dims])
    dims2 = convert(Vector{Csize_t}, [d[2] for d in site_dims])
    ptr = ccall(
        Libdl.dlsym(lib, :t4a_mpo_f64_new_constant),
        Ptr{Cvoid},
        (Ptr{Csize_t}, Ptr{Csize_t}, Csize_t, Cdouble),
        dims1,
        dims2,
        length(site_dims),
        value,
    )
    return MPOF64(ptr)
end

"""
    identity(::Type{Float64}, site_dims::Vector{Int})

Create an identity MPO.
"""
function identity(::Type{Float64}, site_dims::Vector{Int})
    lib = get_lib()
    dims = convert(Vector{Csize_t}, site_dims)
    ptr = ccall(
        Libdl.dlsym(lib, :t4a_mpo_f64_new_identity),
        Ptr{Cvoid},
        (Ptr{Csize_t}, Csize_t),
        dims,
        length(dims),
    )
    return MPOF64(ptr)
end

# ============================================================================
# Constructors - ComplexF64
# ============================================================================

"""
    zeros(::Type{ComplexF64}, site_dims::Vector{Tuple{Int,Int}})

Create a complex MPO representing the zero operator.
"""
function zeros(::Type{ComplexF64}, site_dims::Vector{Tuple{Int,Int}})
    lib = get_lib()
    dims1 = convert(Vector{Csize_t}, [d[1] for d in site_dims])
    dims2 = convert(Vector{Csize_t}, [d[2] for d in site_dims])
    ptr = ccall(
        Libdl.dlsym(lib, :t4a_mpo_c64_new_zeros),
        Ptr{Cvoid},
        (Ptr{Csize_t}, Ptr{Csize_t}, Csize_t),
        dims1,
        dims2,
        length(site_dims),
    )
    return MPOC64(ptr)
end

"""
    constant(::Type{ComplexF64}, site_dims::Vector{Tuple{Int,Int}}, value::ComplexF64)

Create a complex MPO representing a constant operator.
"""
function constant(::Type{ComplexF64}, site_dims::Vector{Tuple{Int,Int}}, value::ComplexF64)
    lib = get_lib()
    dims1 = convert(Vector{Csize_t}, [d[1] for d in site_dims])
    dims2 = convert(Vector{Csize_t}, [d[2] for d in site_dims])
    ptr = ccall(
        Libdl.dlsym(lib, :t4a_mpo_c64_new_constant),
        Ptr{Cvoid},
        (Ptr{Csize_t}, Ptr{Csize_t}, Csize_t, Cdouble, Cdouble),
        dims1,
        dims2,
        length(site_dims),
        real(value),
        imag(value),
    )
    return MPOC64(ptr)
end

"""
    identity(::Type{ComplexF64}, site_dims::Vector{Int})

Create a complex identity MPO.
"""
function identity(::Type{ComplexF64}, site_dims::Vector{Int})
    lib = get_lib()
    dims = convert(Vector{Csize_t}, site_dims)
    ptr = ccall(
        Libdl.dlsym(lib, :t4a_mpo_c64_new_identity),
        Ptr{Cvoid},
        (Ptr{Csize_t}, Csize_t),
        dims,
        length(dims),
    )
    return MPOC64(ptr)
end

# ============================================================================
# Properties - Float64
# ============================================================================

function Base.length(mpo::MPOF64)
    lib = get_lib()
    out_len = Ref{Csize_t}(0)
    status = ccall(
        Libdl.dlsym(lib, :t4a_mpo_f64_len),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}),
        mpo.ptr,
        out_len,
    )
    status == T4A_SUCCESS || error("Failed to get length: status = $status")
    return Int(out_len[])
end

"""
    rank(mpo::MPOF64)

Get the maximum bond dimension (rank) of the MPO.
"""
function rank(mpo::MPOF64)
    lib = get_lib()
    out_rank = Ref{Csize_t}(0)
    status = ccall(
        Libdl.dlsym(lib, :t4a_mpo_f64_rank),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}),
        mpo.ptr,
        out_rank,
    )
    status == T4A_SUCCESS || error("Failed to get rank: status = $status")
    return Int(out_rank[])
end

# ============================================================================
# Properties - ComplexF64
# ============================================================================

function Base.length(mpo::MPOC64)
    lib = get_lib()
    out_len = Ref{Csize_t}(0)
    status = ccall(
        Libdl.dlsym(lib, :t4a_mpo_c64_len),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}),
        mpo.ptr,
        out_len,
    )
    status == T4A_SUCCESS || error("Failed to get length: status = $status")
    return Int(out_len[])
end

# ============================================================================
# Contraction - Float64
# ============================================================================

"""
    contract_naive(a::MPOF64, b::MPOF64)

Contract two MPOs using naive (exact) algorithm.
"""
function contract_naive(a::MPOF64, b::MPOF64)
    lib = get_lib()
    ptr = ccall(
        Libdl.dlsym(lib, :t4a_mpo_f64_contract_naive),
        Ptr{Cvoid},
        (Ptr{Cvoid}, Ptr{Cvoid}),
        a.ptr,
        b.ptr,
    )
    return MPOF64(ptr)
end

"""
    contract_zipup(a::MPOF64, b::MPOF64; tolerance=1e-12, max_bond_dim=0)

Contract two MPOs using zip-up algorithm with compression.
"""
function contract_zipup(
    a::MPOF64,
    b::MPOF64;
    tolerance::Float64 = 1e-12,
    max_bond_dim::Int = 0,
)
    lib = get_lib()
    ptr = ccall(
        Libdl.dlsym(lib, :t4a_mpo_f64_contract_zipup),
        Ptr{Cvoid},
        (Ptr{Cvoid}, Ptr{Cvoid}, Cdouble, Csize_t),
        a.ptr,
        b.ptr,
        tolerance,
        max_bond_dim,
    )
    return MPOF64(ptr)
end

"""
    contract_fit(a::MPOF64, b::MPOF64; tolerance=1e-12, max_bond_dim=0, max_sweeps=0)

Contract two MPOs using variational fitting algorithm.
"""
function contract_fit(
    a::MPOF64,
    b::MPOF64;
    tolerance::Float64 = 1e-12,
    max_bond_dim::Int = 0,
    max_sweeps::Int = 0,
)
    lib = get_lib()
    ptr = ccall(
        Libdl.dlsym(lib, :t4a_mpo_f64_contract_fit),
        Ptr{Cvoid},
        (Ptr{Cvoid}, Ptr{Cvoid}, Cdouble, Csize_t, Csize_t),
        a.ptr,
        b.ptr,
        tolerance,
        max_bond_dim,
        max_sweeps,
    )
    return MPOF64(ptr)
end

"""
    contract(a::MPOF64, b::MPOF64; algorithm=:naive, tolerance=1e-12, max_bond_dim=0)

Contract two MPOs using the specified algorithm.

# Arguments
- `a, b`: MPOs to contract
- `algorithm`: Contraction algorithm (:naive, :zipup, or :fit)
- `tolerance`: Relative tolerance for truncation
- `max_bond_dim`: Maximum bond dimension (0 for unlimited)
"""
function contract(
    a::MPOF64,
    b::MPOF64;
    algorithm::Symbol = :naive,
    tolerance::Float64 = 1e-12,
    max_bond_dim::Int = 0,
)
    lib = get_lib()
    alg_cint = algorithm_to_cint(algorithm)
    ptr = ccall(
        Libdl.dlsym(lib, :t4a_mpo_f64_contract),
        Ptr{Cvoid},
        (Ptr{Cvoid}, Ptr{Cvoid}, Cint, Cdouble, Csize_t),
        a.ptr,
        b.ptr,
        alg_cint,
        tolerance,
        max_bond_dim,
    )
    return MPOF64(ptr)
end

# ============================================================================
# Contraction - ComplexF64
# ============================================================================

"""
    contract_naive(a::MPOC64, b::MPOC64)

Contract two complex MPOs using naive (exact) algorithm.
"""
function contract_naive(a::MPOC64, b::MPOC64)
    lib = get_lib()
    ptr = ccall(
        Libdl.dlsym(lib, :t4a_mpo_c64_contract_naive),
        Ptr{Cvoid},
        (Ptr{Cvoid}, Ptr{Cvoid}),
        a.ptr,
        b.ptr,
    )
    return MPOC64(ptr)
end

"""
    contract_zipup(a::MPOC64, b::MPOC64; tolerance=1e-12, max_bond_dim=0)

Contract two complex MPOs using zip-up algorithm with compression.
"""
function contract_zipup(
    a::MPOC64,
    b::MPOC64;
    tolerance::Float64 = 1e-12,
    max_bond_dim::Int = 0,
)
    lib = get_lib()
    ptr = ccall(
        Libdl.dlsym(lib, :t4a_mpo_c64_contract_zipup),
        Ptr{Cvoid},
        (Ptr{Cvoid}, Ptr{Cvoid}, Cdouble, Csize_t),
        a.ptr,
        b.ptr,
        tolerance,
        max_bond_dim,
    )
    return MPOC64(ptr)
end

"""
    contract_fit(a::MPOC64, b::MPOC64; tolerance=1e-12, max_bond_dim=0, max_sweeps=0)

Contract two complex MPOs using variational fitting algorithm.
"""
function contract_fit(
    a::MPOC64,
    b::MPOC64;
    tolerance::Float64 = 1e-12,
    max_bond_dim::Int = 0,
    max_sweeps::Int = 0,
)
    lib = get_lib()
    ptr = ccall(
        Libdl.dlsym(lib, :t4a_mpo_c64_contract_fit),
        Ptr{Cvoid},
        (Ptr{Cvoid}, Ptr{Cvoid}, Cdouble, Csize_t, Csize_t),
        a.ptr,
        b.ptr,
        tolerance,
        max_bond_dim,
        max_sweeps,
    )
    return MPOC64(ptr)
end

"""
    contract(a::MPOC64, b::MPOC64; algorithm=:naive, tolerance=1e-12, max_bond_dim=0)

Contract two complex MPOs using the specified algorithm.

# Arguments
- `a, b`: MPOs to contract
- `algorithm`: Contraction algorithm (:naive, :zipup, or :fit)
- `tolerance`: Relative tolerance for truncation
- `max_bond_dim`: Maximum bond dimension (0 for unlimited)
"""
function contract(
    a::MPOC64,
    b::MPOC64;
    algorithm::Symbol = :naive,
    tolerance::Float64 = 1e-12,
    max_bond_dim::Int = 0,
)
    lib = get_lib()
    alg_cint = algorithm_to_cint(algorithm)
    ptr = ccall(
        Libdl.dlsym(lib, :t4a_mpo_c64_contract),
        Ptr{Cvoid},
        (Ptr{Cvoid}, Ptr{Cvoid}, Cint, Cdouble, Csize_t),
        a.ptr,
        b.ptr,
        alg_cint,
        tolerance,
        max_bond_dim,
    )
    return MPOC64(ptr)
end

# ============================================================================
# Copy
# ============================================================================

function Base.copy(mpo::MPOF64)
    lib = get_lib()
    ptr = ccall(
        Libdl.dlsym(lib, :t4a_mpo_f64_clone),
        Ptr{Cvoid},
        (Ptr{Cvoid},),
        mpo.ptr,
    )
    return MPOF64(ptr)
end

function Base.copy(mpo::MPOC64)
    lib = get_lib()
    ptr = ccall(
        Libdl.dlsym(lib, :t4a_mpo_c64_clone),
        Ptr{Cvoid},
        (Ptr{Cvoid},),
        mpo.ptr,
    )
    return MPOC64(ptr)
end

# ============================================================================
# Exports
# ============================================================================

export MPOF64, MPOC64
export zeros, constant, identity
export rank
export contract_naive, contract_zipup, contract_fit, contract

end # module MPO
