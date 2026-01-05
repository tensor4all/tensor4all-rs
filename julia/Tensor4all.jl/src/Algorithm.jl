"""
    Algorithm

Submodule for algorithm selection types (corresponds to tensor4all-core-common).

Provides enum-like types for selecting algorithms in various tensor operations,
following patterns from ITensors.jl.

# Basic Usage

```julia
using Tensor4all.Algorithm

# Factorization algorithms
alg = FactorizeAlgorithm.SVD
alg = FactorizeAlgorithm.LU

# Contraction algorithms
alg = ContractionAlgorithm.Naive
alg = ContractionAlgorithm.ZipUp

# Compression algorithms
alg = CompressionAlgorithm.SVD
alg = CompressionAlgorithm.Variational
```
"""
module Algorithm

using Libdl
import ..C_API

# ============================================================================
# Factorize Algorithm
# ============================================================================

"""
    FactorizeAlgorithm

Algorithm for matrix factorization / decomposition.

Used in compression, truncation, and various tensor operations.

# Values
- `SVD`: Singular Value Decomposition (default, optimal truncation)
- `LU`: LU decomposition with partial pivoting (faster)
- `CI`: Cross Interpolation / Skeleton decomposition (adaptive)
"""
module FactorizeAlgorithm
    const SVD = Int32(0)
    const LU = Int32(1)
    const CI = Int32(2)

    """Default algorithm."""
    default() = SVD

    """Get algorithm name as string."""
    function name(alg::Int32)
        if alg == SVD
            return "svd"
        elseif alg == LU
            return "lu"
        elseif alg == CI
            return "ci"
        else
            error("Unknown FactorizeAlgorithm: $alg")
        end
    end

    """Parse algorithm from string."""
    function from_name(s::AbstractString)
        lower = lowercase(s)
        if lower == "svd"
            return SVD
        elseif lower == "lu"
            return LU
        elseif lower in ("ci", "cross", "crossinterpolation")
            return CI
        else
            error("Unknown FactorizeAlgorithm name: $s")
        end
    end
end

# ============================================================================
# Contraction Algorithm
# ============================================================================

"""
    ContractionAlgorithm

Algorithm for tensor train contraction (TT-TT or MPO-MPO).

# Values
- `Naive`: Exact contraction followed by compression (default)
- `ZipUp`: On-the-fly compression during contraction (memory efficient)
- `Fit`: Variational fitting (best for low target rank)
"""
module ContractionAlgorithm
    const Naive = Int32(0)
    const ZipUp = Int32(1)
    const Fit = Int32(2)

    """Default algorithm."""
    default() = Naive

    """Get algorithm name as string."""
    function name(alg::Int32)
        if alg == Naive
            return "naive"
        elseif alg == ZipUp
            return "zipup"
        elseif alg == Fit
            return "fit"
        else
            error("Unknown ContractionAlgorithm: $alg")
        end
    end

    """Parse algorithm from string."""
    function from_name(s::AbstractString)
        lower = lowercase(s)
        if lower == "naive"
            return Naive
        elseif lower in ("zipup", "zip_up", "zip-up")
            return ZipUp
        elseif lower in ("fit", "variational")
            return Fit
        else
            error("Unknown ContractionAlgorithm name: $s")
        end
    end
end

# ============================================================================
# Compression Algorithm
# ============================================================================

"""
    CompressionAlgorithm

Algorithm for tensor train compression.

# Values
- `SVD`: SVD-based compression (default, optimal)
- `LU`: LU-based compression (faster)
- `CI`: Cross Interpolation based compression
- `Variational`: Variational compression (sweeping optimization)
"""
module CompressionAlgorithm
    const SVD = Int32(0)
    const LU = Int32(1)
    const CI = Int32(2)
    const Variational = Int32(3)

    """Default algorithm."""
    default() = SVD

    """Get algorithm name as string."""
    function name(alg::Int32)
        if alg == SVD
            return "svd"
        elseif alg == LU
            return "lu"
        elseif alg == CI
            return "ci"
        elseif alg == Variational
            return "variational"
        else
            error("Unknown CompressionAlgorithm: $alg")
        end
    end

    """Parse algorithm from string."""
    function from_name(s::AbstractString)
        lower = lowercase(s)
        if lower == "svd"
            return SVD
        elseif lower == "lu"
            return LU
        elseif lower in ("ci", "cross", "crossinterpolation")
            return CI
        elseif lower in ("variational", "fit")
            return Variational
        else
            error("Unknown CompressionAlgorithm name: $s")
        end
    end
end

# ============================================================================
# Truncation tolerance utilities
# ============================================================================

"""
    get_default_svd_rtol()

Get the default SVD relative tolerance from the Rust library.

This is the default value used for truncation when no tolerance is specified.
"""
function get_default_svd_rtol()
    lib = C_API.libhandle()
    return ccall(
        Libdl.dlsym(lib, :t4a_get_default_svd_rtol),
        Cdouble,
        (),
    )
end

"""
    resolve_truncation_tolerance(; cutoff=nothing, rtol=nothing)

Resolve truncation tolerance from `cutoff` (ITensor style) or `rtol` (tensor4all style).

# Arguments
- `cutoff`: ITensor-style squared error tolerance: Σ_{discarded} σ²ᵢ / Σᵢ σ²ᵢ ≤ cutoff
- `rtol`: tensor4all-style relative Frobenius error: ‖A - A_approx‖_F / ‖A‖_F ≤ rtol

# Returns
The effective `rtol` value.

# Conversion
`cutoff = rtol²`, so `rtol = √cutoff`

# Behavior
- If neither specified: returns the default rtol from Rust
- If only `cutoff` specified: returns `√cutoff`
- If only `rtol` specified: returns `rtol`
- If both specified: throws an error

# Examples
```julia
# Use default
rtol = resolve_truncation_tolerance()

# ITensor style
rtol = resolve_truncation_tolerance(cutoff=1e-24)  # returns 1e-12

# tensor4all style
rtol = resolve_truncation_tolerance(rtol=1e-12)
```
"""
function resolve_truncation_tolerance(; cutoff = nothing, rtol = nothing)
    if cutoff !== nothing && rtol !== nothing
        throw(ArgumentError(
            "Cannot specify both `cutoff` and `rtol`. Use one or the other.\n" *
            "- cutoff (ITensor style): Σ_{discarded} σ²ᵢ / Σᵢ σ²ᵢ ≤ cutoff\n" *
            "- rtol (tensor4all style): ‖A - A_approx‖_F / ‖A‖_F ≤ rtol\n" *
            "Conversion: cutoff = rtol², so rtol = √cutoff"
        ))
    end

    if cutoff !== nothing
        return sqrt(cutoff)
    elseif rtol !== nothing
        return rtol
    else
        return get_default_svd_rtol()
    end
end

# ============================================================================
# Exports
# ============================================================================

export FactorizeAlgorithm, ContractionAlgorithm, CompressionAlgorithm
export get_default_svd_rtol, resolve_truncation_tolerance

end # module Algorithm
