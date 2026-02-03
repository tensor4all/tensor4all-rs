# Julia documentation examples: algorithm selection + tolerance utilities
#
# Run with:
#   julia --project=julia/Tensor4all.jl docs/examples/julia/algorithm.jl

using Tensor4all
using Tensor4all.Algorithm

# ANCHOR: factorize
@assert FactorizeAlgorithm.default() == FactorizeAlgorithm.SVD
@assert FactorizeAlgorithm.name(FactorizeAlgorithm.SVD) == "svd"
@assert FactorizeAlgorithm.from_name("LU") == FactorizeAlgorithm.LU
# ANCHOR_END: factorize

# ANCHOR: contraction
@assert ContractionAlgorithm.default() == ContractionAlgorithm.Naive
@assert ContractionAlgorithm.name(ContractionAlgorithm.ZipUp) == "zipup"
@assert ContractionAlgorithm.from_name("fit") == ContractionAlgorithm.Fit
# ANCHOR_END: contraction

# ANCHOR: compression
@assert CompressionAlgorithm.default() == CompressionAlgorithm.SVD
@assert CompressionAlgorithm.name(CompressionAlgorithm.Variational) == "variational"
@assert CompressionAlgorithm.from_name("ci") == CompressionAlgorithm.CI
# ANCHOR_END: compression

# ANCHOR: tolerance
rtol0 = get_default_svd_rtol()
@assert rtol0 > 0
@assert isapprox(resolve_truncation_tolerance(cutoff=1e-24), 1e-12; rtol=0, atol=0)
@assert isapprox(resolve_truncation_tolerance(rtol=1e-9), 1e-9; rtol=0, atol=0)
# ANCHOR_END: tolerance

