# Python documentation examples: algorithm selection + tolerance utilities
#
# Run with:
#   python docs/examples/python/algorithm.py

from tensor4all import (
    FactorizeAlgorithm,
    ContractionAlgorithm,
    CompressionAlgorithm,
    get_default_svd_rtol,
    resolve_truncation_tolerance,
)


# ANCHOR: factorize
assert FactorizeAlgorithm.default() == FactorizeAlgorithm.SVD
assert FactorizeAlgorithm.from_name("lu") == FactorizeAlgorithm.LU
assert FactorizeAlgorithm.SVD.name() == "svd"
# ANCHOR_END: factorize


# ANCHOR: contraction
assert ContractionAlgorithm.default() == ContractionAlgorithm.Naive
assert ContractionAlgorithm.from_name("zipup") == ContractionAlgorithm.ZipUp
assert ContractionAlgorithm.ZipUp.name() == "zipup"
# ANCHOR_END: contraction


# ANCHOR: compression
assert CompressionAlgorithm.default() == CompressionAlgorithm.SVD
assert CompressionAlgorithm.from_name("variational") == CompressionAlgorithm.Variational
assert CompressionAlgorithm.Variational.name() == "variational"
# ANCHOR_END: compression


# ANCHOR: tolerance
rtol0 = get_default_svd_rtol()
assert rtol0 > 0.0
assert resolve_truncation_tolerance(cutoff=1e-24) == 1e-12
assert resolve_truncation_tolerance(rtol=1e-9) == 1e-9
# ANCHOR_END: tolerance

