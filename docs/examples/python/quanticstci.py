# Python documentation examples: QuanticsTCI (quantics tensor cross interpolation)
#
# Run with:
#   python docs/examples/python/quanticstci.py

from tensor4all import (
    QuanticsTensorCI2,
    quanticscrossinterpolate,
    quanticscrossinterpolate_discrete,
    DiscretizedGrid,
)


# ANCHOR: discrete
# Interpolate a simple function on a 2D discrete grid
# Grid indices are 1-indexed: i ∈ {1, ..., 4}, j ∈ {1, ..., 4}
def f(i, j):
    return float(i + j)


qtci = quanticscrossinterpolate_discrete([4, 4], f, tolerance=1e-10, unfoldingscheme="fused")
assert qtci.rank > 0
# ANCHOR_END: discrete


# ANCHOR: evaluate
# Evaluate at grid indices (1-indexed)
val = qtci(3, 4)
assert abs(val - 7.0) < 1e-6
# ANCHOR_END: evaluate


# ANCHOR: sum
s = qtci.sum()
assert s > 0
# ANCHOR_END: sum


# ANCHOR: to_tt
tt = qtci.to_tensor_train()
assert tt.n_sites > 0
# ANCHOR_END: to_tt


# ANCHOR: continuous
# Interpolate on a continuous domain using a DiscretizedGrid
grid = DiscretizedGrid(ndims=1, R=[4], lower=[0.0], upper=[1.0])


def g(x):
    return x**2


qtci2 = quanticscrossinterpolate(grid, g, tolerance=1e-8)
assert qtci2.rank > 0

# Integral of x^2 from 0 to 1 should be ≈ 1/3
integ = qtci2.integral()
assert abs(integ - 1 / 3) < 0.1  # discretization introduces some error
# ANCHOR_END: continuous
