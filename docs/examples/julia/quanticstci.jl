# Julia documentation examples: QuanticsTCI (quantics tensor cross interpolation)
#
# Run with:
#   julia --project=julia/Tensor4all.jl docs/examples/julia/quanticstci.jl

using Tensor4all
using Tensor4all.QuanticsTCI
using Tensor4all.SimpleTT: evaluate

# ANCHOR: discrete
# Interpolate a simple function on a 2D discrete grid
# Grid indices are 1-indexed: i ∈ {1, ..., 4}, j ∈ {1, ..., 4}
f(i, j) = Float64(i + j)
qtci = quanticscrossinterpolate_discrete([4, 4], f; tolerance=1e-10, unfoldingscheme=:fused)
@assert QuanticsTCI.rank(qtci) > 0
# ANCHOR_END: discrete

# ANCHOR: evaluate
# Evaluate at grid indices (1-indexed)
val = qtci(3, 4)
@assert abs(val - 7.0) < 1e-6
# ANCHOR_END: evaluate

# ANCHOR: sum
s = QuanticsTCI.sum(qtci)
@assert isfinite(s)
@assert s > 0
# ANCHOR_END: sum

# ANCHOR: to_tt
tt = to_tensor_train(qtci)
@assert length(tt) > 0
# ANCHOR_END: to_tt

# ANCHOR: continuous
# Interpolate on a continuous domain using a DiscretizedGrid
using Tensor4all.QuanticsGrids
grid = DiscretizedGrid(1, [4], [0.0], [1.0])
g(x) = x^2
qtci2 = quanticscrossinterpolate(grid, g; tolerance=1e-8)
@assert QuanticsTCI.rank(qtci2) > 0

# Integral of x^2 from 0 to 1 should be ≈ 1/3
integ = integral(qtci2)
@assert abs(integ - 1/3) < 0.1  # discretization introduces some error
# ANCHOR_END: continuous
