# Julia documentation examples: QuanticsGrids
#
# Run with:
#   julia --project=julia/Tensor4all.jl docs/examples/julia/quanticsgrids.jl

using Tensor4all
using Tensor4all.QuanticsGrids

# ANCHOR: discretized
grid = DiscretizedGrid(2, [3, 3], [0.0, 0.0], [1.0, 1.0])
@assert ndims(grid) == 2
@assert length(local_dims(grid)) > 0
# ANCHOR_END: discretized

# ANCHOR: coord_conversion
# Original coordinates -> quantics indices -> back
coord = [0.25, 0.75]
q = origcoord_to_quantics(grid, coord)
@assert length(q) > 0

coord_back = quantics_to_origcoord(grid, q)
@assert all(isapprox.(coord, coord_back; atol=0.2))
# ANCHOR_END: coord_conversion

# ANCHOR: inherent_discrete
igrid = InherentDiscreteGrid(1, [4], [0])
@assert ndims(igrid) == 1

# Integer coordinates -> quantics -> back
q2 = origcoord_to_quantics(igrid, [2])
coord_int = quantics_to_origcoord(igrid, q2)
@assert coord_int[1] == 2
# ANCHOR_END: inherent_discrete
