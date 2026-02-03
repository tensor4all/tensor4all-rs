# Python documentation examples: QuanticsGrids
#
# Run with:
#   python docs/examples/python/quanticsgrids.py

from tensor4all import DiscretizedGrid, InherentDiscreteGrid


# ANCHOR: discretized
grid = DiscretizedGrid(ndims=2, R=[3, 3], lower=[0.0, 0.0], upper=[1.0, 1.0])
assert grid.ndims == 2
assert len(grid.local_dimensions) > 0
# ANCHOR_END: discretized


# ANCHOR: coord_conversion
# Original coordinates -> quantics indices -> back
coord = [0.25, 0.75]
q = grid.origcoord_to_quantics(coord)
assert len(q) > 0

coord_back = grid.quantics_to_origcoord(q)
assert all(abs(a - b) < 0.2 for a, b in zip(coord, coord_back))
# ANCHOR_END: coord_conversion


# ANCHOR: inherent_discrete
igrid = InherentDiscreteGrid(ndims=1, R=[4], origin=[0])
assert igrid.ndims == 1

# Integer coordinates -> quantics -> back
q2 = igrid.origcoord_to_quantics([2])
coord_int = igrid.quantics_to_origcoord(q2)
assert coord_int[0] == 2
# ANCHOR_END: inherent_discrete
