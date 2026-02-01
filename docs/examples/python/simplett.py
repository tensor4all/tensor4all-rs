# Python documentation examples: SimpleTT (SimpleTensorTrain)
#
# Run with:
#   python docs/examples/python/simplett.py

import math

from tensor4all import SimpleTensorTrain


# ANCHOR: create
tt = SimpleTensorTrain.constant([2, 3, 4], 1.0)
zz = SimpleTensorTrain.zeros([2, 3, 4])
assert tt.n_sites == 3
assert zz.sum() == 0.0
# ANCHOR_END: create


# ANCHOR: properties
assert tt.site_dims == [2, 3, 4]
assert len(tt.link_dims) == tt.n_sites - 1
assert tt.rank >= 1
# ANCHOR_END: properties


# ANCHOR: evaluate
assert tt(0, 1, 2) == 1.0
# ANCHOR_END: evaluate


# ANCHOR: operations
assert tt.sum() == 24.0
assert tt.norm() == math.sqrt(24.0)
core0 = tt.site_tensor(0)
assert core0.shape[1] == 2
tt2 = tt.copy()
assert tt2.sum() == tt.sum()
# ANCHOR_END: operations

