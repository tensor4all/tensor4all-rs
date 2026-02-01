# Python documentation examples: TensorCI (cross interpolation)
#
# Run with:
#   python docs/examples/python/tensorci.py

from tensor4all import TensorCI2, crossinterpolate2, crossinterpolate2_tci


# ANCHOR: basic
def f(i: int, j: int, k: int) -> float:
    return float((1 + i) * (1 + j) * (1 + k))


tt, err = crossinterpolate2(f, [2, 2, 2], tolerance=1e-12)
assert tt.n_sites == 3
assert err >= 0.0
# ANCHOR_END: basic


# ANCHOR: evaluate
assert tt(0, 0, 0) == 1.0
assert tt(1, 1, 1) == 8.0
assert tt(1, 0, 1) == f(1, 0, 1)
# ANCHOR_END: evaluate


# ANCHOR: manual
tci, err2 = crossinterpolate2_tci(f, [2, 2, 2], tolerance=1e-12)
assert tci.n_sites == 3
assert err2 >= 0.0
tt2 = tci.to_tensor_train()
assert tt2.n_sites == 3
# ANCHOR_END: manual
