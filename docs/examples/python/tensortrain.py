# Python documentation examples: TensorTrain (ITensorLike MPS/MPO)
#
# Run with:
#   python docs/examples/python/tensortrain.py

from __future__ import annotations

import numpy as np

from tensor4all import Index, Tensor, TensorTrain, tt_linsolve


# ANCHOR: create
s0 = Index(2, tags="Site,n=1")
s1 = Index(2, tags="Site,n=2")
l01 = Index(3, tags="Link,l=1")

t0 = Tensor([s0, l01], np.ones((2, 3)))
t1 = Tensor([l01, s1], np.ones((3, 2)))
tt = TensorTrain([t0, t1])
assert len(tt) == 2
# ANCHOR_END: create


# ANCHOR: accessors
assert tt.bond_dims == [3]
assert tt.maxbonddim == 3
assert tt.linkind(0) == l01
_ = tt[0]
# ANCHOR_END: accessors


# ANCHOR: orthogonalize
tt.orthogonalize(0)
assert tt.is_ortho is True
assert tt.orthocenter == 0
# ANCHOR_END: orthogonalize


# ANCHOR: truncate
tt.truncate(maxdim=2, rtol=1e-12)
assert tt.maxbonddim <= 2
# ANCHOR_END: truncate


# ANCHOR: operations
tt_a = tt.copy()
tt_b = tt.copy()
tt_sum = tt_a + tt_b
assert len(tt_sum) == len(tt)

# MPO × MPO contraction: only the "inner" indices (s0m/s1m) are shared
s0m = Index(2, tags="Site,n=1,Mid")
s1m = Index(2, tags="Site,n=2,Mid")
la = Index(2, tags="Link,a")
lb = Index(2, tags="Link,b")

mpo_a = TensorTrain([
    Tensor([s0, s0m, la], np.ones((2, 2, 2))),
    Tensor([la, s1, s1m], np.ones((2, 2, 2))),
])
s0out = Index(2, tags="Site,n=1,Out")
s1out = Index(2, tags="Site,n=2,Out")
mpo_b = TensorTrain([
    Tensor([s0m, s0out, lb], np.ones((2, 2, 2))),
    Tensor([lb, s1m, s1out], np.ones((2, 2, 2))),
])
mpo_contracted = mpo_a.contract(
    mpo_b, method="zipup", maxdim=8, rtol=1e-12,
)
assert len(mpo_contracted) == len(mpo_a)

# Verify via dense arrays: contract shared indices with einsum
# mpo_a dense indices: [s0, s0m, s1, s1m]
# mpo_b dense indices: [s0m, s0out, s1m, s1out]
arr_a = mpo_a.to_dense().to_numpy()
arr_b = mpo_b.to_dense().to_numpy()
arr_c = mpo_contracted.to_dense().to_numpy()
expected = np.einsum("ijkl,jmln->imkn", arr_a, arr_b)
assert np.allclose(arr_c, expected)

assert tt_a.norm() > 0.0
_ = tt_a.inner(tt_a)
_ = tt_a.to_dense()
# ANCHOR_END: operations


# ANCHOR: linsolve
# 1-site example (matches the C-API test structure):
# operator: 2-index tensor (matrix)
# rhs/init: 1-index tensors (vectors).
s = Index(2, tags="s")
sp = Index(2, tags="sP")

op_t = Tensor([s, sp], np.eye(2, dtype=np.float64))
op = TensorTrain([op_t])

rhs_t = Tensor([sp], np.array([3.0, 4.0], dtype=np.float64))
rhs = TensorTrain([rhs_t])

init_t = Tensor([s], np.array([1.0, 1.0], dtype=np.float64))
init = TensorTrain([init_t])

sol = tt_linsolve(
    op, rhs, init,
    nhalfsweeps=4, maxdim=10, rtol=1e-10,
    krylov_tol=1e-12, convergence_tol=1e-8,
)
assert len(sol) == 1
# ANCHOR_END: linsolve
