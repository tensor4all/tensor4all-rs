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

tt_contracted = tt_a.contract(tt_b, method="zipup", maxdim=8, rtol=1e-12)
assert len(tt_contracted) == len(tt)
assert tt_a.norm() > 0.0
_ = tt_a.inner(tt_a)
_ = tt_a.to_dense()
# ANCHOR_END: operations


# ANCHOR: linsolve
# 1-site example (matches the C-API test structure):
# operator is a 2-index tensor (matrix), rhs/init are 1-index tensors (vectors).
s = Index(2, tags="s")
sp = Index(2, tags="sP")

op_t = Tensor([s, sp], np.eye(2, dtype=np.float64))
op = TensorTrain([op_t])

rhs_t = Tensor([sp], np.array([3.0, 4.0], dtype=np.float64))
rhs = TensorTrain([rhs_t])

init_t = Tensor([s], np.array([1.0, 1.0], dtype=np.float64))
init = TensorTrain([init_t])

sol = tt_linsolve(op, rhs, init, nhalfsweeps=4, maxdim=10, rtol=1e-10, krylov_tol=1e-12, convergence_tol=1e-8)
assert len(sol) == 1
# ANCHOR_END: linsolve
