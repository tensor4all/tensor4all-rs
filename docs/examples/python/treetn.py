# Python documentation examples: TreeTN (MPS / MPO / TreeTensorNetwork)
#
# Run with:
#   python docs/examples/python/treetn.py

from __future__ import annotations

import numpy as np

from tensor4all import Index, Tensor, TreeTensorNetwork, ttn_inner, ttn_contract, ttn_linsolve


# ANCHOR: create
s0 = Index(2, tags="Site,n=1")
s1 = Index(2, tags="Site,n=2")
l01 = Index(3, tags="Link,l=1")

t0 = Tensor([s0, l01], np.ones((2, 3)))
t1 = Tensor([l01, s1], np.ones((3, 2)))
mps = TreeTensorNetwork([t0, t1])
assert len(mps) == 2
# ANCHOR_END: create


# ANCHOR: accessors
assert mps.bond_dims == [3]
assert mps.maxbonddim == 3
assert mps.linkind(0) == l01
assert mps.num_vertices == 2
assert mps.num_edges == 1
_ = mps[0]
# ANCHOR_END: accessors


# ANCHOR: orthogonalize
mps.orthogonalize(0)
assert mps.canonical_form == 0  # Unitary
# ANCHOR_END: orthogonalize


# ANCHOR: truncate
mps.truncate(maxdim=2, rtol=1e-12)
assert mps.maxbonddim <= 2
# ANCHOR_END: truncate


# ANCHOR: operations
mps_a = mps.copy()
mps_b = mps.copy()
mps_sum = mps_a + mps_b
assert len(mps_sum) == len(mps)

# MPO x MPO contraction: only the "inner" indices (s0m/s1m) are shared
s0m = Index(2, tags="Site,n=1,Mid")
s1m = Index(2, tags="Site,n=2,Mid")
la = Index(2, tags="Link,a")
lb = Index(2, tags="Link,b")

mpo_a = TreeTensorNetwork([
    Tensor([s0, s0m, la], np.ones((2, 2, 2))),
    Tensor([la, s1, s1m], np.ones((2, 2, 2))),
])
s0out = Index(2, tags="Site,n=1,Out")
s1out = Index(2, tags="Site,n=2,Out")
mpo_b = TreeTensorNetwork([
    Tensor([s0m, s0out, lb], np.ones((2, 2, 2))),
    Tensor([lb, s1m, s1out], np.ones((2, 2, 2))),
])
mpo_contracted = ttn_contract(
    mpo_a, mpo_b, method="zipup", maxdim=8, rtol=1e-12,
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

assert mps_a.norm() > 0.0
_ = ttn_inner(mps_a, mps_a)
_ = mps_a.to_dense()
# ANCHOR_END: operations


# ANCHOR: linsolve
# 1-site example (matches the C-API test structure):
# operator: 2-index tensor (matrix)
# rhs/init: 1-index tensors (vectors).
s = Index(2, tags="s")
sp = Index(2, tags="sP")

op_t = Tensor([s, sp], np.eye(2, dtype=np.float64))
op = TreeTensorNetwork([op_t])

rhs_t = Tensor([sp], np.array([3.0, 4.0], dtype=np.float64))
rhs = TreeTensorNetwork([rhs_t])

init_t = Tensor([s], np.array([1.0, 1.0], dtype=np.float64))
init = TreeTensorNetwork([init_t])

sol = ttn_linsolve(
    op, rhs, init,
    nsweeps=4, maxdim=10, rtol=1e-10,
)
assert len(sol) == 1
# ANCHOR_END: linsolve
