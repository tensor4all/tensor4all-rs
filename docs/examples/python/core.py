# Python documentation examples: core types (Index, Tensor)
#
# Run with:
#   python docs/examples/python/core.py

from __future__ import annotations

import numpy as np

from tensor4all import Index, Tensor, StorageKind
from tensor4all.index import (
    sim,
    hascommoninds,
    commoninds,
    uniqueinds,
    replaceinds,
)


# ANCHOR: index_basic
i = Index(2, tags="Site,n=1")
assert i.dim == 2
assert i.tags == "Site,n=1"
assert isinstance(i.id, int)
assert i.has_tag("Site")
# ANCHOR_END: index_basic


# ANCHOR: index_utils
j = Index(3, tags="Link,l=1")
k = Index(2, tags="Site,n=2")

j_sim = sim(j)
assert j_sim.dim == j.dim
assert j_sim.tags == j.tags
assert j_sim.id != j.id

assert hascommoninds([i, j], [j, k])
assert commoninds([i, j], [j, k]) == [j]
assert uniqueinds([i, j], [j, k]) == [i]

new_j = sim(j)
assert replaceinds([i, j, k], [j], [new_j]) == [i, new_j, k]
# ANCHOR_END: index_utils


# ANCHOR: tensor_basic
data = np.arange(6, dtype=np.float64).reshape(2, 3)
t = Tensor([i, j], data)
assert t.rank == 2
assert t.dims == (2, 3)
assert t.storage_kind == StorageKind.DenseF64
# ANCHOR_END: tensor_basic


# ANCHOR: tensor_onehot
oh = Tensor.onehot((i, 1), (j, 2))  # 0-based positions
oh_arr = oh.to_numpy()
assert oh_arr[1, 2] == 1.0
assert float(np.sum(oh_arr)) == 1.0
# ANCHOR_END: tensor_onehot


# ANCHOR: tensor_numpy
arr = t.to_numpy()
assert arr.shape == (2, 3)
assert arr[0, 1] == data[0, 1]
# ANCHOR_END: tensor_numpy

