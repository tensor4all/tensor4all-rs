# Python documentation examples: HDF5 I/O (ITensors.jl compatible)
#
# Run with:
#   python docs/examples/python/hdf5.py

from __future__ import annotations

import tempfile

import numpy as np

from tensor4all import Index, Tensor
from tensor4all.hdf5 import save_itensor, load_itensor


# ANCHOR: save_load_tensor
with tempfile.TemporaryDirectory() as tmp:
    path = f"{tmp}/tensor.h5"
    i = Index(2, tags="Site,n=1")
    j = Index(3, tags="Link,l=1")
    t = Tensor([i, j], np.ones((2, 3), dtype=np.float64))

    save_itensor(path, "my_tensor", t)
    t2 = load_itensor(path, "my_tensor")

    assert t2.dims == (2, 3)
    assert t2.indices[0].tags == "Site,n=1"
# ANCHOR_END: save_load_tensor

