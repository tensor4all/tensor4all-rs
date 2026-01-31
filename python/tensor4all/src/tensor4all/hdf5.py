"""HDF5 save/load functions for tensor4all (ITensors.jl compatible format).

Provides functions to save and load tensors in HDF5 format that is
compatible with ITensors.jl and ITensorMPS.jl.

Examples
--------
>>> from tensor4all import Index, Tensor
>>> from tensor4all.hdf5 import save_itensor, load_itensor
>>> import numpy as np
>>>
>>> i = Index(2, tags="Site,n=1")
>>> j = Index(3, tags="Link,l=1")
>>> t = Tensor([i, j], np.ones((2, 3)))
>>>
>>> save_itensor("tensor.h5", "my_tensor", t)
>>> loaded = load_itensor("tensor.h5", "my_tensor")
"""

from ._ffi import ffi
from ._capi import get_lib, check_status
from .tensor import Tensor


def save_itensor(filepath: str, name: str, tensor: Tensor) -> None:
    """Save a tensor to an HDF5 file in ITensors.jl-compatible format.

    Parameters
    ----------
    filepath : str
        Path to the HDF5 file (will be created/overwritten).
    name : str
        Name of the HDF5 group to write the tensor to.
    tensor : Tensor
        Tensor to save.
    """
    lib = get_lib()
    status = lib.t4a_hdf5_save_itensor(
        filepath.encode("utf-8"),
        name.encode("utf-8"),
        tensor._ptr,
    )
    check_status(status, "save_itensor")


def load_itensor(filepath: str, name: str) -> Tensor:
    """Load a tensor from an HDF5 file in ITensors.jl-compatible format.

    Parameters
    ----------
    filepath : str
        Path to the HDF5 file.
    name : str
        Name of the HDF5 group containing the tensor.

    Returns
    -------
    Tensor
        The loaded tensor.
    """
    lib = get_lib()
    out = ffi.new("t4a_tensor**")
    status = lib.t4a_hdf5_load_itensor(
        filepath.encode("utf-8"),
        name.encode("utf-8"),
        out,
    )
    check_status(status, "load_itensor")
    return Tensor._from_ptr(out[0])
