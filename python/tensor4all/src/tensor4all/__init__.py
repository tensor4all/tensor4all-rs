"""tensor4all - Python bindings for tensor4all Rust library.

This package provides Python bindings to the tensor4all library,
which implements tensor operations for quantum physics applications.

Examples
--------
>>> from tensor4all import Index, Tensor
>>> import numpy as np
>>>
>>> # Create indices
>>> i = Index(2, tags="Site")
>>> j = Index(3, tags="Link")
>>>
>>> # Create a tensor
>>> data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
>>> t = Tensor([i, j], data)
>>>
>>> # Access tensor properties
>>> print(t.rank)  # 2
>>> print(t.dims)  # (2, 3)
>>> print(t.to_numpy())
"""

# Core types (tensor4all-core-common, tensor4all-core-tensor)
from .index import Index
from .tensor import Tensor, StorageKind

# Algorithm types (tensor4all-core-common)
from .algorithm import (
    FactorizeAlgorithm,
    ContractionAlgorithm,
    CompressionAlgorithm,
    get_default_svd_rtol,
    resolve_truncation_tolerance,
)

# SimpleTT types (tensor4all-simplett)
from .simplett import SimpleTensorTrain

# TensorCI types (tensor4all-tensorci)
from .tensorci import TensorCI2, crossinterpolate2

# ITensorLike types (tensor4all-itensorlike)
from .tensortrain import TensorTrain, MPS, MPO
from .tensortrain import linsolve as tt_linsolve
from .tensortrain import contract as tt_contract

# HDF5 functions (ITensors.jl compatible)
from .hdf5 import save_itensor, load_itensor

# Exceptions
from ._capi import (
    T4AError,
    NullPointerError,
    InvalidArgumentError,
    TagOverflowError,
    TagTooLongError,
    BufferTooSmallError,
    InternalError,
)

__version__ = "0.1.0"

__all__ = [
    # Core types
    "Index",
    "Tensor",
    "StorageKind",
    # Algorithm types
    "FactorizeAlgorithm",
    "ContractionAlgorithm",
    "CompressionAlgorithm",
    "get_default_svd_rtol",
    "resolve_truncation_tolerance",
    # SimpleTT types
    "SimpleTensorTrain",
    # TensorCI types
    "TensorCI2",
    "crossinterpolate2",
    # ITensorLike types
    "TensorTrain",
    "MPS",
    "MPO",
    "tt_linsolve",
    "tt_contract",
    # HDF5 functions
    "save_itensor",
    "load_itensor",
    # Exceptions
    "T4AError",
    "NullPointerError",
    "InvalidArgumentError",
    "TagOverflowError",
    "TagTooLongError",
    "BufferTooSmallError",
    "InternalError",
]
