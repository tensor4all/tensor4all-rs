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
from .tensorci import TensorCI2, crossinterpolate2, crossinterpolate2_tci

# TreeTN types (tree tensor network: MPS, MPO, TTN)
from .treetn import TreeTensorNetwork, MPS, MPO
from .treetn import inner as ttn_inner
from .treetn import lognorm as ttn_lognorm
from .treetn import linsolve as ttn_linsolve
from .treetn import contract as ttn_contract

# HDF5 functions (ITensors.jl compatible)
from .hdf5 import save_itensor, load_itensor, save_mps, load_mps

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
    "crossinterpolate2_tci",
    # TreeTN types
    "TreeTensorNetwork",
    "MPS",
    "MPO",
    "ttn_inner",
    "ttn_lognorm",
    "ttn_linsolve",
    "ttn_contract",
    # HDF5 functions
    "save_itensor",
    "load_itensor",
    "save_mps",
    "load_mps",
    # Exceptions
    "T4AError",
    "NullPointerError",
    "InvalidArgumentError",
    "TagOverflowError",
    "TagTooLongError",
    "BufferTooSmallError",
    "InternalError",
]
