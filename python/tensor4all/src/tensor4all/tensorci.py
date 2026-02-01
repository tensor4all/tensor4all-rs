"""TensorCI - Tensor Cross Interpolation algorithms.

This module provides Python bindings for TensorCI2 from tensor4all-tensorci,
implementing tensor cross interpolation for approximating high-dimensional
functions as tensor trains.

Examples
--------
>>> from tensor4all.tensorci import crossinterpolate2
>>> import numpy as np
>>>
>>> # Define a function to interpolate
>>> def f(i, j, k):
...     return float((1 + i) * (1 + j) * (1 + k))
>>>
>>> # Perform cross interpolation
>>> tt, error = crossinterpolate2(f, [2, 2, 2], tolerance=1e-10)
>>> print(tt.n_sites)  # 3
>>> print(tt(0, 0, 0))  # 1.0
>>> print(tt(1, 1, 1))  # 8.0
"""

from __future__ import annotations

from typing import Callable, Sequence, Tuple, Optional

from ._ffi import ffi
from ._capi import get_lib, check_status
from .simplett import SimpleTensorTrain


class TensorCI2:
    """A TCI (Tensor Cross Interpolation) object for 2-site algorithm.

    This wraps the Rust TensorCI2 and provides access to the interpolated
    tensor train representation of a function.

    Currently only supports Float64 values.

    Parameters
    ----------
    _ptr : cdata
        Internal pointer to the Rust object.

    Examples
    --------
    >>> tci = TensorCI2([2, 3, 4])
    >>> print(tci.n_sites)  # 3
    >>> print(tci.rank)  # 0 (empty TCI)
    """

    def __init__(self, local_dims_or_ptr, *, _from_ptr: bool = False):
        """Create a TensorCI2 object.

        Parameters
        ----------
        local_dims_or_ptr : Sequence[int] or cdata
            Either a sequence of local dimensions, or an internal pointer.
        _from_ptr : bool
            If True, treat first argument as a pointer (internal use).
        """
        if _from_ptr:
            if local_dims_or_ptr == ffi.NULL:
                raise ValueError("Cannot create TensorCI2 from null pointer")
            self._ptr = local_dims_or_ptr
            self._local_dims = None  # Will be fetched lazily
        else:
            local_dims = local_dims_or_ptr
            lib = get_lib()
            n_sites = len(local_dims)
            dims_arr = ffi.new("size_t[]", local_dims)
            ptr = lib.t4a_tci2_f64_new(dims_arr, n_sites)
            if ptr == ffi.NULL:
                raise RuntimeError("Failed to create TensorCI2")
            self._ptr = ptr
            self._local_dims = list(local_dims)

    def __del__(self):
        """Release the TCI when garbage collected."""
        if hasattr(self, "_ptr") and self._ptr != ffi.NULL:
            get_lib().t4a_tci2_f64_release(self._ptr)
            self._ptr = ffi.NULL

    @property
    def n_sites(self) -> int:
        """Number of sites."""
        lib = get_lib()
        out_len = ffi.new("size_t*")
        status = lib.t4a_tci2_f64_len(self._ptr, out_len)
        check_status(status, "n_sites")
        return out_len[0]

    def __len__(self) -> int:
        """Number of sites."""
        return self.n_sites

    @property
    def rank(self) -> int:
        """Current maximum bond dimension (rank)."""
        lib = get_lib()
        out_rank = ffi.new("size_t*")
        status = lib.t4a_tci2_f64_rank(self._ptr, out_rank)
        check_status(status, "rank")
        return out_rank[0]

    @property
    def link_dims(self) -> list[int]:
        """List of link (bond) dimensions."""
        lib = get_lib()
        n = self.n_sites
        if n <= 1:
            return []
        n_links = n - 1
        out_dims = ffi.new("size_t[]", n_links)
        status = lib.t4a_tci2_f64_link_dims(self._ptr, out_dims, n_links)
        check_status(status, "link_dims")
        return [out_dims[i] for i in range(n_links)]

    @property
    def max_sample_value(self) -> float:
        """Maximum sample value encountered during interpolation."""
        lib = get_lib()
        out_value = ffi.new("double*")
        status = lib.t4a_tci2_f64_max_sample_value(self._ptr, out_value)
        check_status(status, "max_sample_value")
        return out_value[0]

    @property
    def max_bond_error(self) -> float:
        """Maximum bond error from the last sweep."""
        lib = get_lib()
        out_value = ffi.new("double*")
        status = lib.t4a_tci2_f64_max_bond_error(self._ptr, out_value)
        check_status(status, "max_bond_error")
        return out_value[0]

    def add_global_pivots(self, pivots: Sequence[Sequence[int]]) -> None:
        """Add global pivots to the TCI.

        Parameters
        ----------
        pivots : Sequence[Sequence[int]]
            List of pivots, where each pivot is a list of indices (0-based).
        """
        if not pivots:
            return

        lib = get_lib()
        n_sites = self.n_sites
        n_pivots = len(pivots)

        # Flatten pivots
        flat_pivots = []
        for pivot in pivots:
            if len(pivot) != n_sites:
                raise ValueError(f"Pivot length {len(pivot)} must match n_sites {n_sites}")
            flat_pivots.extend(pivot)

        pivots_arr = ffi.new("size_t[]", flat_pivots)
        status = lib.t4a_tci2_f64_add_global_pivots(self._ptr, pivots_arr, n_pivots, n_sites)
        check_status(status, "add_global_pivots")

    def to_tensor_train(self) -> SimpleTensorTrain:
        """Convert the TCI to a SimpleTensorTrain.

        Returns
        -------
        SimpleTensorTrain
            The tensor train representation.
        """
        lib = get_lib()
        tt_ptr = lib.t4a_tci2_f64_to_tensor_train(self._ptr)
        if tt_ptr == ffi.NULL:
            raise RuntimeError("Failed to convert TCI to TensorTrain")
        return SimpleTensorTrain(tt_ptr)

    def __repr__(self) -> str:
        return f"TensorCI2(n_sites={self.n_sites}, rank={self.rank})"


# Store callback references to prevent garbage collection
_callback_refs = {}


def crossinterpolate2(
    f: Callable[..., float],
    local_dims: Sequence[int],
    *,
    initial_pivots: Optional[Sequence[Sequence[int]]] = None,
    tolerance: float = 1e-8,
    max_bonddim: int = 0,
    max_iter: int = 20,
) -> Tuple[SimpleTensorTrain, float]:
    """Perform cross interpolation of a function to obtain a tensor train approximation.

    Parameters
    ----------
    f : Callable[..., float]
        A function that takes n_sites integer arguments (0-based indices) and returns a float.
    local_dims : Sequence[int]
        List of local dimensions for each site.
    initial_pivots : Sequence[Sequence[int]], optional
        Initial pivots. Default is [[0, 0, ...]].
    tolerance : float, optional
        Relative tolerance for convergence (default: 1e-8).
    max_bonddim : int, optional
        Maximum bond dimension. 0 means unlimited (default: 0).
    max_iter : int, optional
        Maximum number of iterations (default: 20).

    Returns
    -------
    tt : SimpleTensorTrain
        The resulting tensor train approximation.
    final_error : float
        The final error estimate.

    Examples
    --------
    >>> def f(i, j, k):
    ...     return float((1 + i) * (1 + j) * (1 + k))
    >>> tt, err = crossinterpolate2(f, [2, 2, 2], tolerance=1e-10)
    >>> print(tt(0, 0, 0))  # 1.0
    >>> print(tt(1, 1, 1))  # 8.0
    """
    tci, err = crossinterpolate2_tci(
        f,
        local_dims,
        initial_pivots=initial_pivots,
        tolerance=tolerance,
        max_bonddim=max_bonddim,
        max_iter=max_iter,
    )
    return tci.to_tensor_train(), err


def crossinterpolate2_tci(
    f: Callable[..., float],
    local_dims: Sequence[int],
    *,
    initial_pivots: Optional[Sequence[Sequence[int]]] = None,
    tolerance: float = 1e-8,
    max_bonddim: int = 0,
    max_iter: int = 20,
) -> Tuple[TensorCI2, float]:
    """Perform cross interpolation of a function and return the underlying TensorCI2 object.

    This is the low-level form of :func:`crossinterpolate2` that also exposes the
    intermediate `TensorCI2` state.
    """
    lib = get_lib()
    n_sites = len(local_dims)

    if n_sites < 2:
        raise ValueError("local_dims must have at least 2 elements")

    # Prepare local_dims
    dims_arr = ffi.new("size_t[]", local_dims)

    # Prepare initial pivots
    if initial_pivots is None:
        initial_pivots = [[0] * n_sites]

    n_initial_pivots = len(initial_pivots)
    flat_pivots = []
    for pivot in initial_pivots:
        if len(pivot) != n_sites:
            raise ValueError(f"Pivot length {len(pivot)} must match n_sites {n_sites}")
        flat_pivots.extend(pivot)

    if flat_pivots:
        pivots_arr = ffi.new("size_t[]", flat_pivots)
    else:
        pivots_arr = ffi.NULL

    # Create callback wrapper
    # The callback receives indices as int64_t array
    @ffi.callback("int(int64_t*, size_t, double*, void*)")
    def eval_callback(indices_ptr, n_indices, result_ptr, user_data):
        try:
            indices = [indices_ptr[i] for i in range(n_indices)]
            result_ptr[0] = float(f(*indices))
            return 0
        except Exception as e:
            import sys
            print(f"Error in TCI callback: {e}", file=sys.stderr)
            return -1

    callback_id = id(f)
    _callback_refs[callback_id] = eval_callback

    try:
        out_tci = ffi.new("t4a_tci2_f64**")
        out_final_error = ffi.new("double*")

        status = lib.t4a_crossinterpolate2_f64(
            dims_arr,
            n_sites,
            pivots_arr,
            n_initial_pivots,
            eval_callback,
            ffi.NULL,  # user_data
            tolerance,
            max_bonddim,
            max_iter,
            out_tci,
            out_final_error,
        )

        check_status(status, "crossinterpolate2_tci")
        return TensorCI2(out_tci[0], _from_ptr=True), out_final_error[0]
    finally:
        del _callback_refs[callback_id]
