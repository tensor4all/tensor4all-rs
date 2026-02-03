"""QuanticsTCI - Quantics Tensor Cross Interpolation.

This module provides Python bindings for QuanticsTCI from tensor4all-quanticstci,
combining TCI with quantics grid representations for efficient function interpolation.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

from ._ffi import ffi
from ._capi import get_lib, check_status
from .simplett import SimpleTensorTrain
from .quanticsgrids import DiscretizedGrid

# Store callback references to prevent garbage collection
_callback_refs = {}


class QuanticsTensorCI2:
    """Quantics Tensor Cross Interpolation result.

    Wraps the Rust QuanticsTensorCI2<f64> type.
    """

    def __init__(self, ptr, *, _from_ptr: bool = True):
        if ptr == ffi.NULL:
            raise ValueError("Cannot create QuanticsTensorCI2 from null pointer")
        self._ptr = ptr

    def __del__(self):
        if hasattr(self, "_ptr") and self._ptr != ffi.NULL:
            get_lib().t4a_qtci_f64_release(self._ptr)
            self._ptr = ffi.NULL

    @property
    def rank(self) -> int:
        lib = get_lib()
        out = ffi.new("size_t*")
        status = lib.t4a_qtci_f64_rank(self._ptr, out)
        check_status(status, "rank")
        return out[0]

    @property
    def link_dims(self) -> list[int]:
        lib = get_lib()
        buf = ffi.new("size_t[1024]")
        status = lib.t4a_qtci_f64_link_dims(self._ptr, buf, 1024)
        check_status(status, "link_dims")
        result = [buf[i] for i in range(1024)]
        # Find actual length
        try:
            last = max(i for i, v in enumerate(result) if v > 0)
            return result[:last + 1]
        except ValueError:
            return []

    def evaluate(self, *indices: int) -> float:
        """Evaluate at grid indices (1-indexed for discrete, grid index for continuous)."""
        lib = get_lib()
        idx = ffi.new("int64_t[]", list(indices))
        out = ffi.new("double*")
        status = lib.t4a_qtci_f64_evaluate(self._ptr, idx, len(indices), out)
        check_status(status, "evaluate")
        return out[0]

    def __call__(self, *indices: int) -> float:
        return self.evaluate(*indices)

    def sum(self) -> float:
        """Compute factorized sum over all grid points."""
        lib = get_lib()
        out = ffi.new("double*")
        status = lib.t4a_qtci_f64_sum(self._ptr, out)
        check_status(status, "sum")
        return out[0]

    def integral(self) -> float:
        """Compute integral over continuous domain."""
        lib = get_lib()
        out = ffi.new("double*")
        status = lib.t4a_qtci_f64_integral(self._ptr, out)
        check_status(status, "integral")
        return out[0]

    def to_tensor_train(self) -> SimpleTensorTrain:
        lib = get_lib()
        ptr = lib.t4a_qtci_f64_to_tensor_train(self._ptr)
        if ptr == ffi.NULL:
            raise RuntimeError("Failed to convert QTCI to TensorTrain")
        return SimpleTensorTrain(ptr)

    def __repr__(self) -> str:
        return f"QuanticsTensorCI2(rank={self.rank})"


def quanticscrossinterpolate(
    grid: DiscretizedGrid,
    f: Callable[..., float],
    *,
    tolerance: float = 1e-8,
    max_bonddim: int = 0,
    max_iter: int = 200,
) -> QuanticsTensorCI2:
    """Perform quantics cross interpolation on a continuous domain.

    Parameters
    ----------
    grid : DiscretizedGrid
        Grid describing the function domain.
    f : Callable
        Function taking float coordinates, returns float.
    tolerance : float
        Convergence tolerance.
    max_bonddim : int
        Maximum bond dimension (0 = unlimited).
    max_iter : int
        Maximum iterations.
    """
    lib = get_lib()

    @ffi.callback("int(double*, size_t, double*, void*)")
    def eval_callback(coords_ptr, ndims, result_ptr, user_data):
        try:
            coords = [coords_ptr[i] for i in range(ndims)]
            result_ptr[0] = float(f(*coords))
            return 0
        except Exception as e:
            import sys
            print(f"Error in QTCI callback: {e}", file=sys.stderr)
            return -1

    callback_id = id(f)
    _callback_refs[callback_id] = eval_callback

    try:
        out_qtci = ffi.new("t4a_qtci_f64**")
        status = lib.t4a_quanticscrossinterpolate_f64(
            grid._ptr,
            eval_callback,
            ffi.NULL,
            tolerance,
            max_bonddim,
            max_iter,
            out_qtci,
        )
        check_status(status, "quanticscrossinterpolate")
        return QuanticsTensorCI2(out_qtci[0])
    finally:
        del _callback_refs[callback_id]


def quanticscrossinterpolate_discrete(
    sizes: Sequence[int],
    f: Callable[..., float],
    *,
    tolerance: float = 1e-8,
    max_bonddim: int = 0,
    max_iter: int = 200,
    unfoldingscheme: str = "interleaved",
) -> QuanticsTensorCI2:
    """Perform quantics cross interpolation on a discrete integer domain.

    Parameters
    ----------
    sizes : Sequence[int]
        Grid sizes per dimension (must be powers of 2).
    f : Callable
        Function taking 1-indexed integer indices, returns float.
    tolerance : float
        Convergence tolerance.
    max_bonddim : int
        Maximum bond dimension (0 = unlimited).
    max_iter : int
        Maximum iterations.
    unfoldingscheme : str
        "interleaved" or "fused".
    """
    lib = get_lib()
    scheme = 0 if unfoldingscheme == "fused" else 1

    @ffi.callback("int(int64_t*, size_t, double*, void*)")
    def eval_callback(indices_ptr, ndims, result_ptr, user_data):
        try:
            indices = [indices_ptr[i] for i in range(ndims)]
            result_ptr[0] = float(f(*indices))
            return 0
        except Exception as e:
            import sys
            print(f"Error in QTCI callback: {e}", file=sys.stderr)
            return -1

    callback_id = id(f)
    _callback_refs[callback_id] = eval_callback

    try:
        sizes_arr = ffi.new("size_t[]", list(sizes))
        out_qtci = ffi.new("t4a_qtci_f64**")
        status = lib.t4a_quanticscrossinterpolate_discrete_f64(
            sizes_arr,
            len(sizes),
            eval_callback,
            ffi.NULL,
            tolerance,
            max_bonddim,
            max_iter,
            scheme,
            out_qtci,
        )
        check_status(status, "quanticscrossinterpolate_discrete")
        return QuanticsTensorCI2(out_qtci[0])
    finally:
        del _callback_refs[callback_id]
