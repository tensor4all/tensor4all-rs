"""QuanticsTransform - Quantics transformation operators.

Provides shift, flip, Fourier, phase rotation, and cumulative sum
operators for quantics tensor train representations.
"""

from __future__ import annotations
from enum import IntEnum

from ._ffi import ffi
from ._capi import get_lib, check_status
from .treetn import TreeTensorNetwork


class BoundaryCondition(IntEnum):
    PERIODIC = 0
    OPEN = 1


class LinearOperator:
    """A linear operator (MPO) for quantics transformations."""

    def __init__(self, ptr):
        if ptr == ffi.NULL:
            raise ValueError("Cannot create LinearOperator from null pointer")
        self._ptr = ptr

    def __del__(self):
        if hasattr(self, "_ptr") and self._ptr != ffi.NULL:
            get_lib().t4a_linop_release(self._ptr)
            self._ptr = ffi.NULL

    def apply(self, state: TreeTensorNetwork, *,
              method: str = "naive", rtol: float = 0.0,
              maxdim: int = 0) -> TreeTensorNetwork:
        """Apply operator to a TreeTensorNetwork (MPS).

        Parameters
        ----------
        state : TreeTensorNetwork
            Input MPS/TreeTN.
        method : str
            "naive", "zipup", or "fit".
        rtol : float
            Relative tolerance (0.0 = default).
        maxdim : int
            Maximum bond dimension (0 = unlimited).
        """
        lib = get_lib()
        method_int = {"naive": 0, "zipup": 1, "fit": 2}.get(method)
        if method_int is None:
            raise ValueError(f"Unknown method: {method}")

        out = ffi.new("t4a_treetn**")
        status = lib.t4a_linop_apply(
            self._ptr, state._handle, method_int, rtol, maxdim, out
        )
        check_status(status, "apply")

        # Get number of vertices for node names
        n_out = ffi.new("size_t*")
        status = lib.t4a_treetn_num_vertices(out[0], n_out)
        check_status(status, "num_vertices")
        n = n_out[0]

        return TreeTensorNetwork._from_handle(out[0], list(range(n)))

    def __repr__(self):
        return "LinearOperator()"


def shift_operator(r: int, offset: int, *,
                   bc: BoundaryCondition = BoundaryCondition.PERIODIC) -> LinearOperator:
    """Create shift operator: f(x) = g(x + offset) mod 2^r.

    Parameters
    ----------
    r : int
        Number of quantics bits.
    offset : int
        Shift offset (can be negative).
    bc : BoundaryCondition
        Boundary condition (PERIODIC or OPEN).
    """
    lib = get_lib()
    out = ffi.new("t4a_linop**")
    status = lib.t4a_qtransform_shift(r, offset, int(bc), out)
    check_status(status, "shift_operator")
    return LinearOperator(out[0])


def flip_operator(r: int, *,
                  bc: BoundaryCondition = BoundaryCondition.PERIODIC) -> LinearOperator:
    """Create flip operator: f(x) = g(2^r - x).

    Parameters
    ----------
    r : int
        Number of quantics bits.
    bc : BoundaryCondition
        Boundary condition (PERIODIC or OPEN).
    """
    lib = get_lib()
    out = ffi.new("t4a_linop**")
    status = lib.t4a_qtransform_flip(r, int(bc), out)
    check_status(status, "flip_operator")
    return LinearOperator(out[0])


def phase_rotation_operator(r: int, theta: float) -> LinearOperator:
    """Create phase rotation operator: f(x) = exp(i*theta*x) * g(x).

    Parameters
    ----------
    r : int
        Number of quantics bits.
    theta : float
        Phase angle.
    """
    lib = get_lib()
    out = ffi.new("t4a_linop**")
    status = lib.t4a_qtransform_phase_rotation(r, theta, out)
    check_status(status, "phase_rotation_operator")
    return LinearOperator(out[0])


def cumsum_operator(r: int) -> LinearOperator:
    """Create cumulative sum operator: y_i = sum_{j<i} x_j.

    Parameters
    ----------
    r : int
        Number of quantics bits.
    """
    lib = get_lib()
    out = ffi.new("t4a_linop**")
    status = lib.t4a_qtransform_cumsum(r, out)
    check_status(status, "cumsum_operator")
    return LinearOperator(out[0])


def fourier_operator(r: int, *, forward: bool = True,
                     maxbonddim: int = 0,
                     tolerance: float = 0.0) -> LinearOperator:
    """Create Quantics Fourier Transform operator.

    Parameters
    ----------
    r : int
        Number of quantics bits.
    forward : bool
        True for forward FT, False for inverse.
    maxbonddim : int
        Maximum bond dimension (0 = default=12).
    tolerance : float
        Construction tolerance (0.0 = default=1e-14).
    """
    lib = get_lib()
    out = ffi.new("t4a_linop**")
    status = lib.t4a_qtransform_fourier(r, 1 if forward else 0, maxbonddim, tolerance, out)
    check_status(status, "fourier_operator")
    return LinearOperator(out[0])
