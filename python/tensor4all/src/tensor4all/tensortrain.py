"""TensorTrain (MPS/MPO) - ITensorLike tensor train
with orthogonality tracking.

This module provides a Python interface to the tensor4all-itensorlike
TensorTrain type, which is inspired by ITensorMPS.jl.

Examples
--------
>>> from tensor4all import Index, Tensor, TensorTrain
>>> import numpy as np
>>>
>>> # Create indices
>>> s0 = Index(2)
>>> l01 = Index(3)
>>> s1 = Index(2)
>>>
>>> # Create tensors for a 2-site tensor train
>>> t0 = Tensor([s0, l01], np.ones((2, 3)))
>>> t1 = Tensor([l01, s1], np.ones((3, 2)))
>>> tt = TensorTrain([t0, t1])
>>>
>>> print(len(tt))         # 2
>>> print(tt.bond_dims)    # [3]
>>> print(tt.maxbonddim)   # 3
"""

from __future__ import annotations

from typing import Optional, Sequence

from ._ffi import ffi
from ._capi import get_lib, check_status, T4A_INVALID_ARGUMENT
from .index import Index
from .tensor import Tensor


# Canonical form constants (matching Rust enum)
CANONICAL_UNITARY = 0
CANONICAL_LU = 1
CANONICAL_CI = 2

# Contract method constants
CONTRACT_ZIPUP = 0
CONTRACT_FIT = 1
CONTRACT_NAIVE = 2

_CONTRACT_METHODS = {
    "zipup": CONTRACT_ZIPUP,
    "fit": CONTRACT_FIT,
    "naive": CONTRACT_NAIVE,
}


def _resolve_nhalfsweeps(nsweeps: int, nhalfsweeps: int) -> int:
    """Resolve full/half sweep parameters.

    ITensorMPS.jl convention:
    - nsweeps: number of full sweeps
    - nhalfsweeps: number of half-sweeps (forward or backward)
      where a full sweep = 2 half-sweeps.

    Policy here: explicit `nhalfsweeps` takes precedence over derived `nsweeps`.
    """
    if nhalfsweeps > 0:
        return nhalfsweeps
    if nsweeps > 0:
        return nsweeps * 2
    return 0


class TensorTrain:
    """ITensorLike tensor train (MPS/MPO) with orthogonality tracking.

    This class wraps the Rust TensorTrain from tensor4all-itensorlike,
    which is inspired by ITensorMPS.jl. It supports orthogonality tracking,
    truncation with cutoff/rtol/maxdim, contraction, addition, and linear
    solving.

    Parameters
    ----------
    tensors : list of Tensor
        Tensors forming the tensor train. Adjacent tensors must share
        exactly one common index (the link index).
    """

    __slots__ = ("_ptr",)

    def __init__(self, tensors: Sequence[Tensor]):
        """Create a TensorTrain from a list of Tensor objects."""
        lib = get_lib()
        if len(tensors) == 0:
            ptr = lib.t4a_tt_new_empty()
        else:
            tensor_ptrs = ffi.new("t4a_tensor*[]", [t._ptr for t in tensors])
            ptr = lib.t4a_tt_new(tensor_ptrs, len(tensors))
        if ptr == ffi.NULL:
            raise ValueError(
                "Failed to create TensorTrain: invalid tensor structure"
            )
        self._ptr = ptr

    @classmethod
    def _from_ptr(cls, ptr) -> TensorTrain:
        """Create a TensorTrain from a raw C pointer (internal use)."""
        if ptr == ffi.NULL:
            raise ValueError("Cannot create TensorTrain from null pointer")
        obj = object.__new__(cls)
        obj._ptr = ptr
        return obj

    def __del__(self):
        if hasattr(self, "_ptr") and self._ptr != ffi.NULL:
            get_lib().t4a_tensortrain_release(self._ptr)
            self._ptr = ffi.NULL

    # ========================================================================
    # Accessors
    # ========================================================================

    def __len__(self) -> int:
        """Number of sites in the tensor train."""
        out = ffi.new("size_t*")
        check_status(get_lib().t4a_tt_len(self._ptr, out))
        return out[0]

    @property
    def n_sites(self) -> int:
        """Number of sites in the tensor train."""
        return len(self)

    @property
    def is_empty(self) -> bool:
        """Check if the tensor train is empty."""
        result = get_lib().t4a_tt_is_empty(self._ptr)
        if result < 0:
            raise RuntimeError("Error checking if TensorTrain is empty")
        return result == 1

    def __getitem__(self, site: int) -> Tensor:
        """Get the tensor at a site (0-indexed)."""
        n = len(self)
        if site < 0:
            site += n
        if not (0 <= site < n):
            raise IndexError(f"Site {site} out of bounds [0, {n})")
        ptr = get_lib().t4a_tt_tensor(self._ptr, site)
        if ptr == ffi.NULL:
            raise RuntimeError(f"Failed to get tensor at site {site}")
        return Tensor._from_ptr(ptr)

    def __setitem__(self, site: int, tensor: Tensor):
        """Set the tensor at a site (0-indexed). Invalidates orthogonality."""
        n = len(self)
        if site < 0:
            site += n
        if not (0 <= site < n):
            raise IndexError(f"Site {site} out of bounds [0, {n})")
        check_status(get_lib().t4a_tt_set_tensor(self._ptr, site, tensor._ptr))

    @property
    def bond_dims(self) -> list[int]:
        """Bond dimensions of the tensor train (length = n_sites - 1)."""
        n = len(self)
        if n <= 1:
            return []
        out = ffi.new("size_t[]", n - 1)
        check_status(get_lib().t4a_tt_bond_dims(self._ptr, out, n - 1))
        return [out[i] for i in range(n - 1)]

    @property
    def maxbonddim(self) -> int:
        """Maximum bond dimension across all links."""
        out = ffi.new("size_t*")
        check_status(get_lib().t4a_tt_maxbonddim(self._ptr, out))
        return out[0]

    def linkind(self, site: int) -> Optional[Index]:
        """Get the link index between sites `site` and `site+1` (0-indexed).

        Returns None if no link exists.
        """
        ptr = get_lib().t4a_tt_linkind(self._ptr, site)
        if ptr == ffi.NULL:
            return None
        return Index._from_ptr(ptr)

    # ========================================================================
    # Orthogonality tracking
    # ========================================================================

    @property
    def is_ortho(self) -> bool:
        """Check if the tensor train has a single orthogonality center."""
        result = get_lib().t4a_tt_isortho(self._ptr)
        if result < 0:
            raise RuntimeError("Error checking orthogonality")
        return result == 1

    @property
    def orthocenter(self) -> Optional[int]:
        """Get the orthogonality center (0-indexed).

        Returns None if not orthogonalized.
        """
        out = ffi.new("size_t*")
        status = get_lib().t4a_tt_orthocenter(self._ptr, out)
        if status == T4A_INVALID_ARGUMENT:
            return None
        check_status(status)
        return out[0]

    @property
    def llim(self) -> int:
        """Left orthogonality limit."""
        out = ffi.new("int*")
        check_status(get_lib().t4a_tt_llim(self._ptr, out))
        return out[0]

    @property
    def rlim(self) -> int:
        """Right orthogonality limit."""
        out = ffi.new("int*")
        check_status(get_lib().t4a_tt_rlim(self._ptr, out))
        return out[0]

    @property
    def canonical_form(self) -> Optional[int]:
        """Get the canonical form (0=Unitary, 1=LU, 2=CI).

        Returns None if not set.
        """
        out = ffi.new("int*")
        status = get_lib().t4a_tt_canonical_form(self._ptr, out)
        if status == T4A_INVALID_ARGUMENT:
            return None
        check_status(status)
        return out[0]

    # ========================================================================
    # Operations
    # ========================================================================

    def orthogonalize(self, site: int, *, form: str = "unitary") -> TensorTrain:
        """Orthogonalize the tensor train in-place.

        Parameters
        ----------
        site : int
            Target site for orthogonality center (0-indexed).
        form : str
            Canonical form: "unitary" (default), "lu", or "ci".

        Returns
        -------
        self
        """
        form_map = {
            "unitary": CANONICAL_UNITARY,
            "lu": CANONICAL_LU,
            "ci": CANONICAL_CI,
        }
        form_int = form_map.get(form.lower())
        if form_int is None:
            raise ValueError(
                f"Unknown canonical form: {form}. Use 'unitary', 'lu', or 'ci'"
            )
        check_status(
            get_lib().t4a_tt_orthogonalize_with(self._ptr, site, form_int)
        )
        return self

    def truncate(
        self,
        *,
        rtol: float = 0.0,
        cutoff: float = 0.0,
        maxdim: int = 0,
    ) -> TensorTrain:
        """Truncate bond dimensions in-place.

        Parameters
        ----------
        rtol : float
            Relative tolerance (0.0 = not set).
        cutoff : float
            ITensorMPS.jl cutoff (0.0 = not set). Converted to
            rtol = sqrt(cutoff).
            If both rtol and cutoff are positive, cutoff takes precedence.
        maxdim : int
            Maximum bond dimension (0 = no limit).

        Returns
        -------
        self
        """
        check_status(get_lib().t4a_tt_truncate(self._ptr, rtol, cutoff, maxdim))
        return self

    def norm(self) -> float:
        """Compute the norm of the tensor train."""
        out = ffi.new("double*")
        check_status(get_lib().t4a_tt_norm(self._ptr, out))
        return out[0]

    def inner(self, other: TensorTrain) -> complex:
        """Compute the inner product <self|other>.

        Parameters
        ----------
        other : TensorTrain
            The other tensor train.

        Returns
        -------
        complex
            The inner product.
        """
        out_re = ffi.new("double*")
        out_im = ffi.new("double*")
        check_status(
            get_lib().t4a_tt_inner(self._ptr, other._ptr, out_re, out_im)
        )
        return complex(out_re[0], out_im[0])

    def contract(
        self,
        other: TensorTrain,
        *,
        method: str = "zipup",
        maxdim: int = 0,
        rtol: float = 0.0,
        cutoff: float = 0.0,
        nsweeps: int = 0,
        nhalfsweeps: int = 0,
    ) -> TensorTrain:
        """Contract with another tensor train.

        Parameters
        ----------
        other : TensorTrain
            The other tensor train (must share site indices).
        method : str
            Contraction method: "zipup" (default), "fit", or "naive".
        maxdim : int
            Maximum bond dimension (0 = no limit).
        rtol : float
            Relative tolerance (0.0 = not set).
        cutoff : float
            ITensorMPS.jl cutoff (0.0 = not set).
        nsweeps : int
            Number of full sweeps for fit method (0 = use default).
        nhalfsweeps : int
            Number of half-sweeps (0 = use default). If > 0, overrides nsweeps.

        Returns
        -------
        TensorTrain
            The contraction result.
        """
        method_int = _CONTRACT_METHODS.get(method.lower())
        if method_int is None:
            raise ValueError(
                f"Unknown contract method: {method}. Use 'zipup', 'fit', or 'naive'"
            )
        actual_nhalfsweeps = _resolve_nhalfsweeps(nsweeps, nhalfsweeps)
        ptr = get_lib().t4a_tt_contract(
            self._ptr, other._ptr, method_int,
            maxdim, rtol, cutoff, actual_nhalfsweeps,
        )
        if ptr == ffi.NULL:
            raise RuntimeError("Tensor train contraction failed")
        return TensorTrain._from_ptr(ptr)

    def add(self, other: TensorTrain) -> TensorTrain:
        """Add another tensor train using direct-sum construction.

        The result has bond dimensions that are the sum of the inputs'.

        Parameters
        ----------
        other : TensorTrain
            The other tensor train.

        Returns
        -------
        TensorTrain
            The sum.
        """
        ptr = get_lib().t4a_tt_add(self._ptr, other._ptr)
        if ptr == ffi.NULL:
            raise RuntimeError("Tensor train addition failed")
        return TensorTrain._from_ptr(ptr)

    def __add__(self, other: TensorTrain) -> TensorTrain:
        return self.add(other)

    def to_dense(self) -> Tensor:
        """Convert to a dense tensor by contracting all link indices.

        Returns
        -------
        Tensor
            The dense tensor with only site indices.
        """
        ptr = get_lib().t4a_tt_to_dense(self._ptr)
        if ptr == ffi.NULL:
            raise RuntimeError("to_dense failed")
        return Tensor._from_ptr(ptr)

    def copy(self) -> TensorTrain:
        """Create a deep copy."""
        ptr = get_lib().t4a_tensortrain_clone(self._ptr)
        if ptr == ffi.NULL:
            raise RuntimeError("Failed to clone TensorTrain")
        return TensorTrain._from_ptr(ptr)

    def __repr__(self) -> str:
        n = len(self)
        max_bd = self.maxbonddim if n > 1 else 0
        return f"TensorTrain(sites={n}, maxbonddim={max_bd})"


# Type aliases (ITensorMPS.jl compatibility)
MPS = TensorTrain
MPO = TensorTrain


def linsolve(
    operator: TensorTrain,
    rhs: TensorTrain,
    init: TensorTrain,
    *,
    nsweeps: int = 5,
    nhalfsweeps: int = 0,
    maxdim: int = 0,
    rtol: float = 0.0,
    cutoff: float = 0.0,
    krylov_tol: float = 0.0,
    krylov_maxiter: int = 0,
    krylov_dim: int = 0,
    a0: float = 0.0,
    a1: float = 1.0,
    convergence_tol: float = -1.0,
) -> TensorTrain:
    """Solve (a0 + a1 * A) * x = b for x.

    Uses DMRG-like sweeps with local GMRES.

    Parameters
    ----------
    operator : TensorTrain
        The operator A (MPO).
    rhs : TensorTrain
        The right-hand side b (MPS).
    init : TensorTrain
        Initial guess for x (MPS). Cloned internally.
    nsweeps : int
        Number of full sweeps (default 5). Overridden by nhalfsweeps if > 0.
    nhalfsweeps : int
        Number of half-sweeps (0 = use nsweeps * 2).
    maxdim : int
        Maximum bond dimension (0 = no limit).
    rtol : float
        Relative tolerance (0.0 = not set).
    cutoff : float
        ITensorMPS.jl cutoff (0.0 = not set).
    krylov_tol : float
        GMRES tolerance (0.0 = use default 1e-10).
    krylov_maxiter : int
        Max GMRES iterations (0 = use default 100).
    krylov_dim : int
        Krylov subspace dimension (0 = use default 30).
    a0 : float
        Coefficient a0 in (a0 + a1*A)*x = b.
    a1 : float
        Coefficient a1 in (a0 + a1*A)*x = b.
    convergence_tol : float
        Early termination tolerance (negative = disabled).

    Returns
    -------
    TensorTrain
        The solution x.
    """
    actual_nhalfsweeps = _resolve_nhalfsweeps(nsweeps, nhalfsweeps)
    ptr = get_lib().t4a_tt_linsolve(
        operator._ptr, rhs._ptr, init._ptr,
        actual_nhalfsweeps, maxdim, rtol, cutoff,
        krylov_tol, krylov_maxiter, krylov_dim,
        a0, a1, convergence_tol,
    )
    if ptr == ffi.NULL:
        raise RuntimeError("linsolve failed")
    return TensorTrain._from_ptr(ptr)


def contract(
    tt1: TensorTrain,
    tt2: TensorTrain,
    **kwargs,
) -> TensorTrain:
    """Contract two tensor trains (free function form).

    See TensorTrain.contract() for parameter documentation.
    """
    return tt1.contract(tt2, **kwargs)
