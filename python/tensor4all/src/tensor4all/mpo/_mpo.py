"""MPO (Matrix Product Operator) implementations."""

from __future__ import annotations

from enum import IntEnum
from typing import Literal, Sequence

from .._capi import get_lib, check_status


class ContractionAlgorithm(IntEnum):
    """Contraction algorithm for MPO operations."""

    NAIVE = 0
    ZIPUP = 1
    FIT = 2


def _algorithm_to_int(algorithm: str | ContractionAlgorithm) -> int:
    """Convert algorithm specification to integer."""
    if isinstance(algorithm, ContractionAlgorithm):
        return int(algorithm)
    if isinstance(algorithm, str):
        alg_map = {
            "naive": ContractionAlgorithm.NAIVE,
            "zipup": ContractionAlgorithm.ZIPUP,
            "fit": ContractionAlgorithm.FIT,
        }
        alg_lower = algorithm.lower()
        if alg_lower not in alg_map:
            raise ValueError(
                f"Unknown algorithm '{algorithm}'. "
                "Use 'naive', 'zipup', or 'fit'."
            )
        return int(alg_map[alg_lower])
    raise TypeError(f"Expected str or ContractionAlgorithm, got {type(algorithm)}")


class MPOF64:
    """Matrix Product Operator with float64 elements.

    An MPO represents a tensor with a Matrix Product structure,
    commonly used in quantum physics and tensor network computations.

    Examples
    --------
    >>> mpo = MPOF64.identity([2, 2])
    >>> len(mpo)
    2
    """

    __slots__ = ("_ptr", "_lib", "_ffi")

    def __init__(self, ptr, lib, ffi):
        """Initialize from a C pointer (internal use only)."""
        if ptr == ffi.NULL:
            raise ValueError("Failed to create MPOF64 (null pointer)")
        self._ptr = ptr
        self._lib = lib
        self._ffi = ffi

    def __del__(self):
        """Release the underlying C object."""
        if hasattr(self, "_ptr") and self._ptr != self._ffi.NULL:
            self._lib.t4a_mpo_f64_release(self._ptr)

    @classmethod
    def zeros(cls, site_dims: Sequence[tuple[int, int]]) -> MPOF64:
        """Create an MPO representing the zero operator.

        Parameters
        ----------
        site_dims : Sequence[tuple[int, int]]
            Site dimensions as (input_dim, output_dim) pairs.

        Returns
        -------
        MPOF64
            Zero MPO.
        """
        lib, ffi = get_lib()
        dims1 = ffi.new("size_t[]", [d[0] for d in site_dims])
        dims2 = ffi.new("size_t[]", [d[1] for d in site_dims])
        ptr = lib.t4a_mpo_f64_new_zeros(dims1, dims2, len(site_dims))
        return cls(ptr, lib, ffi)

    @classmethod
    def constant(
        cls, site_dims: Sequence[tuple[int, int]], value: float
    ) -> MPOF64:
        """Create an MPO representing a constant operator.

        Parameters
        ----------
        site_dims : Sequence[tuple[int, int]]
            Site dimensions as (input_dim, output_dim) pairs.
        value : float
            Constant value.

        Returns
        -------
        MPOF64
            Constant MPO.
        """
        lib, ffi = get_lib()
        dims1 = ffi.new("size_t[]", [d[0] for d in site_dims])
        dims2 = ffi.new("size_t[]", [d[1] for d in site_dims])
        ptr = lib.t4a_mpo_f64_new_constant(dims1, dims2, len(site_dims), value)
        return cls(ptr, lib, ffi)

    @classmethod
    def identity(cls, site_dims: Sequence[int]) -> MPOF64:
        """Create an identity MPO.

        Parameters
        ----------
        site_dims : Sequence[int]
            Site dimensions (same for input and output).

        Returns
        -------
        MPOF64
            Identity MPO.
        """
        lib, ffi = get_lib()
        dims = ffi.new("size_t[]", list(site_dims))
        ptr = lib.t4a_mpo_f64_new_identity(dims, len(site_dims))
        return cls(ptr, lib, ffi)

    def __len__(self) -> int:
        """Return the number of sites."""
        out_len = self._ffi.new("size_t*")
        status = self._lib.t4a_mpo_f64_len(self._ptr, out_len)
        check_status(status)
        return out_len[0]

    def rank(self) -> int:
        """Return the maximum bond dimension."""
        out_rank = self._ffi.new("size_t*")
        status = self._lib.t4a_mpo_f64_rank(self._ptr, out_rank)
        check_status(status)
        return out_rank[0]

    def copy(self) -> MPOF64:
        """Create a copy of this MPO."""
        ptr = self._lib.t4a_mpo_f64_clone(self._ptr)
        return MPOF64(ptr, self._lib, self._ffi)

    def contract_naive(self, other: MPOF64) -> MPOF64:
        """Contract with another MPO using naive algorithm.

        Parameters
        ----------
        other : MPOF64
            MPO to contract with.

        Returns
        -------
        MPOF64
            Contracted MPO.
        """
        ptr = self._lib.t4a_mpo_f64_contract_naive(self._ptr, other._ptr)
        return MPOF64(ptr, self._lib, self._ffi)

    def contract_zipup(
        self,
        other: MPOF64,
        *,
        tolerance: float = 1e-12,
        max_bond_dim: int = 0,
    ) -> MPOF64:
        """Contract with another MPO using zip-up algorithm.

        Parameters
        ----------
        other : MPOF64
            MPO to contract with.
        tolerance : float, optional
            Truncation tolerance.
        max_bond_dim : int, optional
            Maximum bond dimension (0 for unlimited).

        Returns
        -------
        MPOF64
            Contracted MPO.
        """
        ptr = self._lib.t4a_mpo_f64_contract_zipup(
            self._ptr, other._ptr, tolerance, max_bond_dim
        )
        return MPOF64(ptr, self._lib, self._ffi)

    def contract_fit(
        self,
        other: MPOF64,
        *,
        tolerance: float = 1e-12,
        max_bond_dim: int = 0,
        max_sweeps: int = 0,
    ) -> MPOF64:
        """Contract with another MPO using variational fitting.

        Parameters
        ----------
        other : MPOF64
            MPO to contract with.
        tolerance : float, optional
            Convergence tolerance.
        max_bond_dim : int, optional
            Maximum bond dimension (0 for unlimited).
        max_sweeps : int, optional
            Maximum number of sweeps (0 for default).

        Returns
        -------
        MPOF64
            Contracted MPO.
        """
        ptr = self._lib.t4a_mpo_f64_contract_fit(
            self._ptr, other._ptr, tolerance, max_bond_dim, max_sweeps
        )
        return MPOF64(ptr, self._lib, self._ffi)

    def contract(
        self,
        other: MPOF64,
        *,
        algorithm: str | ContractionAlgorithm = "naive",
        tolerance: float = 1e-12,
        max_bond_dim: int = 0,
    ) -> MPOF64:
        """Contract with another MPO using the specified algorithm.

        Parameters
        ----------
        other : MPOF64
            MPO to contract with.
        algorithm : str or ContractionAlgorithm, optional
            Algorithm to use: "naive", "zipup", or "fit".
        tolerance : float, optional
            Truncation tolerance.
        max_bond_dim : int, optional
            Maximum bond dimension (0 for unlimited).

        Returns
        -------
        MPOF64
            Contracted MPO.
        """
        alg_int = _algorithm_to_int(algorithm)
        ptr = self._lib.t4a_mpo_f64_contract(
            self._ptr, other._ptr, alg_int, tolerance, max_bond_dim
        )
        return MPOF64(ptr, self._lib, self._ffi)


class MPOC64:
    """Matrix Product Operator with complex128 elements.

    An MPO represents a tensor with a Matrix Product structure,
    commonly used in quantum physics and tensor network computations.

    Examples
    --------
    >>> mpo = MPOC64.identity([2, 2])
    >>> len(mpo)
    2
    """

    __slots__ = ("_ptr", "_lib", "_ffi")

    def __init__(self, ptr, lib, ffi):
        """Initialize from a C pointer (internal use only)."""
        if ptr == ffi.NULL:
            raise ValueError("Failed to create MPOC64 (null pointer)")
        self._ptr = ptr
        self._lib = lib
        self._ffi = ffi

    def __del__(self):
        """Release the underlying C object."""
        if hasattr(self, "_ptr") and self._ptr != self._ffi.NULL:
            self._lib.t4a_mpo_c64_release(self._ptr)

    @classmethod
    def zeros(cls, site_dims: Sequence[tuple[int, int]]) -> MPOC64:
        """Create an MPO representing the zero operator.

        Parameters
        ----------
        site_dims : Sequence[tuple[int, int]]
            Site dimensions as (input_dim, output_dim) pairs.

        Returns
        -------
        MPOC64
            Zero MPO.
        """
        lib, ffi = get_lib()
        dims1 = ffi.new("size_t[]", [d[0] for d in site_dims])
        dims2 = ffi.new("size_t[]", [d[1] for d in site_dims])
        ptr = lib.t4a_mpo_c64_new_zeros(dims1, dims2, len(site_dims))
        return cls(ptr, lib, ffi)

    @classmethod
    def constant(
        cls, site_dims: Sequence[tuple[int, int]], value: complex
    ) -> MPOC64:
        """Create an MPO representing a constant operator.

        Parameters
        ----------
        site_dims : Sequence[tuple[int, int]]
            Site dimensions as (input_dim, output_dim) pairs.
        value : complex
            Constant value.

        Returns
        -------
        MPOC64
            Constant MPO.
        """
        lib, ffi = get_lib()
        dims1 = ffi.new("size_t[]", [d[0] for d in site_dims])
        dims2 = ffi.new("size_t[]", [d[1] for d in site_dims])
        ptr = lib.t4a_mpo_c64_new_constant(
            dims1, dims2, len(site_dims), value.real, value.imag
        )
        return cls(ptr, lib, ffi)

    @classmethod
    def identity(cls, site_dims: Sequence[int]) -> MPOC64:
        """Create an identity MPO.

        Parameters
        ----------
        site_dims : Sequence[int]
            Site dimensions (same for input and output).

        Returns
        -------
        MPOC64
            Identity MPO.
        """
        lib, ffi = get_lib()
        dims = ffi.new("size_t[]", list(site_dims))
        ptr = lib.t4a_mpo_c64_new_identity(dims, len(site_dims))
        return cls(ptr, lib, ffi)

    def __len__(self) -> int:
        """Return the number of sites."""
        out_len = self._ffi.new("size_t*")
        status = self._lib.t4a_mpo_c64_len(self._ptr, out_len)
        check_status(status)
        return out_len[0]

    def copy(self) -> MPOC64:
        """Create a copy of this MPO."""
        ptr = self._lib.t4a_mpo_c64_clone(self._ptr)
        return MPOC64(ptr, self._lib, self._ffi)

    def contract_naive(self, other: MPOC64) -> MPOC64:
        """Contract with another MPO using naive algorithm.

        Parameters
        ----------
        other : MPOC64
            MPO to contract with.

        Returns
        -------
        MPOC64
            Contracted MPO.
        """
        ptr = self._lib.t4a_mpo_c64_contract_naive(self._ptr, other._ptr)
        return MPOC64(ptr, self._lib, self._ffi)

    def contract_zipup(
        self,
        other: MPOC64,
        *,
        tolerance: float = 1e-12,
        max_bond_dim: int = 0,
    ) -> MPOC64:
        """Contract with another MPO using zip-up algorithm.

        Parameters
        ----------
        other : MPOC64
            MPO to contract with.
        tolerance : float, optional
            Truncation tolerance.
        max_bond_dim : int, optional
            Maximum bond dimension (0 for unlimited).

        Returns
        -------
        MPOC64
            Contracted MPO.
        """
        ptr = self._lib.t4a_mpo_c64_contract_zipup(
            self._ptr, other._ptr, tolerance, max_bond_dim
        )
        return MPOC64(ptr, self._lib, self._ffi)

    def contract_fit(
        self,
        other: MPOC64,
        *,
        tolerance: float = 1e-12,
        max_bond_dim: int = 0,
        max_sweeps: int = 0,
    ) -> MPOC64:
        """Contract with another MPO using variational fitting.

        Parameters
        ----------
        other : MPOC64
            MPO to contract with.
        tolerance : float, optional
            Convergence tolerance.
        max_bond_dim : int, optional
            Maximum bond dimension (0 for unlimited).
        max_sweeps : int, optional
            Maximum number of sweeps (0 for default).

        Returns
        -------
        MPOC64
            Contracted MPO.
        """
        ptr = self._lib.t4a_mpo_c64_contract_fit(
            self._ptr, other._ptr, tolerance, max_bond_dim, max_sweeps
        )
        return MPOC64(ptr, self._lib, self._ffi)

    def contract(
        self,
        other: MPOC64,
        *,
        algorithm: str | ContractionAlgorithm = "naive",
        tolerance: float = 1e-12,
        max_bond_dim: int = 0,
    ) -> MPOC64:
        """Contract with another MPO using the specified algorithm.

        Parameters
        ----------
        other : MPOC64
            MPO to contract with.
        algorithm : str or ContractionAlgorithm, optional
            Algorithm to use: "naive", "zipup", or "fit".
        tolerance : float, optional
            Truncation tolerance.
        max_bond_dim : int, optional
            Maximum bond dimension (0 for unlimited).

        Returns
        -------
        MPOC64
            Contracted MPO.
        """
        alg_int = _algorithm_to_int(algorithm)
        ptr = self._lib.t4a_mpo_c64_contract(
            self._ptr, other._ptr, alg_int, tolerance, max_bond_dim
        )
        return MPOC64(ptr, self._lib, self._ffi)
