"""SimpleTensorTrain - Simple tensor train wrapper for TCI operations.

This module provides a Python interface to the SimpleTT tensor train
from tensor4all-simplett, designed for tensor cross interpolation.

Examples
--------
>>> from tensor4all.simplett import SimpleTensorTrain
>>> import numpy as np
>>>
>>> # Create a constant tensor train
>>> tt = SimpleTensorTrain.constant([2, 3, 4], value=1.5)
>>> print(tt.n_sites)  # 3
>>> print(tt.site_dims)  # [2, 3, 4]
>>> print(tt.sum())  # 36.0 (= 1.5 * 2 * 3 * 4)
>>>
>>> # Evaluate at specific indices
>>> print(tt(0, 0, 0))  # 1.5
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ._ffi import ffi
from ._capi import get_lib, check_status


class SimpleTensorTrain:
    """A simple tensor train (MPS) wrapper for TCI operations.

    This class wraps the Rust SimpleTT from tensor4all-simplett, providing
    a simple interface for tensor trains created by cross interpolation.

    Currently only supports Float64 values.

    Parameters
    ----------
    _ptr : cdata
        Internal pointer to the Rust object. Use class methods to create instances.

    Examples
    --------
    >>> tt = SimpleTensorTrain.constant([2, 2, 2], 1.0)
    >>> print(tt.n_sites)  # 3
    >>> print(tt.rank)  # 1
    >>> print(tt.sum())  # 8.0
    """

    def __init__(self, _ptr):
        """Create a SimpleTensorTrain from a C pointer.

        Do not call this directly. Use class methods like `constant()` or `zeros()`.
        """
        if _ptr == ffi.NULL:
            raise ValueError("Cannot create SimpleTensorTrain from null pointer")
        self._ptr = _ptr

    def __del__(self):
        """Release the tensor train when garbage collected."""
        if hasattr(self, "_ptr") and self._ptr != ffi.NULL:
            get_lib().t4a_simplett_f64_release(self._ptr)
            self._ptr = ffi.NULL

    @classmethod
    def constant(cls, site_dims: Sequence[int], value: float = 1.0) -> SimpleTensorTrain:
        """Create a constant tensor train.

        All elements of the tensor train will have the specified value.

        Parameters
        ----------
        site_dims : Sequence[int]
            Dimensions for each site.
        value : float, optional
            The constant value (default: 1.0).

        Returns
        -------
        SimpleTensorTrain
            A new tensor train with all elements equal to `value`.

        Examples
        --------
        >>> tt = SimpleTensorTrain.constant([2, 3, 4], 2.0)
        >>> print(tt.sum())  # 48.0 (= 2.0 * 2 * 3 * 4)
        """
        lib = get_lib()
        n_sites = len(site_dims)
        dims_arr = ffi.new("size_t[]", site_dims)
        ptr = lib.t4a_simplett_f64_constant(dims_arr, n_sites, float(value))
        if ptr == ffi.NULL:
            raise RuntimeError("Failed to create constant tensor train")
        return cls(ptr)

    @classmethod
    def zeros(cls, site_dims: Sequence[int]) -> SimpleTensorTrain:
        """Create a zero tensor train.

        Parameters
        ----------
        site_dims : Sequence[int]
            Dimensions for each site.

        Returns
        -------
        SimpleTensorTrain
            A new tensor train with all elements equal to zero.

        Examples
        --------
        >>> tt = SimpleTensorTrain.zeros([2, 3, 4])
        >>> print(tt.sum())  # 0.0
        """
        lib = get_lib()
        n_sites = len(site_dims)
        dims_arr = ffi.new("size_t[]", site_dims)
        ptr = lib.t4a_simplett_f64_zeros(dims_arr, n_sites)
        if ptr == ffi.NULL:
            raise RuntimeError("Failed to create zero tensor train")
        return cls(ptr)

    def copy(self) -> SimpleTensorTrain:
        """Create a deep copy of this tensor train.

        Returns
        -------
        SimpleTensorTrain
            A new tensor train with copied data.
        """
        lib = get_lib()
        new_ptr = lib.t4a_simplett_f64_clone(self._ptr)
        if new_ptr == ffi.NULL:
            raise RuntimeError("Failed to clone tensor train")
        return SimpleTensorTrain(new_ptr)

    @property
    def n_sites(self) -> int:
        """Number of sites in the tensor train."""
        lib = get_lib()
        out_len = ffi.new("size_t*")
        status = lib.t4a_simplett_f64_len(self._ptr, out_len)
        check_status(status, "n_sites")
        return out_len[0]

    def __len__(self) -> int:
        """Number of sites in the tensor train."""
        return self.n_sites

    @property
    def site_dims(self) -> list[int]:
        """List of site (physical) dimensions."""
        lib = get_lib()
        n = self.n_sites
        if n == 0:
            return []
        out_dims = ffi.new("size_t[]", n)
        status = lib.t4a_simplett_f64_site_dims(self._ptr, out_dims, n)
        check_status(status, "site_dims")
        return [out_dims[i] for i in range(n)]

    @property
    def link_dims(self) -> list[int]:
        """List of link (bond) dimensions. Returns n-1 values for n sites."""
        lib = get_lib()
        n = self.n_sites
        if n <= 1:
            return []
        n_links = n - 1
        out_dims = ffi.new("size_t[]", n_links)
        status = lib.t4a_simplett_f64_link_dims(self._ptr, out_dims, n_links)
        check_status(status, "link_dims")
        return [out_dims[i] for i in range(n_links)]

    @property
    def rank(self) -> int:
        """Maximum bond dimension (rank)."""
        lib = get_lib()
        out_rank = ffi.new("size_t*")
        status = lib.t4a_simplett_f64_rank(self._ptr, out_rank)
        check_status(status, "rank")
        return out_rank[0]

    def evaluate(self, indices: Sequence[int]) -> float:
        """Evaluate the tensor train at a given multi-index.

        Parameters
        ----------
        indices : Sequence[int]
            The indices for each site (0-based).

        Returns
        -------
        float
            The value at the specified indices.

        Examples
        --------
        >>> tt = SimpleTensorTrain.constant([2, 3], 5.0)
        >>> print(tt.evaluate([0, 1]))  # 5.0
        """
        lib = get_lib()
        n = len(indices)
        idx_arr = ffi.new("size_t[]", indices)
        out_value = ffi.new("double*")
        status = lib.t4a_simplett_f64_evaluate(self._ptr, idx_arr, n, out_value)
        check_status(status, "evaluate")
        return out_value[0]

    def __call__(self, *indices: int) -> float:
        """Evaluate the tensor train at a given multi-index.

        This is a convenience method equivalent to `evaluate()`.

        Parameters
        ----------
        *indices : int
            The indices for each site (0-based).

        Returns
        -------
        float
            The value at the specified indices.

        Examples
        --------
        >>> tt = SimpleTensorTrain.constant([2, 3], 5.0)
        >>> print(tt(0, 1))  # 5.0
        """
        return self.evaluate(indices)

    def sum(self) -> float:
        """Compute the sum over all tensor train elements.

        Returns
        -------
        float
            The sum of all elements.

        Examples
        --------
        >>> tt = SimpleTensorTrain.constant([2, 3], 1.0)
        >>> print(tt.sum())  # 6.0 (= 2 * 3)
        """
        lib = get_lib()
        out_value = ffi.new("double*")
        status = lib.t4a_simplett_f64_sum(self._ptr, out_value)
        check_status(status, "sum")
        return out_value[0]

    def norm(self) -> float:
        """Compute the Frobenius norm of the tensor train.

        Returns
        -------
        float
            The Frobenius norm.
        """
        lib = get_lib()
        out_value = ffi.new("double*")
        status = lib.t4a_simplett_f64_norm(self._ptr, out_value)
        check_status(status, "norm")
        return out_value[0]

    def site_tensor(self, site: int) -> np.ndarray:
        """Get the site tensor at a specific site.

        Parameters
        ----------
        site : int
            The site index (0-based).

        Returns
        -------
        np.ndarray
            A 3D array with shape (left_dim, site_dim, right_dim).
        """
        lib = get_lib()
        n = self.n_sites
        if site < 0 or site >= n:
            raise IndexError(f"Site index {site} out of range [0, {n})")

        # Get dimensions
        sdims = self.site_dims
        ldims = self.link_dims

        left_dim = 1 if site == 0 else ldims[site - 1]
        site_dim = sdims[site]
        right_dim = 1 if site == n - 1 else ldims[site]

        total_size = left_dim * site_dim * right_dim
        out_data = ffi.new("double[]", total_size)
        out_left = ffi.new("size_t*")
        out_site = ffi.new("size_t*")
        out_right = ffi.new("size_t*")

        status = lib.t4a_simplett_f64_site_tensor(
            self._ptr, site, out_data, total_size, out_left, out_site, out_right
        )
        check_status(status, "site_tensor")

        # Convert to numpy array (Rust uses row-major, need to reshape and transpose)
        data = np.array([out_data[i] for i in range(total_size)])
        # Rust stores as (right, site, left) in row-major, so reshape and permute
        tensor = data.reshape(out_right[0], out_site[0], out_left[0])
        tensor = tensor.transpose(2, 1, 0)  # (left, site, right)
        return tensor

    def __repr__(self) -> str:
        return f"SimpleTensorTrain(n_sites={self.n_sites}, rank={self.rank})"
