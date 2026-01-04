"""TensorTrain classes for f64 and c64 element types."""

from __future__ import annotations

from typing import List, TYPE_CHECKING

from .._ffi import ffi
from .._capi import get_lib, check_status

if TYPE_CHECKING:
    import numpy as np


def _has_numpy() -> bool:
    """Check if numpy is available."""
    try:
        import numpy  # noqa: F401
        return True
    except ImportError:
        return False


class TensorTrainF64:
    """
    A tensor train (Matrix Product State) with float64 elements.

    Wraps the Rust `TensorTrain<f64>` type via the C API.

    Parameters
    ----------
    ptr : ffi pointer
        Internal pointer to the Rust object. Users should use class methods
        like `zeros()` or `constant()` to create instances.

    Examples
    --------
    >>> tt = TensorTrainF64.constant([2, 3, 2], 1.0)
    >>> print(tt.sum())  # 12.0
    >>> print(tt.site_dims)  # [2, 3, 2]
    """

    def __init__(self, ptr):
        """Initialize from a raw pointer (internal use)."""
        if ptr == ffi.NULL:
            raise RuntimeError("Failed to create TensorTrainF64 (null pointer)")
        self._ptr = ptr

    def __del__(self):
        """Release the tensor train."""
        if hasattr(self, "_ptr") and self._ptr != ffi.NULL:
            lib = get_lib()
            lib.t4a_tt_f64_release(self._ptr)
            self._ptr = ffi.NULL

    @classmethod
    def zeros(cls, site_dims: List[int]) -> "TensorTrainF64":
        """
        Create a tensor train representing the zero function.

        Parameters
        ----------
        site_dims : List[int]
            List of site dimensions.

        Returns
        -------
        TensorTrainF64
            A new tensor train with all elements equal to zero.
        """
        lib = get_lib()
        dims = ffi.new("size_t[]", site_dims)
        ptr = lib.t4a_tt_f64_new_zeros(dims, len(site_dims))
        return cls(ptr)

    @classmethod
    def constant(cls, site_dims: List[int], value: float) -> "TensorTrainF64":
        """
        Create a tensor train representing a constant function.

        Parameters
        ----------
        site_dims : List[int]
            List of site dimensions.
        value : float
            The constant value.

        Returns
        -------
        TensorTrainF64
            A new tensor train with all elements equal to `value`.
        """
        lib = get_lib()
        dims = ffi.new("size_t[]", site_dims)
        ptr = lib.t4a_tt_f64_new_constant(dims, len(site_dims), value)
        return cls(ptr)

    def __len__(self) -> int:
        """Get the number of sites in the tensor train."""
        lib = get_lib()
        out_len = ffi.new("size_t*")
        status = lib.t4a_tt_f64_len(self._ptr, out_len)
        check_status(status)
        return out_len[0]

    @property
    def site_dims(self) -> List[int]:
        """Get the site (physical) dimensions."""
        lib = get_lib()
        n = len(self)
        out_dims = ffi.new("size_t[]", n)
        status = lib.t4a_tt_f64_site_dims(self._ptr, out_dims, n)
        check_status(status)
        return list(out_dims)

    @property
    def link_dims(self) -> List[int]:
        """Get the link (bond) dimensions."""
        lib = get_lib()
        n = len(self)
        if n <= 1:
            return []
        out_dims = ffi.new("size_t[]", n - 1)
        status = lib.t4a_tt_f64_link_dims(self._ptr, out_dims, n - 1)
        check_status(status)
        return list(out_dims)

    @property
    def rank(self) -> int:
        """Get the maximum bond dimension (rank)."""
        lib = get_lib()
        out_rank = ffi.new("size_t*")
        status = lib.t4a_tt_f64_rank(self._ptr, out_rank)
        check_status(status)
        return out_rank[0]

    def evaluate(self, indices: List[int]) -> float:
        """
        Evaluate the tensor train at a given index set.

        Parameters
        ----------
        indices : List[int]
            List of indices (0-based).

        Returns
        -------
        float
            The value at the given indices.
        """
        lib = get_lib()
        idx = ffi.new("size_t[]", indices)
        out_value = ffi.new("double*")
        status = lib.t4a_tt_f64_evaluate(self._ptr, idx, len(indices), out_value)
        check_status(status)
        return out_value[0]

    def sum(self) -> float:
        """Compute the sum of all elements."""
        lib = get_lib()
        out_sum = ffi.new("double*")
        status = lib.t4a_tt_f64_sum(self._ptr, out_sum)
        check_status(status)
        return out_sum[0]

    def norm(self) -> float:
        """Compute the Frobenius norm."""
        lib = get_lib()
        out_norm = ffi.new("double*")
        status = lib.t4a_tt_f64_norm(self._ptr, out_norm)
        check_status(status)
        return out_norm[0]

    def log_norm(self) -> float:
        """
        Compute the logarithm of the Frobenius norm.

        More numerically stable than `log(norm())` for very large or small norms.
        """
        lib = get_lib()
        out_log_norm = ffi.new("double*")
        status = lib.t4a_tt_f64_log_norm(self._ptr, out_log_norm)
        check_status(status)
        return out_log_norm[0]

    def scale(self, factor: float) -> None:
        """Scale the tensor train in place."""
        lib = get_lib()
        status = lib.t4a_tt_f64_scale_inplace(self._ptr, factor)
        check_status(status)

    def scaled(self, factor: float) -> "TensorTrainF64":
        """
        Create a new tensor train scaled by a factor.

        The original tensor train is unchanged.
        """
        lib = get_lib()
        ptr = lib.t4a_tt_f64_scaled(self._ptr, factor)
        return TensorTrainF64(ptr)

    def to_numpy(self) -> "np.ndarray":
        """
        Convert the tensor train to a NumPy array.

        Returns the full tensor as a dense array.
        Warning: This can be very large for high-dimensional tensors!

        Returns
        -------
        np.ndarray
            The full tensor as a dense array.

        Raises
        ------
        ImportError
            If NumPy is not installed.
        """
        if not _has_numpy():
            raise ImportError("NumPy is required for to_numpy()")

        import numpy as np

        lib = get_lib()

        # Query length
        out_len = ffi.new("size_t*")
        status = lib.t4a_tt_f64_fulltensor(self._ptr, ffi.NULL, 0, out_len)
        check_status(status)

        # Get data (row-major from Rust)
        data = ffi.new("double[]", out_len[0])
        status = lib.t4a_tt_f64_fulltensor(self._ptr, data, out_len[0], out_len)
        check_status(status)

        # Convert to numpy array and reshape
        arr = np.array(list(data), dtype=np.float64)
        dims = self.site_dims
        return arr.reshape(dims)

    def copy(self) -> "TensorTrainF64":
        """Create a copy (clone) of the tensor train."""
        lib = get_lib()
        ptr = lib.t4a_tt_f64_clone(self._ptr)
        return TensorTrainF64(ptr)

    def clone(self) -> "TensorTrainF64":
        """Alias for copy() to match Julia API."""
        return self.copy()

    def fulltensor(self) -> "np.ndarray":
        """Alias for to_numpy() to match Julia API."""
        return self.to_numpy()

    def __copy__(self) -> "TensorTrainF64":
        return self.copy()

    def __deepcopy__(self, memo) -> "TensorTrainF64":
        return self.copy()

    def __add__(self, other: "TensorTrainF64") -> "TensorTrainF64":
        """Add two tensor trains element-wise."""
        lib = get_lib()
        ptr = lib.t4a_tt_f64_add(self._ptr, other._ptr)
        return TensorTrainF64(ptr)

    def __sub__(self, other: "TensorTrainF64") -> "TensorTrainF64":
        """Subtract two tensor trains element-wise."""
        lib = get_lib()
        ptr = lib.t4a_tt_f64_sub(self._ptr, other._ptr)
        return TensorTrainF64(ptr)

    def __neg__(self) -> "TensorTrainF64":
        """Negate the tensor train."""
        lib = get_lib()
        ptr = lib.t4a_tt_f64_negate(self._ptr)
        return TensorTrainF64(ptr)

    def __mul__(self, other):
        """Multiply: scalar * tt or tt * tt (Hadamard product)."""
        if isinstance(other, TensorTrainF64):
            return self.hadamard(other)
        else:
            return self.scaled(float(other))

    def __rmul__(self, other):
        """Right multiply: scalar * tt."""
        return self.scaled(float(other))

    def reverse(self) -> "TensorTrainF64":
        """Reverse the tensor train (swap left and right)."""
        lib = get_lib()
        ptr = lib.t4a_tt_f64_reverse(self._ptr)
        return TensorTrainF64(ptr)

    def hadamard(self, other: "TensorTrainF64") -> "TensorTrainF64":
        """
        Compute the Hadamard (element-wise) product of two tensor trains.

        For each index i: result[i] = self[i] * other[i]

        The resulting bond dimension is the product of the input bond dimensions.
        Use compress() afterward to reduce the bond dimension.
        """
        lib = get_lib()
        ptr = lib.t4a_tt_f64_hadamard(self._ptr, other._ptr)
        return TensorTrainF64(ptr)

    def dot(self, other: "TensorTrainF64") -> float:
        """
        Compute the inner product (dot product) of two tensor trains.

        Returns: sum over all indices i of self[i] * other[i]
        """
        lib = get_lib()
        out_dot = ffi.new("double*")
        status = lib.t4a_tt_f64_dot(self._ptr, other._ptr, out_dot)
        check_status(status)
        return out_dot[0]

    def compress(self, tolerance: float = 1e-12, max_bond_dim: int = 0) -> None:
        """
        Compress the tensor train in-place.

        Parameters
        ----------
        tolerance : float
            Relative tolerance for truncation (default: 1e-12).
        max_bond_dim : int
            Maximum bond dimension (0 for unlimited).
        """
        lib = get_lib()
        status = lib.t4a_tt_f64_compress(self._ptr, tolerance, max_bond_dim)
        check_status(status)

    def compressed(
        self, tolerance: float = 1e-12, max_bond_dim: int = 0
    ) -> "TensorTrainF64":
        """
        Create a compressed copy of the tensor train.

        Parameters
        ----------
        tolerance : float
            Relative tolerance for truncation (default: 1e-12).
        max_bond_dim : int
            Maximum bond dimension (0 for unlimited).

        Returns
        -------
        TensorTrainF64
            A new compressed tensor train.
        """
        lib = get_lib()
        ptr = lib.t4a_tt_f64_compressed(self._ptr, tolerance, max_bond_dim)
        return TensorTrainF64(ptr)


class TensorTrainC64:
    """
    A tensor train (Matrix Product State) with complex128 elements.

    Wraps the Rust `TensorTrain<Complex64>` type via the C API.
    """

    def __init__(self, ptr):
        """Initialize from a raw pointer (internal use)."""
        if ptr == ffi.NULL:
            raise RuntimeError("Failed to create TensorTrainC64 (null pointer)")
        self._ptr = ptr

    def __del__(self):
        """Release the tensor train."""
        if hasattr(self, "_ptr") and self._ptr != ffi.NULL:
            lib = get_lib()
            lib.t4a_tt_c64_release(self._ptr)
            self._ptr = ffi.NULL

    @classmethod
    def zeros(cls, site_dims: List[int]) -> "TensorTrainC64":
        """Create a tensor train representing the zero function."""
        lib = get_lib()
        dims = ffi.new("size_t[]", site_dims)
        ptr = lib.t4a_tt_c64_new_zeros(dims, len(site_dims))
        return cls(ptr)

    @classmethod
    def constant(cls, site_dims: List[int], value: complex) -> "TensorTrainC64":
        """Create a tensor train representing a constant function."""
        lib = get_lib()
        dims = ffi.new("size_t[]", site_dims)
        ptr = lib.t4a_tt_c64_new_constant(
            dims, len(site_dims), value.real, value.imag
        )
        return cls(ptr)

    def __len__(self) -> int:
        """Get the number of sites in the tensor train."""
        lib = get_lib()
        out_len = ffi.new("size_t*")
        status = lib.t4a_tt_c64_len(self._ptr, out_len)
        check_status(status)
        return out_len[0]

    @property
    def site_dims(self) -> List[int]:
        """Get the site (physical) dimensions."""
        lib = get_lib()
        n = len(self)
        out_dims = ffi.new("size_t[]", n)
        status = lib.t4a_tt_c64_site_dims(self._ptr, out_dims, n)
        check_status(status)
        return list(out_dims)

    @property
    def link_dims(self) -> List[int]:
        """Get the link (bond) dimensions."""
        lib = get_lib()
        n = len(self)
        if n <= 1:
            return []
        out_dims = ffi.new("size_t[]", n - 1)
        status = lib.t4a_tt_c64_link_dims(self._ptr, out_dims, n - 1)
        check_status(status)
        return list(out_dims)

    @property
    def rank(self) -> int:
        """Get the maximum bond dimension (rank)."""
        lib = get_lib()
        out_rank = ffi.new("size_t*")
        status = lib.t4a_tt_c64_rank(self._ptr, out_rank)
        check_status(status)
        return out_rank[0]

    def evaluate(self, indices: List[int]) -> complex:
        """Evaluate the tensor train at a given index set (0-based indices)."""
        lib = get_lib()
        idx = ffi.new("size_t[]", indices)
        out_re = ffi.new("double*")
        out_im = ffi.new("double*")
        status = lib.t4a_tt_c64_evaluate(
            self._ptr, idx, len(indices), out_re, out_im
        )
        check_status(status)
        return complex(out_re[0], out_im[0])

    def sum(self) -> complex:
        """Compute the sum of all elements."""
        lib = get_lib()
        out_re = ffi.new("double*")
        out_im = ffi.new("double*")
        status = lib.t4a_tt_c64_sum(self._ptr, out_re, out_im)
        check_status(status)
        return complex(out_re[0], out_im[0])

    def norm(self) -> float:
        """Compute the Frobenius norm."""
        lib = get_lib()
        out_norm = ffi.new("double*")
        status = lib.t4a_tt_c64_norm(self._ptr, out_norm)
        check_status(status)
        return out_norm[0]

    def log_norm(self) -> float:
        """Compute the logarithm of the Frobenius norm."""
        lib = get_lib()
        out_log_norm = ffi.new("double*")
        status = lib.t4a_tt_c64_log_norm(self._ptr, out_log_norm)
        check_status(status)
        return out_log_norm[0]

    def scale(self, factor: complex) -> None:
        """Scale the tensor train in place."""
        lib = get_lib()
        status = lib.t4a_tt_c64_scale_inplace(self._ptr, factor.real, factor.imag)
        check_status(status)

    def scaled(self, factor: complex) -> "TensorTrainC64":
        """Create a new tensor train scaled by a factor."""
        lib = get_lib()
        ptr = lib.t4a_tt_c64_scaled(self._ptr, factor.real, factor.imag)
        return TensorTrainC64(ptr)

    def to_numpy(self) -> "np.ndarray":
        """Convert the tensor train to a NumPy array.

        Raises
        ------
        ImportError
            If NumPy is not installed.
        """
        if not _has_numpy():
            raise ImportError("NumPy is required for to_numpy()")

        import numpy as np

        lib = get_lib()

        # Query length
        out_len = ffi.new("size_t*")
        status = lib.t4a_tt_c64_fulltensor(
            self._ptr, ffi.NULL, ffi.NULL, 0, out_len
        )
        check_status(status)

        # Get data
        data_re = ffi.new("double[]", out_len[0])
        data_im = ffi.new("double[]", out_len[0])
        status = lib.t4a_tt_c64_fulltensor(
            self._ptr, data_re, data_im, out_len[0], out_len
        )
        check_status(status)

        # Convert to numpy
        arr_re = np.array(list(data_re), dtype=np.float64)
        arr_im = np.array(list(data_im), dtype=np.float64)
        arr = arr_re + 1j * arr_im
        dims = self.site_dims
        return arr.reshape(dims)

    def copy(self) -> "TensorTrainC64":
        """Create a copy (clone) of the tensor train."""
        lib = get_lib()
        ptr = lib.t4a_tt_c64_clone(self._ptr)
        return TensorTrainC64(ptr)

    def clone(self) -> "TensorTrainC64":
        """Alias for copy() to match Julia API."""
        return self.copy()

    def fulltensor(self) -> "np.ndarray":
        """Alias for to_numpy() to match Julia API."""
        return self.to_numpy()

    def __copy__(self) -> "TensorTrainC64":
        return self.copy()

    def __deepcopy__(self, memo) -> "TensorTrainC64":
        return self.copy()

    def __add__(self, other: "TensorTrainC64") -> "TensorTrainC64":
        """Add two tensor trains element-wise."""
        lib = get_lib()
        ptr = lib.t4a_tt_c64_add(self._ptr, other._ptr)
        return TensorTrainC64(ptr)

    def __sub__(self, other: "TensorTrainC64") -> "TensorTrainC64":
        """Subtract two tensor trains element-wise."""
        lib = get_lib()
        ptr = lib.t4a_tt_c64_sub(self._ptr, other._ptr)
        return TensorTrainC64(ptr)

    def __neg__(self) -> "TensorTrainC64":
        """Negate the tensor train."""
        lib = get_lib()
        ptr = lib.t4a_tt_c64_negate(self._ptr)
        return TensorTrainC64(ptr)

    def __mul__(self, other):
        """Multiply: scalar * tt or tt * tt (Hadamard product)."""
        if isinstance(other, TensorTrainC64):
            return self.hadamard(other)
        else:
            return self.scaled(complex(other))

    def __rmul__(self, other):
        """Right multiply: scalar * tt."""
        return self.scaled(complex(other))

    def reverse(self) -> "TensorTrainC64":
        """Reverse the tensor train (swap left and right)."""
        lib = get_lib()
        ptr = lib.t4a_tt_c64_reverse(self._ptr)
        return TensorTrainC64(ptr)

    def hadamard(self, other: "TensorTrainC64") -> "TensorTrainC64":
        """Compute the Hadamard (element-wise) product of two tensor trains."""
        lib = get_lib()
        ptr = lib.t4a_tt_c64_hadamard(self._ptr, other._ptr)
        return TensorTrainC64(ptr)

    def dot(self, other: "TensorTrainC64") -> complex:
        """Compute the inner product (dot product) of two tensor trains."""
        lib = get_lib()
        out_re = ffi.new("double*")
        out_im = ffi.new("double*")
        status = lib.t4a_tt_c64_dot(self._ptr, other._ptr, out_re, out_im)
        check_status(status)
        return complex(out_re[0], out_im[0])

    def compress(self, tolerance: float = 1e-12, max_bond_dim: int = 0) -> None:
        """Compress the tensor train in-place."""
        lib = get_lib()
        status = lib.t4a_tt_c64_compress(self._ptr, tolerance, max_bond_dim)
        check_status(status)

    def compressed(
        self, tolerance: float = 1e-12, max_bond_dim: int = 0
    ) -> "TensorTrainC64":
        """Create a compressed copy of the tensor train."""
        lib = get_lib()
        ptr = lib.t4a_tt_c64_compressed(self._ptr, tolerance, max_bond_dim)
        return TensorTrainC64(ptr)
