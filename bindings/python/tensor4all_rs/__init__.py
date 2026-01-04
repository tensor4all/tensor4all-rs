"""
Tensor4all-rs Python bindings

Python interface to the tensor4all-rs library via its C API.
"""

import ctypes
import os
import platform
from pathlib import Path
from typing import List, Optional, Tuple, Union, TYPE_CHECKING

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    HAS_NUMPY = False

# Status codes (must match Rust constants)
T4A_SUCCESS = 0
T4A_NULL_POINTER = -1
T4A_INVALID_ARGUMENT = -2
T4A_TAG_OVERFLOW = -3
T4A_TAG_TOO_LONG = -4
T4A_BUFFER_TOO_SMALL = -5
T4A_INTERNAL_ERROR = -6

# Library handle
_lib: Optional[ctypes.CDLL] = None


def init_library(libpath: Optional[str] = None) -> None:
    """
    Initialize the library by loading the shared library.

    Parameters
    ----------
    libpath : str, optional
        Path to the libtensor4all_capi shared library.
        If not provided, attempts to find it in standard locations.

    Raises
    ------
    OSError
        If the library cannot be loaded.
    """
    global _lib
    if _lib is not None:
        return

    if libpath is None:
        # Try to find library in standard locations
        system = platform.system()
        if system == "Darwin":
            libname = "libtensor4all_capi.dylib"
        elif system == "Linux":
            libname = "libtensor4all_capi.so"
        elif system == "Windows":
            libname = "tensor4all_capi.dll"
        else:
            raise OSError(f"Unsupported platform: {system}")

        # Check common locations
        search_paths = [
            Path(__file__).parent / libname,
            Path(__file__).parent.parent.parent.parent / "target" / "release" / libname,
            Path(__file__).parent.parent.parent.parent / "target" / "debug" / libname,
        ]

        for path in search_paths:
            if path.exists():
                libpath = str(path)
                break

        if libpath is None:
            raise OSError(
                f"Could not find {libname}. "
                "Please provide the path or build with: cargo build --release -p tensor4all-capi"
            )

    _lib = ctypes.CDLL(libpath)
    _setup_function_signatures()


def _get_lib() -> ctypes.CDLL:
    """Get the library handle."""
    if _lib is None:
        raise RuntimeError(
            "Library not initialized. Call tensor4all_rs.init_library() first."
        )
    return _lib


def _setup_function_signatures() -> None:
    """Set up ctypes function signatures for type safety."""
    lib = _lib
    if lib is None:
        return

    # TensorTrain F64 functions
    lib.t4a_tt_f64_new_zeros.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t]
    lib.t4a_tt_f64_new_zeros.restype = ctypes.c_void_p

    lib.t4a_tt_f64_new_constant.argtypes = [
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        ctypes.c_double,
    ]
    lib.t4a_tt_f64_new_constant.restype = ctypes.c_void_p

    lib.t4a_tt_f64_release.argtypes = [ctypes.c_void_p]
    lib.t4a_tt_f64_release.restype = None

    lib.t4a_tt_f64_clone.argtypes = [ctypes.c_void_p]
    lib.t4a_tt_f64_clone.restype = ctypes.c_void_p

    lib.t4a_tt_f64_len.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
    lib.t4a_tt_f64_len.restype = ctypes.c_int

    lib.t4a_tt_f64_site_dims.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
    ]
    lib.t4a_tt_f64_site_dims.restype = ctypes.c_int

    lib.t4a_tt_f64_link_dims.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
    ]
    lib.t4a_tt_f64_link_dims.restype = ctypes.c_int

    lib.t4a_tt_f64_rank.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
    lib.t4a_tt_f64_rank.restype = ctypes.c_int

    lib.t4a_tt_f64_evaluate.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.t4a_tt_f64_evaluate.restype = ctypes.c_int

    lib.t4a_tt_f64_sum.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
    lib.t4a_tt_f64_sum.restype = ctypes.c_int

    lib.t4a_tt_f64_norm.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
    lib.t4a_tt_f64_norm.restype = ctypes.c_int

    lib.t4a_tt_f64_log_norm.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
    lib.t4a_tt_f64_log_norm.restype = ctypes.c_int

    lib.t4a_tt_f64_scaled.argtypes = [ctypes.c_void_p, ctypes.c_double]
    lib.t4a_tt_f64_scaled.restype = ctypes.c_void_p

    lib.t4a_tt_f64_scale_inplace.argtypes = [ctypes.c_void_p, ctypes.c_double]
    lib.t4a_tt_f64_scale_inplace.restype = ctypes.c_int

    lib.t4a_tt_f64_fulltensor.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.t4a_tt_f64_fulltensor.restype = ctypes.c_int

    # TensorTrain C64 functions
    lib.t4a_tt_c64_new_zeros.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t]
    lib.t4a_tt_c64_new_zeros.restype = ctypes.c_void_p

    lib.t4a_tt_c64_new_constant.argtypes = [
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        ctypes.c_double,
        ctypes.c_double,
    ]
    lib.t4a_tt_c64_new_constant.restype = ctypes.c_void_p

    lib.t4a_tt_c64_release.argtypes = [ctypes.c_void_p]
    lib.t4a_tt_c64_release.restype = None

    lib.t4a_tt_c64_clone.argtypes = [ctypes.c_void_p]
    lib.t4a_tt_c64_clone.restype = ctypes.c_void_p

    lib.t4a_tt_c64_len.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
    lib.t4a_tt_c64_len.restype = ctypes.c_int

    lib.t4a_tt_c64_site_dims.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
    ]
    lib.t4a_tt_c64_site_dims.restype = ctypes.c_int

    lib.t4a_tt_c64_link_dims.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
    ]
    lib.t4a_tt_c64_link_dims.restype = ctypes.c_int

    lib.t4a_tt_c64_rank.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
    lib.t4a_tt_c64_rank.restype = ctypes.c_int

    lib.t4a_tt_c64_evaluate.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.t4a_tt_c64_evaluate.restype = ctypes.c_int

    lib.t4a_tt_c64_sum.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.t4a_tt_c64_sum.restype = ctypes.c_int

    lib.t4a_tt_c64_norm.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
    lib.t4a_tt_c64_norm.restype = ctypes.c_int

    lib.t4a_tt_c64_log_norm.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
    lib.t4a_tt_c64_log_norm.restype = ctypes.c_int

    lib.t4a_tt_c64_scaled.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double]
    lib.t4a_tt_c64_scaled.restype = ctypes.c_void_p

    lib.t4a_tt_c64_scale_inplace.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double]
    lib.t4a_tt_c64_scale_inplace.restype = ctypes.c_int

    lib.t4a_tt_c64_fulltensor.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.t4a_tt_c64_fulltensor.restype = ctypes.c_int


class TensorTrainF64:
    """
    A tensor train (Matrix Product State) with float64 elements.

    Wraps the Rust `TensorTrain<f64>` type via the C API.
    """

    def __init__(self, ptr: ctypes.c_void_p):
        """Initialize from a raw pointer (internal use)."""
        self._ptr = ptr

    def __del__(self):
        """Release the tensor train."""
        if hasattr(self, "_ptr") and self._ptr is not None:
            lib = _get_lib()
            lib.t4a_tt_f64_release(self._ptr)
            self._ptr = None

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
        lib = _get_lib()
        dims = (ctypes.c_size_t * len(site_dims))(*site_dims)
        ptr = lib.t4a_tt_f64_new_zeros(dims, len(site_dims))
        if ptr is None:
            raise RuntimeError("Failed to create TensorTrainF64")
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
        lib = _get_lib()
        dims = (ctypes.c_size_t * len(site_dims))(*site_dims)
        ptr = lib.t4a_tt_f64_new_constant(dims, len(site_dims), value)
        if ptr is None:
            raise RuntimeError("Failed to create TensorTrainF64")
        return cls(ptr)

    def __len__(self) -> int:
        """Get the number of sites in the tensor train."""
        lib = _get_lib()
        out_len = ctypes.c_size_t()
        status = lib.t4a_tt_f64_len(self._ptr, ctypes.byref(out_len))
        if status != T4A_SUCCESS:
            raise RuntimeError(f"Failed to get length: status = {status}")
        return out_len.value

    @property
    def site_dims(self) -> List[int]:
        """Get the site (physical) dimensions."""
        lib = _get_lib()
        n = len(self)
        out_dims = (ctypes.c_size_t * n)()
        status = lib.t4a_tt_f64_site_dims(self._ptr, out_dims, n)
        if status != T4A_SUCCESS:
            raise RuntimeError(f"Failed to get site_dims: status = {status}")
        return list(out_dims)

    @property
    def link_dims(self) -> List[int]:
        """Get the link (bond) dimensions."""
        lib = _get_lib()
        n = len(self)
        if n <= 1:
            return []
        out_dims = (ctypes.c_size_t * (n - 1))()
        status = lib.t4a_tt_f64_link_dims(self._ptr, out_dims, n - 1)
        if status != T4A_SUCCESS:
            raise RuntimeError(f"Failed to get link_dims: status = {status}")
        return list(out_dims)

    @property
    def rank(self) -> int:
        """Get the maximum bond dimension (rank)."""
        lib = _get_lib()
        out_rank = ctypes.c_size_t()
        status = lib.t4a_tt_f64_rank(self._ptr, ctypes.byref(out_rank))
        if status != T4A_SUCCESS:
            raise RuntimeError(f"Failed to get rank: status = {status}")
        return out_rank.value

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
        lib = _get_lib()
        idx = (ctypes.c_size_t * len(indices))(*indices)
        out_value = ctypes.c_double()
        status = lib.t4a_tt_f64_evaluate(
            self._ptr, idx, len(indices), ctypes.byref(out_value)
        )
        if status != T4A_SUCCESS:
            raise RuntimeError(f"Failed to evaluate: status = {status}")
        return out_value.value

    def sum(self) -> float:
        """Compute the sum of all elements."""
        lib = _get_lib()
        out_sum = ctypes.c_double()
        status = lib.t4a_tt_f64_sum(self._ptr, ctypes.byref(out_sum))
        if status != T4A_SUCCESS:
            raise RuntimeError(f"Failed to compute sum: status = {status}")
        return out_sum.value

    def norm(self) -> float:
        """Compute the Frobenius norm."""
        lib = _get_lib()
        out_norm = ctypes.c_double()
        status = lib.t4a_tt_f64_norm(self._ptr, ctypes.byref(out_norm))
        if status != T4A_SUCCESS:
            raise RuntimeError(f"Failed to compute norm: status = {status}")
        return out_norm.value

    def log_norm(self) -> float:
        """
        Compute the logarithm of the Frobenius norm.

        This is more numerically stable than `log(norm())` for very large or small norms.
        """
        lib = _get_lib()
        out_log_norm = ctypes.c_double()
        status = lib.t4a_tt_f64_log_norm(self._ptr, ctypes.byref(out_log_norm))
        if status != T4A_SUCCESS:
            raise RuntimeError(f"Failed to compute log_norm: status = {status}")
        return out_log_norm.value

    def scale(self, factor: float) -> None:
        """Scale the tensor train in place."""
        lib = _get_lib()
        status = lib.t4a_tt_f64_scale_inplace(self._ptr, factor)
        if status != T4A_SUCCESS:
            raise RuntimeError(f"Failed to scale: status = {status}")

    def scaled(self, factor: float) -> "TensorTrainF64":
        """
        Create a new tensor train scaled by a factor.

        The original tensor train is unchanged.
        """
        lib = _get_lib()
        ptr = lib.t4a_tt_f64_scaled(self._ptr, factor)
        if ptr is None:
            raise RuntimeError("Failed to create scaled TensorTrainF64")
        return TensorTrainF64(ptr)

    def to_numpy(self):
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
        if not HAS_NUMPY:
            raise ImportError("NumPy is required for to_numpy()")

        lib = _get_lib()

        # Query length
        out_len = ctypes.c_size_t()
        status = lib.t4a_tt_f64_fulltensor(self._ptr, None, 0, ctypes.byref(out_len))
        if status != T4A_SUCCESS:
            raise RuntimeError(f"Failed to query fulltensor length: status = {status}")

        # Get data (row-major from Rust)
        data = (ctypes.c_double * out_len.value)()
        status = lib.t4a_tt_f64_fulltensor(
            self._ptr, data, out_len.value, ctypes.byref(out_len)
        )
        if status != T4A_SUCCESS:
            raise RuntimeError(f"Failed to get fulltensor: status = {status}")

        # Convert to numpy array and reshape
        arr = np.array(data, dtype=np.float64)
        dims = self.site_dims
        return arr.reshape(dims)

    def copy(self) -> "TensorTrainF64":
        """Create a copy (clone) of the tensor train."""
        lib = _get_lib()
        ptr = lib.t4a_tt_f64_clone(self._ptr)
        if ptr is None:
            raise RuntimeError("Failed to clone TensorTrainF64")
        return TensorTrainF64(ptr)

    def __copy__(self) -> "TensorTrainF64":
        return self.copy()

    def __deepcopy__(self, memo) -> "TensorTrainF64":
        return self.copy()


class TensorTrainC64:
    """
    A tensor train (Matrix Product State) with complex128 elements.

    Wraps the Rust `TensorTrain<Complex64>` type via the C API.
    """

    def __init__(self, ptr: ctypes.c_void_p):
        """Initialize from a raw pointer (internal use)."""
        self._ptr = ptr

    def __del__(self):
        """Release the tensor train."""
        if hasattr(self, "_ptr") and self._ptr is not None:
            lib = _get_lib()
            lib.t4a_tt_c64_release(self._ptr)
            self._ptr = None

    @classmethod
    def zeros(cls, site_dims: List[int]) -> "TensorTrainC64":
        """Create a tensor train representing the zero function."""
        lib = _get_lib()
        dims = (ctypes.c_size_t * len(site_dims))(*site_dims)
        ptr = lib.t4a_tt_c64_new_zeros(dims, len(site_dims))
        if ptr is None:
            raise RuntimeError("Failed to create TensorTrainC64")
        return cls(ptr)

    @classmethod
    def constant(cls, site_dims: List[int], value: complex) -> "TensorTrainC64":
        """Create a tensor train representing a constant function."""
        lib = _get_lib()
        dims = (ctypes.c_size_t * len(site_dims))(*site_dims)
        ptr = lib.t4a_tt_c64_new_constant(dims, len(site_dims), value.real, value.imag)
        if ptr is None:
            raise RuntimeError("Failed to create TensorTrainC64")
        return cls(ptr)

    def __len__(self) -> int:
        """Get the number of sites in the tensor train."""
        lib = _get_lib()
        out_len = ctypes.c_size_t()
        status = lib.t4a_tt_c64_len(self._ptr, ctypes.byref(out_len))
        if status != T4A_SUCCESS:
            raise RuntimeError(f"Failed to get length: status = {status}")
        return out_len.value

    @property
    def site_dims(self) -> List[int]:
        """Get the site (physical) dimensions."""
        lib = _get_lib()
        n = len(self)
        out_dims = (ctypes.c_size_t * n)()
        status = lib.t4a_tt_c64_site_dims(self._ptr, out_dims, n)
        if status != T4A_SUCCESS:
            raise RuntimeError(f"Failed to get site_dims: status = {status}")
        return list(out_dims)

    @property
    def link_dims(self) -> List[int]:
        """Get the link (bond) dimensions."""
        lib = _get_lib()
        n = len(self)
        if n <= 1:
            return []
        out_dims = (ctypes.c_size_t * (n - 1))()
        status = lib.t4a_tt_c64_link_dims(self._ptr, out_dims, n - 1)
        if status != T4A_SUCCESS:
            raise RuntimeError(f"Failed to get link_dims: status = {status}")
        return list(out_dims)

    @property
    def rank(self) -> int:
        """Get the maximum bond dimension (rank)."""
        lib = _get_lib()
        out_rank = ctypes.c_size_t()
        status = lib.t4a_tt_c64_rank(self._ptr, ctypes.byref(out_rank))
        if status != T4A_SUCCESS:
            raise RuntimeError(f"Failed to get rank: status = {status}")
        return out_rank.value

    def evaluate(self, indices: List[int]) -> complex:
        """Evaluate the tensor train at a given index set (0-based indices)."""
        lib = _get_lib()
        idx = (ctypes.c_size_t * len(indices))(*indices)
        out_re = ctypes.c_double()
        out_im = ctypes.c_double()
        status = lib.t4a_tt_c64_evaluate(
            self._ptr, idx, len(indices), ctypes.byref(out_re), ctypes.byref(out_im)
        )
        if status != T4A_SUCCESS:
            raise RuntimeError(f"Failed to evaluate: status = {status}")
        return complex(out_re.value, out_im.value)

    def sum(self) -> complex:
        """Compute the sum of all elements."""
        lib = _get_lib()
        out_re = ctypes.c_double()
        out_im = ctypes.c_double()
        status = lib.t4a_tt_c64_sum(
            self._ptr, ctypes.byref(out_re), ctypes.byref(out_im)
        )
        if status != T4A_SUCCESS:
            raise RuntimeError(f"Failed to compute sum: status = {status}")
        return complex(out_re.value, out_im.value)

    def norm(self) -> float:
        """Compute the Frobenius norm."""
        lib = _get_lib()
        out_norm = ctypes.c_double()
        status = lib.t4a_tt_c64_norm(self._ptr, ctypes.byref(out_norm))
        if status != T4A_SUCCESS:
            raise RuntimeError(f"Failed to compute norm: status = {status}")
        return out_norm.value

    def log_norm(self) -> float:
        """Compute the logarithm of the Frobenius norm."""
        lib = _get_lib()
        out_log_norm = ctypes.c_double()
        status = lib.t4a_tt_c64_log_norm(self._ptr, ctypes.byref(out_log_norm))
        if status != T4A_SUCCESS:
            raise RuntimeError(f"Failed to compute log_norm: status = {status}")
        return out_log_norm.value

    def scale(self, factor: complex) -> None:
        """Scale the tensor train in place."""
        lib = _get_lib()
        status = lib.t4a_tt_c64_scale_inplace(self._ptr, factor.real, factor.imag)
        if status != T4A_SUCCESS:
            raise RuntimeError(f"Failed to scale: status = {status}")

    def scaled(self, factor: complex) -> "TensorTrainC64":
        """Create a new tensor train scaled by a factor."""
        lib = _get_lib()
        ptr = lib.t4a_tt_c64_scaled(self._ptr, factor.real, factor.imag)
        if ptr is None:
            raise RuntimeError("Failed to create scaled TensorTrainC64")
        return TensorTrainC64(ptr)

    def to_numpy(self):
        """Convert the tensor train to a NumPy array.

        Raises
        ------
        ImportError
            If NumPy is not installed.
        """
        if not HAS_NUMPY:
            raise ImportError("NumPy is required for to_numpy()")

        lib = _get_lib()

        # Query length
        out_len = ctypes.c_size_t()
        status = lib.t4a_tt_c64_fulltensor(
            self._ptr, None, None, 0, ctypes.byref(out_len)
        )
        if status != T4A_SUCCESS:
            raise RuntimeError(f"Failed to query fulltensor length: status = {status}")

        # Get data
        data_re = (ctypes.c_double * out_len.value)()
        data_im = (ctypes.c_double * out_len.value)()
        status = lib.t4a_tt_c64_fulltensor(
            self._ptr, data_re, data_im, out_len.value, ctypes.byref(out_len)
        )
        if status != T4A_SUCCESS:
            raise RuntimeError(f"Failed to get fulltensor: status = {status}")

        # Convert to numpy
        arr_re = np.array(data_re, dtype=np.float64)
        arr_im = np.array(data_im, dtype=np.float64)
        arr = arr_re + 1j * arr_im
        dims = self.site_dims
        return arr.reshape(dims)

    def copy(self) -> "TensorTrainC64":
        """Create a copy (clone) of the tensor train."""
        lib = _get_lib()
        ptr = lib.t4a_tt_c64_clone(self._ptr)
        if ptr is None:
            raise RuntimeError("Failed to clone TensorTrainC64")
        return TensorTrainC64(ptr)

    def __copy__(self) -> "TensorTrainC64":
        return self.copy()

    def __deepcopy__(self, memo) -> "TensorTrainC64":
        return self.copy()


__all__ = [
    "init_library",
    "TensorTrainF64",
    "TensorTrainC64",
    "T4A_SUCCESS",
    "T4A_NULL_POINTER",
    "T4A_INVALID_ARGUMENT",
    "T4A_BUFFER_TOO_SMALL",
    "T4A_INTERNAL_ERROR",
]
