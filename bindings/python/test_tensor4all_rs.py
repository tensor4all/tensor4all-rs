"""
Tests for tensor4all_rs Python bindings.

Run with: python -m pytest test_tensor4all_rs.py -v
"""

import os
import platform
import sys
from pathlib import Path

import numpy as np
import pytest

# Add the bindings directory to the path
sys.path.insert(0, str(Path(__file__).parent))

import tensor4all_rs as t4a

# Find library path
if platform.system() == "Darwin":
    LIBNAME = "libtensor4all_capi.dylib"
elif platform.system() == "Linux":
    LIBNAME = "libtensor4all_capi.so"
elif platform.system() == "Windows":
    LIBNAME = "tensor4all_capi.dll"
else:
    raise RuntimeError(f"Unsupported platform: {platform.system()}")

LIBPATH = Path(__file__).parent.parent.parent / "target" / "release" / LIBNAME


@pytest.fixture(scope="module", autouse=True)
def init_lib():
    """Initialize the library before running tests."""
    if not LIBPATH.exists():
        pytest.skip(
            f"Library not found at {LIBPATH}. "
            "Build with: cargo build --release -p tensor4all-capi"
        )
    t4a.init_library(str(LIBPATH))


class TestTensorTrainF64:
    """Tests for TensorTrainF64."""

    def test_zeros(self):
        """Test creating a zero tensor train."""
        tt = t4a.TensorTrainF64.zeros([2, 3, 2])
        assert len(tt) == 3
        assert tt.site_dims == [2, 3, 2]
        assert tt.rank == 1

    def test_constant(self):
        """Test creating a constant tensor train."""
        tt = t4a.TensorTrainF64.constant([2, 2], 5.0)
        assert len(tt) == 2
        assert tt.site_dims == [2, 2]

        # Sum should be 5.0 * 2 * 2 = 20.0
        assert abs(tt.sum() - 20.0) < 1e-10

        # Evaluate at [0, 1] (0-based)
        assert abs(tt.evaluate([0, 1]) - 5.0) < 1e-10

    def test_scale(self):
        """Test scaling operations."""
        tt = t4a.TensorTrainF64.constant([2, 2], 1.0)

        # Immutable scale
        tt2 = tt.scaled(3.0)
        assert abs(tt2.sum() - 12.0) < 1e-10
        assert abs(tt.sum() - 4.0) < 1e-10  # Original unchanged

        # In-place scale
        tt.scale(2.0)
        assert abs(tt.sum() - 8.0) < 1e-10

    def test_norm(self):
        """Test norm computation."""
        tt = t4a.TensorTrainF64.constant([2, 3], 2.0)

        norm_val = tt.norm()
        # norm = sqrt(sum of squares) = sqrt(6 * 4) = sqrt(24)
        assert abs(norm_val - np.sqrt(24.0)) < 1e-10

        log_norm_val = tt.log_norm()
        assert abs(log_norm_val - np.log(norm_val)) < 1e-10

    def test_to_numpy(self):
        """Test conversion to NumPy array."""
        tt = t4a.TensorTrainF64.constant([2, 3], 5.0)
        arr = tt.to_numpy()

        assert arr.shape == (2, 3)
        assert np.allclose(arr, 5.0)

    def test_copy(self):
        """Test copying."""
        tt = t4a.TensorTrainF64.constant([2, 2], 3.0)
        tt2 = tt.copy()

        assert abs(tt2.sum() - tt.sum()) < 1e-10

        # Modify original, copy should be unchanged
        tt.scale(2.0)
        assert abs(tt.sum() - 24.0) < 1e-10
        assert abs(tt2.sum() - 12.0) < 1e-10


class TestTensorTrainC64:
    """Tests for TensorTrainC64."""

    def test_zeros(self):
        """Test creating a zero complex tensor train."""
        tt = t4a.TensorTrainC64.zeros([2, 3, 2])
        assert len(tt) == 3
        assert tt.site_dims == [2, 3, 2]

    def test_constant(self):
        """Test creating a constant complex tensor train."""
        tt = t4a.TensorTrainC64.constant([2, 2], 3.0 + 4.0j)
        assert len(tt) == 2

        val = tt.evaluate([0, 1])
        assert abs(val.real - 3.0) < 1e-10
        assert abs(val.imag - 4.0) < 1e-10

    def test_scale(self):
        """Test scaling by complex number."""
        tt = t4a.TensorTrainC64.constant([2, 2], 1.0 + 0.0j)

        # Scale by imaginary unit
        tt2 = tt.scaled(1.0j)
        s = tt2.sum()
        assert abs(s.real) < 1e-10
        assert abs(s.imag - 4.0) < 1e-10

    def test_to_numpy(self):
        """Test conversion to NumPy array."""
        tt = t4a.TensorTrainC64.constant([2, 3], 1.0 + 2.0j)
        arr = tt.to_numpy()

        assert arr.shape == (2, 3)
        assert np.allclose(arr, 1.0 + 2.0j)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
