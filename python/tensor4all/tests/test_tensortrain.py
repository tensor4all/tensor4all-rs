"""Tests for TensorTrain classes."""

import numpy as np
import pytest

from tensor4all.tensortrain import TensorTrainF64, TensorTrainC64


class TestTensorTrainF64Basic:
    """Basic TensorTrainF64 tests."""

    def test_zeros(self):
        """Test creating a zeros tensor train."""
        tt = TensorTrainF64.zeros([2, 3, 2])
        assert len(tt) == 3
        assert tt.site_dims == [2, 3, 2]
        assert tt.sum() == pytest.approx(0.0)

    def test_constant(self):
        """Test creating a constant tensor train."""
        tt = TensorTrainF64.constant([2, 3, 2], 1.0)
        assert len(tt) == 3
        assert tt.site_dims == [2, 3, 2]
        assert tt.sum() == pytest.approx(2 * 3 * 2)  # All elements are 1.0
        assert tt.rank == 1  # Constant has bond dimension 1

    def test_evaluate(self):
        """Test evaluating tensor train at indices."""
        tt = TensorTrainF64.constant([2, 3, 2], 5.0)
        # All elements should be 5.0 (0-indexed)
        assert tt.evaluate([0, 0, 0]) == pytest.approx(5.0)
        assert tt.evaluate([1, 2, 1]) == pytest.approx(5.0)
        assert tt.evaluate([0, 1, 0]) == pytest.approx(5.0)


class TestTensorTrainF64Norm:
    """Norm tests for TensorTrainF64."""

    def test_norm(self):
        """Test computing norm."""
        tt = TensorTrainF64.constant([2, 2, 2], 1.0)
        # ||tt||_F = sqrt(sum of |elements|^2) = sqrt(8 * 1^2) = sqrt(8)
        assert tt.norm() == pytest.approx(np.sqrt(8.0))

    def test_log_norm(self):
        """Test computing log norm."""
        tt = TensorTrainF64.constant([2, 2, 2], 1.0)
        assert tt.log_norm() == pytest.approx(np.log(np.sqrt(8.0)))


class TestTensorTrainF64Clone:
    """Clone tests for TensorTrainF64."""

    def test_clone(self):
        """Test cloning a tensor train."""
        tt = TensorTrainF64.constant([2, 3], 3.0)
        tt2 = tt.clone()
        assert tt2.site_dims == [2, 3]
        assert tt2.sum() == pytest.approx(tt.sum())


class TestTensorTrainF64Scale:
    """Scaling tests for TensorTrainF64."""

    def test_scale_inplace(self):
        """Test in-place scaling."""
        tt = TensorTrainF64.constant([2, 2], 1.0)
        tt.scale(2.0)
        assert tt.sum() == pytest.approx(8.0)  # 4 elements * 2.0

    def test_scaled(self):
        """Test creating scaled copy."""
        tt = TensorTrainF64.constant([2, 2], 1.0)
        tt2 = tt.scaled(3.0)
        assert tt.sum() == pytest.approx(4.0)  # Original unchanged
        assert tt2.sum() == pytest.approx(12.0)  # Scaled copy


class TestTensorTrainF64FullTensor:
    """Full tensor conversion tests."""

    def test_fulltensor(self):
        """Test converting to full tensor."""
        tt = TensorTrainF64.constant([2, 3], 2.0)
        arr = tt.fulltensor()
        assert arr.shape == (2, 3)
        np.testing.assert_array_almost_equal(arr, np.full((2, 3), 2.0))


class TestTensorTrainF64Arithmetic:
    """Arithmetic operation tests for TensorTrainF64."""

    def test_add(self):
        """Test addition."""
        tt1 = TensorTrainF64.constant([2, 2], 1.0)
        tt2 = TensorTrainF64.constant([2, 2], 2.0)
        tt3 = tt1 + tt2
        assert tt3.sum() == pytest.approx(12.0)  # 4 * (1 + 2)

    def test_sub(self):
        """Test subtraction."""
        tt1 = TensorTrainF64.constant([2, 2], 5.0)
        tt2 = TensorTrainF64.constant([2, 2], 2.0)
        tt3 = tt1 - tt2
        assert tt3.sum() == pytest.approx(12.0)  # 4 * (5 - 2)

    def test_negate(self):
        """Test negation."""
        tt = TensorTrainF64.constant([2, 2], 3.0)
        neg_tt = -tt
        assert neg_tt.sum() == pytest.approx(-12.0)

    def test_hadamard(self):
        """Test Hadamard (element-wise) product."""
        tt1 = TensorTrainF64.constant([2, 2], 2.0)
        tt2 = TensorTrainF64.constant([2, 2], 3.0)
        tt3 = tt1 * tt2
        assert tt3.sum() == pytest.approx(24.0)  # 4 * (2 * 3)

    def test_scalar_multiplication(self):
        """Test scalar multiplication."""
        tt = TensorTrainF64.constant([2, 2], 1.0)
        tt2 = tt * 5.0
        tt3 = 5.0 * tt
        assert tt2.sum() == pytest.approx(20.0)
        assert tt3.sum() == pytest.approx(20.0)

    def test_dot(self):
        """Test dot product."""
        tt1 = TensorTrainF64.constant([2, 2], 1.0)
        tt2 = TensorTrainF64.constant([2, 2], 2.0)
        # dot(tt1, tt2) = sum(tt1 .* tt2) = 4 * (1 * 2) = 8
        assert tt1.dot(tt2) == pytest.approx(8.0)

    def test_reverse(self):
        """Test reversing site order."""
        tt = TensorTrainF64.constant([2, 3, 4], 1.0)
        tt_rev = tt.reverse()
        assert tt_rev.site_dims == [4, 3, 2]
        assert tt_rev.sum() == pytest.approx(tt.sum())


class TestTensorTrainF64Compression:
    """Compression tests for TensorTrainF64."""

    def test_compress_inplace(self):
        """Test in-place compression."""
        # Addition increases bond dimension
        tt1 = TensorTrainF64.constant([2, 2], 1.0)
        tt2 = TensorTrainF64.constant([2, 2], 2.0)
        tt3 = tt1 + tt2
        assert tt3.rank == 2  # Bond dim increases after addition

        tt3.compress(tolerance=1e-12)
        assert tt3.rank == 1  # Compressed back to 1
        assert tt3.sum() == pytest.approx(12.0)  # Values preserved

    def test_compressed(self):
        """Test creating compressed copy."""
        tt1 = TensorTrainF64.constant([2, 2], 1.0)
        tt2 = TensorTrainF64.constant([2, 2], 2.0)
        tt3 = tt1 + tt2

        tt4 = tt3.compressed(tolerance=1e-12)
        assert tt4.rank == 1
        assert tt3.rank == 2  # Original unchanged


class TestTensorTrainC64Basic:
    """Basic TensorTrainC64 tests."""

    def test_zeros(self):
        """Test creating a zeros tensor train."""
        tt = TensorTrainC64.zeros([2, 3, 2])
        assert len(tt) == 3
        assert tt.site_dims == [2, 3, 2]
        assert tt.sum() == pytest.approx(0.0 + 0.0j)

    def test_constant(self):
        """Test creating a constant tensor train."""
        tt = TensorTrainC64.constant([2, 3, 2], 1.0 + 2.0j)
        assert len(tt) == 3
        assert tt.site_dims == [2, 3, 2]
        assert tt.sum() == pytest.approx((1.0 + 2.0j) * 12)

    def test_evaluate(self):
        """Test evaluating tensor train at indices."""
        val = 3.0 + 4.0j
        tt = TensorTrainC64.constant([2, 2], val)
        assert tt.evaluate([0, 0]) == pytest.approx(val)
        assert tt.evaluate([1, 1]) == pytest.approx(val)


class TestTensorTrainC64Arithmetic:
    """Arithmetic operation tests for TensorTrainC64."""

    def test_add(self):
        """Test addition."""
        tt1 = TensorTrainC64.constant([2, 2], 1.0 + 1.0j)
        tt2 = TensorTrainC64.constant([2, 2], 2.0 + 0.0j)
        tt3 = tt1 + tt2
        assert tt3.sum() == pytest.approx(4 * (3.0 + 1.0j))

    def test_dot(self):
        """Test dot product."""
        tt1 = TensorTrainC64.constant([2, 2], 1.0 + 1.0j)
        tt2 = TensorTrainC64.constant([2, 2], 1.0 - 1.0j)
        # dot computes sum(tt1 .* tt2) element-wise
        # (1+i)(1-i) = 1 - i^2 = 2, so 4 * 2 = 8
        d = tt1.dot(tt2)
        assert d.real == pytest.approx(8.0, abs=1e-10)


class TestTensorTrainC64FullTensor:
    """Full tensor conversion tests for complex."""

    def test_fulltensor(self):
        """Test converting to full tensor."""
        val = 1.0 + 1.0j
        tt = TensorTrainC64.constant([2, 2], val)
        arr = tt.fulltensor()
        assert arr.shape == (2, 2)
        np.testing.assert_array_almost_equal(arr, np.full((2, 2), val))
