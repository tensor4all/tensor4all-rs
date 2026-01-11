"""Tests for TensorCI2 and crossinterpolate2."""

import pytest

from tensor4all import TensorCI2, crossinterpolate2, SimpleTensorTrain


class TestTensorCI2:
    """Tests for TensorCI2 class."""

    def test_creation(self):
        """Test creating a TensorCI2 object."""
        tci = TensorCI2([2, 3, 4])

        assert tci.n_sites == 3
        assert len(tci) == 3
        assert tci.rank == 0  # Empty TCI has rank 0

    def test_repr(self):
        """Test string representation."""
        tci = TensorCI2([2, 3])

        s = repr(tci)
        assert "TensorCI2" in s


class TestCrossinterpolate2:
    """Tests for crossinterpolate2 function."""

    def test_constant_function_2site(self):
        """Test interpolating a constant function with 2 sites."""
        def f(i, j):
            return 1.0

        tt, err = crossinterpolate2(f, [3, 4], tolerance=1e-10)

        assert tt.n_sites == 2
        assert tt.rank == 1  # Constant has rank 1

        # Sum should be 1.0 * 3 * 4 = 12.0
        assert abs(tt.sum() - 12.0) < 1e-8

    def test_product_function_2site(self):
        """Test interpolating a product function (rank-1)."""
        def f(i, j):
            return float((1 + i) * (1 + j))

        tt, err = crossinterpolate2(f, [3, 4], tolerance=1e-10)

        assert abs(tt(0, 0) - 1.0) < 1e-10
        assert abs(tt(1, 2) - 6.0) < 1e-10
        assert abs(tt(2, 3) - 12.0) < 1e-10

        # This is a rank-1 function, so TT should capture it exactly
        assert tt.rank == 1

    def test_with_initial_pivots(self):
        """Test interpolation with custom initial pivots."""
        def f(i, j):
            return float((1 + i) * (2 + j))

        tt, err = crossinterpolate2(
            f, [3, 4],
            initial_pivots=[[1, 1]],
            tolerance=1e-10
        )

        assert abs(tt(0, 0) - 2.0) < 1e-10  # (0+1) * (0+2) = 2
        assert abs(tt(1, 2) - 8.0) < 1e-10  # (1+1) * (2+2) = 8

    def test_3site_constant(self):
        """Test 3-site constant function."""
        def f(i, j, k):
            return 1.0

        tt, err = crossinterpolate2(f, [2, 2, 2], tolerance=1e-10)

        assert tt.n_sites == 3
        assert tt.rank == 1
        assert abs(tt.sum() - 8.0) < 1e-8  # 2^3 = 8
        assert abs(tt(0, 0, 0) - 1.0) < 1e-10

    def test_4site_product(self):
        """Test 4-site product function (rank-1)."""
        def f(i, j, k, l):
            return float((1 + i) * (1 + j) * (1 + k) * (1 + l))

        tt, err = crossinterpolate2(f, [2, 2, 2, 2], tolerance=1e-10)

        assert tt.n_sites == 4
        assert tt.rank == 1  # Product is rank-1
        assert abs(tt(0, 0, 0, 0) - 1.0) < 1e-10
        assert abs(tt(1, 1, 1, 1) - 16.0) < 1e-10

    def test_5site_constant(self):
        """Test 5-site constant function."""
        def f(*args):
            return 2.5

        tt, err = crossinterpolate2(f, [2, 2, 2, 2, 2], tolerance=1e-10)

        assert tt.n_sites == 5
        assert tt.rank == 1
        assert abs(tt.sum() - 80.0) < 1e-8  # 2.5 * 2^5 = 80

    def test_returns_simple_tensor_train(self):
        """Test that crossinterpolate2 returns SimpleTensorTrain."""
        def f(i, j):
            return 1.0

        tt, err = crossinterpolate2(f, [2, 2], tolerance=1e-10)

        assert isinstance(tt, SimpleTensorTrain)

    def test_invalid_local_dims(self):
        """Test that single-site raises error."""
        def f(i):
            return float(i)

        with pytest.raises(ValueError):
            crossinterpolate2(f, [2], tolerance=1e-10)
