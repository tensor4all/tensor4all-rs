"""Tests for SimpleTensorTrain."""

import pytest
import numpy as np

from tensor4all import SimpleTensorTrain


class TestSimpleTensorTrain:
    """Tests for SimpleTensorTrain class."""

    def test_constant_creation(self):
        """Test creating a constant tensor train."""
        tt = SimpleTensorTrain.constant([2, 3, 4], value=1.5)

        assert tt.n_sites == 3
        assert len(tt) == 3
        assert tt.site_dims == [2, 3, 4]
        assert tt.rank == 1  # Constant has rank 1

    def test_zeros_creation(self):
        """Test creating a zero tensor train."""
        tt = SimpleTensorTrain.zeros([2, 3])

        assert tt.n_sites == 2
        assert tt.site_dims == [2, 3]
        assert tt.sum() == 0.0

    def test_sum(self):
        """Test sum computation."""
        tt = SimpleTensorTrain.constant([2, 3, 4], value=1.5)

        # Sum should be value * product of dimensions
        expected_sum = 1.5 * 2 * 3 * 4
        assert abs(tt.sum() - expected_sum) < 1e-10

    def test_evaluate(self):
        """Test tensor evaluation."""
        tt = SimpleTensorTrain.constant([2, 3, 4], value=2.0)

        # All elements should be 2.0
        assert abs(tt.evaluate([0, 0, 0]) - 2.0) < 1e-10
        assert abs(tt.evaluate([1, 2, 3]) - 2.0) < 1e-10

    def test_callable_interface(self):
        """Test callable interface."""
        tt = SimpleTensorTrain.constant([2, 3, 4], value=2.0)

        # Test calling with multiple arguments
        assert abs(tt(0, 1, 2) - 2.0) < 1e-10

    def test_copy(self):
        """Test copying tensor train."""
        tt1 = SimpleTensorTrain.constant([2, 3], value=3.0)
        tt2 = tt1.copy()

        assert tt2.n_sites == tt1.n_sites
        assert tt2.site_dims == tt1.site_dims
        assert abs(tt2.sum() - tt1.sum()) < 1e-10

    def test_link_dims(self):
        """Test link dimensions."""
        tt = SimpleTensorTrain.constant([2, 3, 4], value=1.0)

        ldims = tt.link_dims
        assert len(ldims) == 2  # n_sites - 1

        # For rank-1 constant, link dims should all be 1
        assert all(d == 1 for d in ldims)

    def test_site_tensor(self):
        """Test getting site tensors."""
        tt = SimpleTensorTrain.constant([2, 3], value=1.0)

        # Get site tensor at site 0
        t0 = tt.site_tensor(0)
        assert t0.shape[0] == 1  # left dim
        assert t0.shape[1] == 2  # site dim
        assert t0.shape[2] == 1  # right dim

        # Get site tensor at site 1
        t1 = tt.site_tensor(1)
        assert t1.shape[0] == 1  # left dim
        assert t1.shape[1] == 3  # site dim
        assert t1.shape[2] == 1  # right dim

    def test_repr(self):
        """Test string representation."""
        tt = SimpleTensorTrain.constant([2, 3, 4], value=1.0)

        s = repr(tt)
        assert "SimpleTensorTrain" in s
        assert "3" in s  # n_sites

    def test_norm(self):
        """Test Frobenius norm."""
        tt = SimpleTensorTrain.constant([2, 2], value=1.0)

        # Norm of all-ones tensor with 4 elements
        expected_norm = np.sqrt(4.0)
        assert abs(tt.norm() - expected_norm) < 1e-10
