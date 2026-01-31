"""Tests for TensorTrain (ITensorLike) Python bindings."""

import numpy as np
import pytest

from tensor4all import Index, Tensor, TensorTrain, MPS, MPO


def _make_two_site_tt():
    """Helper: create a 2-site tensor train with site dims (2, 2) and bond dim 3."""
    s0 = Index(2)
    l01 = Index(3)
    s1 = Index(2)

    data0 = np.arange(1, 7, dtype=np.float64).reshape(2, 3)
    data1 = np.arange(1, 7, dtype=np.float64).reshape(3, 2)

    t0 = Tensor([s0, l01], data0)
    t1 = Tensor([l01, s1], data1)
    tt = TensorTrain([t0, t1])
    return tt, [s0, s1]


class TestTensorTrainLifecycle:
    def test_create_empty(self):
        tt = TensorTrain([])
        assert len(tt) == 0
        assert tt.is_empty

    def test_create_from_tensors(self):
        tt, _ = _make_two_site_tt()
        assert len(tt) == 2
        assert not tt.is_empty

    def test_copy(self):
        tt, _ = _make_two_site_tt()
        tt2 = tt.copy()
        assert len(tt2) == 2
        assert tt2.maxbonddim == tt.maxbonddim

    def test_repr(self):
        tt, _ = _make_two_site_tt()
        r = repr(tt)
        assert "TensorTrain" in r
        assert "sites=2" in r


class TestTensorTrainAccessors:
    def test_bond_dims(self):
        tt, _ = _make_two_site_tt()
        assert tt.bond_dims == [3]

    def test_maxbonddim(self):
        tt, _ = _make_two_site_tt()
        assert tt.maxbonddim == 3

    def test_getitem(self):
        tt, _ = _make_two_site_tt()
        t0 = tt[0]
        assert t0.rank == 2
        t1 = tt[1]
        assert t1.rank == 2

    def test_getitem_negative_index(self):
        tt, _ = _make_two_site_tt()
        t_last = tt[-1]
        assert t_last.rank == 2

    def test_getitem_out_of_bounds(self):
        tt, _ = _make_two_site_tt()
        with pytest.raises(IndexError):
            tt[5]

    def test_linkind(self):
        tt, _ = _make_two_site_tt()
        link = tt.linkind(0)
        assert link is not None
        assert link.dim == 3


class TestTensorTrainOrthogonality:
    def test_orthogonalize(self):
        tt, _ = _make_two_site_tt()
        tt.orthogonalize(0)
        assert tt.is_ortho
        assert tt.orthocenter == 0

    def test_orthogonalize_site1(self):
        tt, _ = _make_two_site_tt()
        tt.orthogonalize(1)
        assert tt.is_ortho
        assert tt.orthocenter == 1

    def test_canonical_form(self):
        tt, _ = _make_two_site_tt()
        assert tt.canonical_form is None
        tt.orthogonalize(0, form="unitary")
        assert tt.canonical_form == 0  # Unitary


class TestTensorTrainOperations:
    def test_truncate_with_maxdim(self):
        tt, _ = _make_two_site_tt()
        tt.truncate(maxdim=1)
        assert tt.maxbonddim == 1

    def test_truncate_with_cutoff(self):
        tt, _ = _make_two_site_tt()
        tt.truncate(cutoff=1e-10)
        # Should not fail; bond dim may or may not change
        assert tt.maxbonddim >= 1

    def test_truncate_with_rtol(self):
        tt, _ = _make_two_site_tt()
        tt.truncate(rtol=1e-5)
        assert tt.maxbonddim >= 1

    def test_norm(self):
        tt, _ = _make_two_site_tt()
        n = tt.norm()
        assert n > 0

    def test_inner(self):
        tt, _ = _make_two_site_tt()
        # <tt|tt> should equal norm^2
        ip = tt.inner(tt)
        n = tt.norm()
        assert abs(ip.real - n**2) < 1e-10
        assert abs(ip.imag) < 1e-10

    def test_inner_orthogonal(self):
        """Inner product of orthogonal vectors should be zero."""
        s0 = Index(2)

        data_a = np.array([1.0, 0.0])
        data_b = np.array([0.0, 1.0])

        t0 = Tensor([s0], data_a)
        t1 = Tensor([s0], data_b)

        tt0 = TensorTrain([t0])
        tt1 = TensorTrain([t1])

        ip = tt0.inner(tt1)
        assert abs(ip) < 1e-10


class TestTensorTrainContract:
    def test_contract_zipup(self):
        s0 = Index(2)
        data = np.array([1.0, 2.0])

        t0 = Tensor([s0], data)
        t1 = Tensor([s0], data)

        tt0 = TensorTrain([t0])
        tt1 = TensorTrain([t1])

        result = tt0.contract(tt1, method="zipup")
        assert len(result) == 1

    def test_contract_with_cutoff(self):
        s0 = Index(2)
        data = np.array([1.0, 2.0])

        t0 = Tensor([s0], data)
        t1 = Tensor([s0], data)

        tt0 = TensorTrain([t0])
        tt1 = TensorTrain([t1])

        result = tt0.contract(tt1, cutoff=1e-10)
        assert len(result) == 1

    def test_contract_invalid_method(self):
        tt, _ = _make_two_site_tt()
        with pytest.raises(ValueError, match="Unknown contract method"):
            tt.contract(tt, method="invalid")


class TestTensorTrainAdd:
    def test_add(self):
        tt1, _ = _make_two_site_tt()
        tt2, _ = _make_two_site_tt()

        result = tt1.add(tt2)
        assert len(result) == 2
        # Bond dim should be sum (3 + 3 = 6)
        assert result.maxbonddim == 6

    def test_add_operator(self):
        tt1, _ = _make_two_site_tt()
        tt2, _ = _make_two_site_tt()

        result = tt1 + tt2
        assert len(result) == 2
        assert result.maxbonddim == 6


class TestTensorTrainToDense:
    def test_to_dense(self):
        tt, _ = _make_two_site_tt()
        dense = tt.to_dense()
        assert dense.rank == 2


class TestTypeAliases:
    def test_mps_is_tensortrain(self):
        assert MPS is TensorTrain

    def test_mpo_is_tensortrain(self):
        assert MPO is TensorTrain
