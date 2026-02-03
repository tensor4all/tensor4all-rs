"""Tests for TreeTensorNetwork (TreeTN) Python bindings."""

import numpy as np
import pytest

from tensor4all import Index, Tensor, TreeTensorNetwork, MPS, MPO
from tensor4all import ttn_inner, ttn_contract


def _make_two_site_ttn():
    """Helper: create a 2-site TTN with site dims (2, 2) and bond dim 3."""
    s0 = Index(2)
    l01 = Index(3)
    s1 = Index(2)

    data0 = np.arange(1, 7, dtype=np.float64).reshape(2, 3)
    data1 = np.arange(1, 7, dtype=np.float64).reshape(3, 2)

    t0 = Tensor([s0, l01], data0)
    t1 = Tensor([l01, s1], data1)
    ttn = TreeTensorNetwork([t0, t1])
    return ttn, [s0, s1]


class TestTreeTNLifecycle:
    def test_create_from_tensors(self):
        ttn, _ = _make_two_site_ttn()
        assert len(ttn) == 2

    def test_copy(self):
        ttn, _ = _make_two_site_ttn()
        ttn2 = ttn.copy()
        assert len(ttn2) == 2
        assert ttn2.maxbonddim == ttn.maxbonddim

    def test_repr(self):
        ttn, _ = _make_two_site_ttn()
        r = repr(ttn)
        assert "TreeTensorNetwork" in r
        assert "nv=2" in r


class TestTreeTNAccessors:
    def test_num_vertices(self):
        ttn, _ = _make_two_site_ttn()
        assert ttn.num_vertices == 2

    def test_num_edges(self):
        ttn, _ = _make_two_site_ttn()
        assert ttn.num_edges == 1

    def test_bond_dims(self):
        ttn, _ = _make_two_site_ttn()
        assert ttn.bond_dims == [3]

    def test_maxbonddim(self):
        ttn, _ = _make_two_site_ttn()
        assert ttn.maxbonddim == 3

    def test_getitem(self):
        ttn, _ = _make_two_site_ttn()
        t0 = ttn[0]
        assert t0.rank == 2
        t1 = ttn[1]
        assert t1.rank == 2

    def test_linkind(self):
        ttn, _ = _make_two_site_ttn()
        link = ttn.linkind(0)
        assert link is not None
        assert link.dim == 3

    def test_linkdim(self):
        ttn, _ = _make_two_site_ttn()
        assert ttn.linkdim(0) == 3


class TestTreeTNOrthogonality:
    def test_orthogonalize(self):
        ttn, _ = _make_two_site_ttn()
        ttn.orthogonalize(0)
        assert ttn.canonical_form == 0  # Unitary

    def test_orthogonalize_site1(self):
        ttn, _ = _make_two_site_ttn()
        ttn.orthogonalize(1)
        assert ttn.canonical_form == 0  # Unitary

    def test_canonical_form_none(self):
        ttn, _ = _make_two_site_ttn()
        assert ttn.canonical_form is None

    def test_canonical_form_unitary(self):
        ttn, _ = _make_two_site_ttn()
        ttn.orthogonalize(0, form="unitary")
        assert ttn.canonical_form == 0  # Unitary

    def test_canonical_form_lu(self):
        ttn, _ = _make_two_site_ttn()
        ttn.orthogonalize(0, form="lu")
        assert ttn.canonical_form == 1  # LU

    def test_canonical_form_ci(self):
        ttn, _ = _make_two_site_ttn()
        ttn.orthogonalize(0, form="ci")
        assert ttn.canonical_form == 2  # CI


class TestTreeTNOperations:
    def test_truncate_with_maxdim(self):
        ttn, _ = _make_two_site_ttn()
        ttn.truncate(maxdim=1)
        assert ttn.maxbonddim == 1

    def test_truncate_with_cutoff(self):
        ttn, _ = _make_two_site_ttn()
        ttn.truncate(cutoff=1e-10)
        # Should not fail; bond dim may or may not change
        assert ttn.maxbonddim >= 1

    def test_truncate_with_rtol(self):
        ttn, _ = _make_two_site_ttn()
        ttn.truncate(rtol=1e-5)
        assert ttn.maxbonddim >= 1

    def test_norm(self):
        ttn, _ = _make_two_site_ttn()
        n = ttn.norm()
        assert n > 0

    def test_inner(self):
        ttn, _ = _make_two_site_ttn()
        # <ttn|ttn> should equal norm^2
        ip = ttn_inner(ttn, ttn)
        n = ttn.norm()
        assert abs(ip.real - n**2) < 1e-10
        assert abs(ip.imag) < 1e-10

    def test_inner_orthogonal(self):
        """Inner product of orthogonal vectors should be zero."""
        s0 = Index(2)

        data_a = np.array([1.0, 0.0])
        data_b = np.array([0.0, 1.0])

        t0 = Tensor([s0], data_a)
        t1 = Tensor([s0], data_b)

        ttn0 = TreeTensorNetwork([t0])
        ttn1 = TreeTensorNetwork([t1])

        ip = ttn_inner(ttn0, ttn1)
        assert abs(ip) < 1e-10


class TestTreeTNContract:
    def test_contract_zipup(self):
        s0 = Index(2)
        data = np.array([1.0, 2.0])

        t0 = Tensor([s0], data)
        t1 = Tensor([s0], data)

        ttn0 = TreeTensorNetwork([t0])
        ttn1 = TreeTensorNetwork([t1])

        result = ttn_contract(ttn0, ttn1, method="zipup")
        assert len(result) == 1

    def test_contract_with_cutoff(self):
        s0 = Index(2)
        data = np.array([1.0, 2.0])

        t0 = Tensor([s0], data)
        t1 = Tensor([s0], data)

        ttn0 = TreeTensorNetwork([t0])
        ttn1 = TreeTensorNetwork([t1])

        result = ttn_contract(ttn0, ttn1, cutoff=1e-10)
        assert len(result) == 1

    def test_contract_invalid_method(self):
        ttn, _ = _make_two_site_ttn()
        with pytest.raises(ValueError, match="Unknown contract method"):
            ttn_contract(ttn, ttn, method="invalid")


class TestTreeTNAdd:
    def test_add(self):
        ttn1, _ = _make_two_site_ttn()
        ttn2, _ = _make_two_site_ttn()

        result = ttn1 + ttn2
        assert len(result) == 2
        # Bond dim should be sum (3 + 3 = 6)
        assert result.maxbonddim == 6


class TestTreeTNToDense:
    def test_to_dense(self):
        ttn, _ = _make_two_site_ttn()
        dense = ttn.to_dense()
        assert dense.rank == 2


class TestTypeAliases:
    def test_mps_is_treetn(self):
        assert MPS is TreeTensorNetwork

    def test_mpo_is_treetn(self):
        assert MPO is TreeTensorNetwork
