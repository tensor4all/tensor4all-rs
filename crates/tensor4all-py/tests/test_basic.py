"""Tests for the tensor4all-rs PyO3 bindings.

Run with the extension installed in the active environment::

    maturin develop
    pytest
"""

import numpy as np
import pytest

import tensor4all as t4a


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


def test_index_properties():
    i = t4a.Index(3, tags=["Site", "Link"], plev=2)
    assert i.dim == 3
    assert i.plev == 2
    assert sorted(i.tags) == ["Link", "Site"]


def test_index_identity_and_equality():
    i = t4a.Index(2)
    assert i == i
    assert i.same_id(i)
    assert i == i.noprime()

    primed = i.prime()
    assert primed.plev == 1
    assert primed.same_id(i)
    assert primed != i
    assert i == primed.noprime()

    tagged = t4a.Index(2, plev=1)
    assert not i.same_id(tagged)
    assert tagged != i


def test_index_shared_object_is_contractable():
    i = t4a.Index(2)
    j = t4a.Index(3)
    left = t4a.Tensor([i, j], np.arange(6, dtype=float).reshape(2, 3))
    right = t4a.Tensor([j, i], np.arange(6, dtype=float).reshape(3, 2))
    # `j` and `i` are the same Python objects, so the ids match and both axes
    # are summed, leaving a scalar.
    assert left.contract(right).dims == []


def test_index_invalid_inputs():
    with pytest.raises(ValueError):
        t4a.Index(0)
    with pytest.raises(TypeError):
        t4a.Index(2) < t4a.Index(2)

# ---------------------------------------------------------------------------
# Tensor construction and NumPy interchange
# ---------------------------------------------------------------------------


def test_tensor_roundtrip_c_and_f_layout():
    i, j = t4a.Index(2), t4a.Index(3)
    data = np.arange(6, dtype=np.float64).reshape(2, 3)
    for array in (data, np.asfortranarray(data)):
        assert np.array_equal(t4a.Tensor([i, j], array).to_numpy(), data)


def test_tensor_roundtrip_non_contiguous_and_negative_strides():
    i, j = t4a.Index(2), t4a.Index(3)
    base = np.arange(12, dtype=np.float64).reshape(4, 3)
    strided = base[::2]  # shape (2, 3), non-contiguous
    assert not strided.flags["C_CONTIGUOUS"]
    assert np.array_equal(t4a.Tensor([i, j], strided).to_numpy(), strided)

    reversed_data = np.arange(6, dtype=np.float64).reshape(2, 3)[::-1, ::-1]
    assert np.array_equal(
        t4a.Tensor([i, j], reversed_data).to_numpy(), reversed_data
    )


def test_tensor_roundtrip_rank_three_and_non_square():
    a, b, c = t4a.Index(2), t4a.Index(1), t4a.Index(3)
    data = np.arange(6, dtype=np.float64).reshape(2, 1, 3)
    tensor = t4a.Tensor([a, b, c], data)
    assert tensor.dims == [2, 1, 3]
    assert np.array_equal(tensor.to_numpy(), data)


def test_tensor_complex_roundtrip_and_contraction():
    i, j, k = t4a.Index(2), t4a.Index(3), t4a.Index(4)
    left = np.arange(6, dtype=np.complex128).reshape(2, 3) + 1j
    right = np.arange(12, dtype=np.complex128).reshape(3, 4) - 0.5j
    product = t4a.Tensor([i, j], left).contract(t4a.Tensor([j, k], right))
    assert product.to_numpy().dtype == np.complex128
    assert np.allclose(product.to_numpy(), left @ right)


def test_tensor_mixed_dtype_contracts_to_complex():
    i, j, k = t4a.Index(2), t4a.Index(3), t4a.Index(4)
    real = np.ones((2, 3))
    imag = np.ones((3, 4), dtype=np.complex128) * 2j
    product = t4a.Tensor([i, j], real).contract(t4a.Tensor([j, k], imag))
    assert product.to_numpy().dtype == np.complex128
    assert np.allclose(product.to_numpy(), real @ imag)


def test_tensor_rank_zero_roundtrip():
    scalar = t4a.Tensor([], np.array(3.5))
    assert scalar.dims == []
    value = scalar.to_numpy()
    assert value.shape == ()
    assert value == 3.5


def test_tensor_rejects_non_array_input():
    with pytest.raises(TypeError, match="numpy array"):
        t4a.Tensor([t4a.Index(2)], [1.0, 2.0])


def test_tensor_storage_is_independent_of_input_and_output():
    i, j = t4a.Index(2), t4a.Index(3)
    source = np.zeros((2, 3))
    tensor = t4a.Tensor([i, j], source)
    source[0, 0] = 99.0
    assert tensor.to_numpy()[0, 0] == 0.0

    first = tensor.to_numpy()
    first[0, 0] = 42.0
    assert tensor.to_numpy()[0, 0] == 0.0
    assert tensor.to_numpy()[0, 0] != first[0, 0]


@pytest.mark.parametrize("dtype", [np.float32, np.int64, np.complex64])
def test_tensor_unsupported_dtype(dtype):
    i = t4a.Index(2)
    with pytest.raises(TypeError):
        t4a.Tensor([i], np.zeros(2, dtype=dtype))


def test_tensor_shape_mismatch():
    i, j = t4a.Index(2), t4a.Index(3)
    with pytest.raises(ValueError):
        t4a.Tensor([i, j], np.zeros((3, 2)))


def test_tensor_rejects_duplicate_indices():
    i = t4a.Index(2)
    with pytest.raises(ValueError):
        t4a.Tensor([i, i], np.zeros((2, 2)))


def test_tensor_report_preserves_index_identity():
    i = t4a.Index(3, tags=["Site"])
    tensor = t4a.Tensor([i], np.zeros(3))
    assert tensor.indices[0] == i
    assert not tensor.indices[0].same_id(t4a.Index(3, tags=["Site"]))


def test_tensor_contraction_requires_shared_index():
    i, j = t4a.Index(2), t4a.Index(3)
    other = t4a.Index(3)
    with pytest.raises(ValueError):
        t4a.Tensor([i, j], np.zeros((2, 3))).contract(
            t4a.Tensor([other], np.zeros(3))
        )


def test_tensor_outer_product_is_rejected():
    """Contracting unconnected tensors must not silently become an outer product."""
    i, j = t4a.Index(2), t4a.Index(3)
    with pytest.raises(ValueError, match="[Dd]isconnected"):
        t4a.Tensor([i], np.ones(2)).contract(t4a.Tensor([j], np.ones(3)))


# ---------------------------------------------------------------------------
# TreeTensorNetwork
# ---------------------------------------------------------------------------


def chain_network(seed=0):
    """Two-node chain: MPS-like network with a single bond."""
    rng = np.random.default_rng(seed)
    s0, s1, bond = t4a.Index(2), t4a.Index(2), t4a.Index(3)
    left = rng.standard_normal((2, 3))
    right = rng.standard_normal((3, 2))
    network = t4a.TreeTensorNetwork(
        [t4a.Tensor([s0, bond], left), t4a.Tensor([bond, s1], right)]
    )
    return network, left, right


def test_tree_network_chain_structure_and_dense_contraction():
    network, left, right = chain_network()
    assert network.num_vertices == 2
    assert network.num_edges == 1
    assert network.node_names() == [0, 1]
    assert np.allclose(network.contract_to_tensor().to_numpy(), left @ right)


def aligned(values, expected_axes, result):
    """Transpose `values` (axes `expected_axes`) into the result's axis order."""
    order = [expected_axes.index(index) for index in result.indices]
    return np.transpose(values, order)


def test_tree_network_three_node_star_is_not_a_chain():
    """A Y-shaped tree: center tensor with two branching leaves."""
    center, s0, s1 = t4a.Index(2), t4a.Index(2), t4a.Index(3)
    leaf0, leaf1 = t4a.Index(4), t4a.Index(5)
    root = np.arange(2 * 2 * 3, dtype=np.float64).reshape(2, 2, 3)
    branch0 = np.arange(2 * 4, dtype=np.float64).reshape(2, 4)
    branch1 = np.arange(3 * 5, dtype=np.float64).reshape(3, 5)
    network = t4a.TreeTensorNetwork(
        [
            t4a.Tensor([center, s0, s1], root),
            t4a.Tensor([s0, leaf0], branch0),
            t4a.Tensor([s1, leaf1], branch1),
        ]
    )
    assert network.num_vertices == 3
    assert network.num_edges == 2

    dense = network.contract_to_tensor()
    reference = np.einsum("abc,bd,ce->ade", root, branch0, branch1)
    assert sorted(dense.dims) == [2, 4, 5]
    assert np.allclose(
        dense.to_numpy(), aligned(reference, [center, leaf0, leaf1], dense)
    )


def test_tree_network_contract_naive_matches_known_dense_result():
    """Contract an MPS-like chain with an MPO-like chain."""
    rng = np.random.default_rng(1)
    s0, s1, bond = t4a.Index(2), t4a.Index(2), t4a.Index(3)
    state0 = rng.standard_normal((2, 3))
    state1 = rng.standard_normal((3, 2))
    state = t4a.TreeTensorNetwork(
        [t4a.Tensor([s0, bond], state0), t4a.Tensor([bond, s1], state1)]
    )

    link = t4a.Index(2)
    s0_out, s1_out = t4a.Index(2), t4a.Index(2)
    operator0 = rng.standard_normal((2, 2, 2))
    operator1 = rng.standard_normal((2, 2, 2))
    operator = t4a.TreeTensorNetwork(
        [
            t4a.Tensor([s0, link, s0_out], operator0),
            t4a.Tensor([link, s1, s1_out], operator1),
        ]
    )

    result = state.contract(operator)
    dense = result.contract_to_tensor()
    reference = np.einsum(
        "ab,axc,xbd->cd", state0 @ state1, operator0, operator1
    )
    assert sorted(dense.dims) == [2, 2]
    assert np.allclose(
        dense.to_numpy(), aligned(reference, [s0_out, s1_out], dense)
    )

    # zip-up agrees with the dense reference algorithm.
    zipup = state.contract(operator, method="zipup").contract_to_tensor()
    assert np.allclose(
        zipup.to_numpy(), aligned(reference, [s0_out, s1_out], zipup)
    )


def test_tree_network_contract_zipup_maxdim_truncates():
    """The same contraction truncated to bond dimension 1 still yields two sites."""
    rng = np.random.default_rng(3)
    s0, s1, bond = t4a.Index(2), t4a.Index(2), t4a.Index(3)
    state = t4a.TreeTensorNetwork(
        [
            t4a.Tensor([s0, bond], rng.standard_normal((2, 3))),
            t4a.Tensor([bond, s1], rng.standard_normal((3, 2))),
        ]
    )
    link = t4a.Index(3)
    s0_out, s1_out = t4a.Index(2), t4a.Index(2)
    operator = t4a.TreeTensorNetwork(
        [
            t4a.Tensor(
                [s0, link, s0_out], rng.standard_normal((2, 3, 2))
            ),
            t4a.Tensor(
                [link, s1, s1_out], rng.standard_normal((3, 2, 2))
            ),
        ]
    )
    result = state.contract(operator, method="zipup", maxdim=1)
    assert sorted(result.contract_to_tensor().dims) == [2, 2]


def test_tree_network_invalid_inputs():
    i = t4a.Index(2)
    with pytest.raises(ValueError):
        t4a.TreeTensorNetwork([])
    tensor = t4a.Tensor([i], np.zeros(2))
    with pytest.raises(ValueError):
        t4a.TreeTensorNetwork([tensor], names=[0, 1])
    with pytest.raises(ValueError):
        t4a.TreeTensorNetwork(
            [t4a.Tensor([i], np.zeros(2)), t4a.Tensor([i], np.zeros(2)), t4a.Tensor([i], np.zeros(2))]
        )
    with pytest.raises(KeyError):
        t4a.TreeTensorNetwork([tensor]).tensor(7)
    with pytest.raises(ValueError):
        t4a.TreeTensorNetwork([tensor]).contract(
            t4a.TreeTensorNetwork([tensor]), method="unknown"
        )


def test_tree_network_naive_contraction_respects_dense_limit():
    state, _, _ = chain_network()
    with pytest.raises(ValueError):
        state.contract(state, method="naive", dense_reference_limit=1)
