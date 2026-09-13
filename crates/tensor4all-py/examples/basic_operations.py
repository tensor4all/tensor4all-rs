"""Basic tensor and tree-tensor-network operations through the PyO3 bindings.

Run after installing the extension (see README.md)::

    python examples/basic_operations.py

Everything below forwards to the public Rust API of tensor4all-rs; the Python
side only hands over NumPy buffers and reads results back.
"""

import numpy as np

import tensor4all as t4a


def dense_tensor_contraction():
    """Contract two dense tensors over a shared index."""
    i, j, k = t4a.Index(2), t4a.Index(3), t4a.Index(4)
    left = np.arange(6, dtype=np.float64).reshape(2, 3)
    right = np.arange(12, dtype=np.float64).reshape(3, 4)

    product = t4a.Tensor([i, j], left).contract(t4a.Tensor([j, k], right))

    assert product.dims == [2, 4]
    assert np.allclose(product.to_numpy(), left @ right)
    print("dense contraction matches the NumPy reference")


def complex_contraction():
    """The same contraction with complex128 data."""
    i, j, k = t4a.Index(2), t4a.Index(3), t4a.Index(4)
    left = np.arange(6, dtype=np.complex128).reshape(2, 3) + 1j
    right = np.arange(12, dtype=np.complex128).reshape(3, 4) - 0.5j

    product = t4a.Tensor([i, j], left).contract(t4a.Tensor([j, k], right))

    assert product.to_numpy().dtype == np.complex128
    assert np.allclose(product.to_numpy(), left @ right)
    print("complex contraction matches the NumPy reference")


def chain_network():
    """An MPS-like chain of two tensors, contracted into a dense tensor."""
    s0, s1, bond = t4a.Index(2), t4a.Index(2), t4a.Index(3)
    left = np.arange(6, dtype=np.float64).reshape(2, 3)
    right = np.arange(6, dtype=np.float64).reshape(3, 2)

    network = t4a.TreeTensorNetwork(
        [t4a.Tensor([s0, bond], left), t4a.Tensor([bond, s1], right)]
    )

    assert network.num_vertices == 2
    assert network.num_edges == 1
    assert np.allclose(network.contract_to_tensor().to_numpy(), left @ right)
    print("two-node chain contracts to the expected dense tensor")


def branch_tree():
    """A Y-shaped (non-chain) tree: one center tensor and two leaves."""
    center, leaf0, leaf1 = t4a.Index(2), t4a.Index(4), t4a.Index(5)
    s0, s1 = t4a.Index(2), t4a.Index(3)
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
    dense = network.contract_to_tensor()
    reference = np.einsum("abc,bd,ce->ade", root, branch0, branch1)

    assert network.num_vertices == 3
    assert network.num_edges == 2
    assert sorted(dense.dims) == [2, 4, 5]
    assert np.allclose(dense.to_numpy(), reference)
    print("three-node branch tree matches the einsum reference")


def contract_two_networks():
    """Apply an MPO-like network to an MPS-like network."""
    rng = np.random.default_rng(7)
    s0, s1, bond = t4a.Index(2), t4a.Index(2), t4a.Index(3)
    state0 = rng.standard_normal((2, 3))
    state1 = rng.standard_normal((3, 2))
    state = t4a.TreeTensorNetwork(
        [t4a.Tensor([s0, bond], state0), t4a.Tensor([bond, s1], state1)]
    )

    link = t4a.Index(3)
    s0_out, s1_out = t4a.Index(2), t4a.Index(2)
    operator0 = rng.standard_normal((2, 3, 2))
    operator1 = rng.standard_normal((3, 2, 2))
    operator = t4a.TreeTensorNetwork(
        [
            t4a.Tensor([s0, link, s0_out], operator0),
            t4a.Tensor([link, s1, s1_out], operator1),
        ]
    )

    # "naive" materializes both networks (dense reference), "zipup" uses the
    # structural algorithm. Both must agree on this small problem.
    reference = np.einsum(
        "ab,axc,xbd->cd", state0 @ state1, operator0, operator1
    )
    for method in ("naive", "zipup"):
        dense = state.contract(operator, method=method).contract_to_tensor()
        assert sorted(dense.dims) == [2, 2]
        assert np.allclose(dense.to_numpy(), reference)
    print("tree-network contraction agrees between naive and zipup")


def numpy_boundary():
    """NumPy input is copied, output is independent, and layouts are honoured."""
    i, j = t4a.Index(2), t4a.Index(3)
    source = np.arange(12, dtype=np.float64).reshape(4, 3)[::2]
    tensor = t4a.Tensor([i, j], source)

    assert np.allclose(tensor.to_numpy(), source)
    source[0, 0] = 999.0
    assert tensor.to_numpy()[0, 0] == 0.0

    out = tensor.to_numpy()
    out[0, 0] = 42.0
    assert tensor.to_numpy()[0, 0] == 0.0
    print("NumPy interchange copies in and out, and preserves strided layouts")


if __name__ == "__main__":
    dense_tensor_contraction()
    complex_contraction()
    chain_network()
    branch_tree()
    contract_two_networks()
    numpy_boundary()
    print("all example assertions passed")
