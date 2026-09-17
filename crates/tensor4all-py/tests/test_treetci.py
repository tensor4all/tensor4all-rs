"""Tests for the TreeTCI (cross interpolation) bindings.

The evaluator contract is batched: ``evaluate(points)`` receives a
C-contiguous ``int64`` array of shape ``(n_points, n_sites)`` and returns a
``(n_points,)`` array of ``float64`` or ``complex128``. Point ``p`` is row
``p``, so an axis mix-up shows up immediately in these tests.
"""

import numpy as np
import pytest

import tensor4all as t4a

LINEAR_DIMS = [2, 3, 4]


def linear_function(points):
    """f(i, j, k) = i + 10j + 100k: asymmetric, so a wrong batch axis fails."""
    return (points[:, 0] + 10 * points[:, 1] + 100 * points[:, 2]).astype(np.float64)


def linear_reference(dims=LINEAR_DIMS):
    return np.array(
        [
            i + 10 * j + 100 * k
            for i in range(dims[0])
            for j in range(dims[1])
            for k in range(dims[2])
        ],
        dtype=np.float64,
    ).reshape(tuple(dims))


# ---------------------------------------------------------------------------
# Batch contract
# ---------------------------------------------------------------------------


def test_batch_contract_and_axis_order():
    shapes = []

    def evaluate(points):
        shapes.append(points.shape)
        assert points.ndim == 2
        assert points.dtype == np.int64
        assert points.shape[1] == len(LINEAR_DIMS)
        assert np.all(points >= 0)
        # Vectorised over the batch axis (axis 0); a point is a row.
        return linear_function(points)

    network, ranks, errors = t4a.crossinterpolate(
        evaluate, LINEAR_DIMS, initial_pivots=[[0, 0, 1]], seed=0
    )

    assert len(shapes) > 1
    assert all(shape[0] >= 1 for shape in shapes)
    # A single-point batch keeps its batch axis.
    assert any(shape == (1, 3) for shape in shapes)
    assert isinstance(ranks, list) and isinstance(errors, list)
    assert len(ranks) == len(errors) and len(ranks) >= 1
    assert errors[-1] < 1e-10

    dense = network.contract_to_tensor()
    assert dense.dims == LINEAR_DIMS
    assert np.allclose(dense.to_numpy(), linear_reference())


def test_points_are_rows_of_the_batch():
    """Each returned value must belong to the point in the same row."""
    seen = []

    def evaluate(points):
        for point in points:
            seen.append(tuple(int(value) for value in point))
        return linear_function(points)

    t4a.crossinterpolate(evaluate, LINEAR_DIMS, initial_pivots=[[0, 0, 1]], seed=0)
    assert (0, 0, 1) in seen
    assert all(0 <= i < 2 and 0 <= j < 3 and 0 <= k < 4 for i, j, k in seen)


def test_network_structure_matches_sites():
    network, _, _ = t4a.crossinterpolate(
        lambda points: (points[:, 0] + 10 * points[:, 1]).astype(np.float64),
        [2, 3],
        initial_pivots=[[0, 1]],
        seed=0,
    )
    assert network.num_vertices == 2
    assert network.num_edges == 1
    assert sorted(network.node_names()) == [0, 1]
    # Node k is site k, and its site leg is the first index of its tensor.
    for site, dim in enumerate([2, 3]):
        assert network.tensor(site).indices[0].dim == dim


def test_explicit_chain_edges_match_the_default():
    kwargs = dict(initial_pivots=[[0, 0, 1]], seed=0)
    default_network, _, _ = t4a.crossinterpolate(linear_function, LINEAR_DIMS, **kwargs)
    explicit_network, _, _ = t4a.crossinterpolate(
        linear_function, LINEAR_DIMS, edges=[(0, 1), (1, 2)], **kwargs
    )
    assert np.allclose(
        default_network.contract_to_tensor().to_numpy(),
        explicit_network.contract_to_tensor().to_numpy(),
    )


def test_non_chain_star_graph():
    dims = [4, 2, 2]

    def evaluate(points):
        return (points[:, 0] + 10 * points[:, 1] + 100 * points[:, 2]).astype(np.float64)

    network, _, errors = t4a.crossinterpolate(
        evaluate, dims, edges=[(0, 1), (0, 2)], initial_pivots=[[1, 0, 0]], seed=0
    )
    reference = np.array(
        [
            i + 10 * j + 100 * k
            for i in range(dims[0])
            for j in range(dims[1])
            for k in range(dims[2])
        ],
        dtype=np.float64,
    ).reshape(tuple(dims))
    assert network.num_edges == 2
    assert errors[-1] < 1e-10
    assert np.allclose(network.contract_to_tensor().to_numpy(), reference)


# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------


def test_complex_values():
    def evaluate(points):
        real = points[:, 0] + 10 * points[:, 1]
        return (real * (1.0 + 2.0j)).astype(np.complex128)

    network, _, errors = t4a.crossinterpolate(
        evaluate, [2, 3], initial_pivots=[[0, 1]], seed=0
    )
    dense = network.contract_to_tensor()
    reference = np.array(
        [(i + 10 * j) * (1.0 + 2.0j) for i in range(2) for j in range(3)],
        dtype=np.complex128,
    ).reshape(2, 3)
    assert dense.to_numpy().dtype == np.complex128
    assert errors[-1] < 1e-10
    assert np.allclose(dense.to_numpy(), reference)


def test_real_values_stay_real():
    network, _, _ = t4a.crossinterpolate(
        linear_function, LINEAR_DIMS, initial_pivots=[[0, 0, 1]], seed=0
    )
    assert network.contract_to_tensor().to_numpy().dtype == np.float64


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class Boom(RuntimeError):
    """Raised by the evaluator to check that Python exceptions pass through."""


def test_evaluator_exception_propagates_unchanged():
    def evaluate(points):
        raise Boom("evaluator exploded")

    with pytest.raises(Boom, match="evaluator exploded"):
        t4a.crossinterpolate(evaluate, [2, 3], initial_pivots=[[0, 1]], seed=0)


def test_evaluator_exception_after_first_call_propagates():
    calls = {"n": 0}

    def evaluate(points):
        calls["n"] += 1
        if calls["n"] > 1:
            raise Boom("late failure")
        return np.ones(points.shape[0], dtype=np.float64)

    with pytest.raises(Boom, match="late failure"):
        t4a.crossinterpolate(evaluate, [2, 2], seed=0)


def test_wrong_length_is_rejected():
    def evaluate(points):
        return np.zeros(points.shape[0] + 1, dtype=np.float64)

    with pytest.raises(ValueError, match="values for a batch"):
        t4a.crossinterpolate(evaluate, [2, 2], seed=0)


def test_wrong_dtype_is_rejected():
    def evaluate(points):
        return points[:, 0].astype(np.int64)

    with pytest.raises(TypeError, match="int64"):
        t4a.crossinterpolate(evaluate, [2, 2], seed=0)


def test_non_array_result_is_rejected():
    def evaluate(points):
        return [1.0] * points.shape[0]

    with pytest.raises(TypeError, match="numpy array"):
        t4a.crossinterpolate(evaluate, [2, 2], seed=0)


def test_two_dimensional_result_is_rejected():
    def evaluate(points):
        return np.ones((points.shape[0], 1), dtype=np.float64)

    with pytest.raises(TypeError, match=r"shape \(1,\)"):
        t4a.crossinterpolate(evaluate, [2, 2], seed=0)


def test_scalar_result_is_rejected_even_for_a_single_point_batch():
    def evaluate(points):
        return np.float64(1.0)

    with pytest.raises(TypeError, match="shape"):
        t4a.crossinterpolate(evaluate, [2, 2], seed=0)


def test_dtype_must_not_change_after_the_first_call():
    """A late complex result must never be silently truncated to real."""
    calls = {"n": 0}

    def evaluate(points):
        calls["n"] += 1
        real = np.ones(points.shape[0], dtype=np.float64)
        if calls["n"] == 1:
            return real
        return (real * (1.0 + 1.0j)).astype(np.complex128)

    with pytest.raises(TypeError, match="first call returned float64"):
        t4a.crossinterpolate(evaluate, [2, 2], seed=0)


def test_zero_initial_pivot_is_rejected():
    def evaluate(points):
        return (points[:, 0] + 10 * points[:, 1]).astype(np.float64)

    with pytest.raises(ValueError, match="initial pivots"):
        t4a.crossinterpolate(evaluate, [2, 2], seed=0)


def test_initial_pivot_shape_is_validated():
    def evaluate(points):
        return np.ones(points.shape[0], dtype=np.float64)

    with pytest.raises(ValueError, match="one value per site"):
        t4a.crossinterpolate(evaluate, [2, 2], initial_pivots=[[0, 0, 0]])


def test_invalid_configuration_is_rejected():
    def evaluate(points):
        return np.ones(points.shape[0], dtype=np.float64)

    with pytest.raises(ValueError):
        t4a.crossinterpolate(evaluate, [])
    with pytest.raises(ValueError):
        t4a.crossinterpolate(evaluate, [2, 2], edges=[(0, 5)])
    with pytest.raises(ValueError):
        t4a.crossinterpolate(evaluate, [2, 2], edges=[(0, 1), (1, 0)])
    with pytest.raises(ValueError):
        t4a.crossinterpolate(evaluate, [2, 2], tolerance=-1.0)
    with pytest.raises(ValueError):
        t4a.crossinterpolate(evaluate, [2, 2], max_iter=0)


# ---------------------------------------------------------------------------
# Reproducibility and bonds
# ---------------------------------------------------------------------------


def test_seed_makes_the_run_reproducible():
    kwargs = dict(initial_pivots=[[0, 0, 1]])
    first = t4a.crossinterpolate(linear_function, LINEAR_DIMS, seed=0, **kwargs)
    second = t4a.crossinterpolate(linear_function, LINEAR_DIMS, seed=0, **kwargs)
    assert first[1] == second[1]
    assert first[2] == second[2]
    assert np.array_equal(
        first[0].contract_to_tensor().to_numpy(),
        second[0].contract_to_tensor().to_numpy(),
    )


def test_max_bond_dim_caps_the_rank():
    def evaluate(points):
        return np.where(points[:, 0] == points[:, 1], 2.0, 1.0).astype(np.float64)

    _, ranks, _ = t4a.crossinterpolate(evaluate, [2, 2], max_bond_dim=1, seed=0)
    assert max(ranks) <= 1
