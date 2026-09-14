"""Tests for the quantics TCI (QTT) bindings.

The evaluator contract is batched: ``evaluate(points)`` receives a C-contiguous
array of shape ``(n_points, n_dims)`` (``float64`` coordinates for a continuous
grid, ``int64`` grid indices for a discrete grid) and returns a ``(n_points,)``
array of ``float64`` or ``complex128``. Point ``p`` is row ``p``.
"""

import numpy as np
import pytest

import tensor4all as t4a


def xy_linear(indices):
    """f(i, j) = i + 10j on the grid: asymmetric, so a wrong axis order fails."""
    return (indices[:, 0] + 10 * indices[:, 1]).astype(np.float64)


def xy_reference(sizes):
    return np.array(
        [i + 10 * j for i in range(sizes[0]) for j in range(sizes[1])], dtype=np.float64
    ).reshape(tuple(sizes))


# ---------------------------------------------------------------------------
# Discrete grids
# ---------------------------------------------------------------------------


def test_discrete_batch_contract_and_dense_values():
    sizes = [8, 8]
    shapes = []

    def evaluate(points):
        shapes.append(points.shape)
        assert points.ndim == 2
        assert points.dtype == np.int64
        assert points.shape[1] == len(sizes)
        return xy_linear(points)

    result, ranks, errors = t4a.quanticscrossinterpolate_discrete(
        evaluate, sizes, initial_pivots=[[0, 1]], tolerance=1e-10
    )

    assert len(shapes) > 1
    assert any(shape == (1, 2) for shape in shapes)  # a single-point batch
    assert all(shape[1] == 2 for shape in shapes)
    assert isinstance(ranks, list) and isinstance(errors, list)
    assert errors[-1] < 1e-10

    assert result.shape == sizes
    assert np.allclose(result.to_numpy(), xy_reference(sizes))
    assert result.evaluate([3, 2]) == pytest.approx(3 + 10 * 2)
    assert result.sum() == pytest.approx(xy_reference(sizes).sum())


def test_discrete_ranks_reflect_the_function():
    """A constant is a product state, a genuinely two-dimensional function is not."""
    constant = t4a.quanticscrossinterpolate_discrete(
        lambda points: np.ones(points.shape[0]), [8, 8], initial_pivots=[[1, 0]]
    )[0]
    two_dimensional = t4a.quanticscrossinterpolate_discrete(xy_linear, [8, 8], initial_pivots=[[0, 1]])[
        0
    ]
    assert constant.rank == 1
    assert two_dimensional.rank >= 2


def test_discrete_is_deterministic_by_default():
    """`random_init_pivots` defaults to 0, so runs are reproducible."""
    kwargs = dict(initial_pivots=[[0, 1]])
    first = t4a.quanticscrossinterpolate_discrete(xy_linear, [8, 8], **kwargs)
    second = t4a.quanticscrossinterpolate_discrete(xy_linear, [8, 8], **kwargs)
    assert first[1] == second[1]
    assert first[2] == second[2]
    assert np.array_equal(first[0].to_numpy(), second[0].to_numpy())


def test_discrete_max_bond_dim_caps_the_rank():
    def evaluate(points):
        return np.where(points[:, 0] == points[:, 1], 2.0, 1.0)

    result, ranks, _ = t4a.quanticscrossinterpolate_discrete(
        evaluate, [8, 8], max_bond_dim=1
    )
    assert result.rank == 1
    assert max(ranks) <= 1


def test_discrete_rejects_non_power_of_two_sizes():
    with pytest.raises(ValueError):
        t4a.quanticscrossinterpolate_discrete(xy_linear, [10, 10])


def test_discrete_rejects_unequal_sizes():
    with pytest.raises(ValueError):
        t4a.quanticscrossinterpolate_discrete(xy_linear, [8, 16])


# ---------------------------------------------------------------------------
# Continuous grids
# ---------------------------------------------------------------------------


def test_continuous_coordinates_and_axis_order():
    """f(x, y) = x + 10y over [0, 1) x [0, 1) with 2**3 x 2**4 points."""
    bits = [3, 4]
    seen = []

    def evaluate(points):
        assert points.dtype == np.float64
        assert points.shape[1] == 2
        assert np.all(points >= 0.0) and np.all(points < 1.0)
        seen.append(points)
        return (points[:, 0] + 10 * points[:, 1]).astype(np.float64)

    result, _, errors = t4a.quanticscrossinterpolate(
        evaluate, bits, lower=0.0, upper=1.0, initial_pivots=[[1, 0]], tolerance=1e-10
    )

    coordinates = np.concatenate(seen)
    # Every coordinate must be a grid point of its dimension.
    for dimension, size in enumerate(2 ** np.array(bits)):
        values = coordinates[:, dimension] * size
        assert np.allclose(values, np.round(values))

    x = np.arange(2**bits[0]) / 2**bits[0]
    y = np.arange(2**bits[1]) / 2**bits[1]
    reference = x[:, None] + 10 * y[None, :]
    assert result.shape == [2**bits[0], 2**bits[1]]
    assert errors[-1] < 1e-10
    assert np.allclose(result.to_numpy(), reference)


def test_continuous_sum_and_integral_agree_with_the_grid():
    bits = 4

    def evaluate(points):
        return (points[:, 0] ** 2).astype(np.float64)

    result, _, _ = t4a.quanticscrossinterpolate(
        evaluate, bits, lower=0.0, upper=1.0, initial_pivots=[[1]], tolerance=1e-10
    )
    dense = result.to_numpy()
    assert np.allclose(dense, (np.arange(2**bits) / 2**bits) ** 2)
    assert result.sum() == pytest.approx(dense.sum())
    # integral() is the sum times the grid step.
    assert result.integral() == pytest.approx(dense.sum() / 2**bits)


def test_continuous_unfolding_scheme_does_not_change_grid_semantics():
    def evaluate(points):
        return (points[:, 0] + 10 * points[:, 1]).astype(np.float64)

    args = dict(lower=0.0, upper=1.0, initial_pivots=[[1, 0]], tolerance=1e-10)
    interleaved = t4a.quanticscrossinterpolate(evaluate, [3, 3], unfolding="interleaved", **args)[0]
    fused = t4a.quanticscrossinterpolate(evaluate, [3, 3], unfolding="fused", **args)[0]
    grouped = t4a.quanticscrossinterpolate(evaluate, [3, 3], unfolding="grouped", **args)[0]
    assert np.allclose(interleaved.to_numpy(), fused.to_numpy())
    assert np.allclose(interleaved.to_numpy(), grouped.to_numpy())
    assert interleaved.evaluate([2, 3]) == pytest.approx(fused.evaluate([2, 3]))


def test_discrete_integral_is_the_sum():
    """Discrete grids have step 1, so `integral()` equals `sum()`."""
    discrete = t4a.quanticscrossinterpolate_discrete(xy_linear, [8, 8], initial_pivots=[[0, 1]])[0]
    assert discrete.integral() == pytest.approx(discrete.sum())


def test_continuous_multidimensional_bounds():
    def evaluate(points):
        return (points[:, 0] + points[:, 1]).astype(np.float64)

    result, _, errors = t4a.quanticscrossinterpolate(
        evaluate, [3, 3], lower=[0.0, -1.0], upper=[1.0, 1.0], initial_pivots=[[1, 0]]
    )
    assert errors[-1] < 1e-10
    x = np.linspace(0.0, 1.0, 2**3, endpoint=False)
    y = np.linspace(-1.0, 1.0, 2**3, endpoint=False)
    assert np.allclose(result.to_numpy(), x[:, None] + y[None, :])


# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------


def test_complex_values():
    def evaluate(points):
        return (points[:, 0] * (1.0 + 2.0j)).astype(np.complex128)

    result, _, errors = t4a.quanticscrossinterpolate(
        evaluate, 4, lower=0.0, upper=1.0, initial_pivots=[[1]], tolerance=1e-10
    )
    dense = result.to_numpy()
    assert dense.dtype == np.complex128
    assert errors[-1] < 1e-10
    assert np.allclose(dense, (np.arange(2**4) / 2**4) * (1.0 + 2.0j))
    assert result.evaluate([3]) == pytest.approx((3 / 16) * (1.0 + 2.0j))


def test_real_values_stay_real():
    result, _, _ = t4a.quanticscrossinterpolate(
        lambda points: points[:, 0].astype(np.float64),
        4,
        lower=0.0,
        upper=1.0,
        initial_pivots=[[1]],
    )
    assert result.to_numpy().dtype == np.float64


def test_late_dtype_change_is_rejected():
    calls = {"n": 0}

    def evaluate(points):
        calls["n"] += 1
        values = np.ones(points.shape[0], dtype=np.float64)
        if calls["n"] == 1:
            return values
        return (values * (1.0 + 1.0j)).astype(np.complex128)

    with pytest.raises(TypeError, match="first call returned float64"):
        t4a.quanticscrossinterpolate(
            evaluate, 3, lower=0.0, upper=1.0, initial_pivots=[[1]]
        )


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class Boom(RuntimeError):
    """Raised by the evaluator to check that Python exceptions pass through."""


def test_evaluator_exception_propagates_unchanged():
    def evaluate(points):
        raise Boom("quantics evaluator exploded")

    with pytest.raises(Boom, match="quantics evaluator exploded"):
        t4a.quanticscrossinterpolate_discrete(evaluate, [8, 8])


def test_wrong_length_is_rejected():
    def evaluate(points):
        return np.zeros(points.shape[0] + 1, dtype=np.float64)

    with pytest.raises(ValueError, match="values for a batch"):
        t4a.quanticscrossinterpolate_discrete(evaluate, [8, 8])


def test_non_array_result_is_rejected():
    def evaluate(points):
        return [1.0] * points.shape[0]

    with pytest.raises(TypeError, match="numpy array"):
        t4a.quanticscrossinterpolate_discrete(evaluate, [8, 8])


def test_wrong_dtype_result_is_rejected():
    def evaluate(points):
        return points[:, 0].astype(np.int64)

    with pytest.raises(TypeError, match="int64"):
        t4a.quanticscrossinterpolate_discrete(evaluate, [8, 8])


def test_invalid_configuration_is_rejected():
    def evaluate(points):
        return np.ones(points.shape[0], dtype=np.float64)

    with pytest.raises(ValueError):
        t4a.quanticscrossinterpolate_discrete(evaluate, [])
    with pytest.raises(ValueError):
        t4a.quanticscrossinterpolate(evaluate, [], lower=0.0, upper=1.0)
    with pytest.raises(ValueError, match="one entry per dimension"):
        t4a.quanticscrossinterpolate(evaluate, [3, 3], lower=[0.0, 0.0, 0.0], upper=1.0)
    with pytest.raises(ValueError, match="unfolding"):
        t4a.quanticscrossinterpolate_discrete(evaluate, [8], unfolding="nope")
    with pytest.raises(ValueError, match="one grid index per dimension"):
        t4a.quanticscrossinterpolate_discrete(evaluate, [8, 8], initial_pivots=[[0, 0, 0]])
    with pytest.raises(ValueError):
        t4a.quanticscrossinterpolate_discrete(evaluate, [8, 8], initial_pivots=[[0, 99]])


def test_to_numpy_refuses_large_grids():
    def evaluate(points):
        return np.ones(points.shape[0], dtype=np.float64)

    with pytest.raises(ValueError, match="element limit"):
        # A 30-bit grid would need 2**30 elements.
        big, _, _ = t4a.quanticscrossinterpolate_discrete(
            evaluate, [1 << 30], initial_pivots=[[1]]
        )
        big.to_numpy()
