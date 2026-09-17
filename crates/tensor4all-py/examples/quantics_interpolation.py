"""Quantics (QTT) interpolation of batched Python functions.

Run after installing the extension (see README.md)::

    python examples/quantics_interpolation.py

The evaluator is batched: one call per batch of grid points, never one call per
point. For a continuous grid the callback receives original coordinates, for a
discrete grid it receives integer grid indices.
"""

from collections import Counter

import numpy as np

import tensor4all as t4a


def continuous_1d():
    """sin(pi x) on [0, 1) with 2**6 grid points."""
    calls = Counter()

    def evaluate(points):
        calls[points.shape] += 1
        assert points.ndim == 2 and points.dtype == np.float64
        return np.sin(np.pi * points[:, 0])

    # sin(pi * 0) = 0, so the default initial pivot would be rejected.
    result, ranks, errors = t4a.quanticscrossinterpolate(
        evaluate, 6, lower=0.0, upper=1.0, initial_pivots=[[1]], tolerance=1e-10
    )

    print(f"continuous 1D: {sum(calls.values())} calls, batch sizes {sorted(s for s, _ in calls)}")
    print(f"  rank={result.rank} final error={errors[-1]:.2e}")

    x = np.arange(2**6) / 2**6
    assert np.allclose(result.to_numpy(), np.sin(np.pi * x))
    assert abs(result.integral() - 2.0 / np.pi) < 1e-2
    print("  dense values and the integral match the analytic reference")


def continuous_2d():
    """f(x, y) = x + 10y on [0, 1)^2 with 2**4 x 2**4 points."""
    def evaluate(points):
        return (points[:, 0] + 10.0 * points[:, 1]).astype(np.float64)

    result, _, errors = t4a.quanticscrossinterpolate(
        evaluate, [4, 4], lower=0.0, upper=1.0, initial_pivots=[[1, 0]], tolerance=1e-10
    )
    x = np.arange(2**4) / 2**4
    reference = x[:, None] + 10.0 * x[None, :]
    assert errors[-1] < 1e-10
    assert np.allclose(result.to_numpy(), reference)
    # Sum times the grid step is the integral over the unit square.
    assert result.integral() == result.sum() / 2**8
    print("2D: dense values match the reference, integral = sum * step")


def discrete_lattice():
    """f(i, j) = i + 10j on an 8 x 8 integer grid."""
    shapes = []

    def evaluate(points):
        shapes.append(points.shape)
        assert points.dtype == np.int64
        return (points[:, 0] + 10 * points[:, 1]).astype(np.float64)

    result, ranks, errors = t4a.quanticscrossinterpolate_discrete(
        evaluate, [8, 8], initial_pivots=[[0, 1]], tolerance=1e-10
    )
    reference = np.array([[i + 10 * j for j in range(8)] for i in range(8)], dtype=np.float64)
    assert errors[-1] < 1e-10
    assert np.allclose(result.to_numpy(), reference)
    assert abs(result.evaluate([3, 5]) - (3 + 10 * 5)) < 1e-8
    print(f"discrete: rank={ranks[-1]} dense values and evaluate() match")


def complex_target():
    """Complex values keep their phase."""
    def evaluate(points):
        return np.exp(2j * np.pi * points[:, 0]).astype(np.complex128)

    result, _, errors = t4a.quanticscrossinterpolate(
        evaluate, 5, lower=0.0, upper=1.0, initial_pivots=[[1]], tolerance=1e-10
    )
    complex_grid = result.to_numpy()
    assert complex_grid.dtype == np.complex128
    assert errors[-1] < 1e-10
    print(f"complex: sum={result.sum():.6f} (= sum of exp(2 pi i x) over the grid)")


if __name__ == "__main__":
    continuous_1d()
    continuous_2d()
    discrete_lattice()
    complex_target()
    print("all example assertions passed")
