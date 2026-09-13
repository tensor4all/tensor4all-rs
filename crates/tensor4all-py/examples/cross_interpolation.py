"""Tree tensor cross interpolation of a batched Python function.

Run after installing the extension (see README.md)::

    python examples/cross_interpolation.py

The evaluator is batched: TreeTCI asks for many points at once, and the Python
function must be vectorised over the batch axis.
"""

from collections import Counter

import numpy as np

import tensor4all as t4a


def cross_interpolation_chain():
    """Interpolate f(i, j, k) = i + 10j + 100k on a 2 x 3 x 4 chain."""
    dims = [2, 3, 4]
    calls = Counter()

    def evaluate(points):
        # points: (n_points, n_sites) int64, C-contiguous. Row p is point p.
        calls[points.shape] += 1
        assert points.ndim == 2 and points.shape[1] == len(dims)
        return (points[:, 0] + 10 * points[:, 1] + 100 * points[:, 2]).astype(np.float64)

    network, ranks, errors = t4a.crossinterpolate(
        evaluate, dims, initial_pivots=[[0, 0, 1]], seed=0
    )

    print(f"evaluate calls: {sum(calls.values())}")
    print(f"batch sizes   : {sorted(size for size, _ in calls)}")
    print(f"ranks         : {ranks}")
    print(f"final error   : {errors[-1]:.3e}")

    reference = np.array(
        [
            i + 10 * j + 100 * k
            for i in range(dims[0])
            for j in range(dims[1])
            for k in range(dims[2])
        ],
        dtype=np.float64,
    ).reshape(tuple(dims))
    dense = network.contract_to_tensor()
    assert errors[-1] < 1e-10
    assert np.allclose(dense.to_numpy(), reference)
    print("interpolated tensor matches the dense reference")


def cross_interpolation_star():
    """The same scheme on a Y-shaped tree instead of a chain."""
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
    print("non-chain tree graph is interpolated as well")


def cross_interpolation_complex():
    """Complex-valued targets keep their imaginary part."""
    def evaluate(points):
        real = points[:, 0] + 10 * points[:, 1]
        return (real * (1.0 + 2.0j)).astype(np.complex128)

    network, _, errors = t4a.crossinterpolate(
        evaluate, [2, 3], initial_pivots=[[0, 1]], seed=0
    )
    reference = np.array(
        [(i + 10 * j) * (1.0 + 2.0j) for i in range(2) for j in range(3)],
        dtype=np.complex128,
    ).reshape(2, 3)
    dense = network.contract_to_tensor()
    assert dense.to_numpy().dtype == np.complex128
    assert errors[-1] < 1e-10
    assert np.allclose(dense.to_numpy(), reference)
    print("complex-valued target is interpolated without losing the phase")


if __name__ == "__main__":
    cross_interpolation_chain()
    cross_interpolation_star()
    cross_interpolation_complex()
    print("all example assertions passed")
