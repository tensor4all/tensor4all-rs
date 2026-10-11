"""NumPy tree/train utilities for the agreement benchmark.

Fixture cores are dictionaries ``{"neighbors": [...], "array": ndarray}`` whose
array axes are the neighbor bonds (in ``neighbors`` order, ascending node
number) followed by the site. Chain cores in the author's convention are
``(r_left, d, r_right)`` arrays.
"""
import numpy as np


def rooted(n, edges, root):
    adj = {v: [] for v in range(n)}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    parent, order, stack = {root: None}, [], [root]
    while stack:
        v = stack.pop()
        order.append(v)
        for w in sorted(adj[v]):
            if w not in parent:
                parent[w] = v
                stack.append(w)
    children = {v: sorted(w for w in adj[v] if parent.get(w) == v) for v in range(n)}
    return adj, parent, children, order[::-1]  # postorder (children first)


def tree_from_dense(dense, edges, root, rtol, maxrank=None):
    """Hierarchical SVD of a dense tensor (one site per node) onto a tree.

    Each edge discards at most ``rtol**2 * ||dense||**2 / (n - 1)`` squared
    weight, so the total relative error is at most ``rtol``; ``maxrank`` also
    caps every bond (quasi-optimal truncation).
    """
    n = dense.ndim
    adj, parent, children, post = rooted(n, edges, root)
    total = np.linalg.norm(dense)
    budget = (rtol * total) ** 2 / max(n - 1, 1)
    labels = [("s", v) for v in range(n)]
    work = dense
    cores = {}
    for v in post:
        if v == root:
            continue
        group = [("s", v)] + [("b", c) for c in children[v]]
        front = [labels.index(g) for g in group]
        rest = [i for i in range(len(labels)) if i not in front]
        work = np.transpose(work, front + rest)
        gshape = work.shape[: len(front)]
        rshape = work.shape[len(front):]
        u, s, vh = np.linalg.svd(work.reshape(int(np.prod(gshape)), -1), full_matrices=False)
        tail = np.cumsum((s ** 2)[::-1])[::-1]
        keep = max(1, int(np.sum(tail > budget)))
        if maxrank is not None:
            keep = min(keep, maxrank)
        u, s, vh = u[:, :keep], s[:keep], vh[:keep]
        core = u.reshape(*gshape, keep)  # (site, child bonds..., parent bond)
        axis_of = {("s", v): 0}
        for i, c in enumerate(children[v]):
            axis_of[c] = 1 + i
        axis_of[parent[v]] = len(gshape)
        neighbors = sorted(adj[v])
        cores[v] = {"neighbors": neighbors,
                    "array": np.transpose(core, [axis_of[w] for w in neighbors] + [0])}
        work = (s[:, None] * vh).reshape(keep, *rshape)
        labels = [("b", v)] + [labels[i] for i in rest]
    order = [labels.index(("b", c)) for c in sorted(adj[root])] + [labels.index(("s", root))]
    cores[root] = {"neighbors": sorted(adj[root]), "array": np.transpose(work, order)}
    return [cores[v] for v in range(n)]


def centroid(n, edges):
    best = None
    for r in range(n):
        _, _, children, post = rooted(n, edges, r)
        size = {}
        for v in post:
            size[v] = 1 + sum(size[c] for c in children[v])
        worst = max([size[c] for c in children[r]] + [0])
        if best is None or worst < best[0]:
            best = (worst, r)
    return best[1]


def tree_to_dense(cores):
    """Contract fixture cores to a dense array with axes ordered by node."""
    n = len(cores)
    edges = {(min(v, w), max(v, w)) for v, c in enumerate(cores) for w in c["neighbors"]}
    root = centroid(n, sorted(edges))
    _, parent, children, post = rooted(n, sorted(edges), root)
    bond = {e: n + i for i, e in enumerate(sorted(edges))}
    sub = {}
    for v in post:
        c = cores[v]
        operands = [c["array"], [bond[(min(v, w), max(v, w))] for w in c["neighbors"]] + [v]]
        sites = [v]
        for ch in children[v]:
            arr, labs, s = sub.pop(ch)
            operands += [arr, labs]
            sites += s
        sites = sorted(sites)
        out = sites + ([bond[(min(v, parent[v]), max(v, parent[v]))]] if parent[v] is not None else [])
        sub[v] = (np.einsum(*operands, out, optimize=True), out, sites)
    return sub[root][0]


def fixture_from_tt(tt):
    """Author TT cores ``(rl, d, rr)`` -> fixture chain cores."""
    n = len(tt)
    cores = []
    for v, core in enumerate(tt):
        arr = np.transpose(core, (0, 2, 1))  # (rl, rr, d)
        neighbors = []
        if v > 0:
            neighbors.append(v - 1)
        else:
            arr = arr[0]
        if v < n - 1:
            neighbors.append(v + 1)
        else:
            arr = arr[..., 0, :] if v > 0 else arr[0]
        cores.append({"neighbors": neighbors, "array": np.ascontiguousarray(arr)})
    return cores


def tt_from_fixture(cores):
    """Fixture chain cores -> author TT cores ``(rl, d, rr)``."""
    n = len(cores)
    tt = []
    for v, c in enumerate(cores):
        arr = c["array"]
        if n == 1:
            arr = arr[None, None, :]
        elif v == 0:
            arr = arr[None, :, :]
        elif v == n - 1:
            arr = arr[:, None, :]
        tt.append(np.ascontiguousarray(np.transpose(arr, (0, 2, 1))))
    return tt


def tt_norm(tt):
    r = np.ones((1, 1))
    for core in tt:
        m = np.tensordot(r, core, axes=(1, 0))
        q, r = np.linalg.qr(m.reshape(-1, m.shape[-1]))
    return float(abs(r[0, 0]))


def tt_sum(a, b, beta=1.0):
    """Exact TT of ``a + beta * b`` by direct sum of the cores."""
    n = len(a)
    out = []
    for k, (x, y) in enumerate(zip(a, b)):
        if k == 0:
            out.append(np.concatenate([x, beta * y], axis=2))
        elif k == n - 1:
            out.append(np.concatenate([x, y], axis=0))
        else:
            z = np.zeros((x.shape[0] + y.shape[0], x.shape[1], x.shape[2] + y.shape[2]))
            z[: x.shape[0], :, : x.shape[2]] = x
            z[x.shape[0]:, :, x.shape[2]:] = y
            out.append(z)
    return out


def tt_diff_norm(a, b):
    """``||a - b||`` through one orthogonalization sweep of the direct sum."""
    return tt_norm(tt_sum(a, b, -1.0))


def tt_hadamard(a, b):
    return [np.einsum("asb,csd->acsbd", x, y).reshape(x.shape[0] * y.shape[0], x.shape[1],
                                                       x.shape[2] * y.shape[2]) for x, y in zip(a, b)]


def tt_round(tt, rtol, maxrank=None):
    """TT-SVD rounding: right-to-left QR, then left-to-right truncated SVD."""
    tt = [c.copy() for c in tt]
    n = len(tt)
    for k in range(n - 1, 0, -1):
        rl, d, rr = tt[k].shape
        q, r = np.linalg.qr(tt[k].reshape(rl, d * rr).T)
        tt[k] = q.T.reshape(-1, d, rr)
        tt[k - 1] = np.tensordot(tt[k - 1], r.T, axes=(2, 0))
    total = np.linalg.norm(tt[0])
    budget = (rtol * total) ** 2 / max(n - 1, 1)
    for k in range(n - 1):
        rl, d, rr = tt[k].shape
        u, s, vh = np.linalg.svd(tt[k].reshape(rl * d, rr), full_matrices=False)
        tail = np.cumsum((s ** 2)[::-1])[::-1]
        keep = max(1, int(np.sum(tail > budget)))
        if maxrank is not None:
            keep = min(keep, maxrank)
        tt[k] = u[:, :keep].reshape(rl, d, keep)
        tt[k + 1] = np.tensordot(s[:keep, None] * vh[:keep], tt[k + 1], axes=(1, 0))
    return tt


def tt_eval(tt, x):
    """Evaluate a TT at integer points ``x`` of shape ``(samples, n)``."""
    v = np.ones((x.shape[0], 1))
    for k, core in enumerate(tt):
        out = np.empty((x.shape[0], core.shape[2]))
        for s in range(core.shape[1]):  # group samples by site value: one GEMM each
            mask = x[:, k] == s
            out[mask] = v[mask] @ core[:, s, :]
        v = out
    return v[:, 0]


def tt_born_samples(tt, count, rng):
    """Exact samples from ``|tt(x)|^2 / ||tt||^2`` via right canonical form."""
    tt = [c.copy() for c in tt]
    for k in range(len(tt) - 1, 0, -1):
        rl, d, rr = tt[k].shape
        q, r = np.linalg.qr(tt[k].reshape(rl, d * rr).T)
        tt[k] = q.T.reshape(-1, d, rr)
        tt[k - 1] = np.tensordot(tt[k - 1], r.T, axes=(2, 0))
    n = len(tt)
    x = np.zeros((count, n), dtype=np.int64)
    v = np.ones((count, 1))
    for k, core in enumerate(tt):
        cand = np.einsum("pa,asb->psb", v, core)
        p = np.sum(cand ** 2, axis=2)
        p /= p.sum(axis=1, keepdims=True)
        u = rng.random(count)
        s = np.minimum((np.cumsum(p, axis=1) < u[:, None]).sum(axis=1), core.shape[1] - 1)
        x[:, k] = s
        v = cand[np.arange(count), s, :]
        v /= np.linalg.norm(v, axis=1, keepdims=True)
    return x
