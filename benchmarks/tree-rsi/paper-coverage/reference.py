"""Independent bounded NumPy reference operations, outside algorithm timers.

Core layout is (left bond, physical index, right bond). These helpers never
materialize an unbounded physical tensor or an unbounded product bond.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ[_key] = '1'
import numpy as np


def dense(cores, max_elements=1 << 20):
    size = int(np.prod([a.shape[1] for a in cores], dtype=object))
    if size > max_elements:
        raise ValueError(f'dense reference has {size} entries, limit {max_elements}')
    value = np.ones((1, 1), dtype=cores[0].dtype)
    for a in cores:
        value = (value @ a.reshape(a.shape[0], -1)).reshape(-1, a.shape[2])
    return value[:, 0]


def evaluate(cores, points):
    """Batched, prefix-cached amplitudes; never repeat a full contraction per point."""
    points = np.asarray(points, dtype=np.int64)
    _, inverse = np.unique(points, axis=0, return_inverse=True)
    # Cache each distinct prefix once, using grouped GEMM at each site.
    unique = np.unique(points, axis=0)
    state = np.ones((1, 1), dtype=cores[0].dtype)
    previous_inverse = np.zeros(len(unique), dtype=np.int64)
    for site, a in enumerate(cores):
        prefixes, first, current_inverse = np.unique(unique[:, :site+1], axis=0, return_index=True, return_inverse=True)
        result = np.empty((len(prefixes), a.shape[2]), dtype=a.dtype)
        parents = previous_inverse[first]
        for physical in range(a.shape[1]):
            mask = prefixes[:, site] == physical
            result[mask] = state[parents[mask]] @ a[:, physical, :]
        state = result
        previous_inverse = current_inverse
    return state[:, 0][inverse]


def norm(cores):
    """QR norm avoids cancellation from differences of large inner products."""
    carry = np.ones((1, 1), dtype=cores[0].dtype)
    for a in cores:
        block = (carry @ a.reshape(a.shape[0], -1)).reshape(-1, a.shape[2])
        _, carry = np.linalg.qr(block, mode='reduced')
    return float(np.linalg.norm(carry))


def difference(a, b):
    if len(a) != len(b) or len(a) < 2:
        raise ValueError('difference requires matching chains with >=2 sites')
    result = [np.concatenate((a[0], -b[0]), axis=2)]
    for x, y in zip(a[1:-1], b[1:-1]):
        out = np.zeros((x.shape[0]+y.shape[0], x.shape[1], x.shape[2]+y.shape[2]), dtype=np.result_type(x, y))
        out[:x.shape[0], :, :x.shape[2]] = x
        out[x.shape[0]:, :, x.shape[2]:] = y
        result.append(out)
    result.append(np.concatenate((a[-1], b[-1]), axis=0))
    return result


def round_chain(cores, cap, tolerance=0.0):
    """Reference QR/SVD rounding using NumPy LAPACK, not a timed Rust baseline."""
    work = [a.copy() for a in cores]
    for i in range(len(work)-1):
        a = work[i]; q, r = np.linalg.qr(a.reshape(-1, a.shape[2]), mode='reduced')
        work[i] = q.reshape(a.shape[0], a.shape[1], -1)
        work[i+1] = np.tensordot(r, work[i+1], axes=(1, 0))
    discarded = 0.0
    for i in range(len(work)-1, 0, -1):
        a = work[i]; u, s, vh = np.linalg.svd(a.reshape(a.shape[0], -1), full_matrices=False)
        k = min(cap, len(s))
        if tolerance:
            k = min(k, max(1, int(np.count_nonzero(s > s[0]*tolerance))))
        discarded += float(np.sum(s[k:]**2))
        work[i] = vh[:k].reshape(k, a.shape[1], a.shape[2])
        work[i-1] = np.tensordot(work[i-1], u[:, :k]*s[:k], axes=(2, 0))
    return work, discarded**0.5


def dense_to_chain(values, dims, cap):
    if len(values) != int(np.prod(dims, dtype=object)) or len(values) > 1 << 20:
        raise ValueError('bounded reference TT-SVD size mismatch or limit')
    state = np.asarray(values).reshape(1, -1); left = 1; cores = []
    for d in dims[:-1]:
        u, singular, vh = np.linalg.svd(state.reshape(left*d, -1), full_matrices=False)
        right = min(cap, len(singular))
        cores.append(u[:, :right].reshape(left, d, right))
        state = singular[:right, None]*vh[:right]; left = right
    cores.append(state.reshape(left, dims[-1], 1))
    return cores


def product(a, b, max_total_elements=50_000_000):
    count = sum(x.shape[0]*y.shape[0]*x.shape[1]*x.shape[2]*y.shape[2] for x, y in zip(a, b))
    if count > max_total_elements:
        raise ValueError(f'explicit product has {count} elements; exceeds reference budget')
    return [np.einsum('asb,csd->acsbd', x, y).reshape(x.shape[0]*y.shape[0], x.shape[1], x.shape[2]*y.shape[2]) for x, y in zip(a, b)]


def maxabs_upper_bound(cores):
    """Bound entries via bounded prefix frames and remaining spectral norms."""
    work=[a.copy() for a in cores]
    for i in range(len(work)-1,0,-1):
        a=work[i];q,r=np.linalg.qr(a.reshape(a.shape[0],-1).conj().T,mode='reduced')
        work[i]=q.conj().T.reshape(q.shape[1],a.shape[1],a.shape[2])
        work[i-1]=np.tensordot(work[i-1],r.conj().T,axes=(2,0))
    # Enumerate at most2^16 physical prefixes, with at most2^20 frame entries.
    # This tightens the bound while staying far below the full physical grid.
    state=np.ones((1,1),dtype=work[0].dtype);cut=0
    for a in work:
        rows=len(state)*a.shape[1]
        if rows>2**16 or rows*a.shape[2]>2**20:break
        state=(state@a.reshape(a.shape[0],-1)).reshape(rows,a.shape[2]);cut+=1
    bound=float(np.max(np.linalg.norm(state,axis=1)))
    for a in work[cut:]:bound*=max(float(np.linalg.norm(a[:,s,:],ord=2)) for s in range(a.shape[1]))
    return bound


def diagonal_observables(probability):
    """Sum of represented probabilities and adjacent SzSz sum, in one sweep."""
    z = np.array([1.0, 0.0, -1.0]); total = np.ones(1, dtype=probability[0].dtype)
    previous_z = np.zeros_like(total); energy = np.zeros_like(total)
    for a in probability:
        identity = a.sum(axis=1); spin = np.einsum('asb,s->ab', a, z)
        energy, previous_z, total = energy @ identity + previous_z @ spin, total @ spin, total @ identity
    return complex(total[0]), complex(energy[0])


def wavefunction_observables(psi):
    """Independent double-layer MPS contraction; no explicit squared MPS."""
    z = np.array([1.0, 0.0, -1.0]); total = np.ones((1, 1), dtype=psi[0].dtype)
    previous_z = np.zeros_like(total); energy = np.zeros_like(total)
    def transfer(env, a, weights):
        out = np.zeros((a.shape[2], a.shape[2]), dtype=a.dtype)
        for s, weight in enumerate(weights):
            out += weight * (a[:, s, :].conj().T @ env @ a[:, s, :])
        return out
    for a in psi:
        energy, previous_z, total = transfer(energy, a, np.ones(3)) + transfer(previous_z, a, z), transfer(total, a, z), transfer(total, a, np.ones(3))
    return float(total[0, 0].real), float(energy[0, 0].real)


def born_samples(psi, count, seed):
    """Sequential exact Born sampling after right QR; independent of both products."""
    work = [a.copy() for a in psi]
    for i in range(len(work)-1, 0, -1):
        a = work[i]; q, r = np.linalg.qr(a.reshape(a.shape[0], -1).conj().T, mode='reduced')
        work[i] = q.conj().T.reshape(q.shape[1], a.shape[1], a.shape[2])
        work[i-1] = np.tensordot(work[i-1], r.conj().T, axes=(2, 0))
    work[0] /= np.linalg.norm(work[0]); rng = np.random.default_rng(seed)
    state = np.ones((count, 1), dtype=work[0].dtype); points = np.empty((count, len(work)), dtype=np.int64)
    for i, a in enumerate(work):
        branches = (state @ a.reshape(a.shape[0], -1)).reshape(count, a.shape[1], a.shape[2])
        weights = np.sum(np.abs(branches)**2, axis=2)
        probabilities = weights / weights.sum(axis=1, keepdims=True)
        chosen = np.sum(rng.random(count)[:, None] > np.cumsum(probabilities, axis=1), axis=1)
        chosen = np.minimum(chosen, a.shape[1]-1)
        points[:, i] = chosen
        state = branches[np.arange(count), chosen] / np.sqrt(weights[np.arange(count), chosen, None])
    return points
