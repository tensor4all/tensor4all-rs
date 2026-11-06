# RSI agreement benchmark

Checks that the Rust tree RSI (`tensor4all-treersi`) produces the same outputs
as the author's TT RSI, and that it agrees with TreeACI whenever both are
accurate, on chains and on a branched tree. One worker binary runs both Rust
algorithms from the same source tree, so the TreeACI is the one on this
branch's base (`main`).

## Run

From the repository root, with an interpreter that has NumPy, h5py and
TensorLy (the author's code imports TensorLy), and the author checkout
(Recursive-Sketched-Interpolation, pinned `153b25a8`):

```sh
python -B benchmarks/tree-rsi/agreement/run.py --author PATH/TO/Recursive-Sketched-Interpolation
python -B benchmarks/tree-rsi/agreement/run.py --author ... --cases dmrg_n10_chi20,g0_comb
```

`-B` keeps `__pycache__` out of the author checkout. Fixtures, worker copies,
outputs, `results.jsonl` and `provenance.json` (repository head, dirty paths,
worker hash, author head) go only to ignored
`target/tree-rsi/agreement-<timestamp>/`. Never commit them.

## What is compared

For every case and output cap:

| Run | Settings |
|---|---|
| `rsi_s1..3` | Rust RSI library defaults (`rel_tol = 1e-14`, `k = ceil(cap/d) + 5`), seeds 1–3 |
| `aci` | TreeACI defaults (tolerance `1e-12`, 2–20 sweeps, global guard), seed 1 |
| `author` (chains only) | Author `HadamardTT_RSI`, two open cores, `eps = 0`, `k = floor(cap/2) + 10`, NumPy seed 1 |
| `replay` (chains only) | Rust RSI with the author's recorded probe draws, the same `k`, `rel_tol = 0` |

The author code draws its probes from NumPy's global generator; the driver
records those draws during the author run and passes them to Rust as explicit
probes (`TreeRsiProbes`, column-major `d x k` per input and node, nodes
`2..n-1`). Both implementations then perform the same left-to-right sweep
(root `n-1`), the same full-pivot LU row selection, and the same exact-tail
switch (`tail assignments <= k`), so `replay_vs_author` measures
implementation agreement, not sampling variance.

Metrics:

- `error`: relative L2 error against the exact product of the represented
  inputs. Reference modes: `dense` (full grid), `tt` (exact Hadamard TT,
  rounded at `1e-14`; n20 DMRG), `sampled` (n50 DMRG; 20,000 Born samples of
  `psi`, an importance-sampled estimate, not a certificate).
- `replay_vs_author`, `*_vs_aci`: exact relative difference
  `||x - y|| / max(||x||, ||y||)` between two outputs (dense, or one
  orthogonalization sweep of the TT direct sum).

An output is *valid* when its error is at most `1e-6`. Two valid outputs
differ by at most the sum of their errors; the table makes that visible
rather than asserting it.

## Cases

| Case | Kind | Inputs |
|---|---|---|
| `dmrg_n10_chi20`, `dmrg_n20_chi{10,20,30}`, `dmrg_n50_chi{30,60,80,150}` | chain, d = 3 | Author spin-1 DMRG MPS, self product (paper IV.B). Actual bond dimensions are recorded; file names are not trusted. |
| `qtt_gauss` | chain, d = 2 | Paper IV.C Gaussians at 0.25/0.75, sigma 0.15, 2^22 grid |
| `qtt_ring2d` | chain, d = 2 | 2D ring Lorentzian times an oscillating Gaussian, 11+11 interleaved bits |
| `g0_interleaved` | chain, d = 2 | G0-like `(kx, ky, omega)` pair, 3 x 7 bits, gw-rs interleaved order |
| `g0_comb` | tree, d = 2 | Same functions on the gw-rs comb (CTTN): arms joined at their MSBs, junction `ky0` of degree 3 |

Function inputs are compressed from the dense grid by hierarchical SVD
(`rtol = 1e-12`); the reference is the exact product of those compressed
inputs, so input compression error does not enter `error`.

## Timing

Workers run pinned to one CPU (`AGREEMENT_CPU`, default 2) with one backend
thread; `seconds` covers only the product call. The author code is Python
and is not timed against Rust. RSI and ACI times are only comparable at
matched accuracy, which a fixed cap does not provide.
