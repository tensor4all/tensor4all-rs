# ACI Rust API Handoff

Date: 2026-05-22

Branch: `codex/aci-rust-api`

## Scope

This branch ports the core public Rust API for Alternating Cross Interpolation
workflows and aligns the local update path with the upstream Julia algorithm.
The current work focuses on the elementwise two-input TensorTrain case,
MatrixLUCI factor construction, batch evaluation, cache-aware local update
setup, and benchmark instrumentation against
`AlternatingCrossInterpolation.jl`.

No PR has been created from this branch.

## Main Changes

- Added `crates/tensor4all-aci/README.md` with upstream attribution and the
  Ritter paper citation.
- Added repository rules that forbid explicit pivot inverses, local dense
  linear algebra kernels, and copy-pasted scalar-specific implementations where
  a generic helper or macro is appropriate.
- Added system BLAS feature forwarding through relevant crates:
  `tenferro-system-blas`.
- Reworked ACI local update setup:
  - precomputes local input factors per input instead of repeatedly evaluating
    full local TT entries;
  - batches local setup GEMMs when shapes match;
  - batches full local input materialization with `batched_mat_mul_same_shape`;
  - batches frame updates;
  - keeps test-only benchmark gates:
    `T4A_ACI_DISABLE_BATCHED_LOCAL_SETUP=1`,
    `T4A_ACI_DISABLE_BATCHED_MATERIALIZE=1`, and
    `T4A_ACI_DISABLE_BATCHED_FRAME=1`.
- Added cached input-core matrix views in `ElementwiseProblem`, so repeated
  local setup does not rebuild the same dense core matrix views.
- Added `tensor4all-tensorbackend::Matrix` conversion APIs:
  - `into_col_major_vec`;
  - `to_typed_tensor`;
  - `into_typed_tensor`;
  - `try_from_typed_tensor`;
  - `MatrixTensorConversionError`.
- Added owned matrix linalg wrappers:
  - `mat_mul_owned`;
  - `solve_matrix_owned`;
  - `triangular_solve_matrix_owned`.
- Added backend triangular solve wrappers and tests.
- Reworked MatrixLUCI dense factor construction to follow the Julia rrLU-based
  solve formulation:
  - no explicit pivot inverse;
  - uses backend triangular solves;
  - uses owned Matrix-to-TypedTensor conversion where input buffers can be
    consumed.
- Added MatrixLUCI Hilbert timing benchmark and Julia counterpart.
- Added ACI local update step timing benchmark and Julia counterpart.

## Tenferro Notes

Do not assume tenferro changes are present. Another agent was working in a
separate tenferro worktree. This branch currently uses the pinned tenferro API
and its existing `From<TypedTensor<_>> for Tensor` conversions.

A possible future tenferro cleanup is to add a macro-generated
`TensorScalar::into_dynamic` / typed view helper inside tenferro itself, but
that change is not required for this branch.

## Benchmarks Run

The machine was heavily loaded during the last benchmark pass:

- load average was about `48`;
- many unrelated `solve_bse_afm_hf_only_quench` processes were active;
- `mpstat` showed a few mostly idle cores, but measurements still varied.

Treat the following numbers as handoff context, not final performance claims.

### ACI Local Step, `chi = 32`

Command:

```bash
taskset -c 58 env \
RAYON_NUM_THREADS=1 BLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
T4A_STEP_TIMING_REPEATS=50 T4A_STEP_TIMING_CHIS=32 \
cargo test --release -p tensor4all-aci local_update_step_timing -- --ignored --nocapture
```

Result:

```text
rust,32,50,3,33,2.022448,0.004660,0.005720,1.027935,0.986077,0.017761,1.147757,0.118297,3.225273,0.217013,1.325973,8.073430,26,6.172906e-11
```

Julia command:

```bash
taskset -c 58 env \
JULIA_NUM_THREADS=1 BLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
T4A_STEP_TIMING_REPEATS=50 \
julia benchmarks/julia/benchmark_aci_local_steps.jl --chis 32
```

Result:

```text
julia,32,50,3,33,1.7984784999999999,0.0,0.0,0.896193,0.9031885,0.0,1.036934,0.085758,1.6588165,0.07977200000000001,1.076152,5.7409,26,6.172906150931444e-11
```

Earlier in the same session, without CPU pinning, Rust produced `4.457842 ms`
total for `chi=32`, while Julia produced `6.0671765 ms`. Because the machine
load changed during the session, rerun on a quieter host before drawing
conclusions.

### MatrixLUCI Hilbert, size 64

Rust command:

```bash
taskset -c 58 env \
RAYON_NUM_THREADS=1 BLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
T4A_MATRIX_LUCI_REPEATS=100 T4A_MATRIX_LUCI_SIZES=64 \
cargo test --release -p tensor4all-tcicore matrix_luci_hilbert_timing -- --ignored --nocapture
```

Result:

```text
rust,hilbert,64,100,true,0.108844,0.000000,0.046796,0.045056,0.193050,13,9.601802e-12,2.000000e0
rust,hilbert,64,100,false,0.104603,0.000000,0.045361,0.040711,0.187775,13,9.601802e-12,2.000000e0
```

Julia command:

```bash
taskset -c 58 env \
JULIA_NUM_THREADS=1 BLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
T4A_MATRIX_LUCI_REPEATS=100 \
julia benchmarks/julia/benchmark_matrix_luci.jl --sizes 64
```

Result:

```text
julia,hilbert,64,100,true,0.18505549999999998,0.0,0.015735,0.008965,0.212001,13,9.601801849524244e-12,2.0
julia,hilbert,64,100,false,0.174785,0.0,0.008660000000000001,0.0145305,0.198306,13,9.601801849524244e-12,2.0
```

MatrixLUCI selection is close under CPU pinning, but factor construction still
differs by implementation route. The total is close enough that a quieter host
is needed before further tuning.

## Verification Run Before Commit

The latest local verification before commit included:

```bash
cargo fmt --all -- --check
cargo test --release -p tensor4all-tensorbackend --lib -- --nocapture
cargo test --release -p tensor4all-tcicore -- --nocapture
cargo test --release -p tensor4all-aci -- --nocapture
cargo test --doc --release -p tensor4all-tensorbackend
```

Observed results:

- `tensor4all-tensorbackend`: 139 lib tests passed.
- `tensor4all-tcicore`: 101 unit tests passed, 1 ignored; 97 doc tests passed.
- `tensor4all-aci`: 72 unit tests passed, 1 ignored; 4 integration tests
  passed; 15 doc tests passed.
- `tensor4all-tensorbackend` doc tests: 122 passed.

The branch should still be rechecked on the final benchmark host before a PR.

## Recommended Next Steps

1. Re-run the Rust and Julia benchmark commands above on a quieter host with
   the same CPU affinity and one-thread settings.
2. Compare `chi = 16, 32, 64` scaling for ACI local updates; `chi = 32` alone
   is not enough to diagnose scaling.
3. If Rust remains slower, inspect:
   - MatrixLUCI selection timing;
   - frame update timing;
   - `core_update_ms`, where Rust was much slower in the noisy pinned run;
   - whether tenferro-owned linalg wrappers can avoid more typed/dynamic
     boundary overhead.
4. Audit the repository for remaining hand-written dense linalg in feature
   crates, as requested earlier.
5. Decide whether the test-only ACI benchmark gates should remain in-tree or
   move to a benchmark-specific configuration path.
