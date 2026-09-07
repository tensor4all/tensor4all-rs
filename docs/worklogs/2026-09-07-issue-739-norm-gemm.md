# Issue #739 baseline and implementation log

## Baseline

- Baseline branch: detached worktree from `origin/main`.
- Baseline SHA: `1059be68e4a068562eaffc1081aea08429ea13dd`.
- `HEAD` and `origin/main` matched at the start of the baseline run.
- The previously used `fix/treeaci-741-convergence` branch was not included.
- Source worktree: `/root/projects/tensor4all-rust/tensor4all-rs-issue739`.
- Build artifacts: `/root/projects/tensor4all-rust/tensor4all-rs/target`.
- Backend/features: workspace default `tenferro-cpu-faer` (therefore the
  tensorbackend tenferro bridge/global defaults used by the crate).
- CPU/thread controls used for every command below:
  `taskset -c 0`, `RAYON_NUM_THREADS=1`, `BLAS_NUM_THREADS=1`,
  `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and
  `CARGO_BUILD_JOBS=2`.

Focused correctness commands, before source edits:

```text
RAYON_NUM_THREADS=1 BLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CARGO_TARGET_DIR=/root/projects/tensor4all-rust/tensor4all-rs/target CARGO_BUILD_JOBS=2 taskset -c 0 cargo test --release -p tensor4all-itensorlike --test tensortrain_inner -- --nocapture
running 2 tests
test test_inner_empty_tensor_trains ... ok
test test_inner_single_site ... ok
test result: ok. 2 passed; 0 failed; 0 ignored

RAYON_NUM_THREADS=1 BLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CARGO_TARGET_DIR=/root/projects/tensor4all-rust/tensor4all-rs/target CARGO_BUILD_JOBS=2 taskset -c 0 cargo test --release -p tensor4all-itensorlike --test bug_complex_inner -- --nocapture
running 3 tests
test test_inner_wrong_3site_nonstandard ... ok
test test_inner_wrong_with_nonstandard_index_order ... ok
test test_inner_wrong_with_two_site_indices_per_site_nonstandard_order ... ok
test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured

RAYON_NUM_THREADS=1 BLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CARGO_TARGET_DIR=/root/projects/tensor4all-rust/tensor4all-rs/target CARGO_BUILD_JOBS=2 taskset -c 0 cargo test --release -p tensor4all-itensorlike --test bug_norm_oom_large_tt -- --nocapture
running 4 tests
test test_long_tt_residual_uses_norm_without_dense_maxabs ... ok
test test_norm_25_site_tt_matches_local_reference ... ok
test test_norm_90_site_tt_uses_scalable_structured_path ... ok
test test_norm_small_tt_works ... ok
test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured
```

Existing norm benchmark command and output (site ladder, fixed `bd=16`):

```text
RAYON_NUM_THREADS=1 BLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CARGO_TARGET_DIR=/root/projects/tensor4all-rust/tensor4all-rs/target CARGO_BUILD_JOBS=2 taskset -c 0 cargo test --release -p tensor4all-itensorlike --test bench_basic_ops -- --ignored --nocapture
norm(20 sites, bd=16): 0.003s
norm(45 sites, bd=16): 0.007s
norm(90 sites, bd=16): 0.013s
test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured
```

Existing TensorTrain benchmark command and output (chi ladder, `L=32`, `d=2`;
the `tensor4all_inner_mps` rows are the representative TensorTrain rows):

```text
RAYON_NUM_THREADS=1 BLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CARGO_TARGET_DIR=/root/projects/tensor4all-rust/tensor4all-rs/target CARGO_BUILD_JOBS=2 taskset -c 0 cargo run --release -p tensor4all-itensorlike --example benchmark_tt_ops -- --L 32 --d 2 --zipup-L 10 --chis 4,8,16,32,64 --warm-up-time 0.1 --measurement-time 0.2 --min-samples 3 --inner-only
tensor4all_inner_mps,L_32_chi_4_d_2,198,0.865877,0.978368,1.010772,1.299592,0,...
tensor4all_inner_mps,L_32_chi_8_d_2,175,0.920760,1.098464,1.143838,1.637036,0,...
tensor4all_inner_mps,L_32_chi_16_d_2,230,0.667805,0.738098,0.875133,1.458521,0,...
tensor4all_inner_mps,L_32_chi_32_d_2,82,2.000288,2.460934,2.449682,2.853923,0,...
tensor4all_inner_mps,L_32_chi_64_d_2,17,11.594281,12.282173,12.359557,13.072608,0,...
```

The full stdout also reported paired raw eager and sitewise comparison rows
for every chi. The short timing windows are intended as an A/B smoke
measurement, not a stable performance claim.

## Planned implementation

The packed tensor layout is column-major `[left, physical, right]`. The
environment is stored as `[first_link, second_link]`, flattened as
`first_link + left_dim * second_link`, matching the old loop's
`current[left * left_dim + left_conj]` indexing. For each site the backend
formulation is:

```text
mid[a,p,r] = sum_b environment[a,b] * A[b,p,r]
next[s,r]  = sum_{a,p} conj(A[a,p,s]) * mid[a,p,r]
```

This is two backend einsum/GEMM-compatible contractions, with the second
input explicitly conjugated for `Complex64`; it preserves the existing
column-major output ordering. The old nested loop remains as a private
correctness oracle and is used by differential tests. Empty, boundary, and
single-site cases remain explicit.

The whole-train clone used only to normalize site order will be removed by
performing the same per-site permutation while packing. A canonical site is
copied directly; only a noncanonical site allocates a temporary permuted site.
No full train or dense full-train tensor is materialized.

## Candidate

The exact baseline correctness commands were rerun after the edit:

```text
tensortrain_inner: 2 passed; 0 failed
bug_complex_inner: 3 passed; 0 failed
bug_norm_oom_large_tt: 4 passed; 0 failed
packed_norm_backend_matches_oracle_for_real_and_complex_layouts: 1 passed; 0 failed
norm_squared_single_site_has_expected_value: 1 passed; 0 failed
```

Post-edit norm benchmark, same command and settings:

```text
norm(20 sites, bd=16): 0.003s
norm(45 sites, bd=16): 0.005s
norm(90 sites, bd=16): 0.011s
test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured
```

Post-edit TensorTrain benchmark, same command and settings (all
`tensor4all_inner_mps` rows):

```text
tensor4all_inner_mps,L_32_chi_4_d_2,373,0.501582,0.532741,0.536894,0.618962,0,...
tensor4all_inner_mps,L_32_chi_8_d_2,342,0.545916,0.579609,0.585998,1.041076,0,...
tensor4all_inner_mps,L_32_chi_16_d_2,160,1.009808,1.259406,1.252876,1.621597,0,...
tensor4all_inner_mps,L_32_chi_32_d_2,126,1.247614,1.353779,1.594971,2.342150,0,...
tensor4all_inner_mps,L_32_chi_64_d_2,19,10.575740,11.058436,11.005752,11.282277,0,...
```

The focused `--no-default-features` check was also attempted, but fails in
the pre-existing `tenferro-cpu` dependency because no CPU backend is selected
(`compile_error!("enable at least one CPU backend: cpu-faer or cpu-blas")`).
It does not reach the #739 crate code and is not a default-feature regression.
