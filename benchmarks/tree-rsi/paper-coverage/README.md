# Paper workloads and tree extension coverage

This is an experimental benchmark harness, not an assertion that RSI is
accepted for general use. See [coverage.md](coverage.md) for the obligations
and unavailable capabilities. No measured-results synopsis is committed;
generated observations, tensors, author data, source snapshots and reports
belong exclusively under ignored `target/tree-rsi/paper-coverage/`.

## What is compared

- TreeACI from the pristine source snapshot of fetched main commit
  `b881f39d9e72e32c43b9841fe10433fa5f63b167`.
- TreeRSI from the current worktree, identified by source-content and binary
  hashes. A commit hash alone does not identify untracked implementation code.
- CPU faer, f64/c64, one allowed CPU and one backend thread. Neither worker
  uses SimpleTT, tensorci or a legacy chain ACI validation path.
- Author data/scripts pinned to
  `153b25a8aa059d0147b45955d0842b2f32fa5d1d`; paper arXiv:2602.17974v1.
  Author code and historical reports are not correctness oracles.

Older ignored runs retain their own baseline (`dcc91f58` or `9ad67f2`) and
source identities. They lack the build receipts now required by the verifier;
they are historical observations, not verified evidence for the current code.
Do not add new receipts to them or relabel their measurements.

The primary curves use local tolerance `1e-12`, seeds 1/2/3, the stated caps,
TreeACI's enabled global guard and 2–20 sweeps. RSI normally uses
`ceil(cap / minimum_parent_physical_dimension) + oversampling` probes;
the actual width is recorded in diagnostics. In particular, spin-1 DMRG has
physical dimension **3**. The author's public DMRG script instead uses
`floor(cap/2)+10` and `eps=0`. Separate sensitivity cases set those controls
explicitly. They do not replace failed primary observations.

The library default RSI pivot tolerance is `1e-14`; the primary experiment's
explicit `1e-12` is a common local setting, not a global error target.
The author's LU cutoff is absolute, while this implementation exposes a
relative pivot threshold. At zero both disable tolerance-based stopping.

## Reproduction

Use an ignored virtual environment with NumPy and h5py. The recorded run used
NumPy 2.2.6 and h5py 3.16.0. No author Python benchmark is executed wholesale.
From the repository root:

```sh
python3 -m venv --system-site-packages target/tree-rsi/paper-coverage/venv
target/tree-rsi/paper-coverage/venv/bin/pip install h5py==3.16.0
target/tree-rsi/paper-coverage/venv/bin/python benchmarks/tree-rsi/paper-coverage/bootstrap.py
target/tree-rsi/paper-coverage/venv/bin/python benchmarks/tree-rsi/paper-coverage/build.py
```

`bootstrap.py` verifies pinned downloads without running them. The Git object
for the fixed main commit must be available locally. `build.py` reuses the
active worktree's Cargo release cache and builds only isolated workers.
Each build archives its executable, local dependency source closure, resolved
lockfile, Cargo configuration, compiler version and build controls under
`builds/<receipt-hash>/`. Runners verify these inputs against the current
checkout before scheduling and execute the archived binary. A stale or missing
receipt requires rebuilding; the mutable `rsi/worker` copy is never timed.
The original worker source is included so editing it also invalidates a build.

Every protocol records explicit cases (including options), seeds, algorithms,
phase and blocks, plus its harness snapshot and worker build identities.
Summarization checks the exact Cartesian schedule; artifact verification also
checks archived sources and binaries. Trace replay records its full schedule
and compares each outcome against the actual untouched-baseline observation.

Do not compile, prepare heavy fixtures, run diagnostics or run another
benchmark concurrently with a timed batch.

`prepare_fixtures.py` prepares a fresh fixture directory and refuses to
overwrite prior inputs. `fixtures.py` is a historical helper that writes
directly under `target/`; use the fresh-directory script for a reproducible
pipeline. `extend_fixtures.py` constructs narrow Gaussians and oscillatory
inputs independently, plus active matter and GPE, so the product benchmark
has an independent input path. It must run before any measured batch; never
replace fixtures mid-run.
`tree_cases.py` and `convolution.py` prepare the other independent fixtures.
All fixture metadata record the input representation error separately.

```sh
target/tree-rsi/paper-coverage/venv/bin/python benchmarks/tree-rsi/paper-coverage/prepare_fixtures.py
target/tree-rsi/paper-coverage/venv/bin/python -m unittest discover -s benchmarks/tree-rsi/paper-coverage -p test_reference.py
target/tree-rsi/paper-coverage/venv/bin/python benchmarks/tree-rsi/paper-coverage/run.py --group dmrg
target/tree-rsi/paper-coverage/venv/bin/python benchmarks/tree-rsi/paper-coverage/post_validate.py
target/tree-rsi/paper-coverage/venv/bin/python benchmarks/tree-rsi/paper-coverage/run.py --group functions
target/tree-rsi/paper-coverage/venv/bin/python benchmarks/tree-rsi/paper-coverage/run.py --group tree
target/tree-rsi/paper-coverage/venv/bin/python benchmarks/tree-rsi/paper-coverage/repeat.py
target/tree-rsi/paper-coverage/venv/bin/python benchmarks/tree-rsi/paper-coverage/supplement.py
target/tree-rsi/paper-coverage/venv/bin/python benchmarks/tree-rsi/paper-coverage/validate_followups.py
target/tree-rsi/paper-coverage/venv/bin/python benchmarks/tree-rsi/paper-coverage/matched.py
target/tree-rsi/paper-coverage/venv/bin/python benchmarks/tree-rsi/paper-coverage/supplement.py --gpe-tolerance
target/tree-rsi/paper-coverage/venv/bin/python benchmarks/tree-rsi/paper-coverage/gpe_global.py
target/tree-rsi/paper-coverage/venv/bin/python benchmarks/tree-rsi/paper-coverage/validate_followups.py
target/tree-rsi/paper-coverage/venv/bin/python benchmarks/tree-rsi/paper-coverage/calibrate.py
target/tree-rsi/paper-coverage/venv/bin/python benchmarks/tree-rsi/paper-coverage/diagnostics.py
target/tree-rsi/paper-coverage/venv/bin/python benchmarks/tree-rsi/paper-coverage/summarize.py
```

The diagnostic worker reuses the four hooks from `../main-comparison/trace.py`
and creates its print-only main snapshot if absent; the older benchmark suite
does not need to run first. Its patch and binary hashes are recorded. Its
times are excluded; exact output hashes, ranks and diagnostics must match
untouched main. The diagnostic coverage is every fixture/cap/ACI-option
combination at seed1, not all seeds. RSI reports every constructed edge rank.
Local matrix dimensions are recorded separately from actual bond dimensions.

## Validation boundaries

- n10 DMRG and 20-node branching trees: full physical-grid comparison.
- n20 DMRG: bounded local products, independent QR/SVD references and stable
  difference-QR norms. Record reference truncation, cut spectra and sample
  calibration. Never allocate the full `3^20` tensor.
- n50 DMRG and initial GPE checks: independent normalization/observables and disjoint Born
  and uniform holdouts, with a mixture-importance error estimate. This is
  **not a global norm certificate**. Tiny probability tails can make uniform
  relative errors enormous; retain them with that context.
- GPE follow-up: `gpe_global.py` checks all `2^26` probabilities using bounded
  left/right half-chain frames and blocked matrix multiplication. The true
  probability cache is computed once; identical outputs are validated once.
  Completed full-grid records supersede the earlier sampled acceptance.
- Analytic functions: exact products of the represented inputs followed by
  bounded reference QR/SVD, plus separate formula holdouts. Input error and
  reference truncations remain visible. A formula check cannot be replaced
  by a check against the compressed inputs.
- Active matter: the author field is float32. Contract the exact f64-decoded
  worker inputs in f64 for the product reference. The first recorded batch
  used a float32 dense reference; `validation-corrections.json` supersedes
  those 24 validation records while preserving raw outputs and timings.
- Complex convolution: full discrete Fourier-product and inverse FFT
  comparison. NumPy performs the FFT outside timing. This does not validate
  a Rust QTT Fourier operator or reproduce unspecified paper Fig.14 inputs.

Rank curves deliberately include insufficient caps. Interpret failure using
the rank lower bounds, actual output ranks and iteration stopping reasons.
`RankLimited` with a small true error is not a defect. A small pivot or a
successful return does not establish the output error.

Every call, including failed accuracy checks, remains in raw JSONL. Five
fixed timing blocks report variability per seed; a failed approximation
cannot claim an accepted speedup. Validation may be reused only for identical
input bytes/manifest and identical output bytes/core metadata; such reuse is
explicitly attributed. The original five-block run revalidated every output.

## Fidelity and unclosed capabilities

The public author's `tt_sketching_cache` accepts `seed` without seeding the
NumPy generator and samples `normal(0,10)`; the paper specifies standard
normal probes. The Rust implementation uses seeded standard normals. Literal
arrays in author plot scripts are not execution logs. Published Julia timing
numbers cannot be reproduced from the released Python alone.

The current Rust API has no arbitrary nonlinear map: paper ReLU and the old
GW reciprocal callback are not supported. The native downstream GW RSI
wrappers still refer to removed symbols/options; their validation and
provenance require repair before any downstream acceptance. A benchmark of
Hadamard products cannot establish that full pipeline's correctness.
