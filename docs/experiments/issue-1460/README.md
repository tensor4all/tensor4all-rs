# TCI2 scalar sampling ablation

Baseline is `89a4c1a0fac426b31710d1e91bac6e76348ef25e`. A adds the four scalar
buffer-reuse hunks. A+B adds the rook panel map hasher and the workspace
dependency edge to the already locked `rustc-hash` 2.1.3. SHA-256 of the
production diffs used to build the variants:

- A: `5b4e6a5ec541b584ce4ba7ce61bff4ab4808998a562463416edb53966c86c6a1`
- A+B: `7b0e7991c02a97a7ba1cd0053c1017e48df1f60ea5665e0449dc13e60e679b22`

## Reproduce

From the repository root with the implementation applied:

```sh
RAYON_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 docs/experiments/issue-1460/run.py \
  --results target/issue-1460/reproduction --release-tests
```

The runner archives the base, applies the production diffs, and builds each
variant in that same disposable workspace using `cargo build --locked --release`.
It saves the actual diffs, build logs, traces, per-fit times, allocator counts,
and optional focused release test logs in the results directory. It verifies
trace parity within the current run before timing.

## Preservation

The complex masked-product workload comes from [issue #1460](https://github.com/tensor4all/tensor4all-rs/issues/1460).
Each fit starts with a fresh source-value cache and uses six half-sweeps,
a bond cap of 128, seeded pivot search, and a tolerance of `1e-8`.
Every trace contains the complete ordered callback coordinates encoded as
fixed-length base-eight keys, the callback and unique-sample counts, sweep
ranks, final bond dimensions, error history, and 4,096 deterministic complex
readouts. A and A+B matched the baseline byte for byte in each row. Error and
readout bits are compared within one host and compiler. The hashes document
this run.

| Sites × dimension | Pivoting | Callbacks | Unique samples | Cache hits | Scaled max readout residual |
|---|---|---:|---:|---:|---:|
| 6 × 8 | Full | 188,496 | 16,992 | 91.0% | 2.31e-9 |
| 6 × 8 | Rook | 164,233 | 10,706 | 93.5% | 5.40e-9 |
| 10 × 8 | Full | 1,659,967 | 595,185 | 64.1% | 1.10e-8 |
| 10 × 8 | Rook | 923,526 | 210,362 | 77.2% | 1.25e-8 |

The residual is the maximum absolute difference from the source at the
readout points, divided by 32.

| Sites × dimension | Pivoting | Trace SHA-256 |
|---|---|---|
| 6 × 8 | Full | `509534dd49b5129bb57510b1ffe5a2b319c719990a34303e103c5020ce300dfb` |
| 6 × 8 | Rook | `c8fdbf4aa74090fb07693b9e0ec12b3cde8d28bdf89e80697fd6c65fa2f0e956` |
| 10 × 8 | Full | `1f3de6ac1ca6dea1f6c8af05e2480d6c6077b467503397699492b412dd9fae3c` |
| 10 × 8 | Rook | `bc13cc18a2c28f13634dfe07b4b933b46eb292d1e0aed8370744234385d94294` |

## Timing

Apple M3, 8 CPU cores, 16 GiB RAM, macOS 15.6.1 (24G90), AC power.
Rust 1.100.0 (`0ef1bc386059596dada8fb65b50425aab2ea637a`),
aarch64-apple-darwin, LLVM 23.1.5. The backend and threading environment
requested one thread. Variants ran serially, with variant order rotated
between cases. Each variant and case warmed up with repeated fits for at
least one second and recorded nine fresh-cache fits. The timer covers
interpolation, conversion to a tensor train, and source-cache cleanup.
Compilation, trace checks, readout validation, allocation probes, and printing
were outside the timed intervals. Core placement was left to macOS. These
are local wall-clock observations, with x86 measurements pending.

Median milliseconds, with observed min–max in brackets. [Raw timings](timings.csv).

| Pivoting | Sites × dimension | Baseline | A | A+B |
|---|---|---:|---:|---:|
| Full | 6 × 8 | 23.854 [23.731–23.980] | 10.064 [10.027–10.158] | 10.424 [9.931–13.377] |
| Full | 10 × 8 | 335.271 [325.027–345.173] | 175.685 [170.829–180.023] | 160.861 [157.039–164.638] |
| Rook | 6 × 8 | 57.352 [57.089–57.626] | 51.963 [51.747–51.991] | 34.302 [34.124–34.532] |
| Rook | 10 × 8 | 451.831 [444.285–455.978] | 437.618 [418.691–507.882] | 305.263 [301.782–322.421] |

Full pivoting bypasses the rook map, so the full A versus A+B timings give
no evidence of a hasher effect. The rook A-to-A+B reduction was observed at
both sizes on this host.

## Allocation attribution

Separate release executables used a thread-local guard around index construction
in the four targeted loops and a counting allocator forwarding to `System`.
Each scoped call counts allocation and reallocation separately. The guard
ends before invoking the callback, excluding the application cache and all
other allocations, including rook request construction, its panel map, and
factorization workspaces. Each probe executable also reproduced the clean
trace for its variant and case.

Every baseline entry in the four loops allocated. With A, the entries that
allocate correspond to the initial buffer growth once per panel. A and A+B
had identical counts. At 10 × 8:

| Sampling loop | Full baseline → A/A+B | Rook baseline → A/A+B |
|---|---:|---:|
| Two-site panel | 1,315,456 → 54 | — |
| Cleanup | 52,400 → 9 | 52,104 → 9 |
| Reconstruction `pi1` | 256,472 → 60 | 261,280 → 60 |
| Reconstruction `p_mat` | 35,574 → 54 | 36,376 → 54 |

[Raw allocation and reallocation counts](allocations.csv) cover both sizes.
