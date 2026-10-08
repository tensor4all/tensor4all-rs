# TreeTCI target memo paired results

**Decision: PASS.** All predeclared validity, correctness, default-path
non-regression and expensive-target primary gates passed.

The expensive-target geometric mean elapsed-time ratio is **0.2925**
(95% paired bootstrap CI **0.2913–0.2942**), about **3.42x** faster.
This is a synthetic expensive oracle, not a real downstream contraction
throughput claim. Cheap-oracle memo overhead is retained below; memo stays
disabled by default.

Seven alternating pairs, release/faer, CPU 2, provider/Rayon threads 1.
Every case preserved the declared diagnostics and sample bits exactly.
All load observations were <=8, CPU-2 steal fractions <=2%, all descendant
thread affinities matched CPU 2, and every paired-ratio relative SD was <=20%.
No rounds were excluded, retried or replaced.

| Case | Median time ratio | 95% CI | Evaluated / requested | Logical memo bytes |
|---|---:|---|---:|---:|
| chain_cos_129 | 0.9941 | 0.9827–1.0015 | 1.0000 | 0 |
| quantics_chain_r20 | 1.0000 | 0.9801–1.0082 | 1.0000 | 0 |
| tree_3x10_plus_centre | 0.9951 | 0.9816–1.0051 | 1.0000 | 0 |
| cheap_chain_8 | 1.0247 | 1.0133–1.0709 | 0.1079 | 3376 |
| cheap_chain_16 | 1.0417 | 1.0171–1.0534 | 0.2237 | 18160 |
| cheap_branch_8 | 1.0438 | 1.0274–1.0528 | 0.1366 | 9792 |
| cheap_branch_16 | 1.0471 | 1.0317–1.0678 | 0.1944 | 35232 |
| expensive_chain_8 | 0.2278 | 0.2276–0.2296 | 0.1079 | 3376 |
| expensive_chain_16 | 0.3665 | 0.3642–0.3701 | 0.2237 | 18160 |
| expensive_branch_8 | 0.2556 | 0.2527–0.2570 | 0.1366 | 9792 |
| expensive_branch_16 | 0.3429 | 0.3413–0.3457 | 0.1944 | 35232 |

Process peak RSS is recorded for the whole invocation, which contains
multiple cases. It is not a per-case RSS measurement or a logical cache
payload estimate. Complete default suites peaked at roughly 64–66 MiB;
the smaller memo suites peaked at roughly 15 MiB. The memo payload budget
was respected and no successful insertions were dropped in any case.

Reproduction-critical sources/commits, binary hashes and environment:
[manifest.json](manifest.json). Every measured round and host observation:
[rounds.json](rounds.json). Confidence intervals and all ratios:
[summary.json](summary.json). Raw stdout/stderr: the `global-*.txt` and
`memo-*.txt` files in this directory. The source-controlled
[runner](../../rust/run_treetci_memo_pairs.py) implements the
[predeclared protocol](../../2026-10-08-treetci-memo-protocol.md).

```bash
python3 benchmarks/rust/run_treetci_memo_pairs.py \
  --baseline /tmp/t4a-burn3-global-baseline \
  --candidate /tmp/t4a-burn3-global-candidate \
  --memo-candidate /tmp/t4a-burn3-memo-candidate \
  --output /tmp/treetci-memo-complete-rerun
```

Build the immutable baseline/candidate binaries as specified by the
protocol before rerunning; use a new output directory for every experiment.
