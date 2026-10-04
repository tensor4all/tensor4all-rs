# RSI benchmark entry points

## Current high-rank comparison against main

Use [main-comparison/README.md](main-comparison/README.md). This compares an
isolated source snapshot of fetched `origin/main` TreeACI with worktree RSI,
uses signed native TreeTN fixtures with actual ranks 128/256, independently
checks every dense entry, and retains failed accuracy cases in timing output.
Generated results belong only under ignored `target/tree-rsi/`.

The [paper workload harness](paper-coverage/README.md) adds author DMRG inputs,
Gaussian/oscillatory products, active matter, complex GPE data, high-rank
branching trees, fixed timing repetitions and diagnostic rank replays. Read
its validation boundaries and unavailable capabilities before interpreting
any result as acceptance or as a reproduction of the paper.

## Historical small-rank experiment

The workflow below compares both algorithms from the same worktree and uses
ranks 1–16. It does **not** satisfy a comparison against main, high-rank
validation, or acceptance of the current implementation. Its accuracy-based
timing selection excludes failed cases, so use the current workflow for a
complete comparison. It remains here to reproduce the earlier diagnosis.

This private executable depends on both algorithms. The RSI library has no
ACI dependency. Run from the repository root:

```sh
python3 benchmarks/tree-rsi/run.py
python3 -m unittest discover -s benchmarks/tree-rsi -p 'test_*.py'
```

The driver builds a release binary, hashes source contents (including new
untracked source) and the executable, pins one allowed CPU, and sets common
thread controls to one. JSONL attempts and host metadata go under ignored
`target/tree-rsi/`; `--output` selects a new output path. Never commit these
outputs, old benchmark data, generated plots, or protocol snapshots.

Six cases are fixed in source: chain lengths 16/32 at input rank 2, chain
lengths 32/64 at rank 4, a 15-node binary tree at rank 2, and a 64-node star at
rank 1. Inputs use fixed positive random cores, two operands and physical
dimension 2. This is a bounded synthetic experiment, not a broad workload
survey or a test of arbitrary signed/complex application data.

Both algorithms get cap `input_rank²`, local tolerance `1e-12`, root `n-1`,
and seeds 1, 2, 3. No settings are tuned after seeing errors. Each result is
compared with the true product on 2,048 independent uniformly sampled points
at a fixed relative L2 threshold of `1e-8`. Truth is evaluated once per case,
in batches, outside timing; errors use amplitude differences rather than
subtraction of large Gram norms. The accuracy calls also warm the algorithms.

Only cases where all three seeds of both algorithms pass advance to three
alternating paired timing blocks. Every timed result is validated again.
Timing includes only the public product call on prebuilt inputs, including
its RNG initialization and algorithm work. Input construction, reference
contraction, output validation and JSON serialization are outside the timer.

The Python validator independently recomputes acceptance from every recorded
error and rejects nonfinite, missing, duplicate or failed attempts. It never
trusts a saved success boolean. The entire experiment must pass before a timing
summary is produced. Report per-algorithm medians and the median of paired
TreeACI/RSI time ratios; one host run cannot establish universal speedups.

This protocol tests equal sampled accuracy at one fixed setting. It does not
prove a global error bound, compare against the author's Python performance,
or accept downstream GW outputs or iteration convergence.

## Interpreting a failed accuracy gate

Failure at a fixed rank budget is a measurement, not an implementation bug
classification. An exact representation rank bound does not guarantee that an
iterative method converges under that same cap. `RankLimited` and `MaxSweeps`
must be distinguished from convergence, and convergence diagnostics still
need independent output checks. Record actual per-edge ranks from the returned
tree, TreeACI termination and sweep history, and RSI sketch width/local pivots.
These inspections are outside timing. The validator also rejects actual ranks
exceeding the configured cap or missing rank/termination records.

Use the bounded `rank_diagnostic` example for the binary-tree case. It compares
all 32,768 entries for caps 2, 4, 8, 16 and seeds 1, 2, 3, keeping RSI's sketch
width fixed at 7 to isolate cap sensitivity. An additional cap-4 case raises
TreeACI’s minimum sweeps from 2 to 4 to test early stopping:

```sh
cargo run --release -p tree-rsi-benchmark --example rank_diagnostic
```

The protocol also records an untruncated SVD across cut `(0, 1)` of the exact
product. The tail singular-value norm gives a lower bound on the full relative
L2 error of any output with that cut rank. This distinguishes an insufficient
rank budget from interpolation errors without relying on either algorithm's
diagnostics.

It is a diagnostic experiment with no timing claim. Its results do not retune
or replace the original fixed-budget benchmark. A performance comparison must
state achieved accuracy, actual ranks, and termination alongside each method's
parameters; equal configured caps alone do not establish comparable quality.
