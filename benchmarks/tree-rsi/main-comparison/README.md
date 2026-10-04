# Main TreeACI versus worktree RSI

This experiment uses native TreeTN inputs and public product APIs only. It
never imports SimpleTT or TensorCI. It changes no library implementation.
The TreeACI baseline and its tensor dependencies come from an isolated
`git archive` of fetched `origin/main`; RSI and its dependencies come from the
current worktree. A dirty worktree is identified by a source-content hash,
not just its HEAD commit. This is a comparison of those complete source trees,
not an attribution of all performance differences to one algorithm.

## Run

From the repository root, with NumPy installed:

```sh
git fetch origin
python3 benchmarks/tree-rsi/main-comparison/compare.py
# Use the output directory printed by the first command:
python3 benchmarks/tree-rsi/main-comparison/supplement.py target/tree-rsi/main-comparison-TIMESTAMP
python3 benchmarks/tree-rsi/main-comparison/trace.py target/tree-rsi/main-comparison-TIMESTAMP
python3 benchmarks/tree-rsi/main-comparison/summarize.py target/tree-rsi/main-comparison-TIMESTAMP
```

Replace `TIMESTAMP` with that actual directory name. Build logs, binaries,
source snapshots, protocols, raw tensors, observations, traces and summaries
stay in this ignored directory. Never commit them. Do not run compilation or
other benchmark processes concurrently with the timed phase. Each phase has a
180-second timeout; failures and timeouts are retained as observations.

## Cases and reference

All physical dimensions are 2; cores contain signed independent Gaussian
values. Operand seeds are 11 and 29. Each physical cut bounds its stored input
rank, avoiding inflated leaf bond dimensions. Each operand may have different
bond dimensions. Each core uses sorted-neighbor bond axes followed by its
physical axis; flat buffers are column-major.

Initial cases: chains of 16/18 sites and a 19-site heap binary tree, input
caps 128/256, output caps 128/256. These include insufficient-rank cases and
cases using exact complement enumeration on the critical RSI cut.

The supplement adds 20-site chains and branching trees formed by joining two
10-node binary trees. Their central cut has physical dimension 1024 per side.
Operand bond caps are 16 × 16 or 128 × 2; the exact Hadamard product has rank
at most 256 on every edge. Output caps 256 and 512 are both tested, with
seeds 1/2/3. The input caps 16 × 16 case still has a genuine output rank of
256, verified by the reference SVD. No fixture is merely a low-rank positive
signal with a large unused cap.

NumPy contracts each input once and multiplies their dense arrays. The largest
reference contains `2^20` entries. An independent cut SVD records the numerical
input/product ranks and the Eckart–Young relative Frobenius tail bound. This
bound can establish that a cap is insufficient; a small bound on one cut does
not by itself certify that all cuts fit. The supplement's product-rank bound
also follows directly from the two input bond dimensions.

Workers read identical fixture bytes. Their native input contractions are
checked against NumPy before accepting the output measurement. Every output
entry is compared independently in Python; no sampled error or algorithm
success flag is used as acceptance. The relative L2 gate is `1e-8`; local
algorithm tolerances are `1e-12`. The gate is fixed across cases.

## Timing and diagnostics

Release builds use tenferro CPU faer and f64. One allowed CPU is pinned and
common thread controls are set to one. Input contractions initialize the
backend outside the timer. The measured region is the public `hadamard_many`
call, including RNG initialization, with inputs and returned results passed
through `black_box`. Dense output export and validation are outside the timer.

All 14 cases run three blocks of all three seeds, with paired algorithm order
alternating. Report all nine times per algorithm/case, their median and range,
and per-seed repeat CV. A maximum per-seed CV above 10% labels timing unstable;
no selective retry or dropped sample can remove that label. Times for inaccurate
results remain visible, but do not establish speed at equal accuracy. There is
no formal statistical or application-wide speedup claim from this single host.

TreeACI uses the main defaults, including max 20/min 2 sweeps and the global
guard. RSI uses `k = ceil(cap / 2) + 5` (physical dimension 2). Thus increasing
cap also increases sketch width; the cap-512 experiment is **not** an isolated
study of rank-cap sensitivity. Report local matrix shapes, actual returned
bond dimensions and exact/sketch flags alongside accuracy.

TreeACI's sweep maxima alone cannot establish intermediate peaks. After all
timing finishes, `trace.py` builds a separate diagnostic copy with four
print-only instrumentation hooks: initial ranks, every committed edge rank,
global pivot rank injection, and every local matrix shape. Its patch and hash
are saved. It replays all cases/seeds and requires exact output byte hashes,
ranks, termination and diagnostics to match untouched main. Diagnostic times
are excluded. This measures active output bond ranks through the algorithm;
local matrix dimensions are separate and are not called bond dimensions.

RSI constructs each output edge once in postorder; edge reports therefore
cover the peak constructed output bond rank as well as the final rank. They
also expose candidate matrix dimensions. Neither metric is total allocation
size. Process RSS includes inputs and dense reference exports and is recorded
as such, not advertised as the algorithm's peak working memory.

This benchmark does not constitute a proof of randomized tree RSI, complex
scalar validation, or downstream GW acceptance.
