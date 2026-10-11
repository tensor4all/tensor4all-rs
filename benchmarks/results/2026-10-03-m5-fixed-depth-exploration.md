# M5 fixed-depth partition exploration

## Status and provenance

Exploratory data, not decision-grade evidence. The runs were made on
2026-10-03 from a scratch copy of
[`benchmark_m5_fixed_depth_partition.rs`](../rust/benchmark_m5_fixed_depth_partition.rs);
the git revision and build fingerprint were not recorded, so the timings are
not baselines for the current branch. Each workload was run once with one
seed. The raw JSONL files, the generated tables, and their renderer are in
[`m5-fixed-depth/`](m5-fixed-depth/); [`SHA256SUMS`](m5-fixed-depth/SHA256SUMS)
lists their checksums.

The experiment fixes the first `k` quantics bits in a chosen order and runs
one **uncapped** TreeTCI per resulting patch. It does not exercise the
adaptive pQTCI driver and does not compare split-site selectors. Every patch
uses the same absolute tolerance `rtol * analytic max|f|` with TreeTCI's
sampled pivot (max-norm) criterion. The reported relative L2 errors are
estimates from 1,024 deterministic points stratified over the depth-6
partition, not certified bounds.

Ranks are realistic for the target workloads: the unpartitioned runs reach
ranks 26–130 on the branched tree and 38–707 on chains.

## Matched-accuracy comparison

With the same per-patch absolute tolerance, partitioning drifts to a larger
L2 error than the unpartitioned run (up to 17 times on the Lorentzian ridge
tree at depth 4).
Comparisons at one tolerance are therefore not valid timing comparisons. The
unpartitioned runs were repeated at looser tolerances (`mono_*` files, and
`ridge_tree_R8_e0.03_mono_rtol*`). The table pairs each partition with the
unpartitioned run of the nearest **equal or better** sampled L2 error, so
the ratios do not favor partitioning.

| Workload (8 bits per variable, MSB order) | Partitioned run (rel L2) | Unpartitioned reference (rel L2) | Time ratio | Parameter ratio | Evaluation ratio |
|---|---|---|---:|---:|---:|
| Lorentzian ridge, eta 0.03, branched tree | k=4 (5.3e-4) | rtol 1e-3 (4.6e-4) | 0.24 | 0.42 | 0.44 |
| Lorentzian ridge, eta 0.1, branched tree | k=2 (7.1e-5) | rtol 1e-4 (6.1e-5) | 0.56 | 0.61 | 0.78 |
| Four peaks, eta 0.03, branched tree | k=1 (5.7e-4) | rtol 1e-4 (5.1e-4) | 0.52 | 0.73 | 0.51 |
| Four peaks, eta 0.01, branched tree | k=4 (6.0e-3) | rtol 1e-4 (1.7e-3) | 0.75 | 0.53 | 0.17 |
| Lorentzian ridge, eta 0.03, chain | k=4 (1.0e-3) | rtol 3e-4 (6.4e-4) | 6.4 | 5.0 | 3.5 |
| Spectral function, eta 0.3, branched tree | k=4 (5.1e-5) | rtol 1e-4 (4.6e-5) | 2.35 | 7.98 | 1.93 |

The four-peaks eta 0.01 reference is three times more accurate than the
partition; the looser unpartitioned run (rtol 3e-4) is far less accurate
(2.4e-2), so this pair brackets the matched point rather than hitting it.
The Gaussian ridge (width 0.03, branched tree) has no looser unpartitioned
run; at depth 4 its error is 1.2 times the unpartitioned one, with time 0.59
and parameters 0.34 of the unpartitioned run, so that ratio is approximate.

## Findings

- Partitioning helps localized features on the branched tree: at matched
  accuracy the narrow Lorentzian ridge needs about 4 times less TCI time and
  2.4 times fewer parameters at depth 4. At that depth the patches on the
  ridge keep rank 38 while the off-ridge patches fall to 4–5, the
  block-diagonal picture of a patched identity matrix.
- It does not help the chain control: the chain already represents the ridge
  at rank 38, and every partition costs more.
- The delocalized spectral function gains nothing at any depth from 0 to 6;
  patch ranks stay near the unpartitioned rank through depth 4.
- Cost rises again past depth 3–4, probably because every patch carries a
  fixed evaluation floor (not separately measured).
- Quantics bit order matters: on the Lorentzian ridge the x-first order is
  worse than MSB interleaving at depth 6 (6.33 s against 3.14 s at equal
  tolerance).

## Known defects of this data

- **Corner-localized misses.** On the Gaussian ridge with width 0.01, from
  depth 3 on, patches that the ridge enters only near the centre corner
  converged at rank 1; the true L2 error is about 0.1 while the stratified
  sample reported a few `1e-5`. Its rows are excluded from the comparison.
  This is the limitation recorded in
  [`tree-patching-error-contract.md`](../../docs/design/tree-patching-error-contract.md#known-limitation-corner-localized-misses).
- The narrow Gaussian ridge on the chain falsely converges even without
  partitioning (sampled error 1.6e-2); its times are not an accuracy result.
- The four-peaks workloads reach only ranks 26–30 on the tree, so the
  high-rank localized case is untested.
- Single runs, one seed, unknown revision.
- **Thread settings were not recorded.** The related TreeTCI threading
  symptom is tracked in [issue #670](https://github.com/tensor4all/tensor4all-rs/issues/670).
  Oversubscription may inflate these absolute timings. Even if the runs used
  the same thread settings, its effect can differ between the unpartitioned
  network and smaller patches, so the time ratios also need a rerun with
  pinned thread counts before they support a performance decision. Parameter
  and evaluation counts remain descriptive evidence for these recorded runs.

## Reproduction

```bash
cargo run --release -p tensor4all-partitionedtreetn \
  --example benchmark_m5_fixed_depth_partition -- \
  ridge tree 8 0.03 0,1,2,3,4,5,6 msb 16 20
```

Tables are regenerated from the raw files with
`python3 table.py <tree|chain> <baseline file> <file>...` inside
`m5-fixed-depth/`.
