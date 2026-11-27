# Streaming leaf-side enumeration acceptance

## Protocol

Baseline: `7ffa116b` (`origin/main` at the start of #1441), with the same
test-only reporting harness used for the candidate. The candidate changes
only third-stage point generation and retention of its strongest pivots.

The primary contract is bounded point storage with an unchanged sorted,
deduplicated sequence. A 200-arm, 1,001-site structural regression checks
sparse cursor storage before and after its first two points. Small
exhaustive references cover overlaps, repeated anchors,
mixed local dimensions and noncanonical side order. Search tests cover
batch boundaries, equal residuals, retained-pivot limits and failures.

The numerical/cost campaign runs each of these once, with one execution
thread and the same optimized test binary configuration on each revision:

- R=6 and R=7 boxes: seeds 1 through 11, Sampled and PerBond (44 runs).
- #1352 reopened real table: seeds 1 through 11, both rules (22 runs).
- #1153 comb: delta=1 and delta=4, seed 11, both rules (4 runs).

Use the existing tolerances and validation points, with no tolerance changes.
Record target-evaluated points, iterations, rank history, termination and
independent error. All existing accuracy assertions must pass. Compare the
point costs and trajectories across the complete baseline/candidate campaign;
any differences must be reported, never selectively rerun or omitted. Sorted
enumeration is intended to preserve equal-residual tie-breaking as well.

Build through `ai-run build --mem 6G`; run the finished test binary through
`ai-run build --mem 6G` because this campaign counts points and checks values,
not time. Set `RAYON_NUM_THREADS=1`, `OMP_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and `BLAS_NUM_THREADS=1`,
with `--test-threads=1 --nocapture --include-ignored`. That point-count
campaign makes no runtime or RSS claim; the separate RSS diagnostic below
measures only enumeration storage. Logical storage bounds are distinguished from the
unmodified topology metadata and evaluator/memo storage.

## Results

All 70 baseline/candidate cells have identical target-evaluated point counts,
iteration counts, rank histories, termination and independently checked
errors. Both boxes are exact for all 11 seeds under both stop rules. No
tie-breaking change was observed or introduced. The campaign records do not
identify whether stage 3 ran or its pivot cap was binding in each cell.
End-to-end trajectory agreement is therefore scoped to these workloads;
exhaustive unit and injected search tests directly exercise bounded retention,
including eviction and equal-residual ties across batches.

| Case | Seeds | Sampled points, each revision | PerBond points, each revision |
|---|---|---:|---:|
| R=6 box | 1–11 | 47,974 | 47,974 |
| R=7 box | 1–11 | 67,645 | 67,645 |
| #1352 reopened table | 1–11 | 386,341 | 386,341 |
| comb delta=1 | 11 | 45,328 | 46,781 |
| comb delta=4 | 11 | 529,037 | 489,714 |

These are fresh counts on the current baseline, not a restatement of #1440's
historical comb costs. The memo is enabled with the existing fixture settings;
the counts describe oracle points actually evaluated during interpolation,
excluding the independent validation readout.

[comparison.json](comparison.json) retains every cell;
[baseline-output.txt](baseline-output.txt) and
[candidate-output.txt](candidate-output.txt) contain the complete test output.
Their elapsed times are incidental test-run output, not isolated timing
measurements. [provenance.json](provenance.json) records source/binary hashes,
hardware, profile and thread settings. Reproduce the comparison with
`python3 compare.py baseline-output.txt candidate-output.txt --output comparison.json`.

The four mutations in [mutations.json](mutations.json) were applied to an
isolated copy of the production generator and private tests: wrong radix
order, disabled deduplication, omitted candidate cap, and reversed equal-error
ties all fail assertions. The restored copy passes. Production source was
not mutated during the numerical campaign.

The 200-arm regression represents 30,877 unique full-site points (about 247 MB
of index payload if retained eagerly). It asserts that the generator starts
with 1,000 borrowed-anchor cursors and 5,000 varied-coordinate descriptors,
and can return its first two points with that same sparse cursor count.
This is a structural storage check, not an RSS measurement or a full
interpolation on a 200-arm tree.


## Peak RSS on the motivating point-enumeration workload

The separate [declared protocol](rss/protocol.json) uses the exact production
point generators from baseline `f3d0df4c` and candidate `11e2f5c7`, extracted
without edits by [measure_rss.py](measure_rss.py). It fully consumes the
200-arm / 1,001-site / five-anchor workload above with the production
1,047-point batch length. The baseline retains its full vector and reads
borrowed chunks; the candidate collects and releases one batch at a time.

| Arm | Whole-process peak RSS median | Range, three fresh processes |
| --- | ---: | ---: |
| Baseline | 240.00 MiB | 239.93–240.01 MiB |
| Streaming candidate | 10.18 MiB | 10.16–10.21 MiB |

All six processes are CLEAN, with no retry or omitted cell. Every process
emits 30,877 points and the same whole ordered-stream checksum
`15356029994682581386`. The fixed-order checksum is supplementary to the
existing exhaustive production tests for ordering and deduplication.

This isolates the issue's point-storage mechanism. It includes dimensions,
anchors, side lists, enumeration, batch storage and checksum readout; it
excludes TreeTN evaluation, evaluator/memo storage and topology construction.
It is not full-interpolation peak RSS or a runtime speedup measurement.
The retained GNU-time output also shows generation-only user CPU time of
0.08–0.09 s for baseline versus 0.30–0.34 s for streaming, with wall time
0.21 s versus 0.41–0.42 s. This is an observed slowdown in the isolated harness,
not a preregistered runtime comparison or evidence about full interpolation.
TreeTN evaluation is excluded, so its effect on end-to-end runtime is unmeasured.

The single-thread standard-library-only harness uses exact copied production
code, not a substitute tensor implementation.

The [summary](rss/summary.json), [all observations](rss/raw/attempts.json),
per-process GNU-time output and ai-run reports are retained. The
[manifest](rss/raw/manifest.json) pins compiler, source/controller/binary hashes
and the installed ai-run hash. Reproduce with
`taskset -c 0-7 python3 docs/experiments/issue-1441/measure_rss.py`
in a disposable copy without the existing `rss/raw` directory; it builds
through `ai-run build` and runs only the finished binary through exclusive
`ai-run measure`. No numerical tolerance was changed.


Regenerate the RSS medians and ranges without rerunning measurements:
`python3 docs/experiments/issue-1441/report_rss.py`. The helper checks every
accepted cell against its raw guard verdict, stdout and GNU-time peak before
writing `rss/summary.json`. It reports median/minimum/maximum of the three
per-arm KiB peaks divided by 1024; it does not aggregate process timings.
The earlier hand-written summary's platform string is omitted because the
RSS raw manifest did not record it; the declared host remains in the protocol.
The measurement controller and all recorded protocol/source hashes remain
unchanged. A fresh RSS campaign needs both `measure_rss.py` and `report_rss.py`
to recreate the summary.
