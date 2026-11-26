# TCI2 scalar sampling

## Decisions

- Reuse the index storage inside each of the four agreed scalar matrices. Keeping each buffer local to the sampling loop preserves callback order and ownership at the batch boundary.
- Use the workspace FxHasher for the rook cache of internal panel coordinates. Candidate traversal and sampled maxima follow request order. The map supplies only keyed lookup and insertion.

## Verification conclusions and constraints

- Real and complex rank-two fixtures cover full and rook pivoting on nonuniform shapes with and without singleton dimensions. Batch guards cover coordinate order, value placement, and independently owned index vectors.
- Baseline, A, and A+B match exactly on same-host callback traces and numerical readouts. The allocation probe found one allocating entry per active panel after buffer reuse. The timings describe the measured Apple M3 workloads. x86 timing remains for maintainer validation. See the [experiment record](../experiments/issue-1460/README.md).
