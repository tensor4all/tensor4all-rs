# Seed 2026 nesting investigation

This is a private structural diagnostic for #784 within #854, not a change to
TreeACI's algorithm or a performance study. The machine-readable
[results](../treeaci-784-seed2026-nesting.json) retain all seven cases, control
checks, trace hashes, and the main witness's per-edge rank history.

The R10/seed 2026 relative-tolerance baseline and fix exactly reproduce every
maximum rank and local error in the supplied M6a records. The fix records
3,001 states: initialization, both sides of every committed edge update, pass
completion, and Guard completion. A diagnostic-free construction gives the
same rank and error sequences. R9/seed 42, R9/seed 2026 and R10/seed 42 also
match the supplied fix controls exactly.

## What the checks mean

For a directed cut, a component sample records a local coordinate and incoming
child samples. The diagnostic tests whether each child ID belongs to the
current set on the incoming cut. It distinguishes three relations:

- `pc`: selected pivot samples reference current child candidates. This is
  the relation tested by the existing test-only `validate::check_nesting`.
- `pp`: selected pivot samples reference selected child pivots, a stronger
  condition than `pc` when Guard has appended candidates.
- `cc`: all candidate samples reference current child candidates.

`full` checks both directions of every cut. `active` checks directions toward
the current single-node canonical center. `local_pc` checks both sides of the
edge being updated; only its **after-update** value describes the newly
committed cross. Counts are missing child references, not distinct bad cuts.
The snapshot summary counts states containing at least one missing reference.

In the exact R10 relative replay, full nesting first fails during pass 2, but
both active `pc` and active `pp` remain valid throughout. Every freshly
committed cross passes `local_pc`. Candidate prefixes match the selected
pivot order and candidate lengths match output bond dimensions in every case.
Full nesting can fail under reset two-site updates: the chain TCI discussion
in [Sec. 4.2–4.3.1 of the TCI paper](https://arxiv.org/html/2407.02454v3)
distinguishes full nesting from the partial nesting needed by a local update.
The center-directed check is the structural interpretation used here; it is
not a numerical interpolation or conditioning certificate.

## Guard candidate closure witness

After Guard in pass 3, active `cc` fails on directed cut 45 (node 22→21).
Candidate sample 9 references child sample 9 on cut 47 (node 24→22), which
is absent from that cut's current candidate list. The child list has length
16 but only 15 distinct IDs: sample 11 is duplicated. Its raw rank already
equals its algebraic bound, 16. The missing child record exists in the arena;
it is absent specifically from the current candidate list. Updating cut 45
in pass 4 removes the violation.

The source path is `global_guard::inject_global_pivots` followed by
`SampleArena::inject_global_point_masked`. Opposite directions are padded
together, so a projection already present on one side can be appended again
when the opposite side is new. Growth capacity uses raw bond dimension.
Projection constructs component records and their dependencies, including
through inactive child cuts, while activation does not add those child IDs
to saturated candidate lists. This combination permits the observed closure
gap. Duplicates alone are allowed by the current candidate representation.

This gap also occurs in the fresh-pivot baseline and several convergent
controls. It is absent in the R10/seed 42 control. Its causal contribution to
late cycling is unproven: in the main fix replay, it disappears in pass 4,
whereas passes 39–50 still reset the all-edge stability counter to 1 and
Guard finds zero new points. At pass 50, cut 1→3 grows 60→63 and cut 4→6
grows 170→172 despite the maximum rank remaining 372.

The small R4 CTTN control also has ten snapshots with active `pp` failures,
while active `pc` remains valid and it converges in four passes. Thus the
chain's stronger pivot-to-pivot observation must not be generalized to every
tree. This audit does not establish numerical cardinality, interpolation
identities, or conditioning of retained frames.

## Reproduction and limitations

Run `python3 docs/experiments/treeaci-784-nesting/build_probe.py` from a
checkout with the recorded source hashes. The helper verifies those hashes,
temporarily attaches [audit.rs](audit.rs) to the TreeACI crate and installs
[probe.rs](probe.rs) as an example, builds in release mode, copies the binary
to `target/issue784/nesting/nesting-probe`, and restores production source
bytes in `finally`. Release is used because the R10 numerical workload is
impractically slow unoptimized. The helper's `--clean` option builds the same
workload without instrumentation as `nesting-clean-control`.

The binary's arguments are R, topology (`nblock` or `cttn`), temperature, seed,
and a boolean selecting relative tolerance. Use `10 nblock 0.01 2026 true`
for the main witness. Set `T4A_NESTING_HINTS=off` on the instrumented binary
to use the existing empty-hint complete-pivot path. All recorded cases use
CPU 14 and `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, and
`RAYON_NUM_THREADS` set to 1. Stderr contains JSON records prefixed by
`NESTING`, `NESTING_CANDIDATE`, or `NESTING_CONTROL`; stdout contains the
construction result. Initial positive controls delete required children
from cloned sets and check that inward and outward-only failures are
distinguished. They do not alter the construction's state or RNG.

Three exact inputs, explicit session, cap 4000, default Guard, 8 GiB budgets,
two-pass minimum and 50-pass maximum match the M6a setup. The fixture's
frequency offset uses the M6a arithmetic order. Absolute mode uses 0.003183
and is only a supplementary control: it diverges from the supplied maximum
rank history at pass 27, although both stop at 46. The clean and instrumented
versions agree exactly with each other. The origin of that cross-harness
divergence is unresolved. The relative seed-2026 conclusion does not depend
on this supplementary case. Diagnostic timings are excluded from comparisons.
