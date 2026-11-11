# Seed 2026: candidate loss, a Guard intervention, and exactness limits

This investigation follows the [nesting audit](../treeaci-784-nesting/README.md).
The [results](../treeaci-784-root-cause.json) retain every case, negative
outcome, trace hash, accepted-hint count and late growing-edge witness.
Production pivot selection, Guard and stopping conditions are unchanged.
The auxiliary-candidate intervention below is diagnostic only and is **not
promoted** as the fix for #784/#854.

## Continuing mechanism of the late cycle

The instrumented R10/seed 2026 relative run reproduces all 50 original
maximum ranks and local errors exactly. Core records the complete Schur
maximum before each elimination step and the steps accepting old hints.
TreeACI records old pivot rank, remappable pairs, and surviving old rows and
columns before each update. These observations do not change selection.

In passes 39–50, 45 updates increase their edge's rank. Every one has lost
some old pivot rows after neighboring candidate replacement, while all old
columns remain available in that update's directed orientation. At the old
rank, the Schur maximum is 1.0025–2.5375 times the unchanged tolerance.
Increasing the rank is therefore required by the current local truncation
rule. At pass 50 on node 1→3, only 41 of 60 old row/column pairs can be
remapped, 34 old hints are accepted, and the residual after 60 pivots is
2.1283 times tolerance, so the update selects rank 63. Node 4→6 similarly
changes 170→172 with 140 remappable pairs and 114 accepted hints.

The observed loop is neighboring reset selection, loss of previous sample
coordinates, a changed local matrix and pivot path, then regrowth needed to
meet the same residual threshold. A subsequent reverse pass can shrink the
rank again. Strong hints cannot preserve pairs whose rows have disappeared.
The all-edge no-growth counter correctly refuses this sequence; late Guard
discoveries and full-nesting assertions do not explain its reset to 1.
These data identify the continuing mechanism, not a proof that conditioning
or finite-precision effects have no influence on pivot choices.

## Controlled Guard intervention

Immediately after Guard, the diagnostic scans center-directed candidate
dependencies. When a child is absent, it replaces a duplicate **auxiliary**
candidate slot with that child, retaining every selected-pivot prefix, all
dimensions, rank caps and RNG choices. Those auxiliary axes were zero-padded
by the current Guard; their candidate labels are not selected pivots. The
diagnostic adds no matrix factorization and does not change numerical targets.
It is best-effort: a missing dependency without a duplicate auxiliary slot
remains unresolved, so this is not a general mixed-capacity injection design.

| Relative mode, T=0.01 | Original hints | Auxiliary replacement |
|---|---|---|
| R10, seed 2026 | MaxSweeps/50 | Converged/17 |
| R10, seed 42 | Converged/15 | Converged/15; no replacement |
| R9, seed 2026 | Converged/30 | Converged/50 |
| R9, seed 42 | Converged/12 | Converged/9 |

The R10/seed 2026 intervention replaces just one slot after pass 3: on cut
47 (node 24→22), auxiliary slot 15 changes from duplicate sample 11 to missing
sample 9. Candidate closure then holds throughout and the trajectory changes
to convergence at 17. This establishes the early gap as an effective
trajectory trigger in this witness, despite its disappearance during pass 4
in the original run. It does not establish it as the sole cause of every
reset-mode cycle or prove eventual convergence.

Accuracy prevents treating the shorter run as sufficient evidence of a fix.
With the original M6a mixture, identical 5,000 importance points, seed and
analytic normalization, estimated relative L2 error changes from
0.00354242 ± 0.00004996 to 0.00471554 ± 0.00007556 (standard errors), an
increase of 33.1%. The original estimate is reproduced to floating-point
roundoff. The separate 1,094-point peak-aware checks also remain recorded:
relative sampled L2 changes 0.00111791→0.00142705. Neither set is an exhaustive
R10 error certificate. The R9/seed 2026 sweep regression is also retained.
No diagnostic construction times are used as performance evidence.

## Mathematical exactness

There is no unconditional guarantee of eventual exact reconstruction here.
The [TCI paper, Sec. 4.3.3–4.3.6](https://arxiv.org/html/2407.02454v3),
describes reset versus accumulative pivots, deletion propagating through
constricted bonds, and exploration missing sparse nonzero configurations.
It reports useful remedies, not a global convergence theorem for this tree
implementation with finite Guard searches and nonzero truncation.

The public-API [exactness tests](../../../crates/tensor4all-treeaci/tests/exactness.rs)
give two direct counterexamples using unchanged library behavior:

- A fully sampled two-site diagonal matrix with entries 1 and 1e-5 returns
  `Converged` with error 1e-5 at tolerance 1e-4. A tighter 1e-8 target recovers
  both entries. Keeping rank capped at one instead returns `RankLimited`.
- Record the finite query transcript for a constant function on a 12-bit
  grid. Put a spike of height 1024 at an unqueried assignment, then replay the
  same options and seed with default Guard enabled. Both query transcripts
  agree, and both return `Converged`, but independent dense subtraction finds
  maximum error 1024. The true constant-plus-spike tensor has rank at most
  two, so a large required rank is not the explanation.

The second example is also an indistinguishability argument: any deterministic
adaptive finite transcript that leaves an assignment unqueried is compatible
with an arbitrary change at that assignment. A fixed RNG seed makes this
argument apply to the randomized implementation. Random searching can improve
discovery probability; finitely many misses cannot certify its complement.
Positive tolerance, insufficient caps, finite search, and floating-point
arithmetic each require distinguishing approximation from mathematical
exactness. Exact-arithmetic reconstruction on a fully validated domain with
sufficient rank is a conditional statement; this algorithm does not establish
those premises or a monotone global-error descent property.

## Reproduction

Run `python3 docs/experiments/treeaci-784-root-cause/build_probe.py`. The
helper verifies source hashes, assembles the previous audit with
[extra_audit.rs](extra_audit.rs), attaches read-only Core residual/hint logging,
builds the expensive R10 workload in release mode, and restores production
source bytes. The workload shares the previous audit's probe and options;
[importance.rs](importance.rs) retains the M6a sampling formula from benchmark
revision `9dde54ca34032c1b7cbdeeae3d3f49ba3e50c557`. The helper records the
instrumented sources and binary hash under `target/issue784/root-cause/`.

The resulting `root-probe` takes the same arguments as the previous probe.
`T4A_ROOT_REPAIR=on` enables the auxiliary-only intervention;
`T4A_ROOT_IMPORTANCE=on` adds the independent 5,000-point check after
construction. All recorded runs use CPU 14 and the same four thread variables
set to one as the nesting audit. `ROOT_EDGE` and `ROOT_LU` records match in
order. Original and supplementary case lists were recorded before their
respective runs; every outcome is retained. R9/seed 42's baseline history is
reused from the exact earlier nesting replay, not a new timing measurement.
