# Guard candidate abstraction (#1265)

This issue is tracked under #854 and relates to #794 and #784. The branch is
based on current main `66841f68`, independently of the strong-pivot-hint branch.

## Fixture-first scope

Following shinaoka's [review comment](https://github.com/tensor4all/tensor4all-rs/issues/1265#issuecomment-6792048853),
the first commit contains only small characterization fixtures. It does not
modify candidate activation, padding, numerical selection, or stopping rules.
The tests explicitly demonstrate a current defect; the abstraction fix will
turn the same observed failures into required-closure assertions.

A four-node chain reproduces false saturation: a child has two physical slots
but only one distinct candidate after paired padding. A subsequent parent
candidate references an unavailable child. A five-node chain with a valid
rank-two initial guess and two identity sweeps reproduces true saturation: the required child already
has two distinct selected samples at the configured cap, while its parent has
remaining headroom. Both fixtures cover the reflected cut orientation, preserve
selected pivot pairs and represented output values, and respect configured and
algebraic bond dimensions. No slow R10 workload or numerical tolerance change
is required to reproduce the abstraction defect.

## Abstraction work to follow

Useful candidate membership must be represented separately from output-axis
padding. Distinct required children either receive representable candidate
capacity or remain unresolved Guard evidence; a parent cannot be published as
usable when its required child is missing. Selected prefixes, paired output
dimensions, caps and rollback remain the invariants. Reset's required
center-directed nesting must not be confused with full outward nesting.
#784/#854 rank-cycling policies and stopping changes stay outside this diff.

The earlier padding-slot prototype is excluded from this change and is not
promoted as a fix. Its experimental records are retained separately. This
fixture commit changes no production algorithm.


## Rejected prototype and numerical follow-up

The query-pool experiment separates distinct useful queries from selected output
basis dimensions. It preserves the deterministic closure fixtures, selected
prefixes, paired dimensions and transactional rollback. It remains a rejected
prototype: the matched R9/R10 seed-42/2026 study fails its predeclared independent
accuracy gate in both seed-2026 cases. The complete preceding numerical outcomes
are retained with the [follow-up record](../experiments/treeaci-1265-first-divergence/README.md).

That follow-up localizes the R9/seed-2026 regression to adaptive Guard feedback.
A restored child introduces two independent query directions in sweep 2, update
24. Common matrix entries agree exactly and the selected cross condition is
about 51. The prototype correctly repairs two old residuals; the same random
starts consequently produce a different second Guard proposal batch. Replacing
only that batch swaps the final accuracy in both directions, with all later
searches running naturally. Full replay, prefix-three and self-replay controls
agree. This identifies the observed trigger without establishing a production
fix or excluding conditioning effects in later matrices.

The record includes a frozen candidate patch as an experimental input, not a
change to production dispatch. It preserves all fourteen measured follow-up
cells, including failed incomplete probes, and an isolated-worktree reproducer.
The original four-case accuracy gates remain failed. Keep candidate closure,
Guard proposal-quality research and #784/#854 stopping policies distinct;
repaired points must not be labeled unresolved failures to recover a favorable
trajectory. Root README capabilities remain unchanged.

## Landing the abstraction change (2026-11-13)

After #1265 was reopened, the maintainer restated the next step as the owning
abstraction change of scope items 2 and 4, with the characterization fixtures
flipped to required-closure assertions. The query-pool patch archived in the
follow-up record (`candidate.patch`) is that change. It now lands unchanged on
main `501ebb4a`, and it applies without conflicts. The fixtures assert the
required closure in both orientations, including the truly-full child.

The matched four-case study was rerun on `501ebb4a`. Each arm had its own
target directory: the baseline was a detached worktree at `501ebb4a`, the
candidate this branch. The study used the unchanged probe, protocol and
validation sets of the preceding record. All eight runs reproduce the 11-11
values to every printed digit:

| R | Seed | Importance L2, baseline → candidate | Sampled max error | Termination |
|---|------|-------------------------------------|-------------------|-------------|
| 9 | 42 | 0.0031916785 → 0.0031916785 (0.0%) | 1.56e-4 → 1.56e-4 | MaxSweeps/50 both |
| 9 | 2026 | 0.0024271927 → 0.0037736277 (+55.5%) | 1.18e-4 → 2.05e-4 | MaxSweeps/50 both |
| 10 | 42 | 0.0050954290 → 0.0039258703 (−23.0%) | 1.93e-4 → 1.34e-4 | MaxSweeps/50 both |
| 10 | 2026 | 0.0027517176 → 0.0031243721 (+13.5%) | 1.09e-4 → 1.16e-4 | MaxSweeps/50 both |

In every case the maximum rank and the Guard call count are unchanged. The
5% per-case gate was predeclared by this work, not by the issue's acceptance
list. The seed-2026 cases still exceed it. The cause recorded in the
follow-up stands: correctly repaired Guard points change the later random
proposals. The change is landed as the structural fix it is. It is not a
convergence or accuracy improvement, and the #784/#854 rank-cycling work
stays separate. No tolerance, rank floor, Guard setting or sweep limit changed.
