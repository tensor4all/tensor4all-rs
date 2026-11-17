# Clean downstream-pinned remeasurement

## Decisions

Measure unmodified downstream-pinned dependencies against the production source
at `86b27b91d03c0b7027be5547c44c45829220d8de` in a new worktree. The only changes are a public-API
measurement example, its dev references and reproducibility/reporting scripts.
The complete lock diff adds only libc and serde_json dev references. Git-cache
source is checked against the locked commits; Cargo's generated `.cargo-ok`
fetch marker is explicitly distinguished from a source modification. No
withdrawn repair or path override is used, and no latest-upstream conclusion
is claimed. The native scaling grid is separate from the withdrawn MPI work.

## Verification conclusions and constraints

The [complete report](../../benchmarks/results/2026-11-17-clean-pins.md) records
130 independent processes, real/complex five-block sampling, all configurations
and paired speedup intervals. The [original resource-gate report](../../benchmarks/results/2026-11-17-clean-pins-ai-run.json)
has verdict=CLEAN, exit=0 and no contamination reasons. Builds and reporting
checks used ai-run build; the measurement ran only precompiled binaries under
ai-run measure. Source/dependency/binary audits match before and after the
measurement. Oracle results and all cross-configuration work/value checks pass.
No earlier sample is reused. The report generator rejects CONTAMINATED runs,
nonzero exits and incomplete numerical validation.

The clean pin scales the single engine to about 3.4× at four threads on four
physical cores; oversubscription reduces its benefit. Mixed B2 lanes are much
slower than the four-thread engine. All 30 mixed processes observe only CPU8
and CPU10 in the evaluator callbacks; engine construction narrows the creator
from its four-core allocation to CPU8. This is measured behavior of the pinned
stack, not a harness restoration or a local fix. It warrants following the
constructor/placement lifecycle; it does not establish behavior of latest
upstream. The observer's original affinity was restored after the run.

The user authorized a comment on the existing upstream tracking issue and a
push of this clean evidence branch. The
[comment on tenferro #2597](https://github.com/tensor4all/tenferro-rs/issues/2597#issuecomment-6936966152)
records the CLEAN run and the remaining dependency-train constraints. The old
withdrawn prototype branches stay withdrawn; no PR is created.

## Downstream pin conventions and pending version choice

CONTRIBUTING.md requires one provider identity and refreshed feature-fixture
locks. Historical upgrades such as tensor4all-rs commit c0f3ca94 update tenferro
and Hataori together to matching merged Git revisions; they do not require
waiting for a crates.io release of tenferro. The current Hataori adapter still
pins tenferro 32a4e98c, while latest tenferro 0.12/main still locks threadteam
0.5.1. Changing only the top-level tenferro SHA neither imports the affinity fix
nor keeps the adapter's CpuEngine identity unified.

The official merged threadteam fix 6faa2e2 still declares version 0.5.1 and is
API-compatible with the old train. A root crates.io-to-Git patch could therefore
be a temporary workspace-only bridge, provided one source identity is verified.
It is not an established patch convention in the canonical manifest history,
and Cargo dependency patches do not propagate to external Rust consumers. No
such override or local fix is introduced during this convention check.

The measurement pin stays unchanged while the user reviews the version choice:
follow a coherent official upstream train, or explicitly select a temporary
workspace-only source pin. The tracking issue schedules the coordinated 0.6
train after tenferro 0.12. Publishing the baseline evidence does not claim that
this pin update or the upstream repair integration has been completed.


A compatible-version update dry run also locks zero packages: no published
0.5.x patch fix can currently be selected by cargo update. The ordinary
single-identity and fixture-lock checks pass on the retained baseline, as do
release example Clippy, the three fail-closed reporting tests and formatting.
The evidence-only branch is pushed under the user's authorization; dependency
selection remains pending the result of this convention review.
