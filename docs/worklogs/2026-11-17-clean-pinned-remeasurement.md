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

All output remains local; the withdrawn remote branches and issue review
request remain withdrawn. No new PR or upstream publication is performed.
