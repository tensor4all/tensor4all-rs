# Resolve the remaining TreeACI cached-cut overhead

Tracking: [#671](https://github.com/tensor4all/tensor4all-rs/issues/671), under
[#854](https://github.com/tensor4all/tensor4all-rs/issues/854).

## Defect and repair

The public hinted evaluator already chooses an edge cut for its final result.
Nevertheless, on a fully cached query it first constructed assignments for the
whole rooted tree, rebuilt every center-neighbor message, copied packed cache
columns into stacked messages, cloned the selected environment, and decoded
both entire payloads into new typed vectors. An internal center therefore
assembled unused components as well as the two directions actually contracted.
The reverse side already had a direct assignment probe; it did not always
rebuild the whole reverse postorder.

Before any production edit, two owning-module regressions independently failed
on pristine merged `fb6ab269931fed3324edbaa3c64bc7714861ad3c`, in an isolated Cargo
target. A five-site warm internal-center call visited three directed messages
instead of two. At identical 64-point assignments, increasing all bonds from
2 to 32 increased allocator-observed requested bytes from 34,248 to 54,408.
These are independent actual-work and allocation observations, not timing
inferences or counters inserted solely at the intended optimization point.

The evaluator now probes the same selected cut in both directions using its
existing canonical directed-component key builder. Only a complete two-sided
hit contracts borrowed packed columns. Cold queries, refused admission, partial
hits, scalar promotion and singleton networks retain the original computation.
Its selected environment now also borrows the caller-owned map entry instead
of cloning the complete stacked payload; no mutable alias is introduced. No
new persistent topology or message cache is introduced. Tentative lookups leave
hit/miss counters unchanged; successful assembly commits only the two used
directions. Packed offsets, lengths and scalar decoding remain checked. The checked work-count multiplication precedes result allocation and assembly. Owned
and borrowed assembly share an ordered, unconjugated dot product, preserving
the previous multiply/add order for f32, f64, Complex32 and Complex64.

The transient payload is independent of bond dimension; arithmetic remains
`points * selected_bond_dimension`. Physical assignments and lookup metadata
still depend on the component sites and unique assignments. There is no claim
that a general tree evaluator must equal TTCache's chain-specialized cost.

## Numerical and resource boundaries

The tests compare degree-2/3/4 borrowed assembly exactly against the legacy
owned assembly for all four scalar kinds, with unequal bonds, permuted tensor
axes, reordered and duplicate assignments. Retained key count, logical payload
and owned-storage estimates are unchanged. Both possible partial-hit directions
match legacy cache accounting and values. Additional checks cover wide heap
keys on both sides of a 260-site cut, invalid warm coordinates, malformed packed
columns, dtype/length mismatch, and two-hit diagnostics without message kernels.
Existing cold/warm, zero-budget, chunking, multi-physical-axis, scalar-promotion
and independent generic-contraction matrices continue to exercise the fallback.
The center-changing effort assertion changes from three hits to the two actually
used messages; numerical tolerances and coverage requirements are unchanged.

## Downstream investigation boundary

The R=10, T=0.1, mu=0.5, U=2 checkpoints are replayed through current public SGW
operations, at absolute tolerance 1e-4 and the original seeds, sweep/rank limits
and topology-dependent memory budgets. Current transforms prepare shared inputs;
transforms/loading and independent oracle checks stay outside operation timing.
Each output is checked at 1,086 fixed points through the generic TreeTNEvaluator,
with the existing global margin of ten times the stage tolerance. This is a
sampled correctness gate, not a whole-grid certificate.

The original fixed 5--6.5x Comb penalty is absent on the current-main replay:
Pi and Sigma converge on both topologies; W converges on Comb but returns
MaxSweeps at 20 on NBlock. The initial three-replay full matrix remains
INCONCLUSIVE because its predeclared gate required Converged in every cell.
Its numerical/count repeats are exact and its load/frequency/dispersion checks
pass. This failure is retained; it is not discarded to close a performance issue.

A separate predeclared before/after experiment tests exact preservation of
baseline values, point/sweep counts and termination, including that existing
NBlock W MaxSweeps. It answers whether this repair preserves behavior and
changes elapsed cost; it makes no claim to repair or certify W convergence.
The W observation is a quality follow-up; identity with #784's synthetic rank
cycling remains unproven. #784 and #794 remain open, so #854 remains open.

## W convergence follow-up

A separate public-API prefix replay keeps every original W option, including
`min_sweeps=2`, and varies only the requested cap from 2 to 20. History prefixes
are checked exactly, and every returned output is independently sampled at the
same 1,086 points. Comb converges at pass 10. On NBlock, Guard returns no new
pivots from pass 4 onward, every local error is below 1e-4, and every pass from
5 through 20 grows at least one output edge. The schedule requires two passes
without edge growth, so it correctly retains MaxSweeps under its current policy.
The late maximum rank fluctuates between 72 and 76 and the sampled residual
stays around 1.1e-4--1.4e-4. After the last Guard injection, the returned edge
ranks reflect the local state without an injection-cleanup transformation.

This is a new real-world near-threshold rank fluctuation witness related to
#784's investigation. It does not reproduce #784's exact late rank-vector
recurrence: no complete returned rank vector repeats after pass 4. It therefore
identifies the failed stopping prerequisite, not a common numerical root cause
or a justified new stopping rule. Neither loosening tolerance nor dropping the
rank-growth condition is part of this repair. The expensive low-temperature
R=9 regression protects the smaller-cut growth that condition must detect.

## Validation and measurements

The primary repair passes 1,271 debug tests/doctests across TreeTN and TreeACI,
plus focused feature-enabled diagnostics and the expensive R=9 low-temperature
regression in release mode. After the final fallback ownership cleanup, all 96
cached-evaluator diagnostics tests and all 205 TreeACI unit tests pass again.
The final work-count boundary reorder passes the 96 diagnostics tests and deny-warning Clippy again. Rustdoc, formatting and repository-rules preview also pass.
Cargo nextest is unavailable locally, so ordinary crate suites use cargo test.
The initial overly broad debug command was stopped before the costly R=9 case;
its partial log is preserved, and the complete debug suite skips that one case
only because it is separately verified in release mode.

Distinct source-stamped release binaries, an isolated pristine baseline and
complete controlled/real comparisons preserve the numerical and resource
boundaries described above. [Results, phase attribution and all case summaries](../../benchmarks/results/2026-10-09-treeaci-671-cached-cut.md)
retain inconclusive runs alongside separately declared confirmations. Timings
are descriptive; no tolerance, validity gate or coverage requirement changes.

The performance issue #671 can close when this repair merges: the requested
current-main R=10 cost attribution is supplied, its fixed branching signature
is absent, and the remaining independently reproduced complete-hit/copy defect
is repaired and tested. This closure does not assert universal optimality or
close the separately recorded W/DMRG quality and mixed-capacity investigations.
RSI is excluded from the experiments and no stopping/tolerance policy changes.
