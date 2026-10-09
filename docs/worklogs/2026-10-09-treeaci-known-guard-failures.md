# Retain established TreeACI guard failures

Tracking: [#860](https://github.com/tensor4all/tensor4all-rs/issues/860),
[#686](https://github.com/tensor4all/tensor4all-rs/issues/686), under
[#854](https://github.com/tensor4all/tensor4all-rs/issues/854).

## Decisions

An enabled guard must not forget a residual it already proved significant
because a later randomized search misses that assignment. The six-node
rank-one delta counterexample at seed 16 gives guard history `[0, 1, 0, 0]`
and `Converged` on pristine `0e1c49ba`, despite dense maximum error 1.
The returned points now survive to the next guard and are revalidated against
the current output and threshold. Still-failing points retain their slots;
resolved points release them for new discoveries. The pool is bounded by
`max_nglobal_pivots`, with combined retained/start/evaluation storage checked
before allocation and every extra target included in evaluated-point counts.

Immediate termination after the first saturated failure would also block
later local updates that can repair the assignment without growing rank.
A permanent rank-limit latch would fail to recognize such repairs. Fresh
residual evaluation distinguishes these cases. Retained failures take priority
over new, stronger discoveries so a later repair of the stronger point cannot
expose another forgotten failure. Random draws and walks retain their existing
order when there are no known points. Guard counts include unresolved retained
points, and remain distinct from actual injection counts.

The remaining #686 repairs share immutable directional schedules instead of
copying every path per pass, and reuse already resolved axes and physical
slice offsets when a multiincoming candidate group takes the scalar route.
The scalar accumulator and its reduction order are unchanged. No persistent
layout cache or new numerical policy was added.

## Verification conclusions and constraints

The public counterexample regression fails against separately built pristine
main and passes for f32/f64/Complex32/Complex64 on the candidate: it now reports
`RankLimited` with history `[0, 1, 1]`. Its approximation still has error 1;
this repairs false acceptance, not saturated-candidate recovery. The target
itself is representable at rank one. Revalidation tests cover actual repair,
slot priority/deduplication, accounting, malformed inputs, non-finite targets,
checked overflow, and aggregate working-budget refusal.

The scalar fallback regression checks exact values and identical core reads
for all four scalar types, three/four incoming branches, both packed layouts,
duplicate/reordered candidates and a budget immediately below batching.
Axis discovery stays constant as the candidate count grows. Schedule retention
shares the immutable plans and preserves forward/reverse walk validation.

The changed crate's debug tests/doctests, diagnostic feature checks and focused
R=9 low-temperature release regression pass: 256 checks in total. The expensive
R=9 test uses optimization because its unoptimized numerical workload is
impractical; its tolerance is unchanged. Deny-warning all-targets Clippy,
formatting and deterministic repository-rules review pass. Removed paths were
reviewed for coverage: scalar validation and contraction remain in the common
helper, with invalid-axis/frame tests; scheduler validation tests retain their
error-path exercise through explicitly owned test mutations.

The original DMRG outputs after passes 3 and 20 remain byte-identical to the
pristine baseline, including all output core values. Its guards return no
pivots and the run still ends with `MaxSweeps`. The
[per-edge recurrence record](2026-10-09-treeaci-quality-trace.md) remains an
investigation, not a numerical repair or a closure recommendation for #784.

A separately built pristine-main baseline and source-stamped candidate passed
all oracle and validity gates of the existing 120-case paired release matrix
(real/complex, degrees 2/3/4, ACI plus cold/warm queries, three alternating
pairs and three measurements after warmup). Results are descriptive, with no
predeclared regression threshold or general speedup claim. The default matrix
does not establish a wall-time gain for the low-budget group fallback; its
specific evidence is the exact-value/core-read/layout-discovery regression.
The baseline uses an isolated target; candidate source is `38e3ef03`.

#686 remains partially open because individual scalar frame construction has
separate layout preparation costs. #794's mixed-capacity injection no-op has
not been reproduced: #860 is a distinct established-failure retention defect.
#671 still needs representative R=10 branching evidence. RSI algorithms,
existing numerical tolerances and coverage thresholds are unchanged.
