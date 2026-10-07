# Core pivot scale invariance (#779)

## Decisions

- Incorporate #819's narrow absolute-floor repair with its original author
  retained. Complete the core extreme-scale repair here rather than publishing
  overlapping floor implementations. #819 is referenced for supersession;
  its external-fork gate is not waived or modified.
- Preserve the ordinary squared-magnitude pivot scan. When the largest square
  is outside the scalar component's normal range, rescan robust magnitudes in
  the same column-major tie order. Report pivot errors with robust magnitudes.
- Select direct division or normalized reciprocal multiplication once per
  pivot. Monomorphized row/column loops contain no runtime mode branch.
  Guard only exact zero or a reciprocal unrepresentable in the scalar type.
  `Scalar::min_positive` distinguishes f32 from f64 arithmetic without
  scalar-specific factorization entry points.
- Reject non-finite input magnitudes and residuals as numerical errors,
  including input entries that a pivot selection or zero-rank cap could skip.
  Lazy sources validate the residual blocks they request.

## Verification and limits

- All 1,151 unit/integration tests and 435 doctests of the four affected
  crates pass. Strict Clippy, strict rustdoc and deterministic repository rules
  checks pass. Tree/quantics validation was repeated after main gained #839.
  The ten public regressions fail on main at `6ddfa204` before the repair.
  Six pre-existing documentation-link warnings exposed by the strict build
  were corrected without relaxing a lint or changing executable examples.

- Reconstruction covers rrLU and both public LUCI facades, both orientations,
  square/rectangular shapes, and real/complex single/double precision.
  Exact binary rescaling compares rank, pivot indices and scaled pivot errors.
  Tests cover squared underflow/overflow, representable subnormals,
  unrepresentable reciprocals, exact zero, explicit truncation and non-finite
  values. Errors are divided by the input scale so an all-zero result fails.
- The algorithm cannot preserve information lost when input entries or
  elimination intermediates cease to be representable. Reciprocal guards are
  arithmetic safety, not an extra relative or absolute truncation tolerance.
- No performance improvement is claimed. The normal scan remains the fast
  path; input validation adds one matrix scan. Extreme inputs require a second
  pivot scan and reciprocal preparation. The changed inner loops were reviewed
  for allocation and runtime dispatch; no new unsafe or dense-network path is
  introduced.
- A nearby backend `submatrix_argmax` also compares squared magnitudes. It is
  not used by these rrLU/LUCI paths and remains outside this issue's kernel
  repair; it warrants a separate utility audit.
