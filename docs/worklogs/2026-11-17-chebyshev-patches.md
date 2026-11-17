# Piecewise Chebyshev Patches on Adaptive Dyadic Boxes

## Session summary

`tensor4all-interpolativeqtt` gained `interpolate_chebyshev_patches[_on_lanes]`,
`ChebyshevPatch`, `ChebyshevPatches`, `ChebyshevPatchOptions` and
`ExtraValidation` (issue #1350). The module keeps the Chebyshev-Lobatto node
values of every accepted dyadic box and builds no QTT, so a box is accepted or
split purely on its interpolation residual. A box's validation grid is exactly
the node grid of its `2^D` children, so a child inherits its node values from its
parent's validation buffer instead of resampling them: bit for bit, verified by
test. `InterpolativeQttError` gained `ChebyshevNotConverged`, the breaking
addition the issue approved. `interpolate_patched_nd` is untouched apart from
four crate-private helpers becoming reusable.

## Context reviewed

- Issue #1350 (design, decisions, open questions) and `origin/main` `ffc229ad`.
- `crates/tensor4all-interpolativeqtt/src/{patched,interval,basis,interpolation,error}.rs`,
  the `patched/tests` suite, `Cargo.toml`, `README.md`, `prelude.rs`, `lib.rs`.
- `NInterval::split`'s midpoint expression and `Plan::apply_stencil`'s axis
  rotation, which the new module reuses rather than reimplements.
- `AGENTS.md`, `CONTRIBUTING.md` (local validation tiers), `REPOSITORY_RULES.md`
  (file organisation, file-growth advisory), `docs/design/build-profiles.md`.

## Coordinate and inheritance contract

- Node coordinates of `[a, b]`: `j = 0 -> a`, `j = n -> b`, otherwise
  `a + (b - a) * x_j` with `x_j = (1 - cos(jπ/n))/2`. The endpoint rule matters:
  `a + (b - a) * 1.0` need not round to `b`.
- Validation coordinates of `[a, b]`: `mid = a + (b - a)/2` (the same expression
  as `NInterval::split`, so the midpoint is bitwise the children's corner), then
  the left half's node coordinates followed by the right half's without the
  duplicated midpoint: `2n + 1` per axis.
- Child `mask`, node index `j` on axis `d` is validation index
  `j + n * ((mask >> d) & 1)`: a strided gather, integers only.
- Prediction: one-axis stencil `W[i][α] = L_α(t_i)` of shape `[2n+1, n+1]` with
  `t_i` the normalized validation coordinates (`x_j/2`, then `1/2 + x_j/2`),
  applied axis by axis through the engine's session (the existing
  `apply_stencil`), so no dense Kronecker product is formed.

## Evidence

Local checks, all green, on `origin/main` `ffc229ad` plus this change:

```bash
cargo fmt --all -- --check
cargo clippy -p tensor4all-interpolativeqtt --all-targets --features parallel -- -D warnings -D clippy::missing_errors_doc -D clippy::missing_panics_doc
cargo nextest run -p tensor4all-interpolativeqtt
cargo nextest run -p tensor4all-interpolativeqtt --features parallel
cargo test -p tensor4all-interpolativeqtt --doc
cargo test -p tensor4all-interpolativeqtt --doc --features parallel
cargo doc -p tensor4all-interpolativeqtt --no-deps --features parallel
python3 scripts/repository-rules-review.py --base origin/main --worktree --dry-run --file-growth
python3 scripts/check-public-error-docs.py
```

`cargo nextest ... --features parallel` ran 100 tests (1 skipped by an existing
`#[ignore]`), `--doc --features parallel` 44 doctests. Test names cover the
issue's list: `polynomial_is_one_patch`, `smooth_function_is_few_patches`,
`localized_feature_refines_locally`, `pointwise_error_within_tolerance`,
`patches_partition_the_domain`, `no_value_is_evaluated_twice`,
`inherited_node_values_are_bitwise_correct`, `evaluate_reproduces_node_values`,
`lanes_match_serial_bitwise`, `ranges_and_external_evaluators_agree`,
`extra_validation_*`, and the error paths (`evaluator_error_is_preserved`,
`wrong_batch_length_is_reported`, `non_finite_value_is_reported`,
`broken_engine_is_reported`, `non_convergence_reports_the_prefix`,
`patch_limit_is_reported`, `invalid_arguments_are_rejected`,
`evaluate_rejects_invalid_points`).

`python3 scripts/audit-library-panics.py` could not run on this host: it builds
the workspace and stopped at `hdf5-metno-sys` ("Unable to locate HDF5 root
directory and/or headers"). That is an environment gap, not a finding.

### Benchmark (report only, no threshold)

`cargo bench -p tensor4all-interpolativeqtt --bench chebyshev_patches`, one
sample, one construction per constructor, `CpuBackend::with_threads(1)`,
function `1/((x-0.3)^2 + (y-0.6)^2 + 1e-2)` on `[0,1]^2`, degree 8,
tolerance `1e-4`, `num_bits = 10` for the QTT path:

| constructor | patches | evaluations | wall time |
| --- | --- | --- | --- |
| `interpolate_chebyshev_patches` | 55 | 21178 (reported and counted) | 8.5 ms |
| `interpolate_patched_nd` | 19 | 40500 | 328 ms |

The two constructors make different trade-offs, so the numbers are context, not
a comparison of like with like: the QTT path compresses each box by SVD and also
pays eigen/singular-value work per box, while the patch path keeps the node
values and evaluates no linear algebra beyond the validation stencil. The counted
evaluations equal `ChebyshevPatches::num_evaluations` exactly, which independently
confirms the bookkeeping.

## Design decisions carried from the issue, and deviations

1. `ExtraValidation::Fixed` point coordinates were left open by the issue. They
   are the centres of `points_per_axis` equal subintervals per axis, i.e. the
   quarter points for `points_per_axis = 2`, which include the midpoints the next
   refinement level would sample.
2. `ExtraValidation::Greedy` is a step-halving uphill pattern search along the
   finite-difference gradient direction, one probe per sweep, budget
   `max_steps` evaluations per start. The issue offered "e.g. golden-section or
   parabolic fit"; a pattern search keeps the batched-per-sweep shape, spends at
   most `starts * max_steps` evaluations, and can only raise the estimate, so the
   acceptance decision stays conservative.
3. The issue's test list asks for "no duplicate coordinates". A box's own
   midpoint is both a node (index `n/2` for even `n`) and a validation point
   (index `n`), and the issue states that such coincidences are harmless. The
   test asserts the count `(n+1)^D + boxes*(2n+1)^D`, a single node-grid batch,
   and `(2n+1)^D` in every other batch instead.
4. `sample`, `checked_product`, `checked_power` and `product_indices` became
   crate-private, and `Plan::apply_stencil` was extracted into a free
   `pub(crate)` function so the new module reuses them. No behaviour change;
   `patched.rs`'s own suite still passes unchanged.
5. The parent's own corners and nodes are prediction-exact, so their residual is
   zero and they are not deduplicated, as the issue says. The midpoint a split
   introduces is predicted, not sampled, so its residual is a genuine residual of
   the new child corner.
6. The stencil maps the *sampled* physical validation coordinates to local ones
   with the same expression `evaluate` uses, instead of assuming each half is
   exactly half width. `a + (b - a) / 2` rounds for some bounds (`[1, 1 + 3u]`
   has no representable midpoint), and the rounding would otherwise be reported
   as interpolation error: an affine function would fail at `max_depth = 0`.

## Review record

- Plan review (before implementation): `reviewer-glm` subagent, model
  `ollama-cloud/glm-5.3-flash` with thinking `high`. Verdict
  `VERDICT: CHANGES`, no BLOCKER, one IMPORTANT (the duplicate-coordinate
  assertion is not a theorem) and five MINOR findings (evaluate exactness at
  nodes, two missing error paths, lane bounds, greedy doc wording, the
  quarter-point wording). All were addressed; deviation 3 above is that fix, and
  `evaluate` snaps a bitwise node coordinate to its exact grid node.
- Independent post-implementation review: `openai/gpt-6.1-sol`, read-only, on the
  committed diff and the source. Verdict `VERDICT: CHANGES`: no BLOCKER, four
  IMPORTANT and five MINOR findings. Fixed: `evaluate` now rejects a non-rank-two
  point array instead of panicking or flattening it; `degree + 1` and
  `2 * degree + 1` are checked, so `degree = usize::MAX` is `InvalidArgument`
  rather than an overflow panic; the validation stencil uses the sampled physical
  coordinates (see deviation 6); the lane suite compares both parts of a complex
  and exercises both extra-validation modes and canonical first-error selection;
  a point that is bitwise a node of the selected patch now returns the stored
  value (which also keeps a negative zero); the node-exactness guarantee is
  restricted to the owning patch; the extra-validation test now splits a box the
  grid accepted. Deleted per the review: the eagerly built validation
  multi-index table (`Plan::tests`, decoded on demand instead) and the direction
  renormalisation in `Walk::accept`. Wording fixed in `README.md`, `lib.rs`,
  `num_evaluations` and this worklog.
