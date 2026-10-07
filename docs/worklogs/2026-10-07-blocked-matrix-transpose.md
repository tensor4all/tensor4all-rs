# Matrix transpose traversal

## Decisions

- Preserve the public generic `Clone + Zero` contract, including types whose
  clone may unwind. Uninitialized typed slots avoid overwritten zero fills;
  only types with destructors track live slots for unwind cleanup. Ownership
  transfer occurs after every slot is written.
- Keep the original simple loop at 4096 elements or less and clone single-axis
  buffers directly. Use 16x16 blocks for larger square/wide inputs. Tall inputs
  retain contiguous reads through 2 MiB of payload, then use 16x4 tiles to
  avoid the regression observed on a large rrLU factor. The narrower-column
  cache-conflict rationale is an inference, not a hardware-counter result.
- Keep simple and large helpers separate without forcing the public wrapper
  to inline. Avoid moving this implementation downstream or adding scalar APIs.

## Verification conclusions and constraints

- Tests cover four float/complex bit patterns, NaN payloads, signed zero,
  empty/skinny shapes, dispatch boundaries, and partial tiles. Generic unwind
  cleanup covers all traversals and successful ownership transfer; zero-sized
  types with and without destructors retain ordinary semantics.
- The [complete paired experiment](../../benchmarks/results/2026-10-07-matrix-transpose.md)
  passes every predeclared gate against the committed source. Large-square
  helper time falls about 47%; all four public rrLU orientation conversions
  improve. This does not establish a complete factorization/workspace speedup.
- Preserve every rejected experiment: helper code organization affected small
  cases, and the large tall fixture needed a different traversal. The final
  thresholds remain machine-dependent policies, not universally optimal values.
- Existing panic-baseline assertions are unchanged; only their line locations
  move, with no new allowance. CPU/default-faer validation is local; CUDA and
  other providers remain hosted validation limits.
- Final backend verification passes 237 tests, five focused optimized
  transpose tests, the affected doctest, strict Clippy/rustdoc, compiler-backed
  default/all-feature workspace panic audits, and the CPU explicit-context
  configuration. Hosted CI/coverage and rules review remain authoritative.
