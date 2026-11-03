# Opt-in N-ary exact sum (#1107)

Step after #1113 (binary `direct_sum` kernel) in the order agreed on #1107:
the N-ary API and an opt-in engine for exact sums. The defaults stay
binary; switching the summing routines over is a later step.

## Decisions

- `TensorContractionLike::direct_sum_many` is a *provided* method that folds
  the required binary `direct_sum` from the left, so implementors outside the
  workspace keep compiling. `IdxTensor` overrides it with a one-pass
  construction on the #1113 block-copy kernel. A test double that implements
  only the binary method exercises the provided path.
- The validation of `direct_sum` and `direct_sum_many` is one N-operand
  `setup_direct_sum`. With two operands it produces the binary path's former
  error messages word for word, so the existing binary tests are unchanged.
- `TreeTN::add_many` and `TreeTN::add` share one node loop; `add` still calls
  the binary `direct_sum` per node, so its behavior does not change.
- `SumEngine` chooses how `TreeTN::exact_sum` builds an exact sum: `Binary`
  (default, the add chain the summing routines use today), `Nary` (no size
  check) and `NaryWithin { limit_bytes }`, which falls back to `Binary` above
  the limit. The result reports `used_nary` and the estimate, so a fallback is
  visible to the caller rather than only in a log.
- The estimate is the size of the built N-ary network: per node, the product
  of its site dimensions and of its summed bond dimensions, times 16 bytes
  (`Complex64`, the largest supported element). For chains this is the
  `16 sum_k d_k B^2` of #1107. The default limit
  `SumEngine::DEFAULT_NARY_LIMIT_BYTES` is 256 MiB, an amount a library call
  may allocate unasked; it is not derived from the measurements below.
- f32/Complex32 support and the routine-level engine options are separate
  steps.

## Measurements recorded, not used as defaults

- #1107 body: peak resident memory of the whole reproducer process, every
  method in one process: 911 MiB at R = 12, N = 64, and 2.2 GiB at R = 20,
  N = 16 (the latter not isolated).
- #1107 comment by @maxsroda, one method per process at 3e8971f, random
  family, D = 10 (VmHWM, one run each): N-ary sum built 142 / 27 MiB, N-ary
  sum then truncate 224 / 84 MiB, sequential exact chain then truncate
  287 / 84 MiB, sequential pairwise truncating 34 / 144 MiB, summands only
  16 / 12 MiB (R = 12, N = 64 / R = 20, N = 16). The truncation sweep adds a
  transient of 0.65 and 3.8 times the built sum in these two cases, with no
  closed form yet; the `SumEngine` documentation asks for that margin in
  `limit_bytes` when the sum is truncated next.

## Verification conclusions and constraints

- `direct_sum_many` equals repeated binary `direct_sum` on operands of
  different shapes, and matches an independent element-by-element reference
  for three operands with different axis orders.
- `exact_sum` with `Binary` and `Nary` produces node tensors equal element for
  element; `NaryWithin` switches exactly at `estimated_nary_bytes`, which the
  tests check against a hand count on a degree-3 tree.
- The binary path gains no allocation from the N-operand setup. The groups
  are read in place through a pairs-or-lists view instead of converting
  `pairs` into lists, dims are computed once per operand, and error strings
  are built only on the error path. Measured with a counting allocator,
  binary `direct_sum` of two complex chain cores makes 55 allocations per
  call against 60 on `main` (1bedcbc0), and the checked-in criterion bench
  (`--bench direct_sum`, one core, built first, then timed one after the other) gives
  3.03 / 3.16 / 3.72 / 13.6 / 158 µs against 3.12 / 3.25 / 3.82 / 13.8 /
  158 µs on `main` at D = 1 / 4 / 10 / 40 / 160.
- No timing claim is made beyond #1107's reproducer figures quoted in the
  `SumEngine` documentation; this step does not change any default path.
