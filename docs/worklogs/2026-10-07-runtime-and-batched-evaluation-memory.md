# Plain tensor and cached evaluation memory

## Decisions

- Plain native tensors use the existing untracked eager-value adoption seam.
  Creating a semantic leaf with `requires_grad=false` still registers weak
  runtime records that survive the tensor. `enable_grad` remains the explicit
  tracked-leaf boundary. Apply the same seam to adjacent plain payload,
  selection, gradient-readback, and context-scoped wrapping paths.
- Bound cached evaluation's temporary batches independently of the caller's
  point count, retaining output order and evaluator reuse. Center selection
  uses the first internal batch so its assignment metadata is bounded too.
- Use finite defaults for the existing persistent payload budgets: chunking
  alone would leave caches free to grow across batches. Keep explicit zero
  and unlimited policies available. These budgets exclude allocator capacity
  and metadata; they are not exact process-RSS limits.
- Do not add eviction algorithms or a new cache API in this change. The
  existing admission policy remains in place, with bounded temporary groups.

## Verification conclusions and constraints

- The allocator regression reproduced 69,088 retained bytes after 256 dropped
  plain f64 tensors and now requires zero retained bytes for four scalar kinds
  in global and caller-owned CPU contexts, plus diagonal readout.
- Existing reverse AD, compact gradient, and context-isolation tests exercise
  the preserved tracked-leaf boundary. No functional JVP/VJP facade on plain
  `IdxTensor` leaves was found in the affected core/backend APIs.
- Chunk tests cover boundaries, duplicate ordering, four scalar kinds,
  branched and scalar-only networks, automatic and hinted centers, disabled
  caches, invalid limits, and errors in a later chunk followed by reuse.
- The prepared-slice reuse test explicitly selects whole-batch processing
  because its subject is the large-group slice path, independently of defaults.
- Performance results will follow the [predeclared protocol](../../benchmarks/results/2026-10-07-cached-batch-memory-protocol.md).
  CUDA wrapping changes share the seam, but device execution is unavailable on
  this CPU-only host. The separate CUDA transfer module and backend eigensolver
  also wrap native values; they remain adjacent audit targets.
