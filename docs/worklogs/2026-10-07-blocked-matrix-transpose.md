# Blocked matrix transpose

## Decisions

- Preserve the public generic `Clone + Zero` contract, including types whose
  clone may unwind. Float/complex payloads use uninitialized typed slots with
  no bookkeeping; values with destructors track initialized slots for cleanup.
  Ownership transfer occurs only after every tile has been written.
- Tile square/wide inputs larger than 4096 elements with 16x16 blocks. Clone
  single-axis payloads directly (their flat order is unchanged), and preserve
  the existing traversal for small and tall matrices: exploratory
  measurements found that these shapes do not benefit from tiling.
- Keep this change in the owning matrix backend. No downstream transpose,
  dtype-specific APIs, or row-major conversion are introduced.

## Verification conclusions and constraints

- Bit-pattern tests cover four float/complex kinds, signed zero, NaN payloads,
  infinities, empty/skinny shapes, dispatch boundaries, and partial tiles.
- Clone-unwind tests prove initialized outputs release their shared ownership
  after early, mid-tile and late failures; successful generic and zero-sized
  values retain ordinary ownership semantics.
- The rrLU orientation conversion need gate and tile tuning are recorded in
  the [protocol](../../benchmarks/results/2026-10-07-matrix-transpose-protocol.md).
  They establish the cost of that operation, not its fraction of a complete
  factorization or a general workspace speedup.
- The first confirmation rejected regressions in the simple path; keep the
  blocked code out of that path and avoid redundant initialization/copy on
  single-axis inputs. Repeat every case with unchanged acceptance gates.
- Confirmatory paired results and validation will be linked here before PR
  handoff. Existing panic-baseline assertions are moved by unchanged-source
  mapping only, with no new allowance.
