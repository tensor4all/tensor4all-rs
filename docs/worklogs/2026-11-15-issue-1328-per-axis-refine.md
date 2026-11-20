# Per-axis output refinement in `schedule_merge_refine`

## Decisions

- The per-axis stop is derived from the region key instead of a stored mask: a child
  writes `depth + 1` on the axes it extends, so the axes that still refine at a
  level are exactly those whose prefix is one bit short of it
  (`ScheduleGeometry::refining_axes`). One source of truth keeps the stop one-way,
  needs no state that has to survive a rebase, and leaves the insertion point of
  the concurrent input-side change (#1327) alone.
- A region with a rank goal probes each refining axis on its own and fixes output
  bits only on the axes that pass. After the rebase onto #1331 (`e0b45cd8`) the
  per-axis test is #1331's storage guard applied per axis: the region first merges
  its own input siblings, stops every refining axis without a probe when that
  candidate fits the goal, and otherwise passes an axis only when that axis'
  children store no more parameters in total than the merged candidate. The
  original per-axis test (pre-merge parent rank against the axis' child rank) was
  the comparison #1331 replaced, so it is not kept. Every
  refining axis is probed, including one that then stops, so a two-axis region
  still ranges over the same four probes as before; when exactly one axis passes
  its probe children are kept, and two or more passing axes release them and
  materialise the joint children over just those axes. With
  `target_bond_dim: None` nothing is probed and the trajectory is unchanged.
- `stopped_regions` counts one region-axis decision, and `refined_regions` still
  counts a region and level once however many of its axes refined. Single-axis
  runs report the same numbers as before.
- `level_charge_bound` keeps `children + 1` per region for a single refining axis
  and adds the joint child count only when two or more axes can pass, so the
  allocated share of a single-axis run is bit-identical to the previous formula.
- Rejected: an explicit stop mask keyed by region (a second source of truth that
  the key already encodes); deciding from the joint product's child ranks (it
  cannot express that only some axes pay); advancing one axis per level (it
  changes the uniform trajectory and the reported depth semantics).

## Verification conclusions and constraints

- All existing `merge_refine` tests and the multi-axis front-end tests pass
  unmodified, including the reversed-axis and adaptive-bound cases in
  `crates/tensor4all-quanticstransform/tests/patched_fourier.rs`.
- New unit tests pin `refining_axes` (lagging, stopped, exhausted) and
  `child_keys` (one bit per listed axis, unlisted prefixes preserved, filter).
- New integration tests pin the decision on the full transform for two and three
  axes: a rank goal above the root's retained rank stops every axis on its own
  decision (after the rebase: the merged candidate already fits the goal, so every
  axis stops without a probe), so one level keeps the root region instead of splitting into `2^axes`
  children (four and eight in the uniform runs) and the next level merges the
  region's input groups at the same key. Both stay inside the reported bound of
  the axis-by-axis reference, in the L2 norm and per coordinate.
- Consequence to keep in mind for three or more axes: a coarse level can stop
  every axis even when only the joint split would have lowered the rank, because
  no single axis pays on its own. The region then keeps its prefix and merges its
  input siblings; the retained bound stays inside the allowance.
- A *mixed* per-axis decision (one axis refines while another stops) is not pinned
  by an integration test. Restricting a single axis lowers the region's retained
  rank only through a truncation that fits the level's remaining-tolerance share,
  so the outcome moves with the share and with backend floating point: a fixture
  that refined one axis locally stopped both axes on the CI runner. The mixed-prefix
  mechanism is pinned by the `child_keys` unit test (a stopped axis' prefix is kept
  while the refining axis deepens), and the schedule-level evidence is the uniform
  (all axes refine) and rank-goal (all axes stop) trajectories.
- The input side still merges on every axis. The refine criterion is #1327's
  (merged candidate and storage guard, merged in #1331), applied per axis here.
- Rebase onto `e0b45cd8` (#1331): conflicts in `schedule.rs` (the level loop and
  its docs), `schedule/tests.rs` (imports), `qft_reconstruction.rs` (both sides
  appended tests; all kept) and two docs. #1331's `stopped` set is dropped because
  the region key already carries the stop. The charge bound needs no change: it
  already reserves the region's own merge plus one probe per axis child (plus the
  joint children for two or more axes), which covers the merged candidate first.
  `stopped_regions` keeps the per-region-axis count, so a candidate that fits the
  goal at the root of an `N`-axis schedule counts `N`.
