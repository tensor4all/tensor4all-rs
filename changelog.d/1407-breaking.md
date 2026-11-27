- `tensor4all-partitionedtreetn`: replace `MergeRefineOptions::target_bond_dim` with
  `merge_policy: MergePolicy`: `Exact` (default), `RankGoal(NonZeroUsize)`, or `Tolerance`
  (#1130). Tolerance-driven merges choose ranks and adaptive output splits within a fixed
  global allowance. Reports now expose the selected policy, leaf-application bound and
  per-level allocation, retained error and carried allowance. Exact and rank-goal numerical
  decisions are unchanged. The greedy reconstructor's `ReconstructionOptions::target_bond_dim`
  is unchanged. Migration: `None -> MergePolicy::Exact`, `Some(n) -> MergePolicy::RankGoal(n.try_into()?)`.
  New `MergeRefineReport` fields also require updates to exhaustive struct literals.
  Addition and truncation-attempt counters include executed discarded probes;
  they can exceed retained-work counts and should not be compared as retained counts.
