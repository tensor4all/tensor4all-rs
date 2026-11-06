# Sum engine on the partitionedtreetn summing routines (#1107)

Step after #1114 (opt-in `Nary` exact sum) in the order agreed on #1107: the
routines that form exact sums of many patches take a `SumEngine`. The
default stays `Binary` everywhere. The design was confirmed point by point on
#1107 before the change; `docs/design/partitioned-treetn.md` ("Sum engine")
records the result.

## Decisions

- Routines with an N-term exact sum take the engine: `contract` and
  `contract_selected` through `PartitionedContractionOptions { contraction,
  sum_engine }`, `contract_adaptive(_selected)` and `add_with_patching`
  through `PatchingOptions::sum_engine`, and `to_treetn_with(engine)`.
  treetn's `ContractionOptions` gains no field, because a single treetn
  contraction never sums; the two contraction methods take
  `impl Into<PartitionedContractionOptions>`, so existing calls with a plain
  `ContractionOptions` keep compiling and mean `Binary`.
- The four sum sites go through one crate-private
  `SubDomainTreeTN::exact_sum(terms, engine)`, which checks every term as
  `add` checks its operands and keeps the first term's budget metadata;
  `SubDomainTreeTN::add` is that helper with `Binary`.
- Return types do not change. A `NaryWithin` fallback is not reported by the
  partition routines; `to_treetn_with` and `TreeTN::exact_sum` report it. A
  statistics type was left out so as not to fix a caller API before a need.
- `PartitionedTreeTN::add`, `reconstruct` and `schedule_merge_refine` take no
  engine. Every sum they form has two terms, and their docs say why.

## Verification conclusions and constraints

- Every engine gives the same result in `contract`, `contract_selected`,
  `contract_adaptive` and `add_with_patching`, checked against a dense
  reference on partitions whose outputs receive three contributions;
  `to_treetn_with` reports the method and the estimate, and switches exactly at
  `limit_bytes`.
- Whether a routine forwards its engine cannot be observed through values,
  since every engine gives the same sum; it rests on review and on the
  `to_treetn_with` / `TreeTN::exact_sum` tests.
- A `NaryWithin` fallback costs what `Binary` costs. In a partitioned
  MPO x MPO contraction (R = 8, complex, bond D in {2, 4, 8}, N = 2^m in
  {4, 8, 16} contributions to one output projector, default contraction
  options, one core, median of 5), `contract` with `NaryWithin { limit_bytes:
  0 }` stayed within -6% to +1% of `Binary` in all nine cases. `Nary` was at
  most 17% faster there (2.24 s to 1.86 s at D = 8, N = 16): the pair
  contractions dominate.
- The default 256 MiB limit did not apply in that grid; the largest estimate
  was 162 MiB (D = 8, N = 16). By the estimate it applies from about N = 32 at
  D = 8, or at larger bonds. No measurement on production workloads was made,
  so whether the default should rise stays open.
