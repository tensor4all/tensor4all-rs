# Sibling merging of patched interpolation outputs (M6b)

## Status

Proposal for milestone M6b of
[tree-adaptive-patching-roadmap.md](./tree-adaptive-patching-roadmap.md).
Not implemented. The maintainer decided on 2026-11-10 that M6b is
implemented now, without waiting for an M9 measurement, because
overpatching has already been observed to be frequent.

This record covers interpolation outputs only. Merging the outputs of the
patched algebra (`contract_adaptive`, `add_with_patching`,
`truncate_adaptive`) and of M6c waits for M3b, as the roadmap already orders:
those outputs carry no per-patch error, so the rule below has nothing to
spend ([Algebra outputs](#algebra-outputs)).

## Problem

`patched_interpolate` splits a patch whenever it cannot be accepted, and never
undoes a split. The children of a split often end up much smaller than the
cap: the parent may have failed because of one localized feature, a sampling
miss, or an engine that stopped early, while the children together would fit
in one network below the cap. The result has more patches than needed, and
every later operation (contraction pair enumeration, element-wise operations,
evaluation) pays per patch.

TCIAlgorithms.jl removes such patches after its adaptive matrix product with
`_mergesmallpatches` (`src/adaptivematmul.jl`): recursively over the patch
tree, all values below a node are summed with `add(x, y; tolerance,
maxbonddim)`, and the sum replaces them unless its largest bond dimension
equals `maxbonddim`. Each addition truncates with the local `tolerance`;
nothing accounts the truncations against a global error. This crate needs
that accounting, because the producer's result carries the M3 L2 contract
([error contract](./tree-patching-error-contract.md)).

### Why not `reconstruction::reconstruct`

`reconstruct(ReconstructionTarget::from_partition(..))` with
`target_bond_dim = cap` already merges a partition top-down under a global
allowance. It is not the M6b operation, for three reasons:

1. **It needs a second budget.** Its allowance bounds the change from the
   partition, `||F~ - R|| <= delta_rec`, so the error against `f` becomes
   `E + delta_rec`. A caller who wants the original tolerance must run the
   interpolation with a tighter one in advance. M6b instead spends the slack
   that the producer's measurements left inside each region, so the caller's
   tolerance is unchanged and no budget is reserved.
2. **Its allocation is much tighter.** It gives each child `1 / d` of the
   parent's allowance (an l1 split), so a region at depth `k` of a binary
   split gets `delta / 2^k`, against the volume-proportional
   `delta / 2^(k/2)` of the M3 contract.
3. **It may return superpositions.** A region can keep several terms, which
   `into_partition` rejects; M6b always returns a strict partition.

M6b reuses its measurement primitive, not its driver
([Algorithm](#algorithm)).

## Allowance rule

Notation, all from the producer's report: `X` is the domain, `tau` the pinned
RMS allowance (`NormReport::L2::tau`), so `delta = tau * sqrt(|X|)`. A leaf
`P` (an accepted or a zero patch) has the approximation `g_P` (zero for a zero
patch), the deciding measured error `e_P = rms_P * sqrt(|P|)` from its
acceptance measurement, and the allowance `a_P = tau * sqrt(|P|)`.

Let `Q` be the region of a complete sibling group with members
`P_1, ..., P_m`, and `g_Q` the truncated sum of their approximations. The
members are disjoint and cover `Q`, so

```text
||f - g_Q||_Q <= ||f - sum_i g_i||_Q + ||sum_i g_i - g_Q||
              =  sqrt(sum_i e_i^2)   + t_Q
```

where the first term is exact because the residuals have disjoint supports,
and `t_Q` is the norm of the explicit difference network between the exact
sum and its truncation. The merge is accepted only if

```text
sqrt(sum_i e_i^2) + t_Q <= a_Q,      a_Q^2 = sum_i a_i^2 = tau^2 |Q|
```

and the merged rank is strictly below the cap ([Cap](#cap)). In RMS units,
which the implementation uses to avoid overflow,
`sqrt(sum_i w_i rms_i^2) + t_Q / sqrt(|Q|) <= tau` with `w_i = |P_i| / |Q|`.

Consequences:

- **The global contract is kept.** Every region of the merged partition meets
  `e_Q <= a_Q`, and the regions are disjoint with volumes summing to `|X|`,
  so `E <= delta` holds exactly as in M3. No new budget, no change of `tau`.
- **Unused budget is spent locally, never moved.** The slack
  `a_Q - sqrt(sum e_i^2)` exists only inside `Q`; a merge cannot borrow from
  a region it does not contain, so the result does not depend on the order in
  which disjoint groups are tried.
- **Recursion is the same rule.** A merged region enters its parent's group
  with `e_Q := sqrt(sum e_i^2) + t_Q`, the bound just established, and
  `a_Q` as above. The triangle inequality is applied once per level, never
  in quadrature across nested levels.
- **Exact merges are always allowed.** With no slack the sum is kept
  untruncated (`t_Q = 0`); it is accepted when its rank is still below the
  cap.

### What kind of claim the merged result makes

The rule uses the same statistic the producer used for its decisions, the
acceptance measurement. `t_Q` is computed from networks, with no sampling, so
a merge never weakens the basis of its members:

- If every member is `Exact` or `Exhaustive`, the merged region is certified
  up to rounding, like its members.
- If some member is estimated (`Sampled` or `ResidualTt`), the merged
  region is an estimate of the same kind; its audit, when every member has
  one, is `sqrt(sum w_i audit_i^2) + t_Q / sqrt(|Q|)` with the members'
  standard errors combined by their weights. `t_Q` is deterministic and adds
  no variance.

The global `GlobalL2Error` of the merged result is assembled by the same
code as the producer's, from the merged regions as contributions. A run
without any accepted merge therefore reports exactly the producer's global
error, which is a test.

Rounding: `t_Q` carries the rounding of the difference network, about
`eps * ||sum g_i||`. It is added to the existing first-order rounding model
of the report with the same `MEASUREMENT_ROUNDING_FACTOR`; it never enters a
decision, as in M3.

### Excluded members

- **`PatchStatus::ToleranceNotMet`.** A retained patch that missed its
  allowance has no slack, and merging it with accurate siblings would hide
  the failure inside a larger region whose rule could still pass. A group
  containing one is not merged, which also blocks its ancestors.
- **`ErrorNorm::SampledMax`.** It makes no L2 claim and has no
  `tau`; the merge returns a typed unsupported-norm error before any work.

## Sibling groups

The producer splits a patch at the next unfixed site of
`PatchedInterpolationOptions::patch_order`, into one child per value of that
site. The split tree is therefore the prefix tree over `patch_order`, and:

- the **parent** of a leaf fixes its projector's sites except the last one in
  `patch_order`;
- a **complete group** is a parent all of whose `d` children are leaves
  (accepted or zero) of the current partition.

Only complete groups are merged. A partial group is not a projector region
when `d > 2`, and for binary sites the only partial group is a single patch.
Merging regions outside the split tree (for example along a site fixed
earlier) would also satisfy the allowance rule, since the rule depends on
volumes only, but it is not attempted: the split tree is what the producer's
records describe.

The producer's report does not record `patch_order` or the cap today. Both
are added to `PatchedInterpolationReport` (it is `#[non_exhaustive]`), so the
merge needs only the result and cannot be given mismatched options.

### Order

Bottom-up by depth, deepest first. Groups at one depth are disjoint, so their
decisions are independent of each other and of the order in which they are
tried; they are tried in canonical patch order for reproducible reports. An
accepted group becomes a leaf at the parent's depth and can join a group
there; a rejected group blocks every ancestor. The number of attempts is at
most the producer's `splits`.

### Zero members

A zero patch is a member with `g = 0` and its measured `e`. A group whose
members are all zero becomes a zero region with
`e_Q = sqrt(sum e_i^2) <= a_Q`, with no network work. This keeps zero
subtrees from blocking their ancestors.

## Cap

The merged network's largest bond dimension must be strictly below the cap,
the producer's `max_bond_dim` unless `SiblingMergeOptions::max_bond_dim`
lowers it. Equality rejects, the same saturation rule as M6a and M6c: a
patch at the cap is one the producers would have split. A larger cap than the
producer's is rejected as invalid, since it would produce patches the
producer could not have accepted.

## Algorithm

For each complete group, deepest first:

1. If any member is `ToleranceNotMet`, reject.
2. If every member is zero, accept as a zero region.
3. Sum the nonzero members exactly (disjoint masked networks, direct sum).
4. `slack = a_Q - sqrt(sum e_i^2)`. If `slack > 0`, truncate the sum toward
   `slack` and measure `t_Q` on the difference network; a candidate whose
   measured `t_Q` exceeds `slack` is discarded and the exact sum kept with
   `t_Q = 0`. This is `reconstruction::engine::truncate_toward_allowance`,
   moved to a shared module and given an execution context; reconstruction
   keeps calling it.
5. Accept if the result's largest bond dimension is below the cap; then
   remask it to `Q`, replace the members, and record the merge.

Every numerical step (sum, canonicalization, SVD, difference norm) runs on
the caller's explicit `ExecutionContext`, as `contract_with_outcome_in` and
`TreeTN::truncate_in` do. `SubDomainTreeTN` has no context-scoped truncate or
norm yet; crate-private ones are added, and a context-scoped norm in
`tensor4all-treetn` if none exists (only `norm_on_session` does today).

Parallel execution over the groups of one depth (lanes, MPI) is left for
M7; the groups are independent, so it needs no change of the rule.

## Public surface

```rust
pub struct SiblingMergeOptions {
    /// Merged rank must be strictly below this; `None` uses the producer's
    /// `max_bond_dim`, a larger value is invalid.
    pub max_bond_dim: Option<usize>,
}

pub struct SiblingMergeRecord {
    pub region: Projector,
    pub members: Vec<Projector>,
    pub members_rms: f64,      // sqrt(sum w_i rms_i^2)
    pub truncation_rms: f64,   // t_Q / sqrt(|Q|)
    pub max_bond_dim: usize,   // 0 for a zero region
}

pub struct SiblingMergeReport {
    pub merges: Vec<SiblingMergeRecord>,   // accepted, in application order
    pub rejected_by_cap: usize,
    pub rejected_by_status: usize,
    pub global: GlobalL2Error,             // of the merged result
}

pub struct SiblingMergeResult<V> {
    pub partition: PartitionedTreeTN<V>,
    pub producer: PatchedInterpolationReport,  // unchanged
    pub merge: SiblingMergeReport,
}

pub fn merge_siblings_in<V>(
    result: PatchedInterpolationResult<V>,
    center: &V,
    options: &SiblingMergeOptions,
    context: &ExecutionContext,
) -> Result<SiblingMergeResult<V>, PartitionedTreeTNError>;
```

Field names are provisional. The producer report is kept unchanged because
its records describe engine runs that still happened; the merged partition's
regions are described by the merge records. `center` is an existing node,
used for canonicalization, as in `reconstruct`.

Errors (typed, no partial result): unsupported norm, unknown center, invalid
cap, non-finite measurements or norms, and propagated TreeTN errors.

## Algebra outputs

The roadmap's M6b outcome says "any `PartitionedTreeTN`". This record narrows
it: the rule needs the per-patch measured errors and `tau`, which only an
interpolation result carries. A bare `PartitionedTreeTN` from the algebra has
neither; M3b will give those outputs per-patch records, and M6b then accepts
them through the same rule. Until then there is no merge for them; a
best-effort mode without a bound is not proposed.

## Tests

On a branched tree (a star with a degree-3 hub, as the M6a layout tests):

- a function whose leaves are overpatched (a site the function does not
  depend on is split): the merge reaches the parent, the dense error against
  `f` is at most `delta`, and the patch count drops;
- multi-level merge to the root;
- a group whose truncation needs more than the slack keeps the exact sum and
  is rejected by the cap;
- cap equality rejects;
- `ToleranceNotMet` blocks the group and its ancestors;
- an all-zero group merges without network work;
- no accepted merge: the global error equals the producer's, bit for bit;
- `SampledMax` and an unknown center return their typed errors.

Unit tests construct results directly inside the crate; one integration test
runs `patched_interpolate` and checks the dense error against `f`.

Exit (roadmap): a merged interpolation output still meets its M3 tolerance,
and a merge exceeding the allowance is rejected.

## Open questions for the user

1. **Input.** Narrow the M6b outcome from "any `PartitionedTreeTN`" to
   interpolation results now, with algebra outputs after M3b
   ([Algebra outputs](#algebra-outputs)). Proposed: yes.
2. **Producer integration.** The roadmap lists "integration into the
   producers". Proposed: ship the post-processing function first and add a
   `PatchedInterpolationOptions` switch only when a measurement shows a
   benefit of merging inside the driver (earlier memory release).
