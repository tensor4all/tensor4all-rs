# Orthogonal-target reconstruction

## Scope and contract

Reconstruction belongs to `tensor4all-partitionedtreetn::reconstruction`.
It reorganizes a fixed target into disjoint output regions, each holding one
or more eagerly masked TreeTN terms. It is separate from the existing local
discarded-weight `PatchingOptions` contract and from QFT operator construction.

Only the unweighted discrete L2 norm is supported. For an operator this is its
Frobenius / Hilbert--Schmidt norm, not the induced spectral norm. Other norms,
weighted grid measures, and norm-strategy traits are outside the present design.

An immutable target is supplied as mutually disjoint patches. Thus
`||T||² = sum_p ||T_p||²`. When a patch is a tensor product `A_p ⊗ B_p` on
independent external indices, `||T_p|| = ||A_p|| ||B_p||`. Target constructors
validate supports, topology, full index identities and dimensions, scalar
homogeneity, and finite norms before reconstruction. The norm is derived from
the target; no public `reference_scale` override or `orthogonal: true` flag is
accepted.

`from_partition` snapshots existing data. `from_tensor_products` retains factor
pairs. In the initial product surface both factors have the same named tree
topology; the output node owns the union of its factors' sites. Every pair has
the same factor site assignments. Different topology placement requires an
explicit upstream restructuring, not an implicit conversion here. Products are
formed only when reconstruction starts, through the existing non-dense
same-topology partial-contraction path, with zero factorization thresholds and
no rank cap. Product construction roundoff has the same status as other backend
roundoff; it is not a user-requested approximation.

## Public surface

- `ReconstructionTarget<V>`: validated immutable source and pinned norm.
- `ReconstructionTolerance { rtol, atol }`: numerical accuracy only.
- `ReconstructionOptions { target_bond_dim, patch_order, split_strategy, max_regions }`:
  representation/search policy only.
- `reconstruct(&target, &center, tolerance, &options)`: owned result, no mutation.
- `ReconstructedTreeTN<V>`: immutable region/term lists and a report.
- `ReconstructionReport`: reference norm, allowance, measured error bound,
  region/term/rank counts, accepted splits and final-region merges.

The absolute allowance is `delta = max(atol, rtol * target.reference_scale())`.
The reference never changes after dropping, splitting, merging, or rounding.
Zero targets are valid. Empty targets have no topology; nonempty zero targets
still validate the requested center and site identities.

Output regions are disjoint; terms within a region need not be orthogonal.
Consequently their norm squares must not simply be added. `regions()` exposes
the terms. `into_partition()` performs structural conversion only and rejects
multi-term regions. It never silently forms a global sum.

Reuse the existing `PatchSplitStrategy` enum and `PatchingOptions::patch_order`
convention; do not introduce a second `PatchingOrder` enum. The reconstruction
options have this signature:

```rust
pub struct ReconstructionOptions {
    pub target_bond_dim: Option<usize>,
    pub patch_order: Vec<DynIndex>,
    pub split_strategy: PatchSplitStrategy,
    pub max_regions: usize,
}
```

An explicit `patch_order` limits the permitted full site indices. An empty list
uses all external indices in deterministic full-identity order, which need not
be bit-significance order. `Sequential` tries only the first unprojected index
with dimension greater than one. If it has no rank gain or its fanout exceeds
the region limit, the region stops; later indices are not tried.
`ExactParameterGain` (the default) searches all permitted candidates and uses
list order to break ties. Both strategies retain the global L2 allowance and
soft rank goal. Reusing candidate selection does not reuse the older local
`cutoff` or hard-rank-cap error semantics.

## Initial reconstruction algorithm

1. Order target terms by descending support depth, then canonical projector
   order, so deeper dyadic siblings are considered before coarse neighbors.
2. Compress individual terms, then try balanced pairwise additions. Retain a
   sum only when its compressed maximum rank is smaller than the sum of operand
   ranks; otherwise freeze both terms as separate list entries for this region.
   This deterministic heuristic does not search all possible pairs.
3. When a retained term exceeds the soft rank goal, probe permitted external
   indices. A split fixes that whole index to each coordinate. Project original
   region sources, then repeat reduction on the children. Do not project an
   already truncated parent approximation as the defining source.
4. Accept only a candidate whose maximum child term rank is strictly smaller
   than the parent's. `Sequential` probes only the next permitted index;
   `ExactParameterGain` chooses the smallest total logical parameter count
   among eligible candidates, with supplied index order breaking ties.
5. Stop on no gain, no available indices, or the region search limit. Retain
   higher rank or superpositions instead of violating the accuracy condition.

Splits are optional; rank-one content is not forced into a preset tiling.
Search holds only one best candidate plus the current probe. Fanout is checked
against `max_regions` before allocating children. There is no global dense
materialization, no initial all-term direct sum, and no hidden default pool.
This initial implementation is serial and makes no QFT butterfly complexity
claim. General projector overlap validation for product targets uses the
existing pairwise metadata checker; the cost is O(M²), without tensor products.

## Error accounting

For M original terms in a region, at most M initial compressions and M-1
successful pair compressions are possible. Each trial gets `region_delta/(2M)`.
Dropping a term costs its measured norm. A compressed candidate is remasked to
the exact region support, then its explicit difference-network norm is measured
against the local uncompressed source. If this residual exceeds its allowance,
the source is retained. Rejected merge/split probes cost no accepted error.

For accepted operations within one region the triangle inequality gives the
sum of measured residual norms, irrespective of term cancellation. Split
children are rebuilt from original sources, so the rejected parent's residual
is replaced, not added. Child budgets use conservative equal l1 allocation
`parent_delta/fanout`. Final disjoint-region errors combine using `hypot`.
Acceptance checks require the resulting numerical bound to be at most delta.
Zero tolerance disables approximate SVD compression.

This is a numerical a posteriori bound. It is not an interval certificate and
does not bound backend floating-point roundoff in products, addition,
factorization, projection, or norm measurement. Reference tests check the
actual target residual as well as the reported bound. No assumption that errors
from repeated compression are orthogonal, and no square-root-of-step-count
allocation, is used.

## QFT integration contract

`ReconstructionTarget::from_subset_operator` applies an existing, already-built
`LinearOperator` to an ordered subset of the preimage's external full site
indices. The Fourier operator itself is constructed by the caller
(`tensor4all-quanticstransform`), so this crate keeps no dependency on the
simplett stack. QFT must accept an explicit subset of external full site
indices, with ordered binary indices for each transformed coordinate. These
need not be contiguous nodes and may share a node with spectator indices. All
remaining indices are spectators: preserve their identity, dimension, and node
assignment. The call rejects a selection that does not match the operator node
count, repeats an index, is absent from the preimage site space, or shares a
node across operator nodes that cannot be fused into one connected group.

Selected indices may share one tree node. The operator MPO nodes carrying the
selected indices of one node are then fused into a single multi-site node with
`LinearOperator::restructure_to`, which contracts the group locally with no
truncation and moves each mapping to the node owning its internal index; a
singleton group is a pure node rename. Fusion requires those operator nodes to
form one connected group inside the operator's own topology, because a group
threaded through another owner's node cannot be fused without absorbing sites
that belong elsewhere. Such a selection is rejected with repair guidance
("split the selection or restructure the operator") rather than silently
mis-binding it. The fused node keeps the preimage node's name, so node
assignment is preserved. Spectator indices on the same node as a selected index
are supported.

The QFT selection is independent of reconstruction's `patch_order`.
Input bits are ordered from most to least significant. A full transform is
the same interface with all desired axes selected. Multidimensional axes each
carry their own ordered subset. They may be applied separately or composed with
`quantics_fourier_operator_multiaxis` and supplied as one scheduled transform
with a `CoordinateGroup` per axis. Normalization is unitary on selected axes. The transform is never silently padded and its length
is never changed: the operator's input and output are the same selected indices, so a
caller that needs a different output space owns that embedding outside this API.

### Bit significance, site placement, and patching order

For a transform from k to r, both k1 and r1 are the most significant bits:
`k = sum_j 2^(R-j) k_j`, `r = sum_j 2^(R-j) r_j`.
With no output bit-reversal permutation, the selected TT positions change from
`[k1, ..., kR]` to `[rR, ..., r1]`. Thus the position previously carrying k_j
carries r_(R+1-j). For subset transforms this assignment affects only selected
indices; spectators keep their identity, dimension, and node assignment.

Contiguous dyadic input patches fix k1, k2, ...; contiguous output patches fix
r1, r2, ... . Set output `patch_order = [r1, ..., rR]` and
`split_strategy = PatchSplitStrategy::Sequential`, independently of the TT's
reversed output placement. For three bits, fixing r1 separates [0,4) from
[4,8), whereas fixing r3 separates even from odd coordinates. Output patching
therefore proceeds from the right end of the reversed TT layout.

For uniform input depth d, the complementary merge-refine schedule removes
input constraints in order k_d, ..., k1 and adds output constraints in order
r1, ..., r_d. At level t, input unions retain the prefix k1, ..., k_(d-t),
and output regions fix r1, ..., r_t. An input patch's transform generally
spreads over the output domain; there is no direct replacement of its fixed
input bits by fixed output bits. Adaptive schedules may stop on no gain.

The map is `F_selected ⊗ I_spectators`, so exact QFT preserves orthogonality and
the original global norm even though transformed supports overlap. The
constructor nonetheless accepts an arbitrary operator, which need not be unitary
and need not map the preimage's disjoint patches to orthogonal images. It
therefore never measures the transformed output norm: neither by summing the
images into one network, nor by a pairwise-overlap Gram sum.

Instead it pins a reference scale from the operator itself,

```text
s = 1                 if unitary is specified
s = ||A||_F           otherwise
reference_scale = s * preimage.reference_scale()
absolute_tolerance = max(atol, rtol * reference_scale)
```

where `A` is the operator restricted to the selected sites. Spectator identity
factors are neither materialized nor counted, because
`||A ⊗ I||_2 = ||A||_2 <= ||A||_F`. For an `N`-dimensional unitary the Frobenius
norm is `sqrt(N)` while the induced 2-norm is `1`; `unitary = true` assumes the
latter as a caller guarantee and does not assert a unit Frobenius norm.
Successive applications multiply these factors rather than recomputing an output
norm. The option is therefore a contract mark, not a checked property.

For a general operator the reference scale is a norm-based upper bound, so `rtol`
is relative to that scale and not to `||A x||_2`. This is deliberate: a measured
output norm would require either the forbidden global direct sum or a
cancellation-prone `O(M^2)` overlap sum. The two scales coincide for a unitary
acting on an exactly known input scale, and the multiplication composes with an
already-propagated preimage scale for operator-derived preimages. The images stay
separate with only their spectator constraints, and operator-construction error,
including approximate-QFT error, remains separate from the reconstruction bound.
Because a temporary and discarded global sum leaves no trace in the returned
target, separate-term retention tests observe the representation only and are not
offered as detection of that prohibited path.

QFT also supplies the input-merge/output-refine schedule and its bit geometry.
The complementary-area invariant of the Fourier algorithm is not a generic
reconstruction invariant. The current greedy reconstruction entry point does
not claim to implement that schedule or QFT itself.

## Level-coupled merge-refine schedule

The schedule is a second, explicitly selected entry point
([`reconstruction::schedule_merge_refine`]); it does not replace the greedy
engine and is not implied by reversing `patch_order` or by calling the greedy
engine once per final block (which would repeat work and can restore an
`M^2` contribution count).

### Contract and geometry

A work item represents `G(A, B) = P_B F_selected P_A w`, or a retained sum
approximating it, with an explicit bound. Spectator indices are unchanged, and
every item is a complete-transform object: no staged application and no mixed
position/frequency basis.

For a uniform binary input partition of depth `d` (`M = 2^d`):

1. Level 0 applies the complete transform to each input leaf, with `B` the whole
   output domain.
2. Level `t` merges input siblings by removing the constraint on `k_(d-t+1)` and
   refines `B` by fixing `r_t`.
3. Each child is `round(P_B' G(A0, B) + P_B' G(A1, B))`: restriction to the child
   region happens before addition.
4. Consumed parent data is released once all its children exist, so only two
   levels are live and transformed partial sums are reused by descendants.

Input significance is `[k1, ..., kd]` in operator node order, while output
significance `r_j` is carried by the selected index that previously carried
`k_(d+1-j)`; output prefix regions therefore fix `[r1, ..., rR]` independently of
physical node order, and an inverse transform uses the corresponding reversed
selection. The uniform baseline holds `#A * #B = M` items per level and `M * d`
pairwise additions. These are structural counts, not a runtime bound: tensor
ranks, local contraction costs, and retained list lengths must be measured
separately, and adaptive stopping changes the counts.

### Geometry and provenance validation

Selected indices must be binary and non-empty, and the preimage must already be
partitioned into dyadic input leaves of those indices, sharing identical spectator
constraints. A preimage that leaves a selected index free, repeats a
selected-coordinate assignment, or disagrees on a spectator constraint is rejected
before any operator is applied; a repair suggestion is required rather than an
implicit normalization or padding step.

Each leaf fixes a *contiguous prefix* of the selected indices, so the input tree may
be nonuniform: leaf depths can differ as long as the leaves form a dyadic prefix
code, where no leaf fixes a prefix of another leaf's indices. The contract is
checked as a Kraft sum in integer units of `2^-depth`, so an overlapping pair and a
missing assignment are distinguished exactly. Every item then ascends one selected
bit per level: a genuine sibling pair is summed, and a leaf whose sibling is absent
keeps its own region, which is already the union of its subtree, so ascending is
exact and free. Because each refined region keeps its own copy of the still
unmerged prefixes, the live item count can grow with the number of regions, so the
work limit is enforced on every level rather than only before the first one.

Missing leaves are a decision, not an assumption: `MergeRefineOptions::coverage`
defaults to `CoverageContract::Complete`, which requires the Kraft sum to be exactly
one, and `CoverageContract::ZeroForMissingLeaves` accepts a sparse preimage in which
every omitted assignment contributes exactly zero. An entirely empty preimage is the
zero target under that contract; under the complete contract it is rejected. Preparation
keeps the preimage patch support (`PreparedImage::source`) next to the image, so
input ancestry is never inferred from spectator-only projectors and no private
target field is exposed.

### Error transport

The global allowance `delta = max(atol, rtol * reference_scale)` is pinned before
scheduling. Level zero applies the complete transform, so the error sources are the
measured application error of each input leaf, the measured residual of every
accepted truncation, and the measured norm of every dropped contribution.

Every component is measured where it happens and recorded once, grouped by the level
and the region that measured it:

- an application error is measured before any split, so it belongs to level zero,
  which has the single root region;
- a truncation residual or a dropped norm is measured inside the child region it is
  accepted in, so it belongs to that level and region.

Two components of one level live in disjoint regions and therefore have disjoint
supports, so their norms combine by the Euclidean norm. Two components of different
levels can be nested, so they add by the triangle inequality. The reported bound is

```text
error_bound = sum over levels of
              hypot over the regions of that level of
              the components measured in that region at that level
```

Counting the application error once at the root is the point: the same error network
is restricted into every refined region, and an earlier per-region `hypot` counted it
once per region, which made the bound grow like `2^(levels/2)` as the number of
levels increased. Measured on an `r`-bit input with `I^⊗r + 0.1·X^⊗r` truncated to
bond dimension one, the reported bound exceeded the true L2 deviation by 3.9x, 7.2x
and 12.8x at two, three and four levels before this change, and by 1.9x, 2.5x and
3.2x afterwards.

Budget allocation stays conservative: the application error is charged first, and the
rest is split over the levels that remain and over the share-sized charges the
current level can make. A charge is one dropped norm or one accepted truncation
residual, so the number of charges follows the operand term counts of the merges
that level performs, including the discarded probe of a region that stops. The
level share is `(remaining / levels_left) / charges`, so late levels are not
starved and a multi-term item cannot spend more than its level's allocation. A
level spends only what it measures (accepted residuals and dropped norms, including
those of discarded probes). The sum of all measured components cannot exceed the
allowance, so `error_bound` never does either. A restriction is nonexpansive and
consumes no budget, and a rejected probe contributes no reported component, so
nothing here needs a `sqrt(fanout)` division.

Still open on this axis: attributing residual components to the final regions they
touch so that residuals of nested levels could combine by the Euclidean norm instead
of the triangle inequality.

### Public surface and dependency placement

`schedule_merge_refine` lives in `tensor4all-partitionedtreetn` and accepts an
existing `LinearOperator`, ordered coordinate groups, and the explicit dyadic
geometry. QFT construction stays in `tensor4all-quanticstransform`, which remains
a path-only cross-layer dev-dependency: this crate keeps no simplett-stack
runtime dependency. `MergeRefineOptions` selects the output depth, the work
limit, the soft rank goal, and the retained term budget; `MergeRefineReport`
carries the pinned scale, the allowance, the measured bound, and the structural
counters, including the refinement stops. Exact scheduling
algebra is shared with any operator, while Fourier-specific rank and performance
expectations are not claimed for arbitrary operators.

### Adaptive refinement

A region's items are a superposition: each retained term is `P_B F P_A w` up to a
measured bound, and a region's items carry the input prefixes they came from, so
input ancestry is never inferred from spectator-only projectors and no private
target field is exposed downstream.

With a rank goal the trajectory becomes adaptive, and the decision is made per
output region so the final region set stays prefix-free:

- An output region is refined only when its retained terms exceed the goal *and*
  refining strictly lowers its maximum retained rank. Without that gain the region
  stops refining, so a rank-one object is not forced into a preset output tiling.
  A stopped region keeps its output key and still merges its input siblings on every
  later level; restricting it to its own region is a no-op, so it runs the merge
  half of the level with the same ledger share. Its measured probe components are
  charged, but only the merge it keeps contributes a reported component.
- Merging a pair combines the two operands into one term when the candidate fits the
  goal or strictly lowers the bond dimension against the naive sum; otherwise both
  stay as separate terms, which keeps the retained ranks small at the cost of a
  longer term list.
- Refinement stops are reported (`refined_regions`, `stopped_regions`), with a
  stopped region counted once rather than once per remaining level, and the
  retained term count is bounded by `max_terms`, whose exhaustion is an explicit
  resource-limit error rather than a silent accuracy relaxation.
- A merged or unpaired contribution is dropped only when its measured norm fits its
  share of the allowance. Dropping costs that norm plus the inherited bound it
  already carried, both of which stay in the region's report, and a region whose
  terms were all dropped is omitted from the result while its bound still counts
  toward `error_bound`. `dropped_terms` and `dropped_error` report how much was
  removed and how much of the bound it accounts for.

### Evidence and measurements

`benchmarks/rust/benchmark_merge_refine.rs` compares the schedule against the greedy
engine on the same subset-operator target and against applying the complete
transform to the whole input at once. It records the structural counters, the
retained and transient ranks, stored parameters, elapsed time, the measured bound,
and the deviation from the dense exact trajectory. The recorded three- and
four-bit families (2026-09-16) predate the #777 change that keeps a stopped region
merging its input siblings, so their adaptive-stop term counts and additions no
longer describe the current schedule; the exact trajectory's operation counts
(`2^d` applications and `2^d * d` additions) are unchanged. Runtime is rank-driven
rather than structure-driven: in the recorded runs at three bits the adaptive
policy was slower despite three times fewer additions, and at four bits it was
about five times faster while the exact trajectory's retained rank reached `256`
and the adaptive runs stayed at or below the goal.
Applying the
complete transform to the whole input is cheaper on those small inputs, which is the
honest baseline the design must beat only in the patched-input regime that the
nonuniform geometry would open. No constant-rank or speedup claim is made for
arbitrary data, and operator-construction error stays outside every bound.

### Deferred boundaries

Per-input-branch refinement depths with lazy reconciliation remain follow-up work.
Unequal input leaf depths are supported through the dyadic prefix-code contract
described above, and item dropping under the global policy is implemented as described
above.

### Coordinate axes

Several coordinate axes can be transformed in one synchronized level.
`MergeRefineOptions::coordinate_groups` supplies them explicitly, and each
`CoordinateGroup` lists its axis' selected indices in input significance order and,
separately, in output significance order, so the placement is stated by the caller
rather than inferred. The groups must assign every selected index exactly once, and a
group's inputs and outputs must be the same indices. `None`, the default, treats
`selection` as one axis whose output order reverses its input order, which is the
one-axis subset-operator convention.

One level advances every non-exhausted axis by one bit: it merges that axis' next
input bit and fixes that axis' next output bit. Two axes therefore combine four input
children and produce four output children, and the pairwise sums are measured
individually as for one axis. A two-axis caller composes per-axis one-dimensional
operators (for example with `compose_exclusive_linear_operators`) and states each
axis' output order accordingly; the two-axis regression applies such a composed
operator and compares it against the two one-dimensional transforms applied in
sequence.

Approximate operator application is available as an opt-in on the schedule:
`MergeRefineOptions::apply_options` applies the operator with the caller's
truncating options *and* always applies it exactly, measures the per-leaf
deviation `||exact - approximate||`, and charges that sum to the item bounds before
any scheduling compression. The exact application is therefore the measured
reference, the truncation never becomes a silent error that the report would
mis-attribute, and a measured application error already above the global allowance
is rejected instead of returning a result that violates the contract. The
immutable-target entry point keeps applying exactly, because a `ReconstructionTarget`
has no place to carry such an error. Automatic padding stays
out of the schedule: a later opt-in padding API must define the embedding, the
extra bits per transformed coordinate, physical spacing and frequency grid,
normalization, and output interpretation, and the selected transform length is
never changed silently.

### Exact application without product-width bonds (#1279)

Status: implemented (#1279).

The exact application that both `from_subset_operator` and the schedule perform
(above) used `ApplyOptions::naive()`. The naive local product gives every site
tensor of `A x` both bonds equal to input bond × operator bond, so a patched
Fourier transform over two interleaved axes at R = 10 (input bond 50, operator
bond 144) forms 7200-wide site tensors, about 1.7 GB each in complex f64, and
fails with an allocation error under an 8 GiB or a 14 GiB limit. The product's
rank at a bond is also bounded by the physical dimension of each side of it,
which is at most 1024 there, so most of that width carries nothing.

**Change.** `prepare_subset_images` applies the operator exactly with zip-up
instead: `ApplyOptions::zipup()` with
`svd_policy = Some(SvdTruncationPolicy::new(0.0))` and no `max_bond_dim`,
centred at the balanced node, the node that minimizes the largest total
`log2` site dimension of the subtrees around it (ties by node order). A
threshold of zero discards no nonzero singular value, so the application is
exact up to backend roundoff, as the naive path is. Since #1285 that request
factorizes every zip-up block by untruncated QR instead of SVD; see
[Exact zip-up by QR](#exact-zip-up-by-qr-1285). The threshold must be
given: `svd_policy = None` resolves to the default relative cutoff of `1e-12`
and is not exact. No path of `prepare_subset_images` keeps the naive
application.

**Why the bonds stay bounded, by tree shape.**

- *Chains* (every node on one path, the QFT layouts): zip-up takes its chain
  schedule. It first puts both operands in QR form at the first sweep site,
  so each operand's bond toward the far end is bounded by the physical
  dimension of the far side, and it then sweeps in one direction with an SVD
  per site, so the swept bond is bounded by the physical dimension of the swept
  side. Every block it factorizes is bounded on both sides although the sweep
  runs one way; the final canonicalization to the centre leaves the image's
  bonds at the smaller side's physical dimension. The centre changes the sweep
  direction, not these bounds.
- *Other trees* (spectator branches, multi-site nodes): zip-up processes the
  nodes from the leaves towards the centre. The operands are not
  pre-canonicalized there, so a block's parent side is the full product bond,
  and only its processed side is bounded by that side's physical dimension;
  the balanced centre keeps every processed side at most about half of the
  total. A spectator subtree, one that holds no operator node and not the
  centre, is emitted unchanged by the exact identity skip of zip-up (#1186)
  and adds no work; a spectator node between operator nodes and the centre is
  factorized like any other. On such trees the peak is therefore of order
  `2^(half the site bits) × input bond × operator bond` rather than the
  chain's two-sided bound, and the centre contracts the remainders of all its
  branches at once, so three wide branches multiply; no QFT layout of the
  front end has this shape and none of it has been measured. On a chain the
  chain schedule factorizes every node, spectators included, within the chain
  bounds above.

**Guard on other trees.** Before any application, `prepare_subset_images`
estimates the largest block of every term on a tree that is not a chain, by
following the tree zip-up from the leaves to the centre: a node whose subtree
holds no operator node is emitted and leaves only its state bond to its parent;
every other node forms `(children's remainders × its site dimensions) ×
(state bond × operator bond)` towards its parent and passes on the smaller
factor, the operator's largest bond standing in for every operator bond; the
centre forms the product of its remainders and site dimensions. A term whose
estimate exceeds `2^28` elements, 4 GiB in complex `f64`, is refused with
`ResourceLimit` instead of allocated; the limit is fixed, and the remedy is a
chain layout or fewer selected sites. Chains are not checked, since their
blocks are bounded. Three branches of 16 binary sites with bonds 128 already
estimate `2^43` at the centre. The guard is an upper estimate of the block the
factorization sees, not of every temporary the contraction planner forms.

**Error accounting is unchanged.** Exact mode still reports
`error_bound == 0`. With `MergeRefineOptions::apply_options` the schedule still
applies the caller's truncating options as well and charges the measured
`||exact - approximate||` of each leaf; the exact image this measurement needs
is now affordable, and the difference network's bond is the sum of the two
images' bonds. Charging the discarded weights of a canonical-form truncation
instead, as first proposed on #1279, is not needed for the bound and is left
out. Restricting the application to requested output regions (option 2 of
#1279) is also left out: it cannot help a caller that requests the whole
output, as the R = 10 interleaved case does, and is a separate optimization.

**Measured** (R = 10, two interleaved axes, 20 binary sites, 4 input
patches, release, one thread, `ulimit -v` 8 GiB, the front end only). The case
completes in 827 s; the process peaks at 4.0 GiB RSS, and the sampled DFT
error, 1.45e-13, is inside the reported bound, 1.66e-8. Every image has bonds
`[2, 4, ..., 512, 1024, 512, ..., 4, 2]`, the physical bound on each side. The
block the chain zip-up factorizes at each sweep site of the first image, in
millions of elements (one element is 16 bytes), measured on the final code
with a temporary probe; the two operands' bonds toward the far end are the
second and fourth dimensions, the remainder's bond the first:

| sweep site | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| block [M elements] | 0.38 | 1.8 | 3.1 | 5.9 | 10.0 | 17.1 | 20.4 | 7.7 | 1.8 | 0.28 |
| dimensions | 64·31·2·96 | 128·49·2·144 | 256·42·2·144 | 512·40·2·144 | 1024·34·2·144 | 2048·29·2·144 | 4096·26·2·96 | 2496·24·2·64 | 1536·18·2·32 | 576·15·2·16 |

Sites 0 to 5 and 16 to 17 are below 0.1M and the final two-site block has 64
elements. The largest block over all four images is 21.2M elements
(0.32 GiB), at site 12 of the third image. The remainder carries the swept
side's physical dimension, up to 4096, while the operands' far-side bonds stay
at most 49 and 144; the image's bonds are at most 1024 only after the closing
canonicalization. On the prototype, the default centre (node 0, sweep from
node 19) reached at least 28.9M elements (0.43 GiB) before its run hit a time
limit after site 13, which is why the balanced centre is used.

**The peak follows the physical bond, not the compressed rank.** The exact
images keep every singular value and reach the physical bound (1024 here),
although the compressed result has rank 8, so the rank goal does not bound the
memory of the exact application. The 4.0 GiB peak is measured, not bounded;
the heavy test therefore runs this case in a child process under an 8 GiB
address-space ceiling and prints its peak RSS, so an allocation above the
ceiling fails the test instead of leaving the runner's OOM killer to end the
job.

The cost is time: on the prototype each exact image took 180 to 197 s, 765 s
of the schedule's 824 s, spent in SVDs of blocks up to 8192 × 2496. The exact
images keep every singular value, including those at roundoff, which is why
their bonds reach the physical bound although the compressed result has rank
8. #1285 replaces those SVDs by QR; see
[Exact zip-up by QR](#exact-zip-up-by-qr-1285).

On the cases that already passed, the zip-up gives the naive path's result up
to roundoff, faster and with less memory (release, one thread, single runs on
one host; the naive column was measured before the change):

| case | reported bound, naive / zip-up | sampled error, naive / zip-up | schedule [s], naive / zip-up | peak RSS [MB], naive / zip-up |
|---|---|---|---|---|
| R = 10, 1D | 7.912518e-9 / 7.912519e-9 | 4.11e-14 / 4.07e-14 | 0.033 / 0.025 | 40 / 31 |
| R = 10, 2D grouped | 7.705055e-9 / 7.705055e-9 | 8.75e-14 / 9.32e-14 | 0.31 / 0.17 | 67 / 36 |
| R = 20, 1D | 3.455769e-10 / 3.455768e-10 | 2.24e-13 / 2.21e-13 | 0.26 / 0.23 | 89 / 61 |
| R = 20, 2D grouped | 6.853537e-9 / 6.853537e-9 | 4.23e-13 / 4.44e-13 | 1.57 / 1.23 | 121 / 102 |

**Tests.**

- `tests/subset_operator_exact.rs` (`tensor4all-partitionedtreetn`): the images
  equal the dense product of the dense operator and the dense state, on a
  chain whose balanced centre is a spectator node and on a branched tree with
  a spectator branch and a node carrying two selected sites, for real and
  complex data. A truncating threshold fails all four.
- The `balanced_center` unit tests: chains, bits rather than node counts,
  non-binary sites, a branched tree, ties. The guard's unit tests: no
  estimate on a chain, a hand-computed estimate on a branched tree with and
  without spectator subtrees, the deep three-branch tree above, and the
  refusal only above the limit.
- The existing schedule tests cover exact mode (`error_bound == 0`) and a
  truncating `apply_options` whose measured deviation is charged. One existing
  test changed: in `subset_operator_scale_is_not_measured_under_destructive_interference`
  the two images are negatives of each other. Whether their roundoff residual
  cancels exactly depends on the platform's SVD: on Linux the zip-up images
  merge into the zero function and no term stays, while on macOS (Accelerate)
  both stay, as both naive images did. The test therefore asserts what holds
  on every platform: nothing is charged, at most the two terms stay, and their
  sum is zero to roundoff.
- Heavy: `heavy_gaussian_2d_interleaved_r10` in
  `tensor4all-quanticstransform/tests/heavy_patched_fourier.rs`. It runs the
  front end only, one schedule (about 14 minutes here with one thread, 332 s
  with four in the `ci` profile), not the separately timed direct schedule as
  well, since the whole nightly heavy job has 60 minutes, 25 of them spent
  building. It runs under the 8 GiB ceiling above; with the ceiling lowered to
  2 GiB it fails with the backend's allocation error.

**Out of scope.** R = 20 interleaved (40 sites): there the physical bound
exceeds the product bond over the middle of the chain, so an exact image can
really be 7200 wide; it needs truncation during the sweep with a proven bound
and is a separate issue.

### Exact zip-up by QR (#1285)

**Problem.** The exact application above spends almost all of its time in
SVDs whose singular values it never uses: with a zero threshold every value
is kept, roundoff included, and the rank reduction happens later, in the
compression.

**Change.** An exact zip-up request (`Unitary`, a zero SVD threshold, no
`max_bond_dim`; the predicate `zipup_is_exact` in
`tensor4all-treetn/src/treetn/contraction.rs`) now factorizes every block by
QR with `qr_rtol = 0`, on both schedules. Every row of `R` is retained, so each
bond is `min(rows, cols)` of its block, the width the zero-threshold SVD
keeps, and the memory picture of #1279 is unchanged. QR has only a
left-canonical form, so the chain schedule factorizes its final two-site block
with the last site's indices on the orthogonal side: `Q` goes to the last
node, `R` to the penultimate one, and the numerical centre stays where the SVD
schedule left it. The closing zero-threshold truncation of the chain schedule
is unchanged; it costs about 8 s of each 75 s image below. A request that is
not exact keeps the SVD schedule unchanged.

**The identity skip (#1186, #1197).** The same predicate gates the
pass-through of untouched leaves. Skipping a node is a relabeling of its
parent's view only when no factorization of the schedule truncates; QR with a
zero tolerance truncates nothing, as the zero-threshold SVD did, so the
condition holds unchanged. The #1197 four-node counterexample is run under
the exact request with the skip on and off, and both give the exact
amplitudes.

**Measured** (release, one thread pinned to one core, `ulimit -v` 8 GiB,
single runs on one host; the SVD column is the #1279 branch tip measured on
the same day):

| case | quantity | SVD | QR |
|---|---|---|---|
| R = 10, two interleaved axes | front end [s] | 837 | 364 |
| | schedule [s] | — | 364 |
| | one exact image [s] | 180 to 197 (#1279 probe) | 73 to 77 |
| | peak RSS [GiB] | 3.96 | 2.37 (front end only), 2.44 (schedule and front end) |
| | sampled error / reported bound | 1.45e-13 / 1.66e-8 | 1.31e-13 / 1.66e-8 |
| R = 10, 1D | schedule [s] | 0.028 | 0.024 |
| R = 10, 2D grouped | schedule [s] | 0.18 | 0.16 |
| R = 20, 1D | schedule [s] | 0.22 | 0.21 |
| R = 20, 2D grouped | schedule [s] | 1.24 | 1.20 |

The reported bounds of the four small cases agree to six or seven digits and
their sampled errors to roundoff; their peak RSS is within 5 MB. The
per-image time comes from a temporary timing probe that is not committed. Of
each 73 to 77 s image, 47 to 49 s go into the QR factorizations, 18 to 20 s
into the block contractions and about 8 s into the closing truncation; the
four images take about 304 s of the 364-s front end. The remaining cost is
the QR of the blocks up to 8192 × 3744 that the exact application forms at
the physical bond, which is inherent to an exact image of that width.

**Tests.**

- `tensor4all-treetn`, `contraction/tests/exact_qr.rs`: an exact request
  factorizes only by QR, and its result equals the untruncated SVD zip-up
  (smallest positive threshold) and the dense product to `1e-12` relative, with
  the same bond dimensions and the whole norm at the centre. It covers the
  chain and a branched tree with two degree-three nodes, several centres, real
  and complex data, and the process-global and context leaf routes.
- `tests/subset_operator_exact.rs`: on the same shapes, `schedule_merge_refine`
  with an untruncated SVD zip-up as `apply_options` measures each QR image
  against its SVD image under `rtol = 1e-12`; a truncating threshold of `0.3`
  fails all four.
- `apply_zipup_never_skips_without_an_explicit_exact_policy` also compares the
  exact result with the skip on and off.

The general-tree schedule sets only the canonical region, not the edge
orientations, for SVD and QR alike, so the tree cases check the isometries
numerically (the centre carries the whole norm) rather than through
`validate_ortho_consistency`.

## Fourier front end and operator node names

The subset reconstruction entry points accept independent state and operator
node-name types. Sorted operator nodes bind positionally to the selected full
indices; owner grouping, index binding and exact restructuring remain owned by
`prepare_subset_images`. This supports abstract integer bit nodes even when
state nodes use names or hold several selected indices.

`tensor4all-quanticstransform` provides `quantics_fourier_operator_multiaxis`
with an explicit axis/bit layout. When every axis is monotone along the path it
embeds the per-axis trains directly, passing the other axes' bonds through, so
the product is exact with bond equal to the product of per-axis bonds. Other
layouts compose adjacent separated axes pairwise and restructure exactly by site
swaps, which can grow intermediate bonds far beyond that product. The optional `partitioned`
feature supplies `patched_fourier_transform`: it derives selection path order
and coordinate groups and delegates to `schedule_merge_refine`, returning the
lazy result directly. Branching selections remain unsupported.

Fourier construction error and reconstruction error are separate. Exact layout
restructuring can increase the bonds beyond the cap used per axis. A normalized
inverse call reverses each axis's index vector to account for output bit
reversal. See the [QFT guide](../book/src/guides/qft.md) for the checked example.

Input generation is independent (#1276). A TT over only a patch's remaining
bits must be extended with fixed prefix one-hot sites and assigned the shared
full indices/topology before it can become a `SubDomainTreeTN`. Projectors mask
existing sites; they never introduce missing indices.
