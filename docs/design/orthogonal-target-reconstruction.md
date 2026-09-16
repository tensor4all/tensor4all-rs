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
count, repeats an index, or is absent from the preimage site space.

Selected indices must currently sit on distinct tree nodes. A node owning two
selected indices would need the operator's MPO nodes merged into one multi-site
node; that is not implemented, and the call rejects it with repair guidance
("transform indices that share a node separately") rather than silently
mis-binding them. Spectator indices on the same node as a selected index are
supported.

The QFT selection is independent of reconstruction's `patch_order`.
Input bits are ordered from most to least significant. A full transform is
the same interface with all desired axes selected. Multidimensional axes each
carry their own ordered subset and are applied as separate calls; normalization
is unitary on selected axes. Do not silently add padding or change grid length:
padding needs an explicit domain/embedding contract.

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
not claim to implement that schedule, automatic zero-padding, or QFT itself.
