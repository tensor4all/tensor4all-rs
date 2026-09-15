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
the target; no public `reference_norm` override or `orthogonal: true` flag is
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
- `ReconstructionOptions { target_bond_dim, split_indices, max_regions }`:
  representation/search policy only.
- `reconstruct(&target, &center, tolerance, &options)`: owned result, no mutation.
- `ReconstructedTreeTN<V>`: immutable region/term lists and a report.
- `ReconstructionReport`: reference norm, allowance, measured error bound,
  region/term/rank counts, accepted splits and final-region merges.

The absolute allowance is `delta = max(atol, rtol * target.reference_norm())`.
The reference never changes after dropping, splitting, merging, or rounding.
Zero targets are valid. Empty targets have no topology; nonempty zero targets
still validate the requested center and site identities.

Output regions are disjoint; terms within a region need not be orthogonal.
Consequently their norm squares must not simply be added. `regions()` exposes
the terms. `into_partition()` performs structural conversion only and rejects
multi-term regions. It never silently forms a global sum.

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
   than the parent's. Among eligible candidates choose the smallest total
   logical parameter count; supplied index order breaks ties.
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

## QFT integration contract (follow-up, not implemented here)

QFT must accept an explicit subset of external full site indices, with ordered
binary indices for each transformed coordinate. These need not be contiguous
nodes and may share a node with spectator indices. All remaining indices are
spectators: preserve their identity, dimension, and node assignment. Reject
duplicates, absent indices, dimension aliases, and nonbinary selected sites.
An empty selection can be an explicitly documented identity operation.

The QFT selection is independent of reconstruction's permitted split indices.
Input bits are ordered from most to least significant. Bind the existing QFT
operator explicitly and document its bit-reversed output mapping; contiguous
frequency blocks must fix the actual high frequency bits. A full transform is
the same interface with all desired axes selected. Multidimensional axes each
carry their own ordered subset; normalization is unitary on selected axes.
Do not silently add padding or change grid length: padding needs an explicit
domain/embedding contract.

The map is `F_selected ⊗ I_spectators`, so exact QFT preserves orthogonality and
the original global norm even though transformed supports overlap. A QFT-owned
prepared target must carry that provenance from its validated preimage. It must
not try to pass overlapping images through `from_partition` or recompute their
global norm by assuming disjoint image supports. Approximation of the QFT MPO
and its application must be accounted separately from reconstruction; approximate
unitarity cannot silently justify an exact error certificate.

QFT also supplies the input-merge/output-refine schedule and its bit geometry.
The complementary-area invariant of the Fourier algorithm is not a generic
reconstruction invariant. The current greedy reconstruction entry point does
not claim to implement that schedule, automatic zero-padding, or QFT itself.
