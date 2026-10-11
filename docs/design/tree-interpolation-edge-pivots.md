# Optional edge pivots for the tree interpolation seam

## Status and scope

Proposal for review, grounded in `feat/tree-adaptive-patching` at
`a70a127dd04d03abd795a32dd6906169fa200067`. No API, collection option,
selector, or implementation described here has been adopted. The implemented
M1 contract remains [tree-interpolation-engine-seam.md](./tree-interpolation-engine-seam.md).

The immediate purpose is to let a future recursive, capped split-selection
study obtain the selected pivots of a particular edge through the engine
seam. That study must retain the patch's identity and active layout. This
record supplies the data contract; choosing a production selector still
requires the analysis and decision described in
[tree-pqtci-split-selection.md](./tree-pqtci-split-selection.md).

The approved patch-size unit is one generalized bit per active site,
irrespective of site dimension. Spatial size and warnings about nonbinary
sites remain deferred under
[tree-pqtci-patch-size-bounds.md](./tree-pqtci-patch-size-bounds.md#definitions).
Collecting pivots does not decide those policies or capped-patch acceptance.

## Existing surface and the missing information

The public API inventory and the owning implementations establish:

- `InterpolationProblem::site_order()` concatenates nodes in ascending caller
  name order and each node's active indices in their given order.
- `InterpolationOutcome::pivots` has shape
  `[n_active_sites, n_seed_points]`. Despite its current rustdoc description as
  "full-domain", its rows cover the active problem, not the original domain's
  fixed indices. The patch driver supplies the latter when evaluating.
- `TreeTciInterpolator` creates one vertex per caller node. A vertex fuses
  that node's active site coordinates column-major, or has dimension one when
  it carries no active sites.
- `TreeTCI2::ijset` maps `SubtreeKey` to column-major arrays of shape
  `[n_subtree_vertices, n_selected_pivots]`. `TreeTciGraph::subregion_vertices`
  returns the two components of an edge, in endpoint order. After a completed
  sweep the two selected column counts agree. Arbitrary freshly injected
  global pivots do not have that guarantee before a sweep.
- The engine's private `joined_pivots` joins corresponding columns of every
  edge, unfolds the vertex coordinates to active site rows, and deduplicates
  complete points in first-seen order. The returned seed list consequently
  loses edge membership and each edge's selected column multiplicity.

These facts come from
[`treetn/src/interpolation.rs`](../../crates/tensor4all-treetn/src/interpolation.rs),
[`treetci/src/interpolator.rs`](../../crates/tensor4all-treetci/src/interpolator.rs),
[`treetci/src/state.rs`](../../crates/tensor4all-treetci/src/state.rs), and
[`treetci/src/graph.rs`](../../crates/tensor4all-treetci/src/graph.rs).

A downstream driver must not recover this information by reaching through
to `TreeTCI2` or manufacturing edge pivots from the union of joined seeds.
Projecting that union onto a component can include selections from other
edges and does not reconstruct the chosen edge's pivot matrix.

## Literature boundary

[Grosso et al., arXiv:2602.22372v3, Appendix C, Algorithm 1](https://arxiv.org/html/2602.22372v3#A3)
uses the selected left and right pivots at one maximal TT bond. It crosses
all selected row/column combinations, overwrites a candidate site with each
local value, and ranks each modified matrix. This is a Cartesian product,
distinct from joining corresponding columns for seed recycling.

Exposing the corresponding data on a tree is an interface proposal. Using
one edge's modified matrices to predict total tree parameter savings is a
further hypothesis: a junction tensor depends on all incident bonds. The
paper's TT heuristic does not establish that prediction for arbitrary trees.
Rank thresholds, cost weights, probe budgets, and matched-accuracy evidence
remain selector questions. This record neither ports the paper's pseudocode
nor changes the repository's provenance policy. A later implementation must
record its actual references and resolve the existing provenance/license
review item in the split-selection record.

Removing a tree edge always gives **two** components, including at a node
with degree greater than two. Fixing one active site of dimension `d` creates
`d` coordinate children. A dimension-four generalized bit therefore creates
four children while reducing the active-site count by one; it does not turn
the selected edge into a graph junction split. Nodes with several sites still
split one chosen site at a time; a site-free junction offers no split site.

## Proposed public amendment

Use the existing `tensor4all_treetn::interpolation` owner. Proposed names are
listed for concrete review, not as an implemented API.

| Item | Proposed contract |
| --- | --- |
| `InterpolationProblem::with_collect_edge_pivots(bool)` | Builder setting, default `false`; requests collection on this run. |
| `InterpolationProblem::collect_edge_pivots()` | Returns the boolean request value; it is independent of joined seed recycling. |
| `InterpolationOutcome::edge_pivots` | `Option<BTreeMap<(V, V), InterpolationEdgePivots>>`; owned data for the returned active problem and network. |
| `InterpolationEdgePivots` | One scalar-independent record with public `left_site_positions: Vec<usize>`, `right_site_positions: Vec<usize>`, `left: ColMajorArray<usize>`, and `right: ColMajorArray<usize>`. |
| `validate_edge_pivots` | Shared validation in `treetn`, borrowing the problem, returned network, and present map, returning `Result<(), InterpolationError>`. |

The outcome remains a plain struct usable by external engines. Its optional
field is populated with `None` by engines that do not provide edge pivots.
One record type and the existing node names and `ColMajorArray` suffice; no
engine-state handles, graph vertex IDs, subtree-key types, cached values,
matrix inverses, scalar-specific variants, or public type aliases cross the
seam. Records implement `Debug`; formatting should summarize shapes and row
positions rather than dump all coordinate payloads.

### Edge identity, sides, and rows

For every original problem edge with endpoints `a` and `b`, store the map key
`(u, v)` with `u < v` under the caller's `V: Ord`. Left is the component
containing `u` after removing that edge; right is the component containing
`v`. This is independent of the optimizer's root, sweep direction, insertion
order, or tensor leg order. Map iteration is the canonical edge order.

`left_site_positions` and `right_site_positions` are increasing positions
into `problem.site_order()`. They are disjoint, jointly cover every active
site exactly once, and match the corresponding components. Position identity
is tied to the complete `DynIndex` values in that particular problem, never
to `index.id()` alone. Site identities sharing an ID but differing in tags or
prime level remain distinct.

If an edge has `r` selected columns, `left` has shape
`[left_site_positions.len(), r]` and `right` has shape
`[right_site_positions.len(), r]`. Every coordinate is zero-based and below
the dimension of its active site. Payloads are column-major: coordinate row
`i` of selected column `c` is at offset `i + n_side_sites * c`.

The common column count is positive and equals the returned network's bond
dimension at that named edge. No redundant rank field is stored. These are
selected interpolation coordinates, not orthonormal bases, singular vectors,
candidate pools, or independent samples. The contract guarantees neither a
nonzero function value nor a nonsingular or well-conditioned pivot matrix.
It does not require re-ranking or deduplication after unfolding coordinates.

Every site's value stays in its own range. Fusing several active sites of
dimensions `d0, d1, ...` into a vertex uses
`x0 + d0*x1 + d0*d1*x2 + ...`; unfolding preserves their given local order.
All that node's active rows lie on the same side. A node with no active sites
contributes no rows, including when it became site-free through projection.
Its internal dimension-one vertex coordinate must be zero.

A whole component may have no active sites. Its array then has shape
`[0, r]`, with an empty payload and the same explicitly stored column count
as the other side. Its selected tuple is empty. Validation must accept this
case without dividing by the row count or inferring `r` from payload length.
Dimension-one active sites still occupy rows with coordinate zero and count
as generalized bits.

### Joined seeds and Cartesian probes

Corresponding columns `left[:, c]` and `right[:, c]`, placed at their active
site positions, produce one valid complete **active-domain** point. The
column order preserves the engine's final selected pairing. For TreeTCI,
concatenating these joined points in canonical edge order and deduplicating
in first-seen order yields the current `outcome.pivots`. Other engines may
provide seed points independently; their seed list need not equal that union.
`None` for one field does not imply `None` for the other.

For a pivot matrix, combine left column `i` with right column `j` for every
pair. Matrix rows index left selected columns; matrix columns index right
selected columns. A scalar matrix buffer uses offset `i + r*j`. The coordinate
batch for that matrix has shape `[n_active_sites, r*r]` in the same point
order. A future probe overwrites only the chosen active-site coordinate.
Nothing in the snapshot stores or evaluates these `r*r` combinations.

At the patch-driver boundary, the existing layout and the patch's fixed
coordinates embed those active points into full original-domain points.
Fixed sites are neither snapshot rows nor eligible candidates. A snapshot
must never be interpreted using a parent's or sibling's active layout.

### Availability and fallback

- Collection disabled: `edge_pivots = None`, with no additional side-pivot
  payload allocation or function evaluations.
- Collection requested, engine unsupported: `None`. This is an available
  interface behavior, not an interpolation failure.
- Collection requested, supported swept state: `Some` contains every edge
  exactly once. Partial maps are invalid; completeness avoids conflating
  missing data with low rank when selecting a maximal edge.
- The TreeTCI exact single-node path has no edges: requested collection
  returns `Some(empty map)`, while its existing seed field may remain `None`.
- A valid engine result with no selected pivots cannot synthesize them from
  its bond dimensions or arbitrary points. It returns `None`. In particular,
  TreeTCI materialization represents a zero selected-column count using a
  dimension-one structural bond; such a state cannot supply a record whose
  positive column count equals that bond. Availability must be tested against
  the actual selected sets.

All-zero initial samples still follow M1's `AllSamplesZero` error contract.
A patch-driver zero-screen result or exact patch path that bypasses the
engine, including a scalar patch with no active sites, has no engine pivot
snapshot. Zero-valued *selected*
points in an otherwise successful interpolation remain valid pivot data.

A future optional selector can fall back to the current first eligible site
in validated split order when the requested snapshot is `None` or there are
no edges. Its experiment/report must
record why scoring was unavailable. If the study requires pivot scoring,
such a case is unavailable evidence rather than an invented score. Malformed
present data and evaluator/backend failures propagate as errors; they must
not trigger a silent fallback. Fallback selection still respects `patch_order`
and any separately approved minimum-size rule. If no eligible site remains,
the existing driver no-split behavior applies; unavailable pivot scores do
not create a new candidate.

## TreeTCI production and lifetime

After `optimize_with_proposer` returns, use its final swept `TreeTCI2` state
and the same `VertexLayout` used to materialize the outcome. For each sorted
edge, obtain the two `SubtreeKey` values via `subregion_vertices`, validate
their stored arrays, unfold the selected vertex coordinates directly to the
side's active site rows, and preserve their final column order. Map vertex
endpoints to the caller's node names within the owning engine crate.

This snapshot requires no evaluator calls, extra optimization sweeps, dense
domain materialization, or clone of `ijset_history`. It is produced from the
same final state for convergence, bond-cap, and iteration-limit termination;
a capped or inaccurate result does not make valid coordinates unavailable.
The caller must still observe the termination and independent error report.

The record owns its arrays. Dropping the engine state or starting another
run cannot mutate a previously returned snapshot. Its semantic validity is
limited to the returned problem/network before topology changes,
projection, truncation, or another interpolation. A consumer that mutates the
network must discard the snapshot; private lifetime enforcement is not
promised by the public mutable outcome fields. Reusing joined points as
child seeds does not reuse the parent's edge snapshot.

## Validation and resource cost

Engine implementations and downstream consumers share `validate_edge_pivots`
at the `treetn` seam. It validates a present snapshot against the borrowed
problem and network before Cartesian construction, evaluator dispatch, or
backend allocation. The engine runs it before returning data; the driver
checks data from arbitrary external implementations before use.

Validation covers the complete canonical edge set, endpoint/component
membership, site-position ordering and partition, array rank and row counts,
equal positive selected column counts, coordinate bounds, the outcome's
matching site identities/topology, and agreement with each network bond
dimension. It reports invalid engine data as `InterpolationError::Engine`
with the edge and violated invariant; a construction-size overflow due to
the problem's dimensions is `InvalidProblem`, consistently with M1. Evaluator
failures remain `Evaluator`. Error causes are preserved, not recovered by
matching diagnostic strings. `None` needs no map validation.

Bond checks belong in this owning helper. The current `TreeTN::edge_between`
and `TreeTN::bond_index` let it identify the named edge and its bond index;
the driver consumes the validated record rather than inspecting the engine's
subtree state. Do not add a parallel downstream rank extractor.

For `S` active sites and edge ranks `r_e`, coordinate payload storage is
`S * sum_e r_e` values, plus `O(S * E)` row-position metadata and edge keys.
Uniform rank `R` gives `O(E * S * R)` storage; on a chain, both `E` and `S`
grow with length, so this is quadratic in length at fixed rank. This explicit
cost motivates collection being off by default. Retain the snapshot only
until that patch's split decision; accepted patches and queued children do
not accumulate copies of parent snapshots. It is owned diagnostic output,
not a persistent coordinate-keyed cache.

Check all array-size products and sums, byte counts, node-fusion products,
Cartesian counts `r*r`, matrix byte lengths, batch sizes `S*r*r`, and probe
counts `sum_candidate d_site*r*r` before allocation or integer conversion.
Use fallible reservations where an allocation failure can be returned with
context. A future scorer needs explicit limits for probe evaluations and
matrix/batch memory; overflow checks alone do not bound feasible work.
Evaluate probes in batches through the existing patch cache and use the
configured tensor backend for rank computations. Sharing row-layout metadata
may reduce storage later, but it must not introduce hidden mutable lifetime
dependencies or another public graph representation.

Snapshot construction must traverse canonical edges and preserve final
selected order. Repeated runs with identical deterministic engine inputs
produce identical records. Validation should prepare component/site mappings
once for use across coordinates, reuse decoding scratch, and avoid cloning
the entire state or constructing per-coordinate heap objects. Work cannot be
constant in tree length because the requested output itself is large.

## Planned validation and documentation impact

Implementation, if approved, needs meaningful coverage of:

- Chain and branched topologies, nontrivial canonical-name ordering, and
  different insertion/sweep orders; known side memberships and coordinates.
- Several sites on one node with mixed dimensions, site-free internal and
  leaf nodes, an empty-active-site component, fixed sites, dimension-one
  active sites, and same-ID indices with distinct full identities.
- Known paired joins versus the full Cartesian probe matrix; column-major
  point/matrix order and consistency with current TreeTCI seed recycling.
- Converged, capped, and iteration-limit swept outcomes; zero-valued selected
  points, all-zero initial samples, and the legitimate no-selection case.
- Disabled collection, an engine returning `None`, requested single-node
  collection, and mock-engine/scorer fallback without guessing edge sets.
- Every malformed-present branch, including reversed/unknown/missing edges,
  overlapping or unsorted rows, wrong component ownership, wrong shapes,
  out-of-range coordinates, rank disagreement, and checked size overflow.
- Snapshot independence after state drop/another run, and no additional
  evaluator calls or side-array allocation when collection is disabled.

Small numeric reference tests materialize each result once and compare whole
residuals; larger cases use structural invariants or bounded batched samples.
No tolerance changes are proposed. Pure coordinate fixtures check actual
known coordinates rather than shape alone.

Adding an outcome field would update every outcome literal, mock engine,
rustdoc example, and seam description in the same implementation branch.
New public items need runnable asserted rustdoc and explicit error conditions.
Clarify the existing seed field's active-domain wording in its owner. Check
README, guides, tutorials, examples, and `skills/use-tensor4all-rs/` for any
new capability claims; advertise edge-pivot support only once it is validated.
No user guide or README capability changes are appropriate for this proposal.

## Decisions still required

1. Approve or revise the opt-in request and complete optional map. Collecting
   only one selected edge could reduce memory, but makes edge selection part
   of the request and would require another policy contract. This proposal
   favors all edges for a general study and documents their cost.
2. Confirm that unavailable edge data allows deterministic sequential
   fallback, and decide the production reporting surface when a selector is
   eventually proposed. Strict study requirements can reject such evidence.
3. Decide feasible snapshot/probe budgets for downstream-scale ranks and
   layouts. A boolean request does not provide a hard byte limit; no numerical
   default budget is justified by the current experiments.
4. Establish the tree selector's rank tolerance and junction cost model,
   then obtain recursive capped, matched-accuracy evidence. These are not
   settled by validating the pivot-coordinate contract.

Prose-only validation of this record checks links and consistency with the
listed API inventory and source revision. It does not constitute an API
implementation test or new selector evidence.
