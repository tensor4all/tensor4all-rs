# Design: Edge-Based Reverse AD for tidu/tenferro

**Date:** 2026-03-29
**Status:** Drafted after cutover review
**Scope:** Detailed reverse-mode design note for removing single-tape leakage from the cutover

## Goal

Replace the current reverse-mode implementation that assumes one shared `Tape` per connected computation with a torch-like edge-based model where:

- each reverse-capable tensor carries an optional edge to the node that produced it
- multi-input ops may consume tensors originating from previously independent graphs
- `ensure_common_reverse_tape(...)` disappears from the public `tenferro` surface
- downstream crates such as `tensor4all-rs` do not need tape-normalization helpers

This note refines the high-level cutover design in
`docs/plans/2026-03-29-tidu-tenferro-cutover-design.md`.

Status update:

- the public tape-normalization helper leak has already been removed
- the remaining work is the internal reverse-graph rewrite and typed-carrier cleanup
- the next blocker is no longer "how do we join independent reverse graphs?" but
  "how do we stop threading `AdTensor<T>` through the higher AD crates?"
- the staged answer to that blocker now lives in
  `docs/plans/2026-03-30-adtensor-carrier-replacement-design.md`

## Complexity Target

Backward execution must remain linear in the size of the reachable reverse
subgraph.

More precisely, the target complexity is:

- `O(|V| + |E|)` in the number of reachable reverse nodes and reverse edges

When typical op arities are bounded, this is effectively linear in the number
of reachable nodes. The design must not introduce repeated ancestor walks,
quadratic graph merges, or per-node hash-based rescans in the hot path.

Practical consequences:

- each reachable reverse node is discovered once while building the task
- each reverse edge is examined once while building dependency counts
- each node is enqueued at most once when it becomes ready
- each node's `pullback` is executed at most once per backward call
- cotangent accumulation is slot-local and append-free after task construction

## PyTorch Reference Points

This design should follow PyTorch's internal split where it materially helps.

Relevant reference points:

- `TensorImpl` stores nullable autograd metadata via `autograd_meta_ = nullptr`
  and treats null as the default no-autograd state.
- `Variable` carries a canonical gradient edge:
  - interior values use `grad_fn + output_nr`
  - leaves canonicalize to a grad accumulator (`AccumulateGrad`)
- `Edge` points directly at a node, not at a graph/tape identity:

```cpp
struct Edge {
  std::shared_ptr<Node> function;
  uint32_t input_nr;
};
```

- `GraphTask` is **per backward execution**, not persistent tensor state
- `InputBuffer` accumulates multiple incoming cotangents before a node runs
- `unpack_dual(non_dual)` returns `(primal, None)` instead of erroring

For `tidu` and `tenferro`, the important lesson is not that every type name
must match PyTorch, but that:

- persistent tensor metadata stores edges, not global tape membership
- leaf accumulation is a sink role, not a graph-identity special case
- backward execution state is transient and separate from persistent graph data

## Problem Statement

The current implementation no longer needs a public tape-normalization helper,
but the internal reverse stack still carries single-tape-era assumptions in its
typed carrier and reverse wrappers.

Observed downstream failure before the helper removal:

- `crates/tensor4all-core/tests/tensor_native_ad.rs::reverse_ad_sequential_qr_absorption_across_three_leaves_preserves_common_tape`
- `crates/tensor4all-core/tests/tensor_native_ad.rs::reverse_ad_sequential_svd_absorption_across_three_leaves_preserves_common_tape`

Both tests create three independent reverse leaves (`left`, `center`, `right`), run factorization on `left` and `right` separately, and then contract their results into one final dense tensor before calling `backward(...)`. This is a valid torch-like usage pattern. The helper-free path is now green, which proves the remaining problem is not graph joining itself but the internal carrier and reverse-wrapping model.

## Current Architecture and Why It Leaks

### tenferro

Current reverse tensors still carry:

- a `ReverseAttachment { node, tape }`
- leaf state keyed by one `Tape<DynTensor>`
- helper methods such as `ensure_reverse_leaf_on(&Tape<_>)`

This means a reverse tensor is not just "produced by some node". It is also a member of one specific tape identity.

Boundary cleanup status:

- `tenferro` now sources `AdTensor` / `NodeId` from `tenferro-internal-ad-core`
- `tenferro-internal-ad-ops` and `tenferro-internal-ad-linalg` no longer
  re-export `AdTensor` from their crate roots; internal modules now import the
  typed carrier directly from `tenferro-internal-ad-core`
- `internal-ad-surface` remains the public transitional surface for `AdMode`, dynamic tensor APIs, and reverse-mode entrypoints
- the next cleanup step is to stop re-exporting `AdTensor` casually from higher crates, but that is a prelude to the real carrier replacement, not the carrier replacement itself

### tidu

The current `Op::apply(...)` implementation computes `common_tape(inputs)` and rejects inputs belonging to different reverse graphs. This is still a tape-centric execution model, even though the public API has already moved toward output slots and high-level ops.

### Consequence

Two independent reverse computations cannot be joined later unless some upstream code first forces them onto one shared tape. That is the exact behavior `ensure_common_reverse_tape(...)` exposes.

## Target Reverse Model

The target model is edge-based rather than tape-based.

### Key property

A reverse-capable value is defined by:

- its primal value
- whether gradients are required
- an optional edge to the op output that produced it

It is **not** defined by membership in a global or shared tape chosen before composition.

### High-level behavior

- leaf tensors created by `with_requires_grad(true)` have no parent edge
- op outputs created from reverse-capable inputs store edges to their producing node/output slot
- a later op may freely consume inputs from previously independent computations
- the new op node simply records parent edges to all those inputs
- backward traversal starts from one or more output edges and recursively walks parents

This matches the user-visible model of PyTorch more closely: tensors carry graph edges, not tape identities.

## Data Model

### tenferro::Tensor

Long-term public shape remains:

```rust
pub struct Tensor {
    primal: /* tenferro-owned dynamic tensor payload */,
    autograd: Option<AutogradMeta>,
}
```

`AutogradMeta` remains reverse-header-only plus public surface state:

```rust
pub struct AutogradMeta {
    requires_grad: bool,
    reverse: Option<ReverseMeta>,
    leaf_grad: Option<Tensor>,
    retain_grad: bool,
    is_leaf: bool,
}
```

The reverse-specific part no longer stores a `Tape`.

```rust
pub struct ReverseMeta {
    edge: Option<ReverseEdge<Tensor>>,
}
```

Leaf tensors have:

- `requires_grad = true`
- `is_leaf = true`
- `edge = None`

Non-leaf reverse tensors have:

- `requires_grad = true`
- `is_leaf = false`
- `edge = Some(...)`

Primal-only tensors have `autograd = None`.

### tidu reverse handles

`tidu` needs an internal reverse handle that does not depend on a shared tape.
Following PyTorch, the handle should reference a node directly.

For phase 1, `tidu` may keep its existing producer-oriented terminology
(`output_slot`) instead of mirroring PyTorch's consumer-oriented `input_nr`.
That is an implementation choice. The important part is that the handle points
to a node, not to a tape identity.

```rust
pub struct ReverseEdge<V: Differentiable> {
    node: Arc<ReverseNode<V>>,
    output_slot: usize,
}
```

Each reverse node owns:

- op rule / saved backward state
- parent inputs as optional edges
- output count
- metadata needed for backward scheduling

This means the persistent reverse graph is an object graph of `Arc<ReverseNode>`
references. It is **not** an arena keyed by one `Tape` chosen before graph
composition.

Conceptually:

```rust
pub struct ReverseNode<V: Differentiable> {
    rule: Box<dyn EngineRule<V>>,
    inputs: Vec<InputRef<V>>,
    output_count: usize,
}
```

Where:

```rust
pub enum InputRef<V: Differentiable> {
    PrimalOnly,
    Leaf(LeafHandle<V>),
    Edge(ReverseEdge<V>),
}
```

`LeafHandle<V>` identifies the user-visible leaf grad accumulator for that tensor handle.

PyTorch models this role with `AccumulateGrad`, which is a sink `Node`. For
phase 1 in `tidu`, a dedicated `LeafHandle<V>` is simpler and preserves the
same semantic split:

- interior values point to producing nodes
- leaf values accumulate into a sink-like object

We do **not** need to materialize full `AccumulateGrad` nodes immediately to
get the correct user-visible behavior.

### Value-level reverse state

The high-level `tidu::Value<V>` reverse state should move away from:

- `tape: Option<Tape<V>>`
- `node_id: Option<NodeId>`

to something like:

```rust
enum ReverseHandle<V: Differentiable> {
    None,
    Leaf(LeafHandle<V>),
    Edge(ReverseEdge<V>),
}
```

with a surrounding state:

```rust
struct ReverseState<V: Differentiable> {
    requires_grad: bool,
    handle: ReverseHandle<V>,
}
```

This matches the public semantics better:

- detached value -> `None`
- leaf requiring grad -> `Leaf(...)`
- op result requiring grad -> `Edge(...)`

and removes the need for `ensure_attached_to(&Tape<_>)`.

### Saved state ownership

Saved backward state stays on the producing node, not on the output tensor. This preserves the earlier cutover decision:

- `tenferro::Tensor` only stores public AD header state
- `tidu` owns graph internals and saved rule state

## Backward Execution Model

Backward no longer walks one tape. It walks the reachable reverse subgraph from
the requested outputs.

Following PyTorch's `GraphTask`, the persistent graph and the execution-time
state are separated.

### Persistent state vs execution state

Persistent state:

- `Value` reverse handles
- `ReverseNode`
- `LeafHandle`
- saved backward state on nodes

Execution-time state for one `backward()` call:

- root seeds
- per-node pending cotangents
- dependency counts / readiness
- final captured gradients
- graph-lifetime flags such as `keep_graph`

Conceptually:

```rust
struct BackwardTask<V: Differentiable> {
    keep_graph: bool,
    not_ready: HashMap<NodeKey, SlotBuffer<V>>,
    dependencies: HashMap<NodeKey, usize>,
    leaf_grads: HashMap<LeafKey, V::Tangent>,
}
```

This is the `tidu` analogue of PyTorch's `GraphTask + InputBuffer`.

### Root seeding

- `backward()` on a scalar output creates one seed cotangent for that output edge
- `backward_with_gradient(seed)` seeds the matching output edge
- multiple roots are allowed when higher-level APIs request them

### Traversal

The engine maintains a transient `BackwardTask` with:

- a slot buffer per reachable node, analogous to PyTorch's `InputBuffer`
- dependency counts used to decide when a node is ready to run
- leaf gradient accumulation

Processing one node:

1. gather `grad_outputs` for all output slots from the node's slot buffer
2. call `rule.pullback(...)`
3. route returned input cotangents:
   - to another node's slot buffer if the input is `InputRef::Edge`
   - to a leaf accumulator if the input is `InputRef::Leaf`
   - nowhere if the input is `PrimalOnly`
4. when a parent node has received everything it needs, enqueue it

This is the point where PyTorch's `InputBuffer` idea matters: fan-in summation
belongs to the execution task, not to persistent tensor metadata.

### Data structures required by the complexity target

To preserve the linear-time target, `GraphTask` should normalize the reachable
subgraph into task-local dense ids during construction.

Recommended shape:

```rust
type TaskNodeId = usize;

struct GraphTask<V: Differentiable> {
    nodes: Vec<TaskNode<V>>,
    node_ids: HashMap<NodeKey, TaskNodeId>,
    ready: VecDeque<TaskNodeId>,
    leaf_grads: HashMap<LeafKey, V::Tangent>,
}

struct TaskNode<V: Differentiable> {
    node: Arc<ReverseNode<V>>,
    grad_outputs: Vec<Option<V::Tangent>>,
    live_output_slots: Vec<bool>,
    remaining_contributions: Vec<usize>,
    pending_output_slots: usize,
    enqueued: bool,
}
```

Notes:

- the `HashMap<NodeKey, usize>` is build-time indexing only
- the hot path after construction should use dense `Vec` indexing
- `remaining_contributions` is per output slot, not per node
- no traversal step should search the graph globally for parents or children

This is another reason not to use `petgraph`: the needed executor state is
slot-buffer centric, and the hot path wants dense task-local arrays rather
than generic graph indirection.

`NodeKey` and `LeafKey` should be stable identity keys derived from the
persistent graph objects, for example pointer identity of `Arc<ReverseNode<V>>`
and `LeafHandle<V>`.

### Exact meaning of `remaining_contributions`

`remaining_contributions[slot]` is **not** a generic dependency count. It is
the exact number of cotangent writes that are still expected to arrive at that
output slot before the owning node is allowed to execute.

For one output slot, the count is initialized as:

```text
downstream_consumers_within_task + explicit_root_seeds_for_that_slot
```

Where:

- `downstream_consumers_within_task` counts reachable child-node inputs that
  reference this output slot
- `explicit_root_seeds_for_that_slot` counts how many backward roots directly
  seed this output slot, usually `0` or `1`

Every time a cotangent is written into `grad_outputs[slot]`, the executor:

1. accumulates into `grad_outputs[slot]`
2. decrements `remaining_contributions[slot]` by exactly `1`

So a slot is complete exactly when:

```text
remaining_contributions[slot] == 0
```

### `live_output_slots` and `pending_output_slots`

Not every output slot of a reachable node participates in the current backward
task.

`live_output_slots[slot]` is `true` iff the slot is part of the reachable
backward subgraph, meaning at least one of:

- the slot is directly seeded by a backward root
- a reachable child-node input references the slot

`pending_output_slots` is the number of live output slots whose
`remaining_contributions[slot] > 0`.

This gives a precise readiness condition:

```text
node is ready <=> pending_output_slots == 0
```

and avoids incorrectly waiting on unused output slots.

### Build-time algorithm

Task construction should proceed in three linear passes.

Pass 1: Reachability

- DFS/BFS from the root edges
- assign each discovered `ReverseNode` a dense `TaskNodeId`
- allocate `TaskNode` with correctly sized vectors

Pass 2: Fan-in counting

- iterate every reachable node's inputs once
- for each `InputRef::Edge(parent_edge)`:
  - mark the parent slot as live
  - increment that parent slot's `remaining_contributions`
- iterate every root edge once:
  - mark the root slot as live
  - increment that slot's `remaining_contributions`

Pass 3: Finalize readiness

- for each task node:
  - compute `pending_output_slots` by counting live slots with
    `remaining_contributions > 0`
- seed roots:
  - write the root cotangent
  - decrement that slot's `remaining_contributions`
  - if a slot reaches zero, decrement `pending_output_slots`
  - enqueue the node when `pending_output_slots == 0`

After this point, the hot path is dense-vector only.

### Grad accumulation

Leaf grad accumulation stays user-visible and torch-like:

- leaf `.grad()` accumulates over repeated backward calls
- retained non-leaf `.grad()` caches are optional and only populated when requested

The edge-based rewrite should keep this user-facing behavior unchanged.

## Why This Fixes the Current Failure

Revisit the failing `left / center / right` sequential factorization case.

With the new model:

1. `left.with_requires_grad(true)` creates one leaf with no parent edge
2. `factorize(left)` creates outputs whose edges point to a factorization node
3. `right.with_requires_grad(true)` does the same independently
4. `contract(center_after_left, right_fact.right)` simply creates a new node with two parent edges, even though those parent edges originated from computations that were previously independent

No pre-merge step is needed because the final op stores direct edges to both parents. There is no notion of "mixed reverse tapes".

## Relationship to Op / Schema

The high-level `tidu::Op` trait does not need to change shape to adopt edge-based reverse execution.

What changes is the internal implementation of:

- `Value<V>` reverse attachment
- `Op::apply(...)`
- the engine node store / backward traversal

The public high-level contract remains:

- runtime `Schema`
- `primal(...)`
- `save_for_backward(...)`
- `backward(...) -> Vec<Option<_>>`
- `jvp(...) -> Vec<Option<_>>`

This is important because the edge-based rewrite is an internal execution refactor, not a new public AD API.

## Unified Public Architecture

The design boundary for phase 1 should now be stricter.

- `tidu` public reverse-mode API is unified around edge-based `Value` / `Op`
- `tenferro` public ops use only that API
- `tidu::expert::Tape` is not kept as a second public reverse API

Rationale:

- early development means there is no compatibility requirement to preserve
  the old public substrate
- keeping both `Value/Op` and `expert::Tape` would preserve the exact
  conceptual split we are trying to remove
- `tenferro` is already forbidden from relying on `tidu::expert`
- a single public reverse model is easier to document, test, and reason about

This means the rewrite is not "high-level path first, compatibility path
later". It is a single unification:

- one public reverse model in `tidu`
- one internal edge-based executor
- no public tape identity in the final design

Low-level internal helpers may still exist, but they should not appear as a
second public AD model.

## Graph Lifetime and `GraphFreed`

PyTorch separates persistent graph structure from per-backward `GraphTask`
lifetime. `tidu` should do the same.

Phase-1 policy:

- persistent `ReverseNode` / `LeafHandle` stay alive as long as some `Value`
  or parent edge keeps them alive
- `BackwardTask` is one-shot execution state
- default `backward()` drops execution-only state after completion
- retained grads remain readable because they live on leaves or retained
  non-leaf caches, not in the task object

`GraphFreed` therefore should stop meaning "the tape object was consumed" and
instead mean "this exact execution state cannot be reused for another backward
without rebuilding or retaining it".

This is much closer to the user-visible torch-like expectation.

## Incremental Migration Strategy

This rewrite is still part of the single `tenferro-rs` cutover PR, but it should be implemented in explicit internal slices.

### Slice 1: Isolate the current leak

- keep the current downstream regression tests
- add focused upstream tests that combine previously independent reverse computations
- document `ensure_common_reverse_tape(...)` as temporary and remove new downstream call sites

### Slice 2: Introduce edge-based reverse internals in tidu

- add edge-based reverse node / edge types
- add `BackwardTask`-like transient execution state
- keep the current forward-mode level system unchanged
- replace public `expert::Tape`-style reverse entrypoints with the unified
  edge-based API rather than maintaining both

### Slice 3: Rewire tenferro Tensor reverse metadata

- replace tape-specific reverse attachment with edge-based metadata
- stop calling `ensure_reverse_leaf_on(&Tape<_>)`
- keep public semantics (`grad`, `retain_grad`, `zero_grad`, `is_leaf`) stable

### Slice 4: Port op application and cleanup

- remove `common_tape(inputs)` assumptions from `Op::apply(...)`
- keep the public helper-free join path
- update tenferro integration tests and downstream tensor4all regression tests

## Non-Goals of This Note

This note does not redesign:

- forward-mode level internals
- PyTorch-style alias/version-counter semantics
- by-construction `chainrules` DSL unification
- public optimizer APIs

Those remain tracked by the broader cutover design or separate follow-up issues.

## Verification Strategy

Required verification for the edge-based rewrite:

- existing `tidu-rs` test suite stays green
- existing `tenferro-rs` test suite stays green
- existing `tensor4all-rs` TreeTN and tensor native AD tests stay green
- the two downstream sequential factorization regressions pass without any public tape-normalization helper
- public `grad`, `retain_grad`, `zero_grad`, and `detach` semantics remain unchanged

## Exit Criteria

The reverse rewrite is considered complete only when all of the following are true:

- no public `Tensor::ensure_common_reverse_tape(...)` remains
- `tidu::Op::apply(...)` no longer rejects previously independent reverse computations solely due to graph identity
- downstream `tensor4all-rs` does not contain AD tape workaround code
- the sequential three-leaf QR/SVD absorption regressions pass
