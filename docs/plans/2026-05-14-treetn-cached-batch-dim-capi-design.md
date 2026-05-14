# TreeTN Cached Batch-Dim C API Design

## Goal

Optimize `TreeTNCachedEvaluator` by batching repeated directed-message
contractions over an explicit retained assignment index, and make the optimized
batch evaluator the C API evaluator path.

## Scope

Backward compatibility is not required. The existing
`t4a_treetn_evaluator` C handle should be repurposed for cached batch
evaluation instead of adding a parallel uncached handle family. The Rust
`TreeTNEvaluator` can remain as a reference/simple Rust API, but C bindings
should use the cached evaluator by default.

## Rust Evaluation Design

The current directed-message cache computes each unique message by slicing one
local tensor and then contracting it with selected child messages one assignment
at a time. This has the right cache structure, but it pays repeated
`TensorDynLen` setup, allocation, and index bookkeeping costs.

The batch-dim path should group messages with the same local contraction shape:

1. Build compact assignment batches as today.
2. For each message node, create a fresh assignment index whose dimension is the
   number of unique subtree assignments for that node.
3. Stack the sliced local tensors along that assignment index.
4. Gather child-message tensors according to the child assignment map, also
   carrying the same assignment index.
5. Contract the stacked local tensor and gathered child tensors while retaining
   the assignment index.
6. Store the resulting stacked message tensor and slice rows from it when the
   center contraction needs one point.

This should use `ContractionOptions::with_retain_indices` for `TensorDynLen`
instead of TreeTN-specific dense loops. A generic `TensorLike` seam is acceptable
only if it has a correct default implementation and a specialized
`TensorDynLen` implementation.

## C API Design

Replace the internal contents of `t4a_treetn_evaluator` with owned cached
evaluation state:

- cloned `InternalTreeTN`
- ordered site indices
- selected cached center, if already chosen
- scalar kind

`t4a_treetn_evaluator_new` validates and stores the tree and index order.
`t4a_treetn_evaluator_evaluate` constructs a temporary
`TreeTNCachedEvaluator` borrowing the stored tree, evaluates the batch, then
stores the selected center back into the C handle. This avoids self-referential
Rust structs while preserving center reuse across repeated C calls.

Because compatibility is not required, `t4a_treetn_evaluator_evaluate` may take
a mutable evaluator pointer if that makes the cached state explicit. The one-shot
`t4a_treetn_evaluate` should call the same cached evaluator path internally.

The C API must preserve current error-detail behavior through `run_catching`,
`run_status`, and `t4a_last_error_message`. Regenerate
`crates/tensor4all-capi/include/tensor4all_capi.h` if any signature changes.

## Testing

Add Rust tests before implementation:

- `TreeTNCachedEvaluator` matches `TreeTN::evaluate` on chains and branching
  trees when the batched message path is enabled.
- A shape-sensitive test verifies that retained assignment indices survive
  child-message contraction and are removed only at scalar extraction.
- C API evaluator tests cover repeated batches, clone/release behavior,
  out-of-range coordinates, null pointers, and complex output requirements.
- Header generation is checked after C API changes.

## Benchmarks

Extend `crates/tensor4all-treetn/benches/cached_evaluator.rs` to report the
batched message path alongside `TTCache` and the pre-batch cached path when
useful. The key acceptance data is bond-dimension scaling for equivalent chains:
the TreeTN/TTCache ratio should decrease as `bond_dim` grows, or profiling
output should identify the remaining dominant cost.
