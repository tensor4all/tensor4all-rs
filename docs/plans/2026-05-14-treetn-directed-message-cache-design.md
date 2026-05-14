# TreeTN Directed Message Cache Design

## Goal

Improve `TreeTNCachedEvaluator` performance without adding a chain-specific fast
path. The cached evaluator should reuse intermediate contractions inside each
subtree, so a linear-chain TreeTN follows the same cache shape as `TTCache`
while still working for arbitrary tree topology.

## Root Cause

The first cached evaluator caches one tensor per unique neighbor-component
assignment. For each unique assignment it slices every node in the component,
builds a temporary `TreeTN`, and calls `contract_to_tensor`. This is correct but
expensive: benchmark breakdowns showed environment construction dominated the
runtime.

`TTCache` is faster because it does not recompute a whole prefix or suffix for
every unique half-assignment. It memoizes intermediate prefix and suffix vectors
recursively, so related assignments share work.

## Approach

Root the tree at the selected evaluation center. For each non-center node, build
a directed message from that node to its parent. A message key is the assignment
restricted to the node's entire rooted subtree. A message value is the contracted
environment tensor from that subtree into the parent bond.

Messages are computed postorder:

1. Slice the local node tensor by the node's site values.
2. Recursively fetch child messages for the same subtree assignment.
3. Contract the sliced local tensor with the child messages.
4. Cache the resulting parent-facing message by `(node, parent, subtree key)`.

The final point evaluation contracts the center tensor, sliced by center-owned
site values, with the selected child messages. This keeps the public API
unchanged and preserves the existing generic `TensorLike` abstraction.

## Tensor Slicing

Add a high-level `TensorLike::select_indices` method with a default
implementation using `onehot + contract`. Override it for `TensorDynLen` to call
the existing dense `select_indices` method. This avoids ad hoc downcasts in
TreeTN and benefits both the cached evaluator and existing evaluators.

## Tests

- A new internal cached-evaluator test constructs a 3-node chain with repeated
  subtree assignments and asserts the directed-message compute count is lower
  than the old component-environment count.
- Existing correctness tests continue to compare against `TreeTN::evaluate`.
- Rustdoc examples for `TensorLike::select_indices` must be runnable.

## Expected Performance

The TreeTN cached evaluator should still be slower than `TTCache` because it
uses generic tensor contractions rather than rank-vector matrix-vector kernels.
However, environment construction should scale with unique directed subtree
messages instead of unique whole component assignments, closing the largest
current gap in a topology-generic way.
