# Einsum Algorithm Comparison: strided-rs vs omeinsum-rs

This document compares the contraction and permutation algorithm optimizations in
**strided-rs** (strided-einsum2 + strided-opteinsum) and **omeinsum-rs**, to guide
the "best of both" merge into **t4a-omeinsum**.

## 1. Binary Contraction Pipeline

### strided-rs (strided-einsum2)

Six-step reshape-to-GEMM pipeline:

1. **Trace reduction** — axes appearing only in one operand are summed out
   *before* GEMM via `reduce_trace_axes()`. Conjugation is materialized during
   the reduce when needed, so the conj flag passed to GEMM becomes `false`.

2. **Permutation to canonical order** — operands are reordered to
   ~~`A[batch, lo, sum]`, `B[batch, sum, ro]`, `C[batch, lo, ro]`~~
   `A[lo, sum, batch]`, `B[sum, ro, batch]`, `C[lo, ro, batch]` via
   `Einsum2Plan::new()`. (Fixed: batch-last for col-major contiguity.)

3. **Element-wise fast path** — when sum, lo, and ro are all empty (pure
   Hadamard product), bypass GEMM entirely and call `zip_map2_into()`.

4. **Fusability check** (`try_fuse_group`) — tests whether dimension groups
   (lo+sum, sum+ro) can be fused into a single contiguous dimension without
   copying. Sorts (dim, stride) pairs by |stride| ascending and verifies
   `stride[i] * dim[i] == stride[i+1]`. If fusable → zero-copy metadata
   extraction; if not → allocate col-major buffer and copy.

5. **GEMM dispatch** — calls `ActiveBackend::bgemm_contiguous_into()`.
   Backend is selected at compile time (faer, CBLAS, or naive).

6. **Copy-back** — if the output was non-contiguous, `finalize_into()` copies
   from the internal buffer back to the original strided view.

**Key features:**
- Fusability checking avoids copies when strides are contiguous.
- Owned-input path (`einsum2_into_owned`) transfers ownership to the operand
  buffer, avoiding a separate allocation when the array is already fusable.
- `BackendConfig` trait declares per-backend requirements
  (`MATERIALIZES_CONJ`, `REQUIRES_UNIT_STRIDE`) so operand preparation adapts
  without per-call `cfg` checks.

### omeinsum-rs

Simpler matricization pipeline:

1. **Mode classification** — batch, left, right, contracted.
2. **Ensure contiguous** — always copies if strided (no fusability check).
3. **Permute to canonical order** — `A[left, contracted, batch]`,
   `B[contracted, right, batch]`. Batch-last layout ensures each batch slice
   is contiguous in column-major memory.
4. **GEMM dispatch** — non-batched or batched (loop of regular GEMMs).
5. **Output permutation** — result permuted from `[left, right, batch]` to
   requested output order.

**Key features:**
- Always-materialize strategy: inputs always made contiguous before GEMM.
  Simpler code, but more memory copies.
- No fusability checking — every contraction goes through full
  permute → GEMM → permute.
- `TypeId`-based runtime dispatch to specialized kernels (tropical-gemm).

### Comparison

| Aspect | strided-rs | omeinsum-rs | t4a-omeinsum recommendation |
|--------|-----------|-------------|----------------------------|
| Copy avoidance | Fusability check (`try_fuse_group`) | Always copy | **Adopt** fusability check |
| Element-wise bypass | `zip_map2_into` for Hadamard | Goes through GEMM | **Adopt** element-wise bypass |
| Trace pre-reduction | Before GEMM, with conj materialization | After classification, as sum-over | **Adopt** strided-rs approach |
| Owned-input optimization | Transfers ownership → zero-copy | Always allocates new buffer | **Adopt** ownership transfer |
| Backend requirements | Compile-time `BackendConfig` trait | Hardcoded | **Adopt** trait-based config |
| Tropical dispatch | Not supported | `TypeId` runtime dispatch | **Adopt** omeinsum approach |
| Batch placement | ~~Batch-first `[batch, lo, sum]`~~ Fixed to batch-last | Batch-last `[left, contracted, batch]` | Both now batch-last |

## 2. N-ary Einsum (Contraction Tree)

### strided-rs (strided-opteinsum)

- **Recursive `eval_node()`** with borrowed-view passthrough:
  - Leaf nodes return operands as borrowed views (no copy).
  - Contract nodes with 1 child: identity passthrough or permutation-only
    (metadata-only, zero-copy).
  - Contract nodes with 2 children: call `eval_pair()`.
  - Contract nodes with 3+ children: build `omeco::EinCode`, run greedy
    optimizer, execute nested tree.

- **Buffer pool** (`BufferPool`): HashMap-indexed free lists keyed by buffer
  size. Freed buffers are returned to the pool after each pairwise contraction;
  subsequent intermediates reuse them.

- **Final contraction writes directly into user's output** via
  `execute_nested_into()` — no extra allocation for the root node.

### omeinsum-rs

- **Recursive `execute_tree()`**:
  - Leaf nodes return `tensor.clone()` (Arc clone, cheap).
  - Node (binary): recurse left and right, then `contract_binary()`.
  - No distinction between 1-child and 2-child nodes.

- **No buffer pool** — each binary contraction allocates a new result tensor.
  Arc-wrapped storage allows cheap tensor cloning but no reuse of freed
  intermediates.

- **Fallback path** (`execute_pairwise`): contracts left-to-right when no
  optimization is performed.

### Comparison

| Aspect | strided-rs | omeinsum-rs | t4a-omeinsum recommendation |
|--------|-----------|-------------|----------------------------|
| Borrowed-view passthrough | Yes (Leaf → borrow) | No (Leaf → Arc clone) | **Adopt** borrowed views |
| Permutation-only detection | Yes (metadata-only) | No | **Adopt** permutation detection |
| Buffer pool | HashMap by size | None | **Adopt** as opt-in option (trades memory for speed) |
| Root writes into user output | Yes (`execute_nested_into`) | No | **Adopt** direct root write |
| Contraction optimizer | omeco greedy | omeco greedy + TreeSA | **Adopt** both optimizers |
| Unoptimized fallback | 3+ child → inline optimize | Left-to-right pairwise | Either acceptable |

## 3. Single-Tensor Operations

### strided-rs (strided-opteinsum `single_tensor.rs`)

Five-step pipeline with two zero-allocation fast paths:

1. **Full trace** (`"ii->"` or `"ijk,ijk->"` with all same index):
   Single loop over diagonal using `diag_stride = sum of all strides`.
   No allocation, single pass.

2. **Partial trace** (`"iij->j"`): Detect one repeated pair, loop directly
   over those two axes' diagonal with other axes iterated normally.
   No allocation, single pass.

3. **General case**: `reduce_axis()` calls for each summed axis, then permute.

4. **Repeat (broadcast)**: stride-0 view for new dimensions, then materialize.

5. **Duplicate** (`"i->ii"`): write to all matching diagonal positions.

### omeinsum-rs

Single function `execute_unary_naive()`:

1. **Index classification**: outer (output) vs inner (summed-over).
2. **Trace handling**: repeated output indices extract diagonal via
   equality check in nested loop.
3. **Summation**: direct nested loop over inner dimensions.
4. **No fast paths**: all single-tensor ops go through the general loop.

### Comparison

| Aspect | strided-rs | omeinsum-rs | t4a-omeinsum recommendation |
|--------|-----------|-------------|----------------------------|
| Full trace fast path | Yes (single loop, no alloc) | No | **Adopt** trace fast path |
| Partial trace fast path | Yes (single loop, no alloc) | No | **Adopt** partial trace fast path |
| General reduce | `reduce_axis()` per axis | Nested loop | strided-rs (cache-optimized kernel) |
| Broadcast/repeat | Stride-0 view + copy | Not needed (einsum semantics) | Conditional |

## 4. Permutation Strategy

### strided-rs

- **Zero-copy views**: `StridedArrayView::permute()` reorders dims/strides
  in metadata only. The underlying data pointer is unchanged.
- **Fusability-aware materialization**: only copies when GEMM backend requires
  contiguous data *and* the dimension group is non-fusable.
- **`StridedView::clone()` is metadata-only** (cheap) — used for
  permute-only paths to avoid double copy.

### omeinsum-rs

- **Zero-copy permute via `Arc<Storage>`**: `Tensor::permute()` creates a new
  `Tensor` with modified shape/strides but shared `Arc<Storage>`. No data copy
  until GEMM needs contiguous input.
- **`ensure_contiguous()`**: copies when `!is_contiguous()`.
- **`permute_data()`**: always allocates new memory for physical reordering.

### Comparison

Both use zero-copy views for permutation. strided-rs's advantage is the
fusability check that can avoid materializing even when the view is
technically non-contiguous (but strides happen to be dense within dimension
groups).

## 5. GPU Strategy

### strided-rs

No GPU support currently. The design document (t4a unified backend) plans:
- CubeCL for map/reduce/broadcast kernels.
- cuTENSOR for einsum (via `TensordotBackend` trait).

### omeinsum-rs

- **cuTENSOR integration**: bypasses reshape-to-GEMM entirely, passes strides
  and modes directly to the cuTENSOR library.
- **Plan caching**: `PlanCache` with `HashMap<CacheKey, Plan>` avoids
  re-creating cuTENSOR plans for repeated contractions.
- **cudarc** for device memory management.

### Recommendation for t4a-omeinsum

Adopt omeinsum-rs's cuTENSOR integration pattern:
- `TensordotBackend` trait wraps cuTENSOR's direct-contraction API.
- Plan caching for repeated contractions.
- Dispatch priority: TensordotBackend > BgemmBackend > naive loop.

## 6. Algebra Extensibility

### strided-rs

- `BgemmBackend<T>` trait: external crates can implement for custom scalar
  types (e.g., tropical semiring).
- Compile-time backend selection via `ActiveBackend` type alias and Cargo
  features.
- No built-in tropical support.

### omeinsum-rs

- `Algebra` trait with `zero()`, `add()`, `mul()`, `to_scalar()` methods.
- `TypeId`-based runtime dispatch to specialized SIMD kernels (tropical-gemm).
- Built-in tropical types: `MaxPlus<T>`, `MinPlus<T>`, `MaxMul<T>`.
- Argmax tracking for tropical backward pass.

### Recommendation for t4a-omeinsum

- Adopt the `Algebra` trait design for semiring extensibility.
- Keep `BgemmBackend<T>` for pluggable GEMM backends.
- Support `TypeId`-based dispatch for performance-critical hot paths (tropical-gemm).
- Argmax tracking as an opt-in feature.

## 7. Summary: Best-of-Both for t4a-omeinsum

### From strided-rs (adopt)

1. **Fusability checking** (`try_fuse_group`) — avoid unnecessary copies when
   dimension groups are already contiguous.
2. **Element-wise bypass** — skip GEMM for Hadamard products.
3. **Trace pre-reduction** — sum trace axes before GEMM with integrated conj
   materialization.
4. **Owned-input optimization** — transfer ownership to avoid allocation.
5. **Buffer pool** (opt-in) — reuse intermediate buffers across pairwise
    contractions. Opt-in because it increases peak memory usage.
6. **Borrowed-view passthrough** — Leaf nodes return borrows, not clones.
7. **Permutation-only detection** — metadata-only transformation in
   contraction tree.
8. **Single-tensor fast paths** — full trace and partial trace zero-allocation
   loops.
9. **Direct root write** — final contraction writes into user's output buffer.
10. **Cache-optimized kernels** — `reduce_axis()` uses blocked, dimension-fused
    iteration from strided-kernel.

### From omeinsum-rs (adopt)

1. **Algebra trait** — semiring-generic `zero()`, `add()`, `mul()` interface.
2. **Tropical-gemm dispatch** — `TypeId`-based runtime specialization for SIMD
   tropical kernels.
3. **Argmax tracking** — tropical backward pass support.
4. **cuTENSOR integration** — direct contraction without reshape-to-GEMM,
   plan caching.
5. **TreeSA optimizer** — simulated annealing for better contraction orders
   (in addition to greedy).
6. **Column-major + batch-last layout** — evaluate whether this is better
   than batch-first for the unified system.

### Neither (new for t4a-omeinsum)

1. **Builder-pattern `BackendConfig`** — unified dispatch across all crates.
2. **Two-tier backend** — `BgemmBackend<T>` (CPU GEMM) +
   `TensordotBackend<T>` (cuTENSOR-like direct contraction).
3. **Global + per-call override** — `einsum()` uses global default,
   `einsum_with()` accepts explicit config.
4. **Custom scalar extensibility tests** — `ModInt<P>` test type to verify
   all three backend tiers work.
