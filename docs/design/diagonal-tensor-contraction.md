# Diagonal Tensor and Contraction Path Optimization Design

This document captures the design decisions and implementation plan for diagonal tensor support and contraction path optimization in tensor4all-rs.

## Background

When contracting tensors with diagonal structure (e.g., SVD: U * s * V where s is diagonal), we want to:
1. Preserve the diagonal structure for memory efficiency
2. Use optimal contraction ordering via omeco
3. Handle "hyperedges" (indices shared by 3+ tensors) correctly

## Key Discoveries

### omeco's Hyperedge Handling

omeco correctly:
- Tracks hyperedges via `IncidenceList::is_external()`
- Computes contraction costs correctly for hyperedges
- Returns optimal contraction paths

omeco's limitation:
- `contraction_output()` in `expr_tree.rs` only checks `final_output`, not remaining tensors
- This causes incorrect intermediate outputs in `NestedEinsum` for hyperedges

Example test case (in `contract.rs`):
```rust
// U(i, j), s(j), V(j, k) - j is hyperedge connecting all 3 tensors
let ixs = vec![vec!['i', 'j'], vec!['j'], vec!['j', 'k']];
let output = vec!['i', 'k'];

// omeco's tree: U*s → output 'i' only (j incorrectly contracted!)
// Correct: U*s → output 'i', 'j' (j must remain for V)
```

### Solution Architecture

Use omeco for path optimization only. Execute contractions ourselves with "external indices" tracking:

```
Tensors → omeco.optimize_code() → NestedEinsum (path only)
                                        ↓
                             Custom execution with:
                             - Track which tensors still need each index
                             - Only contract index when all tensors using it are consumed
```

## DiagOuter Storage Design

### Structure

```rust
/// Outer product of primitive diagonals: δ_{i1,j1} ⊗ δ_{i2,j2} ⊗ ...
///
/// Memory: O(Σ d_k) instead of O(Π d_k) for full tensor
pub struct DiagOuterStorageF64 {
    /// Diagonal components: components[k] has length d_k
    /// Total storage = sum of component lengths
    components: Vec<Vec<f64>>,

    /// Which tensor indices belong to each diagonal group
    /// index_groups[k] = [i, j] means indices i and j form diagonal k
    index_groups: Vec<Vec<usize>>,
}

pub struct DiagOuterStorageC64 {
    components: Vec<Vec<Complex64>>,
    index_groups: Vec<Vec<usize>>,
}
```

### Example: 3D Superdiagonal

δ_{IJK} where I==J==K:
```rust
DiagOuterStorageF64 {
    components: vec![vec![1.0; d]],  // One component of length d
    index_groups: vec![vec![0, 1, 2]],  // All 3 indices in one group
}
```

### Example: SVD diagonal s

s(j) where j is 1D:
```rust
DiagOuterStorageF64 {
    components: vec![s_values.clone()],
    index_groups: vec![vec![0]],  // Single index
}
```

## Contraction Execution with External Tracking

### Algorithm

```rust
fn execute_with_externals(
    tensors: &[Tensor],
    tree: &NestedEinsum,
    remaining_tensors: &HashSet<usize>,  // Tensor indices not yet consumed
) -> Tensor {
    match tree {
        NestedEinsum::Leaf { tensor_index } => tensors[*tensor_index].clone(),
        NestedEinsum::Node { args, .. } => {
            // Recursively evaluate children
            let children: Vec<Tensor> = args.iter()
                .map(|arg| execute_with_externals(tensors, arg, remaining_tensors))
                .collect();

            // Compute which tensors are still remaining after this node
            let consumed_here: HashSet<usize> = collect_leaf_indices(tree);
            let still_remaining: HashSet<usize> = remaining_tensors - consumed_here;

            // Find "external" indices: those needed by still_remaining tensors
            let external_indices: HashSet<Index> = still_remaining.iter()
                .flat_map(|&i| tensors[i].indices())
                .collect();

            // Contract children, but keep external indices uncontracted
            contract_with_externals(&children, &external_indices)
        }
    }
}
```

### Key Functions Needed

1. `collect_leaf_indices(tree)` - Get all tensor indices under a node
2. `contract_with_externals(tensors, externals)` - Contract, preserving external indices
3. Integration with `contract_storage()` to pass external info

## Implementation Plan

### Phase 1: Storage Types (Blocked by mdarray-linalg bug)

1. Add `DiagOuterF64` and `DiagOuterC64` variants to `Storage` enum
2. Implement basic operations:
   - `new(components, index_groups)`
   - `to_dense_storage()` - expand to full dense
   - `contract_diag_outer_*` - specialized contractions

### Phase 2: Tensor Layer

1. Add constructor `TensorDynLen::from_diag_outer()`
2. Update `tensordot()` to detect and handle DiagOuter
3. Optimization: DiagOuter × Dense can be O(n²) instead of O(n³)

### Phase 3: Contraction Path Execution

1. Modify `execute_contraction_tree()` in `contract.rs`
2. Add external indices tracking
3. Test with hyperedge cases (SVD, delta)

### Phase 4: mdarray-linalg Integration

After upstream bug fix:
1. Replace hand-written GEMM with `Naive.contract()`
2. Consider using Faer backend for better performance

## mdarray-linalg Bug

Location: `mdarray-linalg/src/matmul.rs` lines 250-262

Bug: When `keep_shape_a.is_empty()`, code reshapes to `keep_shape_a` (empty) instead of `keep_shape_b`.

Fix (applied locally but not usable due to version mismatch):
```rust
} else if keep_shape_a.is_empty() {
    ab_resh
        .view(0, ..)
        .reshape(keep_shape_b)  // Fixed: was keep_shape_a
        ...
} else if keep_shape_b.is_empty() {
    ab_resh
        .view(.., 0)
        .reshape(keep_shape_a)  // Fixed: was keep_shape_b
        ...
}
```

Issue: Local `extern/mdarray-linalg` uses different mdarray version (git rev) than Cargo.toml (0.7.2), causing type incompatibility.

## Test Cases

### Existing Tests (in contract.rs)

- `test_omeco_hyperedge_delta` - A(i,x) * B(j,x) * C(k,x) * delta(x)
- `test_omeco_hyperedge_svd` - U(i,j) * s(j) * V(j,k)

### Additional Tests Needed

1. DiagOuter creation and dense conversion
2. DiagOuter × Dense contraction
3. DiagOuter × DiagOuter contraction
4. Full hyperedge execution with external tracking
5. Performance comparison: DiagOuter vs Dense for large diagonals

## References

- omeco: https://github.com/GiggleLiu/omeco
- mdarray-linalg: https://github.com/grothesque/mdarray-linalg
- ITensor DiagTensor: https://itensor.github.io/ITensors.jl/
