# Tree Network Zip-up Contraction Implementation Plan

## Overview

Extend the current zip-up contraction implementation from linear chains (TensorTrain) to arbitrary Tree Network structures. The algorithm adopts a "leaf-to-root traversal with accumulated intermediate tensors (environment tensors)" approach.

## Current Implementation Status

### Existing Implementation
- **TensorTrain (linear chain)**: Implemented via `contract_zipup_itensors_like`
  - ITensors.jl-style cumulative R approach
  - Processes sequentially while maintaining environment tensor R
- **TreeTN (generic tree)**: Implemented via `contract_zipup_with`
  - Currently uses "factorize child tensor and absorb into parent" approach per edge
  - Does not maintain intermediate tensors

### Issues
- TreeTN zip-up processes each edge independently, which may be inefficient
- Differs from the required "maintain intermediate tensors while progressing" approach

## Algorithm Design

### Basic Approach
1. **Get edges via Post-order DFS**: Order from leaves towards root
2. **Maintain intermediate tensors (environment tensors)**: Manage accumulated intermediate tensors at each node
3. **Progressive contraction and factorization**: Contract 2 cores at leaves, contract intermediate tensors + 2 cores at internal nodes

### Detailed Algorithm

#### Step 1: Preprocessing
1. Topology verification: Verify both networks have matching structure via `same_topology()`
2. Internal index separation: Isolate bond indices of both networks via `sim_internal_inds()`
3. Root node determination: Use specified center node as root
4. Post-order DFS edge retrieval: Get leaf→root edge order via `edges_to_canonicalize_by_names(center)`

#### Step 2: Leaf Node Processing
For each leaf node (source):

```
Input:
  - A[source]: Source node tensor from network A
  - B[source]: Source node tensor from network B
  - destination: Parent node of source

Processing:
  1. C_temp = contract(A[source], B[source])  // Contract 2 cores
  2. (C[source], R) = factorize(C_temp, left_inds=site_indices(source) + bond_to_destination)
     // Left factor = site indices + bond to destination, right factor = environment tensor R
  3. Update edge with result bond index: replace_edge_bond(edge(source, destination), new_bond)
  4. Register intermediate tensor R at destination (do not contract yet)
  5. Store C[source] as result tensor for source node
```

**Notes:**
- `site_indices(source)` are physical indices of source node (excluding bond indices)
- `bond_to_destination` is the bond index from source to destination
- At this point, **do not contract** destination tensor with R
- Edge bond update (`replace_edge_bond`) is required, similar to existing TreeTN

#### Step 3: Internal Node Processing
For each internal node (source, but not a leaf):

```
Input:
  - R_accumulated: Accumulated intermediate tensors from source (already registered)
  - A[source]: Source node tensor from network A
  - B[source]: Source node tensor from network B
  - destination: Parent node of source

Processing:
  1. Prepare tensor list: [R_accumulated..., A[source], B[source]]
  2. C_temp = contract([R_accumulated..., A[source], B[source]], AllowedPairs::All)
     // Optimal contraction order is automatically selected
  3. (C[source], R_new) = factorize(C_temp, left_inds=site_indices(source) + bond_to_destination)
  4. Update edge with result bond index: replace_edge_bond(edge(source, destination), new_bond)
  5. Register new intermediate tensor R_new at destination (do not contract yet)
  6. Store C[source] as result tensor for source node
```

**Notes:**
- `T::contract(&[tensors...], AllowedPairs::All)` automatically executes contraction in optimal order
- Multiple intermediate tensors are **contracted together** (not pre-contracted) to leverage order optimization
- Edge bond update (`replace_edge_bond`) is required

#### Step 4: Root Node Processing
For root node (where all intermediate tensors aggregate):

```
Input:
  - R_list: List of intermediate tensors from all child nodes connected to root
  - A[root]: Root node tensor from network A
  - B[root]: Root node tensor from network B

Processing:
  1. Prepare tensor list: [R_list..., A[root], B[root]]
  2. C_temp = contract([R_list..., A[root], B[root]], AllowedPairs::All)
     // Optimal contraction order is automatically selected
  3. C[root] = C_temp  // No factorization needed at root (this is the final result)
  4. Store C[root] as result tensor for root node
```

**Notes:**
- Root node may have multiple children (e.g., star structure)
- All intermediate tensors must be contracted together
- Factorization is not needed at root (this is the final tensor)

#### Step 5: Set Canonical Center of Final Result
After all processing is complete, set the canonical center of the result TreeTN:

```
Processing:
  1. After result TreeTN is constructed, set specified center node as canonical center
  2. result.set_canonical_center(std::iter::once(center.clone()))?;
  3. This makes the result TreeTN have the specified center as its canonical center
```

**Notes:**
- Similar to existing `contract_zipup_with` implementation, set canonical center at the end
- `set_canonical_center` only succeeds if center node exists in result
- Setting canonical center makes the result TreeTN canonicalized
- Update edge `ortho_towards` to **point towards center** (consistent with existing implementation)

## Implementation Details

### Data Structures

```rust
// Structure to manage intermediate tensors
struct IntermediateTensors {
    // List of intermediate tensors registered at each node
    // Key: node name, Value: list of intermediate tensors aggregated at that node
    accumulated: HashMap<V, Vec<TensorDynLen>>,
}

// Or more simply
// Manage directly as HashMap<V, Vec<TensorDynLen>>
```

### Function Signature (Proposed)

```rust
impl<T, V> TreeTN<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    /// Zip-up contraction for Tree Network (accumulated intermediate tensor approach)
    pub fn contract_zipup_tree_accumulated(
        &self,
        other: &Self,
        center: &V,
        form: CanonicalForm,
        rtol: Option<f64>,
        max_rank: Option<usize>,
    ) -> Result<Self> {
        // Implementation
    }
}
```

### Implementation Steps

#### Phase 1: Basic Structure Implementation
1. **Preprocessing section**
   - Topology verification
   - Internal index separation
   - Post-order DFS edge retrieval

2. **Intermediate tensor management implementation**
   - Manage intermediate tensors aggregated at each node via `HashMap<V, Vec<TensorDynLen>>`
   - Functions to add/retrieve intermediate tensors

#### Phase 2: Leaf Node Processing Implementation
1. **Leaf node detection**
   - Nodes processed first in post-order DFS are leaves
   - Or detect via `graph.neighbors(node).count() == 1` (excluding root)

2. **Contraction + factorization at leaf nodes**
   - Contract 2 cores via `T::contract(&[a, b], AllowedPairs::All)`
   - Factorize via `factorize_with()`
   - Store left factor as result, register right factor (environment tensor) at parent node

#### Phase 3: Internal Node Processing Implementation
1. **Internal node detection**
   - Nodes that are neither leaves nor root

2. **Accumulated intermediate tensor retrieval**
   - Retrieve intermediate tensors registered at that node
   - If multiple exist, contract them together first to consolidate

3. **Contraction + factorization**
   - Contract via `T::contract(&[r_accumulated, a, b], AllowedPairs::All)`
   - Factorize via `factorize_with()`
   - Store left factor as result, register right factor at parent node

#### Phase 4: Root Node Processing Implementation
1. **Final processing at root node**
   - Retrieve all intermediate tensors
   - Contract via `T::contract(&[r_list..., a_root, b_root], AllowedPairs::All)`
   - Store result as-is (no factorization needed)

2. **Set canonical center of final result**
   - After result TreeTN is constructed, call `result.set_canonical_center(std::iter::once(center.clone()))?`
   - Verify center node exists in result before setting
   - This makes the result TreeTN have the specified center as canonical center

#### Phase 5: Edge Case Handling
1. **Single node network**
   - Case where root = leaf
   - Directly execute `contract(A[root], B[root])`

2. **Two node network**
   - Only root and leaf
   - Factorize at leaf, final contract at root

3. **Star structure (multiple leaves connected to root)**
   - Intermediate tensors aggregate from each leaf
   - Contract all at root

## Technical Considerations

### 1. Contraction Order Optimization
- `T::contract(&[tensors...], AllowedPairs::All)` automatically selects optimal order
- However, order for multiple intermediate tensors may need explicit control

### 2. Factorization Options
- Specify `rtol` and `max_rank` via `FactorizeOptions`
- **Support all three canonical forms**: `Unitary`/`LU`/`CI`
  - Use corresponding `FactorizeAlg` for each `Unitary`/`LU`/`CI`
  - Maintain approach where left factor holds "site indices + bond"

### 3. Index Management
- Site index extraction: Exclude bond indices from `external_indices()`
- Bond index identification: Common indices with adjacent nodes

### 4. Memory Efficiency
- Keep intermediate tensors only for minimal necessary duration
- Early release of processed node tensors

## Test Plan

### Unit Tests
1. **Single node**: Directly contract 2 tensors
2. **2-node chain**: Leaf→root processing
3. **3-node chain**: Leaf→internal→root processing
4. **Star structure**: 3 or more leaves connected to root
5. **Branching structure**: Internal node with multiple children

### Integration Tests
1. **Comparison with existing zip-up implementation**
   - Check if results match `contract_zipup_tree_accumulated` for linear chains
2. **Comparison with naive contraction**
   - Check if results numerically match `contract_naive` (excluding truncation error)

### Performance Tests
1. **Benchmarks on large networks**
   - Measure with varying node count, bond dimension, physical dimension
2. **Memory usage measurement**
   - Verify memory increase due to intermediate tensor retention

## Integration with Existing Code

### Function Placement
- Add to `crates/tensor4all-treetn/src/treetn/contraction.rs`
- Keep existing `contract_zipup_with` (for backward compatibility)
- New function name: `contract_zipup_accumulated` or `contract_zipup_tree_accumulated`

### Option Extensions
- Possibly add new method to `ContractionOptions`
- Or replace implementation of existing `ContractMethod::Zipup`

## Implementation Priority

1. **High priority**: Basic algorithm implementation (Phase 1-4)
2. **Medium priority**: Edge case handling (Phase 5)
3. **Low priority**: Performance optimization, memory efficiency

## References

- ITensors.jl MPO zip-up implementation
- Existing `contract_zipup_tree_accumulated` implementation (for Tree Network)
- `contract_zipup_with` implementation (current TreeTN)

## Notes

- Implement incrementally (test after each phase)
- Maintain compatibility with existing implementations
- Implement proper error handling (error checks at each step)
