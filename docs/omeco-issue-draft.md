# Issue: `compute_contraction_output` does not preserve hyperedge indices connected to other tensors

## Summary

When contracting two tensors that share a hyperedge index (an index appearing in 3+ tensors), `compute_contraction_output` in `greedy.rs` incorrectly contracts the shared index even if it's still connected to other tensors that haven't been contracted yet.

## Reproduction

```rust
use omeco::{EinCode, GreedyMethod, CodeOptimizer};
use std::collections::HashMap;

fn main() {
    // ixs = [[1, 2], [2], [2, 3]]  (tensors: A[i,j], B[j], C[j,k])
    // out = [1, 3]  (output: [i,k])
    // Index 2 (j) is a hyperedge appearing in 3 tensors, contracted

    let ixs = vec![
        vec![1usize, 2],  // tensor 0: indices i, j
        vec![2usize],      // tensor 1: index j only
        vec![2usize, 3],  // tensor 2: indices j, k
    ];
    let out = vec![1usize, 3];  // output: i, k

    let code = EinCode::new(ixs, out);

    let mut sizes = HashMap::new();
    sizes.insert(1usize, 2);  // i
    sizes.insert(2usize, 3);  // j
    sizes.insert(3usize, 2);  // k

    let optimizer = GreedyMethod::default();
    let nested = optimizer.optimize(&code, &sizes).unwrap();
    println!("{:#?}", nested);
}
```

## Actual Output

```
Node {
    args: [
        Leaf { tensor_index: 2 },        // C[j,k]
        Node {
            args: [
                Leaf { tensor_index: 0 },  // A[i,j]
                Leaf { tensor_index: 1 },  // B[j]
            ],
            eins: EinCode {
                ixs: [[1, 2], [2]],
                iy: [1],                   // ← BUG: j (index 2) is contracted here!
            },
        },
    ],
    eins: EinCode {
        ixs: [[2, 3], [1]],
        iy: [2, 3, 1],                     // Result has wrong shape [j,k,i] instead of [i,k]
    },
}
```

## Problem

The inner contraction `(A[i,j] * B[j]) -> [i]` contracts index `j` too early. But `j` is still needed for the contraction with `C[j,k]`!

**What happens:**
1. `A[i,j] * B[j]` contracts `j` because `j` is not in `final_output=[i,k]`
2. Result is `R1[i]` - j is gone!
3. `C[j,k] * R1[i]` produces outer product `[j,k,i]` instead of correct `[i,k]`

**Expected behavior:**
```
Node {
    args: [
        Leaf { tensor_index: 2 },        // C[j,k]
        Node {
            args: [
                Leaf { tensor_index: 0 },  // A[i,j]
                Leaf { tensor_index: 1 },  // B[j]
            ],
            eins: EinCode {
                ixs: [[1, 2], [2]],
                iy: [1, 2],                // ← j should be preserved here!
            },
        },
    ],
    eins: EinCode {
        ixs: [[2, 3], [1, 2]],
        iy: [1, 3],                        // Correct output [i,k]
    },
}
```

## Root Cause

The function `compute_contraction_output` (in `greedy.rs:352-375`) determines which indices to keep:

```rust
fn compute_contraction_output<L: Label>(left: &[L], right: &[L], final_output: &[L]) -> Vec<L> {
    // ...
    for l in left {
        if (!right_set.contains(l) || final_set.contains(l)) && !output.contains(l) {
            output.push(l.clone());
        }
    }
    // ...
}
```

The logic is:
- If `l` is only in `left` → keep
- If `l` is in both `left` and `right`, but also in `final_output` → keep
- If `l` is in both `left` and `right`, and NOT in `final_output` → **contract (remove)**

This is correct for standard pairwise contraction, but **incorrect when the index is a hyperedge connected to other tensors**.

## Note

`IncidenceList` already correctly tracks hyperedges via `is_external()`:

```rust
pub fn is_external(&self, e: &E, vi: &V, vj: &V) -> bool {
    if self.is_open(e) {
        return true;
    }
    if let Some(verts) = self.e2v.get(e) {
        verts.iter().any(|v| v != vi && v != vj)  // Detects hyperedge!
    } else {
        false
    }
}
```

And `ContractionDims::compute` uses this for **cost calculation** (distinguishing `d12` vs `d012`).

However, `compute_contraction_output` in `tree_to_nested_einsum` does NOT use this information.

## Suggested Fix

Modify `compute_contraction_output` to preserve indices connected to other (not-yet-contracted) tensors:

```rust
fn compute_contraction_output<L: Label, V>(
    left: &[L],
    right: &[L],
    final_output: &[L],
    il: &IncidenceList<V, L>,  // Add this parameter
    vi: &V,
    vj: &V,
) -> Vec<L> {
    // ...
    for l in left {
        let in_right = right_set.contains(l);
        let in_final = final_set.contains(l);
        let is_external = il.is_external(l, vi, vj);  // Check hyperedge

        // Keep if: not in right, OR in final output, OR connected to other tensors
        if (!in_right || in_final || is_external) && !output.contains(l) {
            output.push(l.clone());
        }
    }
    // ... similar for right
}
```

## Impact

This affects any einsum with a contracted hyperedge index (index in 3+ tensors, not in output). Common in:
- Tensor network contractions with multi-body interactions
- Quantum circuits with multi-qubit gates
- Factor graphs with higher-order factors

## Environment

- omeco version: fd3763b (git dependency)
- Rust version: stable
