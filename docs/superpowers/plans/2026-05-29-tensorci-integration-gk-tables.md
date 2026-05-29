# TensorCI Integration GK Tables Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand and document Rust's fixed embedded Gauss-Kronrod integration rules.

**Architecture:** Keep integration deterministic with embedded node/weight tables. Add verified tables incrementally, centralize supported-order reporting, and strengthen tests around supported and unsupported orders.

**Tech Stack:** Rust, `tensor4all-tensorci::integration`, release-mode cargo tests.

---

## File Structure

- Modify `crates/tensor4all-tensorci/src/integration.rs`: add tables and supported-order helper.
- Modify `docs/api/tensor4all_tensorci.md`: regenerate via `api-dump`.

### Task 1: Add Supported Order Tests

**Files:**
- Modify: `crates/tensor4all-tensorci/src/integration.rs`

- [ ] **Step 1: Add tests in the existing `mod tests`**

```rust
#[test]
fn test_gk_supported_orders_are_reported() {
    assert!(gk_nodes_weights(15).is_ok());
    assert!(gk_nodes_weights(31).is_ok());
    assert!(gk_nodes_weights(41).is_ok());
    assert!(gk_nodes_weights(51).is_ok());
    assert!(gk_nodes_weights(61).is_ok());
}

#[test]
fn test_gk_order_error_lists_supported_orders() {
    let err = gk_nodes_weights(21).unwrap_err();
    let message = err.to_string();
    assert!(message.contains("21"));
    assert!(message.contains("15, 31, 41, 51, 61"));
}
```

- [ ] **Step 2: Run the tests and confirm they fail**

```bash
cargo test --release -p tensor4all-tensorci test_gk_supported_orders_are_reported test_gk_order_error_lists_supported_orders
```

Expected: fail because `41`, `51`, and `61` are unsupported and the error list is still `15, 31`.

### Task 2: Add Fixed Rule Tables

**Files:**
- Modify: `crates/tensor4all-tensorci/src/integration.rs`

- [ ] **Step 1: Add `GK41_NODES`, `GK41_WEIGHTS`, `GK51_NODES`, `GK51_WEIGHTS`, `GK61_NODES`, and `GK61_WEIGHTS`**

Copy the Kronrod abscissae and weights from the QUADPACK rules `dqk41`,
`dqk51`, and `dqk61`. Store the expanded nodes on `[-1, 1]` in ascending order
to match the existing `GK15` and `GK31` arrays. Store the full symmetric weight
arrays in the same order as the node arrays.

- [ ] **Step 2: Add a supported-order constant**

```rust
const SUPPORTED_GK_ORDERS: &[usize] = &[15, 31, 41, 51, 61];

fn supported_gk_orders_message() -> String {
    SUPPORTED_GK_ORDERS
        .iter()
        .map(|order| order.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}
```

- [ ] **Step 3: Extend `gk_nodes_weights`**

```rust
fn gk_nodes_weights(gk_order: usize) -> Result<(&'static [f64], &'static [f64])> {
    match gk_order {
        15 => Ok((&GK15_NODES, &GK15_WEIGHTS)),
        31 => Ok((&GK31_NODES, &GK31_WEIGHTS)),
        41 => Ok((&GK41_NODES, &GK41_WEIGHTS)),
        51 => Ok((&GK51_NODES, &GK51_WEIGHTS)),
        61 => Ok((&GK61_NODES, &GK61_WEIGHTS)),
        _ => Err(TCIError::InvalidOperation {
            message: format!(
                "GK order {} not supported. Supported orders: {}.",
                gk_order,
                supported_gk_orders_message()
            ),
        }),
    }
}
```

- [ ] **Step 4: Run supported-order tests**

```bash
cargo test --release -p tensor4all-tensorci test_gk_supported_orders_are_reported test_gk_order_error_lists_supported_orders
```

Expected: PASS.

- [ ] **Step 5: Add table sanity tests**

```rust
#[test]
fn test_gk_weights_sum_to_two() {
    for order in [15, 31, 41, 51, 61] {
        let (_nodes, weights) = gk_nodes_weights(order).unwrap();
        let total: f64 = weights.iter().sum();
        assert!(
            (total - 2.0).abs() < 1e-14,
            "GK{order} weights sum to {total}"
        );
    }
}

#[test]
fn test_gk_nodes_are_symmetric_and_sorted() {
    for order in [15, 31, 41, 51, 61] {
        let (nodes, weights) = gk_nodes_weights(order).unwrap();
        assert_eq!(nodes.len(), order);
        assert_eq!(weights.len(), order);
        for pair in nodes.windows(2) {
            assert!(pair[0] < pair[1], "GK{order} nodes are not sorted");
        }
        for i in 0..nodes.len() {
            let j = nodes.len() - 1 - i;
            assert!((nodes[i] + nodes[j]).abs() < 1e-14);
            assert!((weights[i] - weights[j]).abs() < 1e-14);
        }
    }
}
```

### Task 3: Add Numerical Integration Coverage

**Files:**
- Modify: `crates/tensor4all-tensorci/src/integration.rs`

- [ ] **Step 1: Add a GK61 polynomial test**

```rust
#[test]
fn test_integrate_polynomial_with_gk61() {
    let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
    let options = TCI2Options {
        tolerance: 1e-10,
        max_iter: 20,
        seed: Some(42),
        ..TCI2Options::default()
    };
    let result: f64 = integrate(&f, &[0.0, 0.0], &[1.0, 1.0], 61, options).unwrap();
    assert!((result - 2.0 / 3.0).abs() < 1e-8);
}
```

- [ ] **Step 2: Add one-dimensional error behavior test**

```rust
#[test]
fn test_integrate_one_dimensional_reports_tci_requirement() {
    let f = |x: &[f64]| x[0] * x[0];
    let err = integrate::<f64, _>(&f, &[0.0], &[1.0], 15, TCI2Options::default()).unwrap_err();
    assert!(err.to_string().contains("local_dims should have at least 2 elements"));
}
```

- [ ] **Step 3: Run integration tests**

```bash
cargo test --release -p tensor4all-tensorci integration
```

Expected: PASS.

### Task 4: Update Docs And Commit

**Files:**
- Modify: `crates/tensor4all-tensorci/src/integration.rs`
- Modify: `docs/api/tensor4all_tensorci.md`

- [ ] **Step 1: Update module and function rustdoc**

State that Rust supports embedded fixed rules `15, 31, 41, 51, 61`, while Julia obtains rules from `QuadGK.kronrod`.

- [ ] **Step 2: Format and test**

```bash
cargo fmt --all
cargo test --release -p tensor4all-tensorci integration
cargo test --doc --release -p tensor4all-tensorci
```

- [ ] **Step 3: Regenerate API docs**

```bash
cargo run -p api-dump --release -- . -o docs/api
```

- [ ] **Step 4: Commit**

```bash
git add crates/tensor4all-tensorci/src/integration.rs docs/api/tensor4all_tensorci.md
git commit -m "feat(tensorci): add fixed Gauss-Kronrod rules"
```
