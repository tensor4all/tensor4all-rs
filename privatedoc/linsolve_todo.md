# Linsolve TODO

## Completed (WIP)

- [x] Step 1: Add kryst dependency to Cargo.toml
- [x] Step 2: Create linsolve module structure
- [x] Step 3: Implement kryst GMRES integration in LinsolveUpdater
- [x] Step 4: Add linsolve tests
- [x] Step 5: Add integration test with simple MPS example
- [x] Fix clippy warning in local_linop.rs

## Known Limitations (to be addressed in future iterations)

### 1. Panic Risks in matvec (kryst LinOp trait limitation)

Location: `local_linop.rs:154, 157`

```rust
// Line 154: could panic if RwLock is poisoned
let mut proj_op = self.projected_operator.write().unwrap();

// Line 157: could panic if apply fails
let hx = proj_op.apply(...).expect("Failed to apply projected operator");
```

**Issue**: kryst's `LinOp::matvec` signature is `fn matvec(&self, x: &[S], y: &mut [S])` and does not return `Result`. Cannot propagate errors.

**Workaround options**:
- Add validation before GMRES to catch errors early
- Use `catch_unwind` (not recommended)
- Request kryst API change to support fallible operations

### 2. Missing Validation

- No explicit check that operator and state have compatible tree topologies
- No check that site indices match between MPO and MPS
- These failures are caught as runtime errors, but earlier validation would improve UX

**Suggested implementation**:
```rust
fn validate_compatibility(operator: &TreeTN, state: &TreeTN) -> Result<()> {
    // Check same node names
    // Check matching site dimensions
    // Check compatible bond structure
}
```

### 3. Test Coverage

Current test (`test_linsolve_simple_two_site`) only verifies:
- Function runs without panic
- Returns expected number of sweeps

Missing tests:
- [ ] Verify solution correctness (compare with known solution)
- [ ] Test with non-identity operators
- [ ] Test convergence behavior
- [ ] Test with larger systems (3+ sites)
- [ ] Test truncation behavior

### 4. Performance

- `solve_local` clones the entire state on each call (`state.clone()` in `LocalLinOp::new`)
- Acceptable for small systems but inefficient for large states
- Consider passing references or using `Rc<RefCell<>>` pattern

### 5. Error Handling in update()

Location: `updater.rs:215-222`

Multiple `.unwrap()` calls that assume structure is valid:
```rust
let idx_u = subtree.node_index(node_u).unwrap();
let idx_v = subtree.node_index(node_v).unwrap();
// etc.
```

Should be converted to proper error handling with `?` operator.

## Future Enhancements

- [ ] Add preconditioner support
- [ ] Add convergence monitoring/callbacks
- [ ] Implement residual computation for convergence check
- [ ] Add support for complex numbers (Complex64)
- [ ] Parallelize environment computations
- [ ] Add tracing/logging for debugging
