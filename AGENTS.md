# Agent Guidelines for tensor4all-rs

Before starting work, read the repository root `README.md` as well as this `AGENTS.md`.

## Table of Contents

1. [Development Stage](#development-stage)
2. [General Guidelines](#general-guidelines)
3. [Code Style](#code-style)
4. [Error Handling](#error-handling)
5. [Testing](#testing)
6. [API Design](#api-design)
7. [C API Design](#c-api-design)
8. [Language Bindings](#language-bindings)
9. [Dependencies](#dependencies)
10. [Git Workflow](#git-workflow)

---

## Development Stage

**This project is in early development stage.** Backward compatibility is not required at this time:

- Do not keep legacy/deprecated code for backward compatibility
- Remove deprecated methods immediately instead of marking them with `#[deprecated]`
- Prefer clean, simple APIs over maintaining multiple ways to do the same thing

---

## General Guidelines

- Use the same language as in past conversations with the user (if it has been Japanese, use Japanese)
- All source code and documentation must be in English
- Each crate in `crates/` is an independent Rust package with its own `Cargo.toml`, `src/`, and `tests/` directories
- When working on a crate, understand the package structure and be aware of dependencies between crates

### Before Starting Work: API Reference

**Always dump the latest API documentation before starting implementation work:**

```bash
cargo run -p api-dump --release -- . -o docs/api
```

This generates Markdown files in `docs/api/` with function signatures and docstrings for each crate.

**Required workflow:**
1. Run the API dump command above
2. Read the API docs for crates related to your task (e.g., `docs/api/tensor4all_simpletensortrain.md`)
3. Understand existing functions before implementing new ones
4. During implementation, check the API docs to avoid duplicating existing functionality

**When to reference API docs:**
- Before adding a new function: check if similar functionality exists
- When implementing a feature: look for helper functions you can reuse
- When unsure about existing API: read the relevant crate's API doc

---

## Code Style

- Follow standard Rust style guidelines
- Use `rustfmt` for formatting: `cargo fmt`
- Use `clippy` for linting: `cargo clippy`
- Prefer explicit error handling over `unwrap()` or `expect()` in library code

---

## Error Handling

### Prefer `anyhow` for Generic Error Handling

For generic error handling and error context propagation, **prefer using `anyhow`**:

```rust
use anyhow::{Context, Result};

fn some_function() -> Result<()> {
    let value = some_operation()
        .context("Failed to perform operation")?;
    Ok(())
}
```

**Converting errors to custom error types:**
```rust
use anyhow;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum MyError {
    #[error("Operation failed: {0}")]
    OperationError(#[from] anyhow::Error),
}

let result = some_operation()
    .map_err(|e| anyhow::anyhow!("Context: {:?}", e))
    .map_err(MyError::OperationError)?;
```

### When to Use `thiserror` vs `anyhow`

| Use Case | Library |
|----------|---------|
| Public API error types that callers need to match | `thiserror` |
| Error types that need serialization | `thiserror` |
| Internal error handling and propagation | `anyhow` |
| Adding context to errors | `anyhow` |

---

## Testing

### Running Tests

```bash
# Full test suite for a crate
cargo test

# Specific test
cargo test --test test_name

# With output
cargo test -- --nocapture

# All crates in workspace
cargo test --workspace
```

### Test Organization

- **Private functions**: Add tests at the end of the source file using `#[cfg(test)]` modules
- **Integration tests**: Place in `tests/` directory with naming convention:
  - Same name as source: `src/index.rs` → `tests/index.rs`
  - Feature-specific: `src/index.rs` → `tests/index_ops.rs`

**Example for private functions:**
```rust
// src/index.rs
fn internal_helper() -> usize { 42 }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_internal_helper() {
        assert_eq!(internal_helper(), 42);
    }
}
```

### Generic Tests for Real and Complex Types

**Always use generic test functions** instead of duplicating tests for `f64` and `Complex64`:

```rust
// ✅ GOOD: Generic test helper
fn test_operation_generic<T: Scalar + PartialEq + std::fmt::Debug>()
where
    T: From<f64>,
{
    let x = T::from(1.0);
    let y = T::from(2.0);
    assert_eq!(operation(x, y), T::from(3.0));
}

#[test]
fn test_operation_f64() { test_operation_generic::<f64>(); }

#[test]
fn test_operation_c64() { test_operation_generic::<Complex64>(); }
```

---

## API Design

- **Do not make functions `pub` unnecessarily**: Only expose functions that are part of the public API
- Functions should be `pub` only when used by other crates or documented as public interface

---

## C API Design

When working on the C API (`tensor4all-capi`) or language bindings, refer to the [C API Design Guidelines](docs/CAPI_DESIGN.md) for:

- Opaque type patterns and lifecycle management
- Error handling with status codes
- Panic safety requirements
- Row-major data layout conventions
- Memory contiguity requirements
- Column-major to row-major conversion (Julia bindings)

**Important**: All C API functions must follow the patterns documented in `docs/CAPI_DESIGN.md`.

### Ownership Model Summary

| Pattern | Description |
|---------|-------------|
| Creation | `t4a_<TYPE>_new()` - caller owns the object |
| Release | `t4a_<TYPE>_release()` - explicit release required |
| Borrowing | Functions operate on borrowed references |
| In-place | Use `_inplace` suffix for in-place operations |

### Implementation Note

- **tensor4all-rs**: Uses `Box<T>` - `clone()` creates full copy, `release()` immediately frees
- **sparse-ir-rs**: Uses `Box<Arc<T>>` - `clone()` increments refcount (cheap)
- Consider `Arc<T>` if profiling shows `clone()` is a bottleneck

---

## Language Bindings

### Directory Layout

```
tensor4all-rs/
├── julia/
│   └── Tensor4all.jl/
│       ├── src/
│       │   ├── Tensor4all.jl    # Main module
│       │   └── ITensorLike.jl   # Submodule
│       └── test/
├── python/
│   └── tensor4all/
│       ├── __init__.py
│       └── tests/
```

### Namespace Usage

**Julia:**
```julia
using Tensor4all           # Core types (Index, Tensor)
using Tensor4all.ITensorLike  # TensorTrain functionality
```

**Python:**
```python
import tensor4all
from tensor4all import tensortrain
```

### Guidelines

- **One submodule per Rust crate**: Each major Rust crate gets its own submodule
- **Consistent naming**: Same naming convention across languages (adjusted for language style)
- **Re-export common types**: Main module should re-export commonly used types
- **Separate test files**: Each submodule should have its own test file
- **Do not use `bindings/` directory**: Place bindings directly under `julia/` and `python/`

### Truncation Tolerance (cutoff vs rtol)

Language bindings must support **both** parameters:

| Library | Parameter | Semantics |
|---------|-----------|-----------|
| tensor4all-rs | `rtol` | Relative Frobenius error: `‖A - A_approx‖_F / ‖A‖_F ≤ rtol` |
| ITensors.jl | `cutoff` | Squared relative error: `Σ_{discarded} σ²_i / Σ_i σ²_i ≤ cutoff` |

**Conversion:** `cutoff = rtol²` → `rtol = √cutoff`

**Requirements:**
1. Accept both `cutoff` and `rtol`
2. Error if both specified (with helpful message)
3. Read default `rtol` from Rust via C API
4. Internally convert `cutoff` to `rtol = √cutoff`

---

## Dependencies

When adding dependencies, consider:
- Is it already used in another crate?
- Is it a workspace dependency? (check root `Cargo.toml`)
- Does it fit the crate's purpose and scope?

### Array and Matrix Libraries

**Prefer `mdarray` over custom implementations:**

| Type | Use Case |
|------|----------|
| `DTensor<T, 2>` | 2D matrices (e.g., `DTensor<f64, 2>`) |
| `DTensor<T, N>` | Fixed-rank tensors |
| `Tensor<T>` | Dynamic rank tensors |

**Important notes:**
- `mdarray` uses **row-major (C-order)** layout only
- For column-major (Julia), handle conversion at binding level

**Use `mdarray-linalg` for linear algebra:**
- SVD, QR, LU decomposition
- Eigen decomposition
- Solve and inverse
- Supports FAER and LAPACK backends

**⚠️ SVD singular values:** Stored in `s[[0, i]]`, NOT `s[[i, i]]` (LAPACK convention).

---

## Git Workflow

**Important**: Do NOT execute `git push` or `gh pr create` without explicit user instruction. Always wait for user approval before pushing commits or creating PRs.

### Minor Changes vs Large Features

| Change Type | Workflow |
|-------------|----------|
| Minor fixes (README, docs, delete unused files) | Branch + PR with auto-merge |
| Large features, refactoring | Worktree + PR with auto-merge |

### Minor Changes (Branch-based)

For small fixes that don't require isolated development:

```bash
# Create branch
git checkout -b fix-readme
# Make changes, commit
git add -A && git commit -m "Fix README typo"
git push -u origin fix-readme

# Create PR with auto-merge
gh pr create --base main --title "Fix README typo" --body "Description"
gh pr merge --auto --squash --delete-branch
```

### Large Features (Worktree-based)

For substantial development that benefits from isolation:

```bash
# Create worktree for new feature
git worktree add ../tensor4all-rs-feature-name -b feature-name
cd ../tensor4all-rs-feature-name
# Work on feature...
```

Always start from the latest main branch.

### Creating Pull Requests

```bash
# Create PR
gh pr create --base main --title "Feature: name" --body "Description"

# Enable auto-merge (recommended)
gh pr merge --auto --squash --delete-branch
```

### Monitoring CI

```bash
# Check PR status
gh pr checks <PR_NUMBER>

# View detailed status
gh pr view <PR_NUMBER> --json state,statusCheckRollup

# View failed logs
gh run view <RUN_ID> --log-failed
```

**If CI fails:** Fix the issue, push a new commit, and continue monitoring.

### Branch Protection Rules

- **Never push directly to main branch**: All changes must go through PRs
- **Never force push to main branch**: Use feature branches for history rewriting

### Before Creating a PR

**Required**: Before creating a PR, always:
1. Read `README.md` and verify the information is still accurate
2. Update any outdated sections, especially:
   - **Project Structure**: Ensure crate list matches actual `crates/` directory
   - **Sample Code**: Verify examples still compile and reflect current API
3. Check if new features need to be documented

Do not create a PR with outdated documentation.
