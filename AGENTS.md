# Agent Guidelines for tensor4all-rs

Before starting work, read the repository root `README.md` as well as this `AGENTS.md`.

## General Guidelines

- Use the same language as in past conversations with the user (if it has been Japanese, use Japanese)
- All source code and documentation must be in English
- Each crate in `crates/` is an independent Rust package with its own `Cargo.toml`, `src/`, and `tests/` directories. Understand the package structure before making changes
- When working on a crate, navigate into its directory and work as if it were a standalone package. Be aware of dependencies between crates

## Error Handling

### Prefer `anyhow` for Generic Error Handling

For generic error handling and error context propagation, **prefer using `anyhow`** over manual error construction. This provides better error messages and error chaining.

**Recommended pattern:**
```rust
use anyhow::{Context, Result};

fn some_function() -> Result<()> {
    let value = some_operation()
        .context("Failed to perform operation")?;
    Ok(())
}
```

**When converting errors to custom error types:**
```rust
use anyhow;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum MyError {
    #[error("Operation failed: {0}")]
    OperationError(#[from] anyhow::Error),
}

// Usage:
let result = some_operation()
    .map_err(|e| anyhow::anyhow!("Context: {:?}", e))
    .map_err(MyError::OperationError)?;
```

**Benefits:**
- Better error messages with context
- Automatic error chaining
- Consistent error handling across the codebase
- Easy to add context at any point in the call stack

### When to Use `thiserror`

Use `thiserror` for:
- Public API error types that need to be matched by callers
- Error types that need to be serialized or sent across boundaries
- Error types that are part of the public interface

Use `anyhow` for:
- Internal error handling and propagation
- Adding context to errors
- Generic error handling where the specific error type doesn't matter to callers

## Running Tests

**Full test suite:**
```bash
cd /path/to/crate
cargo test
```

**Running specific tests:**
```bash
cargo test --test test_name
```

**Running with output:**
```bash
cargo test -- --nocapture
```

**Running tests for all crates:**
```bash
cargo test --workspace
```

## Code Style

- Follow standard Rust style guidelines
- Use `rustfmt` for formatting (run `cargo fmt`)
- Use `clippy` for linting (run `cargo clippy`)
- Prefer explicit error handling over `unwrap()` or `expect()` in library code

## API Design

- **Do not make functions `pub` unnecessarily**: Only expose functions that are part of the public API. Internal helper functions should remain private.
- Functions should be `pub` only when they need to be used by other crates or are part of the documented public interface.

## C API Design

When working on the C API (`tensor4all-capi`) or language bindings (Julia, Python), refer to the [C API Design Guidelines](CAPI_DESIGN.md) for:

- Opaque type patterns and lifecycle management
- **Ownership model**: "Owned Objects with Explicit Lifecycle" - objects created with `new` must be explicitly released
- Error handling with status codes
- Panic safety requirements
- Row-major data layout conventions
- Memory contiguity requirements
- Column-major to row-major conversion (Julia bindings)
- Function export conventions

**Important**: All C API functions must follow the patterns documented in `CAPI_DESIGN.md` to ensure consistency and safety across language bindings.

**Ownership Model Summary**: 
- Objects created via `t4a_<TYPE>_new()` are owned by the caller
- All objects must be explicitly released using `t4a_<TYPE>_release()`
- Functions never take ownership; they operate on borrowed references
- Both immutable operations (return new objects) and in-place operations (modify existing objects) are available
- Use `_inplace` suffix for in-place operations

**Current Implementation vs sparse-ir-rs:**
- **tensor4all-rs (current)**: Objects use `Box<T>` - `clone()` creates full copy, `release()` immediately frees memory
- **sparse-ir-rs**: Objects use `Box<Arc<T>>` - `clone()` increments reference count (cheap), `release()` only frees when last reference is dropped
- **Note**: tensor4all-rs currently uses the simpler `Box<T>` model. Consider `Arc<T>` if profiling shows `clone()` is a bottleneck and multiple references are common.

## Testing Guidelines

- **Private function testing**: For private functions and internal helpers, add tests at the end of the source file using `#[cfg(test)]` modules.
- **Integration test file naming**: When creating test files in the `tests/` directory, follow the naming convention:
  - Use the same name as the source file: `src/index.rs` → `tests/index.rs`
  - For testing specific features: `src/index.rs` → `tests/index_ops.rs` (where `_ops` indicates operations)
  - The test file name should clearly indicate what is being tested

**Example:**
```rust
// src/index.rs
fn internal_helper() -> usize {
    42
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_internal_helper() {
        assert_eq!(internal_helper(), 42);
    }
}
```

For public API testing, use `tests/index.rs` or `tests/index_ops.rs` as appropriate.

- **Prefer generic tests for real and complex types**: When writing tests for operations that support both real (`f64`) and complex (`Complex64`) types, **always use generic test functions** instead of duplicating tests. This reduces code duplication, ensures consistency, and makes it easier to add new scalar types in the future.

**Example:**
```rust
// ❌ BAD: Duplicating tests for f64 and Complex64
#[test]
fn test_operation_f64() {
    let x: f64 = 1.0;
    let y: f64 = 2.0;
    assert_eq!(operation(x, y), 3.0);
}

#[test]
fn test_operation_c64() {
    let x = Complex64::new(1.0, 0.0);
    let y = Complex64::new(2.0, 0.0);
    assert_eq!(operation(x, y), Complex64::new(3.0, 0.0));
}

// ✅ GOOD: Use generic test helper
fn test_operation_generic<T: Scalar + PartialEq + std::fmt::Debug>()
where
    T: From<f64>,
{
    let x = T::from(1.0);
    let y = T::from(2.0);
    assert_eq!(operation(x, y), T::from(3.0));
}

#[test]
fn test_operation_f64() {
    test_operation_generic::<f64>();
}

#[test]
fn test_operation_c64() {
    test_operation_generic::<Complex64>();
}
```

**Benefits of generic tests:**
- Reduces code duplication and maintenance burden
- Ensures consistency across real and complex type tests
- Makes it easier to add new scalar types (e.g., `f32`, `Complex32`) in the future
- Single source of truth for test logic

## Dependencies

- When adding dependencies, consider:
  - Is it already used in another crate? (check `Cargo.toml` files)
  - Is it a workspace dependency? (check root `Cargo.toml`)
  - Does it fit the crate's purpose and scope?

### Array and Matrix Libraries

- **Prefer `mdarray` over custom array/matrix implementations**: When working with arrays and matrices in Rust, **prefer using `mdarray`** instead of creating custom array or matrix classes. This ensures consistency across the codebase and leverages well-tested, maintained libraries.
- **Use `DTensor<T, 2>` for matrices**: For 2D matrices, use `mdarray::DTensor<T, 2>` where `T` is the element type (e.g., `f64`, `Complex64`). This type provides type-safe matrix operations and integrates well with `mdarray-linalg`.
  - Example: `DTensor<f64, 2>` for real-valued matrices, `DTensor<Complex64, 2>` for complex-valued matrices
  - For higher-dimensional tensors with fixed rank, use `DTensor<T, N>` where `N` is the rank (e.g., `DTensor<T, 3>` for 3D tensors)
  - **For dynamic rank**: Use `mdarray::Tensor<T>` (or `Tensor<T, DynRank>`) when the rank is not known at compile time. This is the default when no rank is specified: `Tensor<T>` is equivalent to `Tensor<T, DynRank>`.
- **Memory layout**: `mdarray` always uses **row-major (C-order)** memory layout. There is no built-in support for column-major (Fortran-order) layout. If column-major data is needed (e.g., for Julia bindings), use `permute` or dimension permutation to convert the data, or handle the conversion at the language binding level.

- **⚠️ mdarray-linalg SVD singular values storage**: When using `mdarray-linalg`'s SVD decomposition, **singular values are stored in `s[[0, i]]`, NOT `s[[i, i]]`**. This is a LAPACK-style convention where the first row of the diagonal matrix is used as the singular value buffer. Both FAER and LAPACK backends follow this convention.
  - ✅ **Correct**: `let sv = s[[0, i]];`
  - ❌ **Wrong**: `let sv = s[[i, i]];`
  - Always use `s[[0, i]]` when extracting singular values from `SVDDecomp<T>`.
  - See `tensor4all-rs/crates/tensor4all/core-linalg/src/svd.rs` for the correct implementation pattern.
- **Use `mdarray-linalg` for linear algebra operations**: The `mdarray-linalg` crate provides linear algebra operations for `mdarray` types, including:
  - **SVD** (Singular Value Decomposition)
  - **QR decomposition**
  - **LU decomposition**
  - **Eigen decomposition**
  - **Solve and inverse**
  - **Cholesky decomposition**
  - **Matrix multiplication** and other matrix/vector operations
- `mdarray-linalg` supports multiple backends (FAER, LAPACK) via feature flags, allowing backend selection based on performance and portability requirements.
- Avoid implementing custom array or matrix types unless there is a specific, well-justified reason that `mdarray` cannot meet the requirements.
- Avoid implementing custom linear algebra algorithms unless there is a specific, well-justified reason that `mdarray-linalg` cannot meet the requirements.

## Git Workflow

- **Use `git worktree` for new features**: When creating a new feature, use a git worktree based on the latest main branch. This allows you to work on the feature in isolation while keeping the main working directory clean:
  ```bash
  # Create a worktree for a new feature branch
  git worktree add ../tensor4all-rs-feature-name -b feature-name
  cd ../tensor4all-rs-feature-name
  # Work on the feature here
  ```
  Always start from the latest main branch to ensure your feature is based on the most recent codebase.

- **Before creating a PR**: Read the README.md file and check if any information has become outdated due to the implementation changes. If outdated information is found, propose updates to keep the documentation accurate.

- **Create PR and enable auto-merge**: After pushing your branch, create a PR using GitHub CLI and enable auto-merge. This allows the PR to be automatically merged once all CI checks pass:
  ```bash
  # Create PR
  gh pr create --base main --title "Feature: your feature name" --body "Description of changes"
  
  # Enable auto-merge (recommended)
  gh pr merge --auto --squash --delete-branch
  ```
  **Benefits of auto-merge**:
  - PR is automatically merged when all CI checks pass
  - No need to manually monitor CI status
  - Branch is automatically deleted after merge
  - Reduces manual intervention and speeds up development cycle

- **Never push directly to main branch**: All changes must be made through pull requests. Create a branch, commit changes, push the branch, and create a PR. Wait for CI workflows to pass before merging.

- **Never use force push to main branch**: Force pushing (including `--force-with-lease`) to main is prohibited. If you need to rewrite history, do it on a feature branch and create a PR.

## Language Bindings Structure

### Directory Layout

Language bindings are placed under the root directory with a unified structure:

```
tensor4all-rs/
├── julia/
│   └── Tensor4all.jl/          # Julia package
│       ├── src/
│       │   ├── Tensor4all.jl   # Main module
│       │   └── TensorTrain/    # Submodule for TensorTrain
│       │       └── TensorTrain.jl
│       └── test/
│           ├── runtests.jl     # Test runner
│           └── test_tensortrain.jl
├── python/
│   └── tensor4all/             # Python package
│       ├── __init__.py         # Main module
│       ├── tensortrain/        # Subpackage for TensorTrain
│       │   └── __init__.py
│       └── tests/
│           ├── test_tensortrain.py
│           └── conftest.py
```

### Namespace Separation

Bindings should use namespace separation by feature/crate:

**Julia:**
```julia
using Tensor4all
using Tensor4all.TensorTrain

# Or access via qualified names
tt = Tensor4all.TensorTrain.zeros(Float64, [2, 3, 2])
```

**Python:**
```python
import tensor4all
from tensor4all import tensortrain

# Or access via qualified names
tt = tensor4all.tensortrain.TensorTrainF64.zeros([2, 3, 2])
```

### Guidelines

- **One submodule per Rust crate**: Each major Rust crate (e.g., `tensor4all-tensortrain`) gets its own submodule in both Julia and Python bindings
- **Consistent naming**: Use the same naming convention across languages (e.g., `TensorTrain` in Julia, `tensortrain` in Python following language conventions)
- **Re-export common types**: The main module should re-export commonly used types for convenience
- **Separate test files**: Each submodule should have its own test file (e.g., `test_tensortrain.jl`, `test_tensortrain.py`)
- **Do not use `bindings/` directory**: Place bindings directly under `julia/` and `python/` at the repository root

