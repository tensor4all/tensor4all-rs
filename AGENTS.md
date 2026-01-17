# Agent Guidelines for tensor4all-rs

Read `README.md` before starting work.

## Development Stage

**Early development** - no backward compatibility required. Remove deprecated code immediately.

## General Guidelines

- Use same language as past conversations (Japanese if previous was Japanese)
- Source code and docs in English
- Each crate in `crates/` is independent with own `Cargo.toml`, `src/`, `tests/`

### API Reference (Check First)

```bash
cargo run -p api-dump --release -- . -o docs/api
```

Read `docs/api/*.md` before source files. Only read source when API doc is insufficient.

## Context-Efficient Exploration

- Use Task tool with `subagent_type=Explore` for open-ended exploration
- Use Grep for structure: `pub fn`, `impl.*for`, `^pub (struct|enum|type)`
- Read specific lines with `offset`/`limit` parameters
- Prefer API docs over full source files

## Code Style

`cargo fmt` for formatting, `cargo clippy` for linting. Avoid `unwrap()`/`expect()` in library code.

**Always run `cargo fmt --all` before committing changes.**

## Error Handling

- `anyhow` for internal error handling and context
- `thiserror` for public API error types

## Testing

```bash
cargo test                    # Full suite
cargo test --test test_name   # Specific test
cargo test --workspace        # All crates
```

- Private functions: `#[cfg(test)]` module in source file
- Integration tests: `tests/` directory
- **Test tolerance changes**: When relaxing test tolerances (unit tests, codecov targets, etc.), always seek explicit user approval before making changes.

## API Design

Only make functions `pub` when truly public API.

### Layering and Maintainability

**Respect crate boundaries and abstraction layers.**

- **Never access low-level APIs or internal data structures from downstream crates.** Use high-level public methods instead of directly manipulating internal representations.
- **Use high-level APIs.** If downstream code needs low-level access, create appropriate high-level APIs rather than exposing internal details.
- **Examples:**
  - Instead of `match scalar { AnyScalar::F64(x) => ... }`, use `scalar.real()`, `scalar.is_complex()`, `scalar.is_zero()`
  - Instead of `AnyScalar::F64(1.0)`, use `AnyScalar::new_real(1.0)`
  - Instead of `AnyScalar::C64(z)`, use `AnyScalar::new_complex(re, im)`

**This applies to both library code and test code.** Tests should also use public APIs to maintain consistency and reduce maintenance burden when internal representations change.

### Code Deduplication

- **Avoid duplicate test code.** Use macros, functions, or generic functions to share test logic.
- **Example pattern for testing f64/Complex64:**

```rust
fn test_op_generic<T: Scalar + From<f64>>() { /* test */ }

#[test]
fn test_op_f64() { test_op_generic::<f64>(); }
#[test]
fn test_op_c64() { test_op_generic::<Complex64>(); }
```

## C API & Language Bindings

See `docs/CAPI_DESIGN.md` for C API patterns. Bindings: `julia/Tensor4all.jl/`, `python/tensor4all/`.

Truncation tolerance: support both `cutoff` (ITensors) and `rtol` (tensor4all-rs). Conversion: `rtol = √cutoff`.

## Dependencies

- Prefer `mdarray` for arrays (row-major only), `mdarray-linalg` for linear algebra
- SVD singular values: `s[[0, i]]` not `s[[i, i]]`

## Git Workflow

**Never push/create PR without user approval.**

### Pre-PR Checks

Before creating a PR, always run lint checks locally:

```bash
cargo fmt --all          # Format all code
cargo clippy --workspace # Check for common issues
cargo test --workspace   # Run all tests
```

| Change Type | Workflow |
|-------------|----------|
| Minor fixes | Branch + PR with auto-merge |
| Large features | Worktree + PR with auto-merge |

```bash
# Minor: branch workflow
git checkout -b fix-name && git add -A && git commit -m "msg"
cargo fmt --all && cargo clippy --workspace  # Lint before push
git push -u origin fix-name
gh pr create --base main --title "Title" --body "Desc"
gh pr merge --auto --squash --delete-branch

# Large: worktree workflow
git worktree add ../tensor4all-rs-feature -b feature

# Check PR before update
gh pr view <NUM> --json state  # Never push to merged PR

# Monitor CI
gh pr checks <NUM>
gh run view <RUN_ID> --log-failed
```

**Before creating PR**: Verify README.md is accurate (project structure, examples).
