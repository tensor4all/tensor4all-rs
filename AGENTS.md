# Agent Guidelines for tensor4all-rs

Read `README.md` before starting work.

## Development Stage

**Early development** - no backward compatibility required. Remove deprecated code immediately.

## General Guidelines

- Use same language as past conversations (Japanese if previous was Japanese)
- Source code and docs in English
- Each crate in `crates/` is independent with own `Cargo.toml`, `src/`, `tests/`
- **Bug fixing**: When a bug is discovered, always check related files for similar bugs and propose to the user to inspect them

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

## Documentation Requirements

### Rustdoc Standards

Every public type, trait, and function **must** have doc comments with the following:

**Types (struct/enum/trait):**
- Summary: what it represents, when to use it (1-2 sentences)
- Related types: relationship to similar types (e.g., "`TensorTrain` is the simple chain version; `TreeTN` is the general tree version")
- `# Examples` section with runnable code and assertions

**Functions/methods:**
- Summary: what it does (1 sentence)
- Arguments: meaning, constraints, typical values for each parameter (especially for `Options` types)
- Returns: what is returned, how to use it
- `# Panics` or `# Errors`: under what conditions it fails
- `# Examples` section with runnable code and assertions

**Options/Config types (critical for usability):**
- Each field: meaning, recommended values, and default behavior
- Field relationships and trade-offs (e.g., `rtol` vs `max_bond_dim`)
- "When in doubt" defaults

### Code Example Rules

- All doc examples **must** be runnable (`ignore` and `no_run` attributes are **prohibited**)
- All doc examples **must** include assertions verifying correctness (not just compilation/execution)
  - Use `assert!`, `assert_eq!`, `approx::assert_abs_diff_eq!`, etc.
- mdBook guide code blocks follow the same rules: runnable with assertions
- mdBook code blocks use hidden lines (`# ` prefix) for `use` statements and `fn main()` wrappers

### CI Verification

- `cargo test --doc --release --workspace` must pass (rustdoc examples)
- `mdbook test docs/book` must pass (mdBook guide examples)

## Error Handling

- `anyhow` for internal error handling and context
- `thiserror` for public API error types

### C API Error Handling (Known Issue)

**IMPORTANT**: The C API (`tensor4all-capi`) currently discards all error details at the FFI boundary. See [#228](https://github.com/tensor4all/tensor4all-rs/issues/228).

- `catch_unwind` catches panics but the panic message is lost (`result.unwrap_or(T4A_INTERNAL_ERROR)`)
- `Err(e)` variants are discarded (`Err(_) => T4A_INTERNAL_ERROR`)
- FFI consumers (Julia, Python) only see a generic "Internal error" with no diagnostic info
- **~76 `catch_unwind` sites** and **~47 `Err(_)` discard sites** across the capi crate need updating
- **Do not add new `catch_unwind` / `Err(_) => T4A_INTERNAL_ERROR` patterns** without preserving error messages
- When #228 is resolved, use the `t4a_last_error_message` API and `run_catching` helper for all new FFI functions

## Testing

```bash
cargo nextest run --release --workspace          # Full suite
cargo nextest run --release --test test_name     # Specific test
cargo nextest run --release -p crate_name        # Single crate
```

**Always use `--release` mode for tests** to avoid excessive execution time in debug builds.

- Private functions: `#[cfg(test)]` module in source file
- Integration tests: `tests/` directory
- Dense whole-result comparisons: avoid per-element re-contraction loops in tests. Materialize once to a dense tensor/matrix, then compare via tensor subtraction and `maxabs()`. `TensorDynLen` subtraction aligns indices by index semantics, so explicit axis reordering is usually unnecessary.
- **Test tolerance changes**: When relaxing test tolerances (unit tests, codecov targets, etc.), always seek explicit user approval before making changes.
- **Dense whole-result comparisons**: When comparing full tensors/operators in tests, do not recompute contractions element-by-element. Materialize once to a dense tensor/matrix, then compare via tensor subtraction and `maxabs()`. `TensorDynLen` subtraction aligns indices by index semantics, so explicit axis reordering is usually unnecessary.

## API Design

Only make functions `pub` when truly public API.

### Layering and Maintainability

**Respect crate boundaries and abstraction layers.**

- **Never access low-level APIs or internal data structures from downstream crates.** Use high-level public methods instead of directly manipulating internal representations.
- **Use high-level APIs.** If downstream code needs low-level access, create appropriate high-level APIs rather than exposing internal details.
- **Prefer type-generic Rust APIs.** If a public Rust library API can be expressed generically, do not introduce scalar-specific entry points such as `*_f64` / `*_c64`. Add or use a generic API instead, and prefer the generic form in library code, tests, examples, and docs. This rule does not apply to the C API / FFI boundary, where scalar-specific names may be appropriate.
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

See `docs/CAPI_DESIGN.md` for C API patterns. Bindings: [Tensor4all.jl](https://github.com/tensor4all/Tensor4all.jl) (separate repo), `python/tensor4all/`.

Truncation tolerance: support both `cutoff` (ITensors) and `rtol` (tensor4all-rs). Conversion: `rtol = √cutoff`.

## Dependencies

- Prefer `mdarray` for arrays (row-major only), `mdarray-linalg` for linear algebra
- SVD singular values: `s[[0, i]]` not `s[[i, i]]`

## Git Workflow

**Never push/create PR without user approval.**

### Pre-PR Checks

Before creating a PR, always run lint checks locally:

```bash
cargo fmt --all                        # Format all code
cargo clippy --workspace               # Check for common issues
cargo nextest run --release --workspace # Run all tests
cargo doc --workspace --no-deps        # Build rustdoc
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
