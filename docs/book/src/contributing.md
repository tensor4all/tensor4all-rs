# Contributing

Thank you for your interest in contributing to tensor4all-rs!

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/tensor4all/tensor4all-rs.git
   cd tensor4all-rs
   ```

2. Build the project:
   ```bash
   cargo build --release
   ```

3. Run tests:
   ```bash
   cargo test --workspace
   ```

## Code Style

- Use `rustfmt` for formatting: `cargo fmt`
- Use `clippy` for linting: `cargo clippy`
- Prefer explicit error handling over `unwrap()` or `expect()` in library code

## Testing

### Running Tests

```bash
# All tests
cargo test --workspace

# Specific crate
cargo test -p tensor4all-core

# With output
cargo test -- --nocapture
```

### Test Organization

- **Unit tests**: In `#[cfg(test)]` modules at the end of source files
- **Integration tests**: In the `tests/` directory of each crate

### Generic Tests

For operations that work on both real and complex types, use generic test functions:

```rust
fn test_operation_generic<T: Scalar>() { /* ... */ }

#[test]
fn test_operation_f64() { test_operation_generic::<f64>(); }

#[test]
fn test_operation_c64() { test_operation_generic::<Complex64>(); }
```

## Pull Requests

1. Create a feature branch from `main`
2. Make your changes with clear commit messages
3. Ensure all tests pass
4. Submit a pull request

## Documentation

- Update relevant documentation when changing APIs
- Add docstrings to public functions
- Run `cargo doc` to verify documentation builds
