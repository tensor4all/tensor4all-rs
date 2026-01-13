# Introduction

**tensor4all-rs** is a Rust implementation of tensor networks designed for **vibe coding** — rapid, AI-assisted development with fast trial-and-error cycles.

## Design Philosophy

- **Modular architecture**: Independent crates with unified core (`tensor4all-core`) enable fast compilation and isolated testing
- **ITensors.jl-like dynamic structure**: Flexible `Index` system and dynamic-rank tensors preserve the intuitive API
- **Static error detection**: Rust's type system catches errors at compile time while maintaining runtime flexibility
- **Multi-language support via C-API**: Full functionality exposed through C-API; initial targets are Julia and Python

## Scope

Initial focus on:
- **QTT (Quantics Tensor Train)**: Efficient representation of high-dimensional functions
- **TCI (Tensor Cross Interpolation)**: Efficient construction of tensor trains from black-box functions

The design is extensible to support Abelian and non-Abelian symmetries in the future.

## Type Correspondence

| ITensors.jl | QSpace v4 | tensor4all-rs |
|-------------|-----------|---------------|
| `Index{Int}` | — | `Index<Id, NoSymmSpace>` |
| `Index{QNBlocks}` | `QIDX` | `Index<Id, QNSpace>` (future) |
| `ITensor` | `QSpace` | `TensorDynLen<Id, Symm>` |
| `Dense` | `DATA` | `Storage::DenseF64/C64` |
| `Diag` | — | `Storage::DiagF64/C64` |
| `A * B` | — | `a.contract(&b)` |

## Truncation Tolerance

| Library | Parameter | Conversion |
|---------|-----------|------------|
| tensor4all-rs | `rtol` | — |
| ITensors.jl | `cutoff` | `rtol = √cutoff` |
