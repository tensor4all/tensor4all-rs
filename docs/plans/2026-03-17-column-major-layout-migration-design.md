# Column-Major Layout Migration Design

**Date:** 2026-03-17

## Goal

Migrate tensor4all-rs to a fully column-major dense layout model across internal storage, reshape/flatten semantics, public flat-buffer APIs, Python bindings, and HDF5 interoperability, while keeping HDF5 compatible with ITensors.jl.

## Motivation

The current codebase mixes layout-aware logic across multiple layers. This creates avoidable ambiguity around:

- flat index to multi-index mapping
- reshape linearization
- NumPy / C API import-export behavior
- tenferro interop
- HDF5 conversion rules

For a high-level tensor library, users should reason primarily in terms of indices and tensor structure, not dense storage order. Layout details should be explicit only at well-defined boundaries. Moving to column-major semantics aligns tensor4all-rs with tenferro-rs, Julia, ITensors.jl, MATLAB, and Eigen's default model, while reducing the number of row/column-major conversion boundaries in the implementation.

## Approved Product Decisions

- Breaking changes are acceptable.
- The final system should be clean, not compatibility-layer driven.
- Internal dense storage will use column-major layout.
- Public flat-buffer semantics will also be column-major.
- Python, C API, and HDF5 documentation must say this explicitly.
- HDF5 will continue to support only the ITensors.jl-compatible format for now.
- Boundary code may temporarily host adapters during migration, but the final state must not keep row-major compatibility shims.

## Architectural Direction

### 1. Layout-aware code is pushed down

Layout-specific logic should live in backend and boundary layers only.

- **Backend** owns stride calculation, flat offset calculation, and reshape linearization.
- **Core / higher-level crates** should become layout-neutral wherever possible.
- **Boundary layers** (C API, Python, HDF5) may mention and enforce layout semantics explicitly.

### 2. Introduce a centralized layout module

Create a small backend-owned layout module that becomes the single source of truth for:

- dense stride generation
- `multi_index -> flat_offset`
- `flat_offset -> multi_index`
- reshape linearization / dense reinterpretation rules

The migration should first route existing code through this module, then flip the module semantics from row-major to column-major. This reduces the number of places that must be audited during the semantic flip.

### 3. Internal-first staged migration

Use an internal-first migration rather than a big-bang rewrite.

**Phase 1: Layout primitive consolidation**
- Centralize layout helpers in backend.
- Remove ad hoc row-major math from high-level crates.

**Phase 2: Internal semantic flip**
- Flip backend dense storage, reshape/flatten semantics, tenferro bridge helpers, and core linearization-dependent algorithms to column-major.
- Keep temporary adapters only at public boundaries.

**Phase 3: Public contract flip**
- Update C API, Python, HDF5 docs and behavior to column-major semantics.
- Remove temporary row-major adapters.

## Public Semantics After Migration

### Rust API

- Dense constructors/exporters and reshape/flatten semantics follow column-major linearization.
- High-level tensor operations remain index-semantic first; users only need to care about layout when dealing with flat buffers or explicit reshape/flatten behavior.

### C API

- Dense input/output buffers are column-major.
- Function documentation explicitly states this.

### Python

- Arrays are copied and normalized at the boundary.
- `from_numpy` interprets values by logical indices, then stores them internally in column-major form.
- `to_numpy` returns arrays consistent with column-major semantics; F-contiguous output is preferred.
- Reshape/flatten behavior matches `order="F"`.

### HDF5

- ITensors.jl compatibility remains the only supported public HDF5 format.
- After migration, HDF5 dense layout should align naturally with internal semantics, reducing conversion complexity.

## README / Documentation Policy

README must explicitly state that tensor4all-rs uses column-major internal dense storage and column-major linearization semantics. This is important because users can otherwise be surprised at API boundaries, especially in Python and C interop.

At the same time:

- high-level API docs should avoid unnecessary layout discussion
- layout details should be concentrated in:
  - README
  - C API docs
  - Python import/export docs
  - HDF5 interoperability docs
  - reshape / flatten / dense constructor docs

## Main Risk Areas

1. **Reshape / flatten regressions**
   - Existing tests and comments already document row-major assumptions in multiple places.

2. **tenferro bridge ambiguity**
   - Current bridge helpers explicitly force row-major behavior to work around reshape bugs and semantic mismatches.

3. **Linalg regressions**
   - QR/SVD flatten tensors into matrices; wrong linearization silently corrupts element ordering.

4. **Boundary mismatch**
   - C API and Python behavior will become breaking changes.

5. **Test fragility**
   - Many tests use flat dense literals that implicitly encode layout assumptions.

## Verification Strategy

Verification must focus on semantics, not just passing tests.

- Golden tests for `multi_index <-> flat_offset`
- reshape / flatten roundtrip tests under column-major rules
- dense constructor/exporter roundtrip tests
- QR/SVD reconstruction tests with known layout-sensitive examples
- tenferro bridge regression tests for unit-dimension reshape edge cases
- HDF5 ITensors.jl roundtrip tests
- Python import/export tests matching `order="F"` reshape semantics
- C API dense get/set tests for column-major buffers

Where possible, tests should compare tensors by logical indices or tensor subtraction and `maxabs()`, not by hand-written flat layouts embedded in multiple places.

## Final State

The intended final state is:

- internal dense layout: column-major
- public flat-buffer semantics: column-major
- HDF5 ITensors.jl compatibility preserved
- high-level code largely layout-neutral
- layout awareness centralized in backend and explicit boundary modules

