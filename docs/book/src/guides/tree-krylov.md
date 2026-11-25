# Krylov solvers over TreeTN vectors

Nonempty `TreeTN<T, V>` implements `tensor4all_core::TensorVectorSpace`, so
`gmres_with_truncation` and `restart_gmres_with_truncation` can solve a global
linear system with a whole tree as each vector. This is distinct from
`square_linsolve`, whose ALS/MALS sweeps solve local projected systems.

`axpby` forms an exact direct sum. Compression is performed only by the
callback passed to the solver: choose its bond cap and tolerance explicitly,
and truncate towards the center needed by subsequent operations. Scaling
preserves a single canonical center. `TensorVectorSpace::norm(&tree)` and
`norm_squared` canonicalize a copy; explicitly call `TreeTN::norm(&mut tree)`
when the input should instead be canonicalized in place.

The operands must have the same topology and full physical indices, including
prime levels and tags. When an operator deliberately creates fresh output
legs, explicitly call `reindex_site_space_like` before returning to the Krylov
space if its node-by-node matching is the intended identification. The trait
never guesses identity from matching dimensions.

This executable example solves a known system on a degree-three star using a
caller-owned CPU context. It uses f64 cores; Complex64 is also supported.
Existing f32/Complex32 QR/SVD limitations still apply when evaluating norms
or truncating. Only the small final reference check materializes a dense
16-element tensor.

```rust
{{#include ../../../../crates/tensor4all-treetn/examples/gmres_tree.rs:2:}}
```

Elementwise `maxabs` and `min_nonzero_abs` return errors, since a scalable
TreeTN cannot compute them by silently visiting the full physical tensor.
Use norm-based residuals for large networks. `is_all_finite` inspects stored
core components; finite cores do not guarantee that contracted values fit in
the scalar range. Affine-GMRES paths that need elementwise range analysis
remain unsupported. The core GMRES-with-truncation variants do not require
those extrema.
The existing host-copy direct sum does not record derivatives, so this workflow
does not provide automatic differentiation through the full Krylov solve.
