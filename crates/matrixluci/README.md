# matrixluci

Low-level LUCI / rank-revealing LU substrate for matrix and tensor cross interpolation.

## Attribution

The dense LU path in `matrixluci` is a tensor4all-owned port derived from `faer`'s
full-pivoting LU implementation and API ideas.

- Keep this upstream attribution explicit in this README and in crate-level docs.
- Treat copyright and authorship carefully when refactoring or splitting this crate.
- Do not describe the dense LU path as if it were independent of the `faer`
  implementation lineage.

`matrixluci` does not vendor `faer` source verbatim; tensor4all owns the LUCI-specific
truncation semantics, pivot-error compatibility behavior, and lazy/block-rook extensions.

