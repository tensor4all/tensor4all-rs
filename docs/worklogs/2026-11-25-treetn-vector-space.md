# TreeTN Krylov vectors (#1423)

## Decisions

- Implement `TensorVectorSpace` at the TreeTN owner, generically over its core
  tensor and ordered node-name types. Reuse exact direct-sum addition, scaling,
  inner products, and norms. Compression belongs exclusively in the solver's
  explicit callback, which should restore the intended canonical center.
- Preserve full site-index identity. Operator callbacks that intentionally
  produce fresh site indices can explicitly use `reindex_site_space_like`;
  the vector-space trait does not infer identity from shape or node names.
- Read-only trait norms canonicalize a copy. Explicit `TreeTN::norm(&mut tree)`
  retains its existing mutating behavior. Scale delegates to the existing
  canonical-center-preserving operation.
- Require nonempty connected vectors. Empty TreeTN products remain supported
  by `inner`, but have no node on which scaling can act. Elementwise extrema
  return unsupported-configuration errors rather than materializing a dense
  tensor. Finiteness inspects stored core components, not contracted values.
- Caller-owned CPU execution exposed three prerequisite ownership bugs: plain
  scalar coefficients could remain in the global runtime, TreeTN scalar
  extraction constructed a foreign scalar-one tensor, and direct-sum host
  results/promotion used global constructors. Fix these at their core owners:
  rehome only untracked scalar coefficients, provide rank-zero extraction on
  `TensorConstructionLike`, and publish direct-sum results in the operand's
  runtime. Tracked foreign coefficients and scalar initialization errors remain
  errors, and same-runtime scalar/tensor gradients must survive.

## Verification conclusions and constraints

- Numerical and documentation validation is pending. No performance speedup
  or general affine-GMRES support is claimed; elementwise range analysis remains
  unsupported for tensor-network vectors.
