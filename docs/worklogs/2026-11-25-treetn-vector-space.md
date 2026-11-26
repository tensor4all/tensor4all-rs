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
- Follow the existing standalone-guide pattern for shared executable sources:
  mdBook expands the GMRES source include, and `book-tests` embeds that same
  source directly for raw rustdoc. Also enable the RSI guide's required
  `global-defaults` feature in its test-harness dependency, so the standalone
  mdBook script does not depend on workspace-wide feature unification.

## Verification conclusions and constraints

- Core unit tests: 794 passed, one existing ignored test skipped. TreeTN tests:
  1,194 passed, five existing ignored tests skipped. The 12 new vector-space
  regressions include real/complex non-Hermitian branched-tree solves,
  restarted GMRES, exact rank growth, index mismatch, invalid vectors, and a
  64-site rank-one network without dense materialization.
- Caller-owned context integration: 19 tests passed. Scalar-context unit tests
  verify ownership, same-runtime tensor/coefficient gradients, foreign tracked
  coefficient rejection, and initialization-failure propagation. The executable
  example also passed in a separate downstream client with all path dependencies'
  default features disabled; its feature tree has no `global-defaults`.
- Core doctests: 429 passed. TreeTN doctests: 231 passed. Workspace Clippy with
  warnings, missing error docs, and missing panic docs denied passed after
  removing an unnecessary `clone()` on the now-`Copy` truncation options.
- API inventory, public error documentation, crate boundaries, implicit-engine
  ratchet, and repository-rules dry run (including file growth) passed. The
  repository-rules review helper's 131 tests passed.
- The standalone mdBook script passed with the dev profile, including the
  GMRES and RSI examples. Standard `cargo doc --no-deps` for core and TreeTN
  passed. An additional `RUSTDOCFLAGS="-D warnings"` run failed on five
  pre-existing core links (unqualified `TensorElement`, two private storage
  links, two redundant explicit targets); all five source sites were verified
  unchanged in `origin/main`. TreeTN produced no rustdoc warnings.
- No performance speedup or general affine-GMRES support is claimed;
  elementwise range analysis remains unsupported for tensor-network vectors.
  Existing f32/Complex32 unitary QR limitations propagate as errors. Direct-sum
  host copies do not record derivatives, so full-solver AD is not promised.
