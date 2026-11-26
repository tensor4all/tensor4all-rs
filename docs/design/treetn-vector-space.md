# TreeTN vector-space boundary

TreeTN owns its `TensorVectorSpace` implementation (#1423), so downstream users
can pass a whole tree to the core Krylov solvers without an orphan-rule newtype.
It remains independent of `TensorLike`: arbitrary tree contraction and
factorization are not required merely to use a network as a vector.

Linear combinations use exact direct sums. Rank control belongs in the
solver's caller-supplied truncation callback, which restores the intended
canonical center before the next Krylov operation. Scaling reuses TreeTN's
canonical-center-preserving primitive. Read-only trait norms canonicalize a
copy; the explicitly qualified mutating norm remains available for callers
that want to update the gauge.

The vector space is defined by topology and full site-index identity. Node
names or equal dimensions do not establish identity. Applications that
intentionally rename physical legs explicitly use `reindex_site_space_like`
before returning operator output to the solver.

Validation requires a nonempty connected tree with consistent site/bond and
orthogonality metadata. Existing factorization dtype restrictions propagate;
no scalar conversion or dense fallback is introduced. Elementwise extrema are
unsupported. Finiteness means finite stored core components, not a bound on
the contracted tensor's values or dynamic range.

See the [user guide](../book/src/guides/tree-krylov.md)
and its executable example for the supported GMRES workflow. This boundary
does not promise the elementwise range analysis needed by every affine-GMRES
overflow-recovery path.

Caller-owned CPU values stay in their existing runtime across scalar
multiplication, rank-zero extraction (`TensorConstructionLike::to_scalar`),
and direct-sum publication. The existing direct-sum host-copy algorithm is not
differentiable; this integration makes no whole-solver AD guarantee. Standalone
scaling and scalar extraction retain supported tracked graphs.
