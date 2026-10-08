# Experimental tree RSI: equations and implementation boundaries

## References and scope

The starting algorithm is Meng, Khoo, Li and Stoudenmire,
[Recursive Sketched Interpolation: Efficient Hadamard Products of Tensor Trains,
arXiv:2602.17974v1](https://arxiv.org/abs/2602.17974v1), III.A.1–4.
The earlier implementation consulted `src/multiply_rsi.py` and `src/sketch.py`
in [Recursive-Sketched-Interpolation at 153b25a8](https://github.com/zmeng137/Recursive-Sketched-Interpolation/tree/153b25a8aa059d0147b45955d0842b2f32fa5d1d).
Those sources inform the algorithm; matching their outputs cannot establish
accuracy, numerical stability, or efficient implementation.

This crate implements Hadamard products on labeled trees. Tree traversal,
exact-complement caching, relative pivot stopping, allocation limits and binary
scaling are repository extensions. There is no unconditional tree accuracy
theorem here. The implementation uses standard-normal probes and seeded
ChaCha8 draws, rather than reproducing the author's random-number stream.

## Local algebra

Cut an edge. For operand `a`, let `B_a(x, alpha_a)` be its exact subtree frame
at candidate physical rows `x`. Define a conceptual lifted frame

`F(x, alpha_1, ..., alpha_m) = product_a B_a(x, alpha_a)`.

The true product across that edge is `M = F C`, where each column of `C`
multiplies complement frames at the **same physical assignment**. Contraction
of each operand against its product-state probes produces environments
`E_a(alpha_a, s_parent, l)`. The local matrix used to select rows is

`Z(x, s_parent, l) = product_a sum_alpha_a B_a(x, alpha_a) E_a(alpha_a, s_parent, l)`.

Thus `Z = F W`, where `W` is the product of those operand environments. Neither
`F` nor `W` is materialized. Sharing probes does not turn this construction into
a squared-probe sketch of `M`: expanding the product of sums includes different
complement assignments for different operands.

Row interpolation fits `Z ≈ U Z[I,:]`. If `F = U F[I,:]`, then
`M = U M[I,:]`; however, an exact fit of `Z` only gives
`(F - U F[I,:]) W = 0`. Transfer to `C` needs an additional condition, such as
injectivity of the sketch on the relevant row space or
`range(C) ⊆ range(W)`. Approximate transfer also depends on conditioning and
error accumulation. Low rank of `M` alone is insufficient.

A concrete counterexample is `A=[[1,2],[2,1]]`, `B=[[2,1],[1,2]]`, with
operand probes `(1,1)` and `(1,-1)`, rank cap one and probe width one. The true
product is constant two, but the local sketched product is `(3,-3)`. Its row ID
selects a sign change. A three-node chain with an identity middle core returns
column-major `[2,-2,2,-2]`, with relative L2 error `sqrt(2)`.
`tests/boundaries.rs` reproduces these values. This is not a probability claim
about Gaussian sketches or a test of default oversampling.

## Recursion and exact complements

Children supply exact input frames evaluated at their selected rows. Replacing
child bond axes with these frames forms the parent's candidate rows. After row
selection, each operand is sliced again at the selected rows; it is never
replaced by the approximated product. Induction therefore preserves exact
selected input samples, up to floating-point error. Root evaluation multiplies
these original sampled inputs. Accuracy elsewhere remains a separate question.

An edge uses exact columns when the physical complement excluding the adjacent
parent's physical group has at most `k` assignments. The parent physical group
is enumerated explicitly in both exact and sketched cases. Directed exact
messages with at most `k` assignments are cached once per direction. This
avoids walking the entire complement for every edge. Exact columns remove
sketch error at that edge, but rank truncation and conditioning still matter.

## Implementation correspondence

| Source | Responsibility | Basis |
|---|---|---|
| `api.rs`, `options.rs` | Context/RNG ownership, rank and resource controls | Repository API; sketch width adapts paper Eq. 7 heuristically |
| `plan.rs` | Labeled topology/index validation, root, required messages, exact cases | Tree extension |
| `engine.rs::sketch` and `sketch_environments` | Operand product-state sketches and directed environments | Paper III.A.1, generalized to trees |
| `engine.rs::candidate` and `run` | Product of local sketches, row ID, exact operand resampling, root product | Paper III.A.2–4, with tree candidate row sets |
| `engine.rs::Exact` | Cached exact component contractions | Tree extension of exact-tail policy |
| `core::matrix_luci_row_interpolation_owned_in` | Rank-revealing LU row interpolant, without an unused right factor | Existing repository MatrixLUCI/backend |
| `dense.rs` | Column-major axis changes and backend GEMM/batched GEMM | Layout implementation |
| `scaled.rs`, `scalar.rs` | Common binary scale per dense block | Numerical engineering |
| `result.rs`, `error.rs` | Local diagnostics and typed failures | Repository API, no success/error certificate |

A single scale per block preserves relative probe-column weights. Scales are
carried through multiplication and contraction and restored at the root;
independently normalizing columns would change the sketch and pivot policy.
Scaling does not guarantee that every representable final answer has
representable intermediate blocks. Detected lost range returns an error.

Traversal uses prefix/suffix products for bounded component sizes, and for
scalar exact messages and rank-one sketch messages on high-degree nodes.
General high-degree cores still incur degree-dependent dense work. There is
no blanket cubic-in-bond-dimension bound for arbitrary trees. The configured
allocation limit bounds each algorithm-owned dense block, not the aggregate
memory of caches or backend decomposition scratch.

## Validation and downstream acceptance

Full dense tests cover real/complex, single/double precision, one through four
operands, chains, binary trees, stars, rerooting, and both exact and sketched
columns. Separate regressions cover long chains, extreme gauges, subnormal
products, invalid inputs and the counterexample above. Tests establish their
specific cases; neither a report nor an author-code replay replaces them.

For larger inputs, evaluate true operand products once on held-out batched
points, then compare the returned product on those points. Reject nonfinite
values, failed calls and errors over a predeclared threshold. Local pivots,
selected-row interpolation and rank limits cannot accept an approximation.
Sampled checks do not prove a global norm bound.

A downstream GW application must separately check the actual G0, Pi and Sigma
objects, reciprocal/iteration residuals and convergence criteria. It must fail
when those checks fail. This rewrite supplies no arbitrary nonlinear sketch
callback or downstream GW acceptance claim. The separate
[benchmark protocol](../../benchmarks/tree-rsi/README.md) records exactly what
is timed and what its sampled error test can establish.
The [paper workload protocol](../../benchmarks/tree-rsi/paper-coverage/README.md)
adds physical DMRG inputs, complex branching trees, independent global
references where bounded, and explicitly scoped sample-only checks for
larger states. Its parameter sensitivity experiments distinguish local
pivot stopping, sketch width and rank capacity; they do not establish a
global tolerance guarantee for this API.

## Benchmark build evidence

Paper-coverage timings execute immutable workers bound to build receipts. A
receipt captures the worker source, local path-dependency files, resolved lock,
Cargo configuration, compiler and build controls before and after compilation.
A changed input requires a new build; an existing binary cannot acquire the
identity of the current worktree. Protocols bind these builds to explicit
case/options, algorithm, seed, phase and block schedules. Count equality alone
does not establish completeness. Historical unreceipted runs remain historical
and do not establish behavior or performance of a newer implementation.

The rrLU extreme-magnitude repair for
[core issue #779](https://github.com/tensor4all/tensor4all-rs/issues/779) is
included through main's #840 changes, carried into the shared factorization
engine used by both full-factor and row-only LUCI. Row-only pivot and residual
diagnostics use robust magnitudes, including at a rank cap. RSI
continues to require independent output validation; a zero local tolerance or
reported zero local residual does not certify exact global reconstruction.
