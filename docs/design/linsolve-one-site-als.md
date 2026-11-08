# One-site ALS linear solves

## Public behavior

TreeTN and itensorlike `LinsolveOptions` expose `with_nsite(1)`; the default stays `nsite=2`. One-site ALS solves one local tensor with the selected projected Krylov solver (GMRES by default) and moves the orthogonality center by untruncated QR. It does not grow bonds or apply SVD truncation. Values other than 1 or 2 and explicit SVD policies or bond caps with nsite=1 are rejected. Oversized/redundant bonds may shrink during canonicalization/QR; insufficient initial ranks can prevent convergence.

Single-node networks are supported when nsite=1; two-site mode retains its minimum of two nodes. Affine coefficients, index mappings (including primed indices and spectator nodes), residual reporting, early stopping, and zero-sweep behavior use the existing infrastructure. Mode options are validated before shortcuts, including identity-term-only solves.

## Implementation

The implementation reuses the existing Euler edge tour and LocalUpdater integration: for each edge, solve only its source tensor, then QR-transport its center to the destination. The two-node transport region ensures existing reference-state synchronization and both environment caches see every changed tensor. A single-node problem gets one one-node update per sweep. The two-site local solve and SVD path are unchanged. There is no new dependency, global sweep framework extension, or dense production materialization.

## Validation

Test real and complex local systems, affine coefficients, full-rank chains and branching trees, repeated sweeps/cache freshness, rank-limited nonconvergence without bond growth, single-node solves, zero sweeps, exact initial guesses, zero RHS and zero initial guesses, invalid modes/truncation combinations, and the MPS wrapper. Check true dense results only in small tests and preserve two-site regression coverage. If an existing GMRES edge case blocks correctness, identify and address its shared cause rather than hiding it in ALS.

## Deferred follow-ups

Local CG/BiCGSTAB selection is described in [Local solver selection](linsolve-local-solver-selection.md). Preconditioning, new affine Krylov wrappers, compression-aware Krylov methods, enrichment/rank growth, benchmarks/performance promises, and unrelated GMRES improvements remain separate.

## Scope decisions

The identity-term-only shortcut remains for two-site solves. One-site ALS must
project even that problem into the initial bond spaces, otherwise copying the
RHS would silently grow ranks. The zero-operator/zero-identity case still fails.

The shared GMRES zero-RHS shortcut previously returned the initial guess. All
four paths now return the zero solution for a zero norm, and nonzero RHS values
are no longer discarded by a fixed 1e-15 cutoff. Both affine coefficients equal
to zero still cause an error. The range of representable norms remains subject
to the shared tensor implementation; the separate CG PR improves that range.

No external solver code is copied. The existing projected-GMRES structure is
also used by [ITensorMPS.jl](https://github.com/ITensor/ITensorMPS.jl/blob/main/src/solvers/linsolve.jl).
