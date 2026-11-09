# One-site ALS linear solves

## Public behavior

TreeTN and itensorlike `LinsolveOptions` expose `with_nsite(1)`; the default stays `nsite=2`. One-site ALS solves one local tensor with the existing projected GMRES and moves the orthogonality center by untruncated QR. Without enrichment it does not grow bonds or apply SVD truncation. Values other than 1 or 2 and explicit SVD policies or bond caps with nsite=1 are rejected. Oversized/redundant bonds may shrink during canonicalization/QR; insufficient initial ranks can prevent convergence unless residual enrichment is enabled.

Single-node networks are supported when nsite=1; two-site mode retains its minimum of two nodes. Affine coefficients, index mappings (including primed indices and spectator nodes), residual reporting, early stopping, and zero-sweep behavior use the existing infrastructure. Mode options are validated before shortcuts, including identity-term-only solves.

## Implementation

The implementation reuses the existing Euler edge tour and LocalUpdater integration: for each edge, solve only its source tensor, then QR-transport its center to the destination. The two-node transport region ensures existing reference-state synchronization and both environment caches see every changed tensor. A single-node problem gets one one-node update per sweep. The two-site local solve and SVD path are unchanged. There is no new dependency, global sweep framework extension, or dense production materialization.

## Residual enrichment

TreeTN `LinsolveOptions::with_enrichment(ResidualEnrichment)` grows one-site bonds from the residual; it is rejected unless nsite=1, and the itensorlike wrapper does not expose it. After the QR transport on edge `(src, dst)`, the updater forms the two-site projected residual `b - (a0 + a1 A) x` on `{src, dst}`, removes its component in the span of the new `src` basis, and keeps up to `max_added` leading left singular vectors whose singular values exceed `cutoff * ||b_local||`. These are appended to the `src` basis by direct sum and re-orthonormalized by QR; the `dst` center is recomputed by projecting the two-site state onto the enlarged basis, so the represented state is unchanged and the next local solve at `dst` uses the new directions. A bond never exceeds the `src`-side dimension or `max_bond_dim`, and enriched bonds are not truncated. Each step costs one two-site operator application and one SVD.

This is a local, per-step expansion driven by the linear-system residual. Global subspace expansion (`gse.rs`) is a different method: it expands the whole state once, before TDVP sweeps, from Krylov reference states.

## Validation

Test real and complex local systems, affine coefficients, full-rank chains and branching trees, repeated sweeps/cache freshness, rank-limited nonconvergence without bond growth, single-node solves, zero sweeps, exact initial guesses, zero RHS and zero initial guesses, invalid modes/truncation combinations, and the MPS wrapper. For enrichment, test rank-1 initial guesses that converge only with enrichment on chains and stars (real and complex), the per-step and total bond limits, skipping zero RHS, single-node problems and negligible residuals, and rejection with nsite=2 or invalid settings. Check true dense results only in small tests and preserve two-site regression coverage. If an existing GMRES edge case blocks correctness, identify and address its shared cause rather than hiding it in ALS.

## Deferred follow-ups

Local CG/BiCGSTAB selection, preconditioning, new affine Krylov wrappers, compression-aware Krylov methods, enrichment in the itensorlike wrapper, benchmarks/performance promises, and unrelated GMRES improvements remain separate.

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
