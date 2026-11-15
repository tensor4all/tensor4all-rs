# TreeTN-native interpolative QTT

## Decisions

- Publish the interpolation algorithm's local arrays directly as a TreeTN
  chain and retain the ordered physical indices alongside it. The [design
  record](../design/patched-interpolative-qtt.md) describes the layout,
  inverse gauge and dyadic prefix contracts.
- Use a full-rank QR sweep followed by an SVD sweep on the caller's engine.
  Keep singular values equal to the relative cutoff and keep the available
  width up to the cap at zero cutoff, matching the previous rank selection.
- Read validated dense local cores in chain order for the inverse. Patch
  acceptance uses a session-local cached evaluator and rejects nonfinite
  residuals before taking their sampled maximum.

## Verification conclusions and constraints

- Direct-construction inverse tests cover the exact and blended routes,
  rescaling, a perturbation at shared endpoints, and an interior rewrite
  that agrees with all stored samples. Invalid site order and dtype return
  errors.
- Core sorts the singular values and permutes both factors together before
  returning a session SVD. The full-rank factorization dispatches all four
  supported dtypes through that path. The shared Core regression exercises
  reordering of a complex spectrum, and the optional bridge tests compare
  the ordered bond ranks and dense results of both compressors on a chain
  with multiple bonds, with and without a cap. QR and SVD can choose
  different gauges, so the comparison uses a numerical residual.
- The default, faer and system BLAS parallel, and provider-injection
  dependency graphs omit both SimpleTT and the optional TreeTN bridge.
  The patched Fourier tutorial builds its prefix embedding from native
  TreeTN operations. Fourier operator construction is tracked in #1333.
- The [paired local measurements](../experiments/2026-11-15-interpolative-qtt-treetn.md)
  found higher time in the narrow 1D fixture, lower time in the wider
  complex 2D fixture, and higher peak RSS in both. The SVD sweep publishes
  full-width factors before selecting the retained bond directions.
