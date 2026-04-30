# QTT Playground Notes

This is the reading map for the repository. The examples build on each other,
but each one can still be run on its own.

## Reading Order

1. [QTT of a scalar function](tutorials/qtt_function_tutorial.md)
   - Start here.
   - Opens `src/bin/qtt_function.rs`.
   - Builds a QTT for `cosh(x)` on the unit interval.

2. [QTT on a physical interval](tutorials/qtt_interval_tutorial.md)
   - Opens `src/bin/qtt_interval.rs`.
   - Replaces integer-grid thinking with `DiscretizedGrid`.
   - Uses `x^2` on `[-1, 2]`.

3. [QTTs for multivariate functions](tutorials/qtt_multivariate_tutorial.md)
   - Opens `src/bin/qtt_multivariate.rs`.
   - Builds a QTT for a two-dimensional function and compares interleaved and grouped layouts.

4. [Definite integrals](tutorials/qtt_integral_tutorial.md)
   - Opens `src/bin/qtt_integral.rs`.
   - Reuses the interval setup and calls `integral()`.

4. [Sweep over bit depth](tutorials/qtt_r_sweep_tutorial.md)
   - Opens `src/bin/qtt_r_sweep.rs`.
   - Compares accuracy and runtime as the quantics depth changes.

5. [Elementwise TreeTN product](tutorials/qtt_elementwise_product_tutorial.md)
   - Opens `src/bin/qtt_elementwise_product.rs`.
   - Builds two QTTs, multiplies them pointwise, and inspects bond growth.

6. [Fourier transform](tutorials/qtt_fourier_tutorial.md)
   - Opens `src/bin/qtt_fourier.rs`.
   - Builds a Gaussian QTT, applies the quantics Fourier operator, and plots the transform.

7. [Affine transformation](tutorials/qtt_affine_tutorial.md)
   - Opens `src/bin/qtt_affine.rs`.
   - Applies a 2D affine pullback operator and compares periodic and open boundary behavior.

8. [2D partial Fourier transform](tutorials/qtt_partial_fourier2d_tutorial.md)
   - Opens `src/bin/qtt_partial_fourier2d.rs`.
   - Applies a Fourier transform to only the `x` coordinate of a 2D QTT.

## Supporting Notes

- [Glossary](glossary.md): short definitions for QTT and tensor-network terms.
- [Repository improvement roadmap](plans/2026-04-15-repository-improvement-roadmap.md): current cleanup plan.

## Output Policy

`docs/data` and `docs/plots` are tracked on purpose. They are curated tutorial
artifacts, not disposable build output.

Routine checks write to `target/check/data` instead, so `./scripts/check.sh`
should not change committed CSVs or plots.
