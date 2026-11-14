# Patched interpolative QTT construction

The `tensor4all-interpolativeqtt` patch driver implements the main construction
of #1276, following generic scalar support in #1302. It returns
`Vec<InterpolativePatch<T>>`: a prefix per axis, a local `SimpleTensorTrain<T>`,
and an absolute local error estimate. Retaining the prefix is necessary to map
the train back to the original grid. There is no TreeTN production dependency.

## Construction and acceptance

The driver processes dyadic boxes breadth first. Each box runs the existing
multiscale Chebyshev core assembly with its remaining bits. The assembly's
sampling boundary accepts fallible batches, shared with the existing pointwise
constructors. Samples on each immediate child's Chebyshev grid also feed a
validation stencil on a grid with twice as many nodes per axis. The stencil is
built once and its contraction runs through the supplied engine's GEMM.

SVD compression uses `SimpleTensorTrain::compress_svd_with_session`. Its sweeps
and retained-rank rules are shared with ordinary compression. The quadrature
sum of discarded singular values bounds Frobenius truncation error, excluding
roundoff; it also bounds each entry's truncation error. A box is accepted when
the maximum checked interpolation residual plus this bound is at most the
absolute tolerance, and its compressed rank is at most the optional cap. The cap
never silently truncates a candidate to make it acceptable.

A failed candidate fixes the next bit of every axis and becomes `2^D` child
boxes. With one site remaining, direct sampling yields an exact local train;
zero-site trains are unnecessary. A configured depth or patch limit returns a
typed error instead of an unaccepted result. Dense storage scales with the
explicitly bounded Chebyshev stencil, not with the full quantics grid.

The Frobenius compression bound is conservative on very large grids and can
trigger additional refinement even for a small pointwise residual. Setting
`compression_tolerance` to zero removes truncation while retaining the full
interpolation ranks.

The error is an estimate for interpolation: an arbitrary feature between all
sample points can be missed. It is not an interval-arithmetic certificate.
Bounds must be finite and strictly increasing, bit depth is limited to 52, and
splits whose midpoints cannot be represented are rejected.

## Execution and ordering

The serial entry uses one configured, serial CPU engine. The optional `parallel`
entry uses `hataori::map_in_lanes` and caller-owned `TenferroLanes`. The caller
chooses P lanes and B threads per engine; neither entry consults the global
backend. Each evaluator batch receives the lane engine's `threadteam::Exec`
token through `CpuEngine::with_exec`, alternating with GEMM and SVD on that same
engine. A wave narrower than P leaves lanes idle; budgets are not rebalanced.

Callbacks must be pure and independent of lane identity, token width and call
order. Wave results (including typed evaluator errors) stay in input order;
pending waves and final output are lexicographically sorted by per-axis prefix.
The first failing box is therefore schedule independent. Regression tests
compare prefixes, error estimates and core values exactly across P/B layouts.
Floating-point backend arithmetic remains subject to roundoff, especially for
acceptance thresholds at machine precision.

`parallel` selects native/faer, whose upstream `native`/`blas` compile-time
exclusion rejects system BLAS, provider injection and Accelerate combinations.
On macOS, disable the default platform provider when enabling `parallel`.
MPI and automatic lane sizing remain outside this API.

## Downstream conversion

Each local site fuses one bit of each axis, axis zero fastest. A caller can
bridge its train to TreeTN, unfuse these sites with column-major ordering and
split to an interleaved binary chain. Insert the prefix's one-hot sites before
creating `SubDomainTreeTN`, which requires all physical axes to remain present.
The executable tutorial demonstrates this conversion and the patched Fourier
transform without changing the TreeTN library or its interpolation trait.
