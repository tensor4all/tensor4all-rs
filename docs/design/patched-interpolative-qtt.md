# Patched interpolative QTT construction

The `tensor4all-interpolativeqtt` patch driver implements #1276, with generic
scalar support from #1302 and native TreeTN output from #1334. A patch carries a
prefix per axis, its local TreeTN with ordered physical indices, and an absolute
per-value error estimate. The prefix maps the local chain back to the original
grid.

## Representation boundary

Chebyshev basis construction, dyadic refinement and sampling operate on dense
local arrays ordered `[left bond, fused physical digit, right bond]` in
column-major layout. TreeTN assigns index identities when publishing those
cores. Site zero is the most significant level. Within a fused digit the bit
for axis zero varies fastest. The ordinary constructors and the patch driver
share the same refinement and compression operations. Output storage supports
real and complex values through the same TreeTN type.

At zero tolerance with no bond cap, the ordinary constructors retain the
assembled cores in their original gauge. Inverse reconstruction reads them in
physical and bond order without canonicalizing. Its finest-level decoding
checks the bond dimensions, the proportional finest core and agreement at
shared endpoints. A compatible rescaling of that core preserves the exact
route. Compression and other changes that fail these checks use the two-point
blend. Deliberately modified cores that preserve the checked samples can still
change interior reconstructed values, as documented by `invert_qtt`.

TreeTN's optional SimpleTT bridge is outside the interpolation dependency
graph. Removing the bridge does not change the interpolation algorithms or
the result types.

## Construction and acceptance

The driver processes dyadic boxes breadth first. Each box runs the existing
multiscale Chebyshev core assembly with its remaining bits. The assembly's
sampling boundary accepts fallible batches, shared with the existing pointwise
constructors. Each batch is tagged as either the Chebyshev nodes of a named
interval or exact last-core grid points, so the sampler never reconstructs a
box from its points. Samples on each immediate child's Chebyshev grid also feed a
validation stencil on a grid with twice as many nodes per axis. The stencil is
stored as one `(2p+2) × (p+1)` matrix, built once. Successive axis
contractions run through the supplied engine's GEMM; column-major rotations
restore the original axis-zero-fastest validation order. All sample coordinates are written
directly into one column-major `[D, n]` buffer per batch, shared by pointwise
and batched node sampling. No dense Kronecker
stencil is allocated.

Compression uses `TreeTN::compress_chain_on_session`, shared with ordinary
construction. It sweeps towards the finest site with full-rank QR, then sweeps
back with SVD using a threshold relative to each bond's largest singular value.
The compressed train is checked per value against the uncompressed train. An axis with at
most `2(2p+2)` grid points is checked at every point. A longer axis is checked,
in each child half, at the grid points nearest to the validation coordinates
mapped onto that half, so both child boundaries (including the patch midpoint,
where the child interpolants meet) and the child interiors are sampled. Because
every box of the driver is safe, the
uncompressed train equals the owning child's Chebyshev interpolant at each grid
point, so the comparison uses the sampled node values and costs
`(degree + 1)^D` products per point rather than an evaluation of the
uncompressed rank. The 1D Lagrange values for the selected child coordinates
are precomputed once and reused on every axis. A call-local
`TreeTNCachedEvaluator` shares message contractions across the batch and runs
on the supplied session. The error estimate is the maximum interpolation residual plus
the maximum compression change. A box is accepted when that estimate is at most
the absolute tolerance and its compressed rank is at most the optional cap. A
cutoff or cap that visibly degrades the train therefore leads to splitting. On
long axes the compression check is sampled, like the interpolation check, so it
is an estimate rather than a bound.

A failed candidate fixes the next bit of every axis and becomes `2^D` child
boxes. With one site remaining, direct sampling yields an exact local train;
zero-site trains are unnecessary. A configured depth or patch limit returns a
typed error instead of an unaccepted result. Validation storage scales with the tensor-product node and test-point
batches plus the single 1D stencil, independently of the full quantics grid.

Acceptance compares only per-value quantities with the per-value tolerance. The
Frobenius norm of the discarded singular values covers the whole patch tensor
and grows roughly with the square root of its grid size, so it is not used.

The error is an estimate for interpolation: an arbitrary feature between all
sample points can be missed. It is not an interval-arithmetic certificate.
Bounds must be finite and strictly increasing, bit depth is limited to 52, and
`NInterval::split` rejects splits whose midpoints cannot be represented. This
same check covers multiscale/adaptive constructors as well as the patch driver.

## Execution and ordering

The serial entry takes the caller's `CpuExecutionContext` and runs the whole
construction in one session of its engine, so re-entry is detected by the
context; it never creates a backend. The optional `parallel`
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

`parallel` enables only the lane executor (`hataori`, `hataori-tenferro`), both
provider-neutral, so it builds with the platform default and every explicit
provider feature. Lane execution is validated on native/faer only: there each
lane's algebra stays within its B threads, and the nightly lane tests assert
bit-identical results across layouts. With a BLAS provider such as Accelerate or
system BLAS, each of the P lanes calls a library with its own threading, which
can oversubscribe the cores and relies on the library's thread safety and
determinism; CI only checks that combination compiles. MPI and automatic lane
sizing remain outside this API.

## Downstream projector embedding

Unfuse the ordered local sites in column-major order and split them into an
interleaved binary chain. Attach the prefix's one-hot sites with dimension-one
bonds before creating `SubDomainTreeTN`, which requires all physical sites to
be present. The executable tutorial demonstrates the embedding and checks a
patched Fourier transform against a dense reference.
