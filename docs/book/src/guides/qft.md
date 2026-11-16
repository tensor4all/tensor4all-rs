# Quantum Fourier Transform

This guide demonstrates how to apply the quantics Fourier transform (QFT)
operator to tensor trains. It corresponds to the Julia
[Quantum Fourier Transform](https://tensor4all.org/T4APlutoExamples/qft.html) notebook.

## Cargo features

The Fourier constructors build TreeTN operators directly. To use them without
SimpleTT, disable the transform crate's defaults and select a CPU provider.
For a local checkout, the unpatched examples below use these dependencies.

```toml
[dependencies]
num-complex = "0.4"
tensor4all-core = { path = "../tensor4all-rs/crates/tensor4all-core", default-features = false, features = ["tenferro-cpu-platform"] }
tensor4all-treetn = { path = "../tensor4all-rs/crates/tensor4all-treetn", default-features = false, features = ["tenferro-cpu-platform"] }
tensor4all-quanticstransform = { path = "../tensor4all-rs/crates/tensor4all-quanticstransform", default-features = false, features = ["tenferro-cpu-platform"] }
```

For patched Fourier, also enable `partitioned` on
`tensor4all-quanticstransform` and declare `tensor4all-partitionedtreetn` with
the same provider and its defaults disabled when importing it directly. On
macOS, use `tenferro-cpu-faer` on all these crates to select faer explicitly.
TreeTN defaults leave `simplett-bridge` off; disabling them is needed only
when selecting a different CPU provider. Enable `simplett-transforms` if you also use
non-Fourier transforms or the transform crate's SimpleTT conversions.

## Background

The QFT operator converts a function from position space to frequency space
(and vice versa) using a matrix product operator (MPO) construction due to
Chen and Lindsey (arXiv:2404.03182). In quantics representation, the DFT
of a function on 2^R grid points is expressed as a compact tensor train with
small bond dimension.

Key properties:
- The output is in **bit-reversed** frequency order.
- Forward transform uses sign = -1 in the exponent; inverse uses +1.
- When `normalize = true` (default), the transform is an isometry.

## Simple QFT Example

`FTCore` prepares forward and inverse operators with the same options. The
example verifies their node and index mappings.

```rust
use tensor4all_quanticstransform::{FourierOptions, FTCore};

let r = 4;

// Create forward and inverse QFT operators
let ft = FTCore::new(r, FourierOptions::default()).unwrap();
let fwd = ft.forward().unwrap();
let bwd = ft.backward().unwrap();

// Each operator has r sites
assert_eq!(fwd.mpo().node_count(), r);
assert_eq!(bwd.mpo().node_count(), r);

// Both operators have input and output mappings for all sites
for i in 0..r {
    assert!(fwd.get_input_mapping(&i).is_some());
    assert!(fwd.get_output_mapping(&i).is_some());
    assert!(bwd.get_input_mapping(&i).is_some());
    assert!(bwd.get_output_mapping(&i).is_some());
}
```

## 1D QFT Application

This example applies the QFT to the product state |0>, which is one at the
origin and zero elsewhere. Its Fourier coefficients all equal 1/sqrt(N).

```rust
use num_complex::Complex64;
use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_quanticstransform::{quantics_fourier_operator, FourierOptions};
use tensor4all_treetn::{apply_linear_operator, ApplyOptions, TreeTN};
# fn main() -> Result<(), Box<dyn std::error::Error>> {

let r = 3usize;
let n = 1usize << r;
let qft_op = quantics_fourier_operator(r, FourierOptions::forward())?;

// Each site uses its operator input index and shares a dimension-one bond
// with each neighbor.
let bonds: Vec<_> = (1..r).map(|_| DynIndex::new_dyn(1)).collect();
let mut tensors = Vec::new();
for i in 0..r {
    let input = qft_op
        .get_input_mapping(&i)
        .ok_or("missing operator input mapping")?
        .true_index
        .clone();
    let mut indices = vec![input];
    if i > 0 {
        indices.push(bonds[i - 1].clone());
    }
    if i + 1 < r {
        indices.push(bonds[i].clone());
    }
    let zero_bit = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
    tensors.push(IdxTensor::from_dense(indices, zero_bit)?);
}
let state = TreeTN::from_tensors(tensors, (0..r).collect())?;

// Apply the QFT with local exact naive apply. This can grow bond dimensions as
// state/operator products, so use it for small exact/debug cases.
let result = apply_linear_operator(&qft_op, &state, ApplyOptions::naive())?;

// This example is intentionally small, so dense verification is acceptable.
// For production-size networks, prefer scalable residual norms or sampled
// `evaluate()` checks rather than `contract_to_tensor()`.
let dense = result.contract_to_tensor()?;
let coefficient = Complex64::new(1.0 / (n as f64).sqrt(), 0.0);
let expected = IdxTensor::from_dense(dense.indices().to_vec(), vec![coefficient; n])?;
let error = dense.sub(&expected)?.maxabs()?;
assert!(error < 1e-6, "maximum coefficient error: {error}");
# Ok(())
# }
```

### Interpreting QFT Output

The QFT output represents the discrete Fourier transform of the input function.
For a function f(x) on N = 2^R points, the k-th Fourier coefficient is:

```text
F(k) = (1/sqrt(N)) * sum_{x=0}^{N-1} f(x) * exp(-2*pi*i*k*x/N)
```

The output is in **bit-reversed** frequency order: the output at site
configuration (b_0, b_1, ..., b_{R-1}) corresponds to frequency index
k = b_{R-1} * 2^{R-1} + ... + b_1 * 2 + b_0 (i.e., the bit-reversal of
b_0 * 2^{R-1} + ... + b_{R-1}).

## Multi-axis QFT

`quantics_fourier_operator_multiaxis` builds a product transform with one
abstract node per selected bit. `AxisBit { axis, bit }` specifies the axis and
input significance (`bit = 0` is the MSB); axis and bit numbers start at zero
without gaps. Each axis has at least two bits. For example, the layout
`[(0,0), (1,0), (0,1), (1,1)]` interleaves two two-bit axes.

```rust
use tensor4all_quanticstransform::{quantics_fourier_operator_multiaxis, AxisBit, FourierOptions};
# fn main() -> Result<(), Box<dyn std::error::Error>> {
let layout = [AxisBit { axis: 0, bit: 0 }, AxisBit { axis: 1, bit: 0 },
              AxisBit { axis: 0, bit: 1 }, AxisBit { axis: 1, bit: 1 }];
let operator = quantics_fourier_operator_multiaxis(&layout, FourierOptions::forward())?;
// The normalized 16-by-16 product has Frobenius norm sqrt(16).
assert!((operator.mpo().clone().norm()? - 4.0).abs() < 1e-10);
# Ok(())
# }
```

Bind the operator's node `j` to the desired selected state index using its
input/output mappings and `apply_linear_operator_to_indices`. Node ordering
is separate from bit significance: the layout can also be grouped or permuted.
Each axis is built with `FourierOptions`. If some axis's bits are not monotone
along the layout, the product is composed in separated axis order and
restructured exactly by site swaps; its intermediate and final bonds can exceed
the per-axis `max_bond_dim`, and reordering adds no intentional truncation.

When every axis's bits are monotone along the layout, as in grouped and
interleaved layouts, the per-axis operators are embedded directly instead: each
site passes the other axes' bonds through unchanged. This is exact and needs no
site swaps; the bond at each cut is the product of the per-axis bonds there. Two
interleaved axes at the default options therefore reach bond `12 * 12 = 144`,
while a grouped layout keeps the per-axis bond `12`.

## Patched QFT on an index subset

Enable `tensor4all-quanticstransform`'s optional `partitioned` feature to use
`patched_fourier_transform`. Pass an existing `PartitionedTreeTN`, a center
node, and each axis's full binary indices in MSB-first order. The selected
indices must lie on one path; they may skip spectator nodes or share a node
with other selected indices or spectators. Branching selections are rejected.
State node names need not be integers.

The input patches must be dyadic leaves of the selected bits: each fixes a
contiguous most-significant prefix of every axis. Spectator constraints may
differ. The scheduler keeps disjoint spectator groups directly; overlapping
constraints use a common coordinate grid of their varying spectator indices.
The grid and refined leaf count are checked against `max_work_items` and
`max_terms` before tensor operations. Each original patch is transformed once,
then its image is restricted exactly to those cells. Output regions retain the
spectator constraints. Complete selected-coordinate coverage is checked in each
represented cell; omitted leaves require `ZeroForMissingLeaves`. Large common
grids can be refused: raise the limits or restrict the producer's patch order
to selected indices. The original application error is charged once globally,
and later errors on disjoint output/spectator cells combine within each level.

`PatchedFourierOptions` defaults to relative reconstruction tolerance `1e-6`
and an exact full-depth schedule. Set `merge_refine.target_bond_dim` for adaptive
compression, and use its output-depth/region and work/term limits as needed.
Leave `coordinate_groups` unset; the front end derives them from the axes.
The returned `MergeRefineResult` keeps output regions and their retained terms
lazy. `into_partition()` requires one final term per region. Requested output
regions select intersecting schedule cells and can return coarser cells.
Repeated exact transforms can grow intermediate bonds substantially. A rank
goal is soft and does not bound the initial operator application, which is
always performed exactly: its bonds are the input bond times the operator bond.
For large transforms set an appropriate output depth/region selection and a
rank goal. The defaults preserve exact behavior and should not
be treated as a scalable memory policy. The small round-trip example uses a
rank goal with a tight reconstruction tolerance.

Output significance reverses **within each axis**. For a normalized inverse
round trip use `axes_inv[a] = axes[a].iter().rev().cloned().collect()` with
`FourierOptions::inverse()`. Reusing the forward axes unchanged is incorrect.
With `normalize: false`, both directions omit scaling and the round trip
multiplies by `2^total_selected_bits`.

`report().error_bound` measures reconstruction against the constructed
operator, excluding operator construction error and backend roundoff. Test
reconstruction with a stated roundoff allowance, and compare to a direct DFT
separately. Neither `FourierOptions::tolerance` nor `normalize` certifies an
exact-DFT error bound for the approximated operator.

The runnable example transforms six k-space patches into 2D r-space
regions, checks all sixteen values against a direct DFT, and verifies the
reversed-axes inverse call. It intentionally uses a tiny dense input/oracle.

```bash
./scripts/check-fourier-dependencies.sh --run
```

The complete [example source](https://github.com/tensor4all/tensor4all-rs/blob/main/crates/tensor4all-quanticstransform/examples/patched_fourier.rs)
is also executed by the guide doctest harness.

Future input generators such as issue #1276 may return a TT over only a patch's
remaining bits. Before forming `SubDomainTreeTN`, restore the fixed prefix bits
as one-hot sites, split any fused levels, and use the same full index identities
and named topology for every patch. A prefix projector alone cannot introduce
missing site indices.

## Large periodic-Gaussian validation

`patched_fourier_large` builds shifted periodic Gaussians by single-scale
interpolative QTT construction, cuts input prefix patches and transforms them
without enumerating a dense grid. InterpolativeQTT returns the input as TreeTN
chains, and the patched Fourier transform takes the resulting TreeTN patches
without a SimpleTT conversion.

The Gaussian is defined by a finite Fourier series with known discrete
coefficients, so its DFT oracle has no continuous-FT
quadrature or aliasing approximation. The 2D function is the product of the two
shifted Gaussians. Inputs and outputs are checked at a fixed seeded set of
random points, including nonzero low frequencies and both frequency signs.

The example uses explicit input/layout rank policies, reconstruction tolerance
and output depth. These are example choices, not a certified construction-error
bound or a new default. Its output reports input/operator bond dimensions,
operator construction time, isolated scheduler time, front-end time (which also
includes operator/target construction), and sampled error alongside the
reconstruction bound. The arguments are the bits per axis, the number of axes
and the layout; a fourth argument `frontend-only` skips the isolated scheduler
run. The small dense examples above remain the primary layout and
bit-significance regression checks.

```bash
RAYON_NUM_THREADS=1 cargo run --release -p tensor4all-quanticstransform --features partitioned --example patched_fourier_large -- 10 1 grouped
RAYON_NUM_THREADS=1 cargo run --release -p tensor4all-quanticstransform --features partitioned --example patched_fourier_large -- 10 2 grouped
```

A small R=10 1D smoke check runs in the ordinary suite. The full R=10/20 1D and
2D grouped cases and the R=10 2D interleaved case live in
`heavy_patched_fourier.rs` and run in the scheduled heavy-test workflow. Run
them locally in release mode:

```bash
RAYON_NUM_THREADS=1 cargo test --release -p tensor4all-quanticstransform --features partitioned --test heavy -- --ignored --nocapture --test-threads=1
```

Two interleaved axes work at R=10 but are slow. The operator itself is cheap
(bond 144, built in well under a second at R=20), but the scheduler applies it
exactly to each input patch as its error reference. That application keeps
every bond at its physical bound, at most 1024 for the R=10 interleaved
Gaussian, and the case completes with a 2.4 GiB peak; it takes about 6
minutes with one thread, almost all of it in those exact applications, which
factorize by QR rather than SVD (issue #1285). At
R=20 the exact application itself can be 50 x 144 = 7200 wide, so that case
is still out of reach. Prefer a grouped layout for patched multi-axis
transforms where the layout is free (issue #1279):

```bash
RAYON_NUM_THREADS=1 cargo run --release -p tensor4all-quanticstransform --features partitioned --example patched_fourier_large -- 10 2 interleaved frontend-only
```
