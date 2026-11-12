# Quantum Fourier Transform

This guide demonstrates how to apply the quantics Fourier transform (QFT)
operator to tensor trains. It corresponds to the Julia
[Quantum Fourier Transform](https://tensor4all.org/T4APlutoExamples/qft.html) notebook.

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

Before working with TCI-constructed states, here is a minimal example that
creates a QFT operator and verifies its structure.

```rust
use tensor4all_quanticstransform::{quantics_fourier_operator, FourierOptions, FTCore};

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

This example applies the QFT to a product state |0> (the uniform function)
and verifies that the result has uniform magnitude 1/sqrt(N) at all
frequency components -- a well-known property of the DFT.

```rust
use num_complex::Complex64;
use num_traits::{One, Zero};
use tensor4all_core::TensorIndex;
use tensor4all_simplett::{types::tensor3_zeros, AbstractTensorTrain, Tensor3Ops, SimpleTensorTrain};
use tensor4all_treetn::{apply_linear_operator, ApplyOptions, tensor_train_to_treetn};
use tensor4all_quanticstransform::{quantics_fourier_operator, FourierOptions};

let r = 3;
let n = 1usize << r; // 8

// Create the |0> product state as a SimpleTensorTrain
// |0> = |0> x |0> x ... x |0>  (all bits zero)
let mut tensors = Vec::new();
for _ in 0..r {
    let mut t = tensor3_zeros(1, 2, 1).unwrap();
    t.set3(0, 0, 0, Complex64::one()); // bit = 0
    tensors.push(t);
}
let mps = SimpleTensorTrain::new(tensors).unwrap();

// Convert MPS to TreeTN
let (treetn, site_indices) = tensor_train_to_treetn(&mps).unwrap();

// Create the forward QFT operator
let qft_op = quantics_fourier_operator(r, FourierOptions::forward()).unwrap();

// Replace TreeTN site indices with operator input indices
let mut state = treetn;
for i in 0..r {
    let op_input = qft_op
        .get_input_mapping(&i)
        .unwrap()
        .true_index
        .clone();
    state = state.replaceind(&site_indices[i], &op_input).unwrap();
}

// Apply the QFT with local exact naive apply. This can grow bond dimensions as
// state/operator products, so use it for small exact/debug cases.
let result = apply_linear_operator(&qft_op, &state, ApplyOptions::naive()).unwrap();

// The result should exist and have the same number of nodes
assert_eq!(result.node_count(), r);

// This example is intentionally small, so dense verification is acceptable.
// For production-size networks, prefer scalable residual norms or sampled
// `evaluate()` checks rather than `contract_to_tensor()`.
let dense = result.contract_to_tensor().unwrap();
let data = dense.to_vec::<Complex64>().unwrap();

// For |0> input, all Fourier coefficients should have magnitude 1/sqrt(N)
let expected_mag = 1.0 / (n as f64).sqrt();
for val in &data {
    let mag = val.norm();
    assert!((mag - expected_mag).abs() < 1e-6,
        "Expected magnitude {}, got {}", expected_mag, mag);
}
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
cargo run -p tensor4all-quanticstransform --features partitioned --example patched_fourier
```

The complete [example source](https://github.com/tensor4all/tensor4all-rs/blob/main/crates/tensor4all-quanticstransform/examples/patched_fourier.rs)
is also executed by the guide doctest harness.

Future input generators such as issue #1276 may return a TT over only a patch's
remaining bits. Before forming `SubDomainTreeTN`, restore the fixed prefix bits
as one-hot sites (through the existing simplett bridge), split any fused levels,
and use the same full index identities and named topology for every patch. A
prefix projector alone cannot introduce missing site indices. Input generation
is separate from this Fourier API.

## Large periodic-Gaussian validation

`patched_fourier_large` builds shifted periodic Gaussians by single-scale
interpolative QTT construction, cuts input prefix patches and transforms them
without enumerating a dense grid. The Gaussian is defined by a finite Fourier
series with known discrete coefficients, so its DFT oracle has no continuous-FT
quadrature or aliasing approximation. The 2D function is the product of the two
shifted Gaussians. Inputs and outputs are checked at a fixed seeded set of
random points, including nonzero low frequencies and both frequency signs.

The example uses explicit input/layout rank policies, reconstruction tolerance
and output depth. These are example choices, not a certified construction-error
bound or a new default. Its output reports input/operator bond dimensions,
operator construction time, isolated scheduler time, front-end time (which also
includes operator/target construction), and sampled error alongside the
reconstruction bound. The small dense examples above remain the primary
layout and bit-significance regression checks.

```bash
RAYON_NUM_THREADS=1 cargo run --release -p tensor4all-quanticstransform --features partitioned --example patched_fourier_large -- 10 1 grouped
RAYON_NUM_THREADS=1 cargo run --release -p tensor4all-quanticstransform --features partitioned --example patched_fourier_large -- 10 2 grouped
```

A small R=10 1D smoke check runs in the ordinary suite. The full R=10/20 1D and
2D grouped cases live in `heavy_patched_fourier.rs` and run in the scheduled
heavy-test workflow. Run them locally in release mode:

```bash
RAYON_NUM_THREADS=1 cargo test --release -p tensor4all-quanticstransform --features partitioned --test heavy_patched_fourier -- --ignored --nocapture --test-threads=1
```

Two interleaved axes are not yet practical at this size. The operator itself
is cheap (bond 144, built in well under a second at R=20), but the scheduler
always applies it exactly to each input patch as its error reference, so that
application has bonds up to input bond times operator bond: 50 x 144 = 7200
for the R=10 interleaved Gaussian, which exhausted a 14 GiB limit. Prefer a
grouped layout for patched multi-axis transforms until the scheduler bounds
that application (issue #1279).
