# Affine Transformation

An affine transformation evaluates a function at shifted or mixed coordinates.
The tutorial uses the pullback point of view: to get the new value at an output
point, look up the old function at the transformed input point.

Runnable source: [`docs/tutorial-code/src/bin/qtt_affine.rs`](../../../../tutorial-code/src/bin/qtt_affine.rs)

## Key API Pieces

`AffineParams` stores the matrix and offset. Boundary conditions say what
happens when transformed coordinates leave the grid. The operator is
applied via `tensor_train_to_treetn`, `align_to_state`, and
`apply_linear_operator`.

```rust
# fn main() -> anyhow::Result<()> {
# use tensor4all_core::TensorIndex;
# use tensor4all_quanticstci::{
#     quanticscrossinterpolate_discrete, QtciOptions, UnfoldingScheme,
# };
# use tensor4all_quanticstransform::{
#     affine_operator, AffineParams, BoundaryCondition,
# };
# use tensor4all_treetn::{apply_linear_operator, tensor_train_to_treetn, ApplyOptions};
let bits = 3;
let source_grid = [8usize, 8];
let source_options = QtciOptions::default()
    .with_nrandominitpivot(0)
    .with_unfoldingscheme(UnfoldingScheme::Fused)
    .with_verbosity(0);
let pivots = vec![vec![1_i64, 1], vec![8, 8]];

let (source, _, _) = quanticscrossinterpolate_discrete::<f64, _>(
    &source_grid,
    |idx| idx[0] as f64 + 2.0 * idx[1] as f64,
    Some(pivots),
    source_options,
)?;

let params = AffineParams::from_integers(vec![1, 1, 0, 1], vec![0, 0], 2, 2)?;
let mut operator = affine_operator(bits, &params, &[BoundaryCondition::Periodic; 2])?
    .transpose();

let source_tt = source.tensor_train();
let (state, _indices) = tensor_train_to_treetn(&source_tt)?;

operator.align_to_state(&state)?;
let result = apply_linear_operator(&operator, &state, ApplyOptions::naive())?;

let external = TensorIndex::external_indices(&result);
assert_eq!(external.len(), 3);
# Ok(())
# }
```

The full tutorial repeats the same workflow for two boundary conditions and
compares periodic and open results.

## What It Computes

The example builds a two-dimensional QTT, creates an affine operator, applies
it to the QTT, and compares the transformed values with a direct reference.

![Affine transformed values](qtt_affine_values.png)

![Affine transformation error](qtt_affine_error.png)

The operator has its own bond dimensions, and the transformed QTT has another
set. Both are useful when judging the cost of the operation.

![Bond dimensions after the affine transformation](qtt_affine_bond_dims.png)

![Bond dimensions of the affine operator](qtt_affine_operator_bond_dims.png)
