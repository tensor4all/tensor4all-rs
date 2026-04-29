# Elementwise Product

This tutorial multiplies two functions after both have been represented as
QTTs. Elementwise means that values at the same grid point are multiplied:
`h(x_i) = f(x_i) g(x_i)`.

Runnable source: [`docs/tutorial-code/src/bin/qtt_elementwise_product.rs`](../../../../tutorial-code/src/bin/qtt_elementwise_product.rs)

## Key API Pieces

The first step is simply to build two QTTs on the same grid. Converting
to TreeTN enables `partial_contract` with `diagonal_pairs` for the
pointwise product.

```rust
# fn main() -> anyhow::Result<()> {
# use tensor4all_quanticstci::{quanticscrossinterpolate_discrete, QtciOptions};
# use tensor4all_treetn::{
#     contraction::ContractionOptions,
#     partial_contract, tensor_train_to_treetn, PartialContractionSpec,
# };
let sizes = [8usize];
let f_a = |idx: &[i64]| -> f64 { (idx[0] as f64).powi(2) };
let f_b = |idx: &[i64]| -> f64 { (10.0 * (idx[0] as f64 - 1.0) / 8.0).sin() };
let options = QtciOptions::default()
    .with_nrandominitpivot(0)
    .with_verbosity(0);
let pivots = vec![vec![1_i64], vec![8]];

let (qtt_a, _, _) = quanticscrossinterpolate_discrete::<f64, _>(
    &sizes, f_a, Some(pivots.clone()), options.clone(),
)?;
let (qtt_b, _, _) = quanticscrossinterpolate_discrete::<f64, _>(
    &sizes, f_b, Some(pivots), options,
)?;

let tt_a = qtt_a.tensor_train();
let tt_b = qtt_b.tensor_train();
let (tn_a, site_indices_a) = tensor_train_to_treetn(&tt_a)?;
let (tn_b, site_indices_b) = tensor_train_to_treetn(&tt_b)?;

let diagonal_pairs: Vec<_> = site_indices_a
    .iter()
    .cloned()
    .zip(site_indices_b.iter().cloned())
    .collect();
let spec = PartialContractionSpec {
    contract_pairs: vec![],
    diagonal_pairs,
    output_order: Some(site_indices_a.clone()),
};
let center = tn_a.node_names()[0];
let product = partial_contract(
    &tn_a, &tn_b, &spec, &center, ContractionOptions::default(),
)?;

assert_eq!(product.node_count(), 3);
# Ok(())
# }
```

The important condition is that both QTTs use compatible grids, so that a site
in one QTT refers to the same grid bit as the paired site in the other QTT.

## What It Computes

The example builds two QTTs, converts them to TreeTN form, pairs matching grid
sites, and contracts those pairs via `partial_contract` to form the product.

![Input factors for the elementwise product](qtt_elementwise_product_factors.png)

![Elementwise product compared with direct values](qtt_elementwise_product_product.png)

The product may need larger bond dimensions than either factor alone, because
it carries information from both inputs.

![Bond dimensions for the elementwise product](qtt_elementwise_product_bond_dims.png)
