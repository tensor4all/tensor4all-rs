//! A sketched chain product checked against a complete dense oracle.
use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_treersi::{hadamard_many, TreeRsiOptions};
use tensor4all_treetn::TreeTN;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sites: Vec<_> = (0..7).map(|_| DynIndex::new_dyn(2)).collect();
    let bonds: Vec<_> = (0..6).map(|_| DynIndex::new_dyn(2)).collect();
    let mut cores = Vec::new();
    for v in 0..7 {
        let mut indices = vec![sites[v].clone()];
        if v > 0 {
            indices.push(bonds[v - 1].clone());
        }
        if v < 6 {
            indices.push(bonds[v].clone());
        }
        // Column-major core entries; nonconstant bond-two input.
        let data = (0..(1 << indices.len()))
            .map(|j| 0.3 + ((j * 13 + v * 7) % 17) as f64 / 20.0)
            .collect();
        cores.push(IdxTensor::from_dense(indices, data)?);
    }
    let input = TreeTN::from_tensors(cores, (0usize..7).collect())?;
    let dense = input.to_dense()?.permute_indices(&sites)?.to_vec::<f64>()?;
    let expected = IdxTensor::from_dense(sites, dense.iter().map(|x| x * x).collect())?;
    let options = TreeRsiOptions {
        max_bond_dim: Some(4),
        sketch_dim: Some(3),
        seed: 7,
        ..Default::default()
    };
    let result = hadamard_many::<f64, _>(&[input.clone(), input], &options)?;
    assert!(result
        .diagnostics
        .edges
        .iter()
        .any(|edge| !edge.exact_columns));
    let error = result.tree.to_dense()?.sub(&expected)?.maxabs()? / expected.maxabs()?;
    assert!(error < 1e-10, "actual product relative max error: {error}");
    Ok(())
}
