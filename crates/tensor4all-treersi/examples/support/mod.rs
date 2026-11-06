//! Deterministic synthetic inputs shared by tests and the benchmark.
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;
use tensor4all_core::{DynIndex, IdxTensor, IndexLike};
use tensor4all_treersi::TreeRsiScalar;
use tensor4all_treetn::TreeTN;
type Tree = TreeTN<IdxTensor, usize>;
pub fn random_tree<T: TreeRsiScalar>(
    sites: &[DynIndex],
    edges: &[(usize, usize)],
    bond_dim: usize,
    seed: u64,
    sample: impl Fn(f64, f64) -> T,
) -> Tree {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let bonds = edges
        .iter()
        .map(|&(a, b)| ((a.min(b), a.max(b)), DynIndex::new_dyn(bond_dim)))
        .collect::<HashMap<_, _>>();
    let tensors = (0..sites.len())
        .map(|v| {
            let mut neighbors = edges
                .iter()
                .filter_map(|&(a, b)| {
                    if a == v {
                        Some(b)
                    } else if b == v {
                        Some(a)
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();
            neighbors.sort();
            let mut indices = neighbors
                .iter()
                .map(|&w| bonds[&(v.min(w), v.max(w))].clone())
                .collect::<Vec<_>>();
            indices.push(sites[v].clone());
            let len = indices.iter().map(|i| i.dim()).product();
            let data = (0..len)
                .map(|_| sample(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)))
                .collect();
            IdxTensor::from_dense(indices, data).unwrap()
        })
        .collect();
    TreeTN::from_tensors(tensors, (0..sites.len()).collect()).unwrap()
}
