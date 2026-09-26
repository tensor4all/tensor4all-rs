//! Deterministic random tree fixtures shared by unit tests and diagnostics.

use tensor4all_core::{DynIndex, IdxTensor, IndexLike};
use tensor4all_treetn::TreeTN;

/// Explicit splitmix64 stream, so the fixture does not depend on `rand`.
pub(crate) struct SplitMix(pub(crate) u64);

impl SplitMix {
    /// Uniform value in `[-1, 1)`.
    pub(crate) fn next_signed(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        2.0 * ((z >> 11) as f64 / (1u64 << 53) as f64) - 1.0
    }

    /// Uniform real and imaginary parts in `[-1, 1)` (real part only for real `T`).
    pub(crate) fn next_scalar<T: crate::TreeAciScalar>(&mut self) -> T {
        let (re, im) = (self.next_signed(), self.next_signed());
        T::from_evaluated_scalar(tensor4all_core::AnyScalar::new_complex(re, im))
            .or_else(|_| T::from_evaluated_scalar(tensor4all_core::AnyScalar::new_real(re)))
            .unwrap()
    }
}

/// Deterministic tree whose bond component `k` is weighted by `decay^k` at
/// each edge's second node, so cross-interpolation pivots decay geometrically.
pub(crate) fn random_decaying_tree<T: crate::TreeAciScalar>(
    edges: &[(usize, usize)],
    physical: &[DynIndex],
    bond: usize,
    decay: f64,
    rng: &mut SplitMix,
) -> TreeTN<IdxTensor, usize> {
    let bonds = edges
        .iter()
        .map(|_| DynIndex::new_dyn(bond))
        .collect::<Vec<_>>();
    let tensors = physical
        .iter()
        .enumerate()
        .map(|(node, site)| {
            let mut indices = vec![site.clone()];
            for (edge, &(left, right)) in edges.iter().enumerate() {
                if left == node || right == node {
                    indices.push(bonds[edge].clone());
                }
            }
            let weighted_axes = edges
                .iter()
                .filter(|&&(left, right)| left == node || right == node)
                .map(|&(_, right)| right == node)
                .collect::<Vec<_>>();
            let len = indices.iter().map(IndexLike::dim).product();
            let values = (0..len)
                .map(|flat| {
                    // Column-major: physical first, then the incident bonds.
                    let mut rest = flat / site.dim();
                    let mut weight = 1.0;
                    for &weighted in &weighted_axes {
                        if weighted {
                            weight *= decay.powi((rest % bond) as i32);
                        }
                        rest /= bond;
                    }
                    rng.next_scalar::<T>() * <T as tensor4all_core::Scalar>::from_f64(weight)
                })
                .collect();
            IdxTensor::from_dense(indices, values).unwrap()
        })
        .collect();
    TreeTN::from_tensors(tensors, (0..physical.len()).collect()).unwrap()
}
