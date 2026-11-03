//! Fixed-rank RSVD and a two-site MPO product with one continuous RNG stream.
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use tensor4all_core::{
    randomized_svd, DynIndex, FactorizeAlg, IdxTensor, RsvdOptions, TensorContractionLike,
};
use tensor4all_treetn::{
    contraction::{contract_with_rng, ContractionOptions},
    TreeTN,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = ChaCha8Rng::seed_from_u64(1108);
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(2);
    let matrix = IdxTensor::from_dense(vec![i.clone(), j], vec![3., 0., 0., 0.])?;
    let result = randomized_svd(&matrix, &[i], &RsvdOptions::new(1), &mut rng)?;
    let reconstructed = result
        .u
        .contract_pair(&result.s)?
        .contract_pair(&result.vh)?;
    assert!(matrix.sub(&reconstructed)?.maxabs()? < 1e-12);

    let shared: Vec<_> = (0..2).map(|_| DynIndex::new_dyn(2)).collect();
    let make = || -> Result<TreeTN<IdxTensor, usize>, Box<dyn std::error::Error>> {
        let bond = DynIndex::new_dyn(1);
        let tensors = shared
            .iter()
            .map(|site| {
                IdxTensor::from_dense(
                    vec![site.clone(), DynIndex::new_dyn(2), bond.clone()],
                    vec![1.; 4],
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(TreeTN::from_tensors(tensors, vec![0, 1])?)
    };
    let (a, b) = (make()?, make()?);
    for options in [
        ContractionOptions::zipup(),
        ContractionOptions::fit().with_nfullsweeps(2),
    ] {
        let options = options
            .with_factorize_alg(FactorizeAlg::RSVD)
            .with_max_bond_dim(1);
        let result = contract_with_rng(&a, &b, &0, options, &mut rng)?;
        // Each shared two-dimensional site contributes 2, hence 2 * 2 = 4.
        assert!(result
            .to_dense()?
            .to_vec::<f64>()?
            .iter()
            .all(|&value| (value - 4.).abs() < 1e-11));
    }
    Ok(())
}
