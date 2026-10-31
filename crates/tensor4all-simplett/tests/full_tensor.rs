//! #1060 explicitly uses pointwise evaluation as an independent oracle on small grids.
use num_complex::Complex64;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use tensor4all_simplett::mpo::{tensor4_from_data, tensor4_zeros, MPOError, MPO};
use tensor4all_simplett::{
    tensor3_from_data, tensor3_zeros, AbstractTensorTrain, EinsumScalar, SimpleTensorTrain,
    SimpleTensorTrainError, TTScalar,
};

fn compare<T: TTScalar>(data: &[T], shape: &[usize], evaluate: impl Fn(&[usize]) -> T) {
    assert_eq!(data.len(), shape.iter().product::<usize>());
    for (flat, &value) in data.iter().enumerate() {
        let mut remaining = flat;
        let indices: Vec<_> = shape
            .iter()
            .map(|&dim| {
                let index = remaining % dim;
                remaining /= dim;
                index
            })
            .collect();
        let expected = evaluate(&indices);
        let error = (value - expected).abs_sq().sqrt();
        assert!(
            error <= 1e-10 * expected.abs_sq().sqrt().max(1.0),
            "{indices:?}: error {error}"
        );
    }
}

fn random_cases<T: TTScalar + EinsumScalar>(mut sample: impl FnMut(&mut ChaCha8Rng) -> T) {
    let mut rng = ChaCha8Rng::seed_from_u64(1060);
    for dims in [vec![3], vec![2, 3], vec![2, 1, 3, 2, 2], vec![2, 2, 2]] {
        for bond in [1, 3, 8, 0] {
            let cores = dims
                .iter()
                .enumerate()
                .map(|(site, &dim)| {
                    let left = if site == 0 { 1 } else { bond };
                    let right = if site + 1 == dims.len() { 1 } else { bond };
                    let data = (0..left * dim * right).map(|_| sample(&mut rng)).collect();
                    tensor3_from_data(data, left, dim, right).unwrap()
                })
                .collect();
            let tt = SimpleTensorTrain::new(cores).unwrap();
            let (data, shape) = tt.full_tensor().unwrap();
            assert_eq!(shape, dims);
            compare(&data, &shape, |indices| tt.evaluate(indices).unwrap());
        }
    }
    for dims in [
        vec![(2, 3)],
        vec![(2, 3), (3, 2)],
        vec![(2, 1), (1, 3), (2, 2), (3, 1), (1, 2)],
        vec![(2, 1); 3],
    ] {
        for bond in [1, 3, 8, 0] {
            let cores = dims
                .iter()
                .enumerate()
                .map(|(site, &(d1, d2))| {
                    let left = if site == 0 { 1 } else { bond };
                    let right = if site + 1 == dims.len() { 1 } else { bond };
                    let data = (0..left * d1 * d2 * right)
                        .map(|_| sample(&mut rng))
                        .collect();
                    tensor4_from_data(data, left, d1, d2, right).unwrap()
                })
                .collect();
            let mpo = MPO::new(cores).unwrap();
            let (data, shape) = mpo.full_tensor().unwrap();
            assert_eq!(
                shape,
                dims.iter()
                    .flat_map(|&(d1, d2)| [d1, d2])
                    .collect::<Vec<_>>()
            );
            compare(&data, &shape, |indices| mpo.evaluate(indices).unwrap());
        }
    }
}

#[test]
fn random_f64_matches_every_point() {
    random_cases(|rng| rng.random_range(-0.5..0.5));
}

#[test]
fn random_complex_matches_every_point_without_conjugating() {
    random_cases(|rng| Complex64::new(rng.random_range(-0.5..0.5), rng.random_range(-0.5..0.5)));
}

fn nonuniform_bonds<T: TTScalar + EinsumScalar>(mut sample: impl FnMut(&mut ChaCha8Rng) -> T) {
    let mut rng = ChaCha8Rng::seed_from_u64(1065);
    let bonds = [1, 3, 2, 1];
    let tt_dims = [2, 3, 2];
    let tt = SimpleTensorTrain::new(
        tt_dims
            .iter()
            .zip(bonds.windows(2))
            .map(|(&dim, pair)| {
                let data = (0..pair[0] * dim * pair[1])
                    .map(|_| sample(&mut rng))
                    .collect();
                tensor3_from_data(data, pair[0], dim, pair[1]).unwrap()
            })
            .collect(),
    )
    .unwrap();
    let (data, shape) = tt.full_tensor().unwrap();
    assert_eq!(shape, tt_dims);
    compare(&data, &shape, |indices| tt.evaluate(indices).unwrap());

    let mpo_dims = [(2, 3), (3, 2), (2, 2)];
    let mpo = MPO::new(
        mpo_dims
            .iter()
            .zip(bonds.windows(2))
            .map(|(&(d1, d2), pair)| {
                let data = (0..pair[0] * d1 * d2 * pair[1])
                    .map(|_| sample(&mut rng))
                    .collect();
                tensor4_from_data(data, pair[0], d1, d2, pair[1]).unwrap()
            })
            .collect(),
    )
    .unwrap();
    let (data, shape) = mpo.full_tensor().unwrap();
    assert_eq!(shape, [2, 3, 3, 2, 2, 2]);
    compare(&data, &shape, |indices| mpo.evaluate(indices).unwrap());
}

#[test]
fn nonuniform_f64_bonds_match_tt_and_mpo_every_point() {
    nonuniform_bonds(|rng| rng.random_range(-0.5..0.5));
}

#[test]
fn nonuniform_complex_bonds_match_tt_and_mpo_every_point() {
    nonuniform_bonds(|rng| {
        Complex64::new(rng.random_range(-0.5..0.5), rng.random_range(-0.5..0.5))
    });
}

#[test]
fn mutable_first_left_bond_keeps_legacy_folding_order() {
    let core: Vec<_> = (1..=8).map(f64::from).collect();
    let mut tt = SimpleTensorTrain::<f64>::constant(&[2, 2], 1.0);
    tt.site_tensors_mut()[0] = tensor3_from_data(core.clone(), 2, 2, 2).unwrap();
    tt.site_tensors_mut()[1] = tensor3_from_data(core.clone(), 4, 2, 1).unwrap();
    assert_eq!(
        tt.full_tensor().unwrap(),
        (vec![44.0, 64.0, 100.0, 152.0], vec![2, 2])
    );
    let mut mpo = MPO::<f64>::constant(&[(2, 1), (2, 1)], 1.0).unwrap();
    mpo.site_tensors_mut()[0] = tensor4_from_data(core.clone(), 2, 2, 1, 2).unwrap();
    mpo.site_tensors_mut()[1] = tensor4_from_data(core, 4, 2, 1, 1).unwrap();
    assert_eq!(
        mpo.full_tensor().unwrap(),
        (vec![44.0, 64.0, 100.0, 152.0], vec![2, 1, 2, 1])
    );
}

#[test]
fn empty_and_zero_physical_dimensions_preserve_shape() {
    assert_eq!(
        SimpleTensorTrain::<f64>::new(vec![])
            .unwrap()
            .full_tensor()
            .unwrap(),
        (vec![], vec![])
    );
    assert_eq!(
        MPO::<f64>::new(vec![]).unwrap().full_tensor().unwrap(),
        (vec![], vec![])
    );
    for dims in [vec![0], vec![0, 2, 3], vec![2, 0, 3], vec![2, 3, 0]] {
        assert_eq!(
            SimpleTensorTrain::<f64>::zeros(&dims)
                .full_tensor()
                .unwrap(),
            (vec![], dims)
        );
    }
    for dims in [
        vec![(0, 2)],
        vec![(2, 0)],
        vec![(0, 2), (3, 2)],
        vec![(2, 3), (2, 0)],
    ] {
        let shape: Vec<_> = dims.iter().flat_map(|&(d1, d2)| [d1, d2]).collect();
        assert_eq!(
            MPO::<f64>::zeros(&dims).unwrap().full_tensor().unwrap(),
            (vec![], shape)
        );
    }
}

#[test]
fn checked_product_keeps_zero_and_overflow_order() {
    let overflow = vec![2; usize::BITS as usize];
    for trailing_zero in [false, true] {
        let mut dims = overflow.clone();
        if trailing_zero {
            dims.push(0);
        }
        assert!(
            matches!(SimpleTensorTrain::<f64>::zeros(&dims).full_tensor(), Err(SimpleTensorTrainError::InvalidOperation { message }) if message == "full tensor element count overflowed usize")
        );
        let mpo_dims: Vec<_> = dims.iter().map(|&d| (d, 1)).collect();
        assert!(
            matches!(MPO::<f64>::zeros(&mpo_dims).unwrap().full_tensor(), Err(MPOError::InvalidOperation { message }) if message == "full tensor element count overflowed usize")
        );
    }
    let mut dims = overflow;
    dims.insert(0, 0);
    assert_eq!(
        SimpleTensorTrain::<f64>::zeros(&dims)
            .full_tensor()
            .unwrap(),
        (vec![], dims.clone())
    );
    let mpo_dims: Vec<_> = dims.iter().map(|&d| (d, 1)).collect();
    let shape: Vec<_> = mpo_dims.iter().flat_map(|&(d1, d2)| [d1, d2]).collect();
    assert_eq!(
        MPO::<f64>::zeros(&mpo_dims).unwrap().full_tensor().unwrap(),
        (vec![], shape)
    );
}

#[test]
fn environment_overflow_is_checked_even_with_zero_storage_cores() {
    let wide = 1usize << (usize::BITS - 4);
    let bonds = [1, 0, wide, 0, 1];
    let dims = [16, 1, 1, 1];
    // Every core has a zero bond, so even the huge logical bond requires no payload.
    let tt = SimpleTensorTrain::<f64>::new(
        dims.iter()
            .zip(bonds.windows(2))
            .map(|(&dim, pair)| tensor3_from_data(vec![], pair[0], dim, pair[1]).unwrap())
            .collect(),
    )
    .unwrap();
    assert!(
        matches!(tt.full_tensor(), Err(SimpleTensorTrainError::InvalidOperation { message })
        if message == "full tensor environment element count overflowed usize")
    );
    let mpo = MPO::<f64>::new(
        dims.iter()
            .zip(bonds.windows(2))
            .map(|(&dim, pair)| tensor4_from_data(vec![], pair[0], dim, 1, pair[1]).unwrap())
            .collect(),
    )
    .unwrap();
    assert!(
        matches!(mpo.full_tensor(), Err(MPOError::InvalidOperation { message })
        if message == "full tensor environment element count overflowed usize")
    );
}

#[test]
fn mixed_bonds_can_resume_in_place_updates() {
    let bonds = [1, 2, 1, 1, 2, 1];
    let cores: Vec<_> = bonds
        .windows(2)
        .enumerate()
        .map(|(site, pair)| {
            (0..pair[0] * 2 * pair[1])
                .map(|i| (i + site) as f64 / 8.0 - 0.5)
                .collect::<Vec<_>>()
        })
        .collect();
    let tt = SimpleTensorTrain::new(
        cores
            .iter()
            .zip(bonds.windows(2))
            .map(|(data, pair)| tensor3_from_data(data.clone(), pair[0], 2, pair[1]).unwrap())
            .collect(),
    )
    .unwrap();
    let mpo = MPO::new(
        cores
            .into_iter()
            .zip(bonds.windows(2))
            .map(|(data, pair)| tensor4_from_data(data, pair[0], 2, 1, pair[1]).unwrap())
            .collect(),
    )
    .unwrap();
    let (data, shape) = tt.full_tensor().unwrap();
    compare(&data, &shape, |indices| tt.evaluate(indices).unwrap());
    let (mpo_data, mpo_shape) = mpo.full_tensor().unwrap();
    compare(&mpo_data, &mpo_shape, |indices| {
        mpo.evaluate(indices).unwrap()
    });
    assert_eq!(data, mpo_data);
}

#[test]
fn broken_bonds_do_not_index_past_the_environment() {
    let mut tt = SimpleTensorTrain::<f64>::constant(&[2, 3], 1.0);
    tt.site_tensors_mut()[1] = tensor3_zeros(2, 3, 1).unwrap();
    assert!(matches!(
        tt.full_tensor(),
        Err(SimpleTensorTrainError::DimensionMismatch { site: 0 })
    ));
    let mut mpo = MPO::<f64>::constant(&[(2, 1), (3, 2)], 1.0).unwrap();
    mpo.site_tensors_mut()[1] = tensor4_zeros(2, 3, 2, 1).unwrap();
    assert!(matches!(
        mpo.evaluate(&[0; 4]),
        Err(MPOError::InvalidOperation { .. })
    ));
    assert!(matches!(
        mpo.full_tensor(),
        Err(MPOError::InvalidOperation { .. })
    ));
}

#[test]
fn a_narrower_left_bond_is_rejected() {
    // A trailing tensor with a left bond narrower than its predecessor's right bond would
    // silently ignore the remaining environment states, so it has to be reported instead.
    let mut tt = SimpleTensorTrain::<f64>::constant(&[2, 3], 1.0);
    tt.site_tensors_mut()[0] = tensor3_zeros(1, 2, 2).unwrap();
    tt.site_tensors_mut()[1] = tensor3_zeros(1, 3, 1).unwrap();
    assert!(matches!(
        tt.full_tensor(),
        Err(SimpleTensorTrainError::DimensionMismatch { site: 0 })
    ));
}

#[test]
fn a_non_scalar_final_environment_is_still_an_error() {
    let mut tt = SimpleTensorTrain::<f64>::constant(&[2, 3], 1.0);
    tt.site_tensors_mut()[1] = tensor3_zeros(1, 3, 2).unwrap();
    assert!(matches!(
        tt.evaluate(&[0, 0]),
        Err(SimpleTensorTrainError::InvalidOperation { .. })
    ));
    assert!(
        matches!(tt.full_tensor(), Err(SimpleTensorTrainError::InvalidOperation { message }) if message.contains("expected 1"))
    );
    let mut mpo = MPO::<f64>::constant(&[(2, 1), (3, 2)], 1.0).unwrap();
    mpo.site_tensors_mut()[1] = tensor4_zeros(1, 3, 2, 2).unwrap();
    assert!(matches!(
        mpo.evaluate(&[0; 4]),
        Err(MPOError::InvalidOperation { .. })
    ));
    assert!(
        matches!(mpo.full_tensor(), Err(MPOError::InvalidOperation { message }) if message.contains("expected 1"))
    );
}
