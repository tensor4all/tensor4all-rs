
use super::super::types::tensor4_from_data;
use super::*;

fn sample_tensor4(scale: f64) -> Tensor4<f64> {
    tensor4_from_data(vec![scale, scale + 1.0], 1, 1, 2, 1)
}

#[test]
fn test_inverse_mpo_accessors_and_rank() {
    let mpo = InverseMPO::from_parts_unchecked(
        vec![sample_tensor4(1.0), sample_tensor4(3.0)],
        vec![vec![2.0, 4.0]],
    );

    assert_eq!(mpo.len(), 2);
    assert!(!mpo.is_empty());
    assert_eq!(mpo.site_dims(), vec![(1, 2), (1, 2)]);
    assert_eq!(mpo.link_dims(), vec![2]);
    assert_eq!(mpo.rank(), 2);
    assert_eq!(mpo.inv_lambda(0), &[2.0, 4.0]);
    assert_eq!(mpo.inv_lambdas(), &[vec![2.0, 4.0]]);
    assert_eq!(*mpo.site_tensor(0).get4(0, 0, 1, 0), 2.0);
    assert_eq!(mpo.site_tensors().len(), 2);
}

#[test]
fn test_inverse_mpo_rank_for_single_site_is_one_and_mutation_works() {
    let mut mpo = InverseMPO::from_parts_unchecked(vec![sample_tensor4(5.0)], Vec::new());
    assert_eq!(mpo.rank(), 1);
    assert!(mpo.link_dims().is_empty());

    *mpo.site_tensor_mut(0).get4_mut(0, 0, 0, 0) = 9.0;
    assert_eq!(*mpo.site_tensor(0).get4(0, 0, 0, 0), 9.0);
}

#[test]
fn test_inverse_mpo_placeholder() {
    // Placeholder test - actual tests will be added when implementation is complete
    let mpo = MPO::<f64>::constant(&[(2, 2), (2, 2)], 1.0);
    let result = InverseMPO::from_mpo(mpo);
    // Currently returns error as not implemented
    assert!(result.is_err());
}

#[test]
fn test_inverse_mpo_into_mpo_is_currently_unimplemented() {
    let mpo = InverseMPO::from_parts_unchecked(vec![sample_tensor4(1.0)], Vec::new());
    assert!(matches!(
        mpo.into_mpo(),
        Err(MPOError::InvalidOperation { .. })
    ));
}
