
use super::*;
use crate::mpo::types::tensor4_zeros;

#[test]
fn test_contract_site_tensors() {
    // Create two simple 4D tensors
    let mut a: Tensor4<f64> = tensor4_zeros(1, 2, 3, 1);
    let mut b: Tensor4<f64> = tensor4_zeros(1, 3, 2, 1);

    // Fill with some values
    for s1 in 0..2 {
        for s2 in 0..3 {
            a.set4(0, s1, s2, 0, (s1 * 3 + s2 + 1) as f64);
        }
    }
    for s1 in 0..3 {
        for s2 in 0..2 {
            b.set4(0, s1, s2, 0, (s1 * 2 + s2 + 1) as f64);
        }
    }

    let result = contract_site_tensors(&a, &b).unwrap();

    // Result should have shape (1, 2, 2, 1)
    assert_eq!(result.left_dim(), 1);
    assert_eq!(result.site_dim_1(), 2);
    assert_eq!(result.site_dim_2(), 2);
    assert_eq!(result.right_dim(), 1);
}

#[test]
fn test_left_environment() {
    let mpo_a = MPO::<f64>::constant(&[(2, 2), (2, 2)], 1.0);
    let mpo_b = MPO::<f64>::constant(&[(2, 2), (2, 2)], 1.0);

    let mut cache = Vec::new();

    // Left environment at site 0 should be [[1]]
    let env0 = left_environment(&mpo_a, &mpo_b, 0, &mut cache).unwrap();
    assert_eq!(env0[[0, 0]], 1.0);

    // Left environment at site 1
    let env1 = left_environment(&mpo_a, &mpo_b, 1, &mut cache).unwrap();
    // Each element contributes 1*1 = 1, sum over 2*2=4 physical indices
    assert!((env1[[0, 0]] - 4.0).abs() < 1e-10);
}

#[test]
fn test_right_environment() {
    let mpo_a = MPO::<f64>::constant(&[(2, 2), (2, 2)], 1.0);
    let mpo_b = MPO::<f64>::constant(&[(2, 2), (2, 2)], 1.0);

    let mut cache = Vec::new();

    // Right environment at last site should be [[1]]
    let env_last = right_environment(&mpo_a, &mpo_b, 1, &mut cache).unwrap();
    assert_eq!(env_last[[0, 0]], 1.0);

    // Right environment at site 0
    let env0 = right_environment(&mpo_a, &mpo_b, 0, &mut cache).unwrap();
    // Each element contributes 1*1 = 1, sum over 2*2=4 physical indices
    assert!((env0[[0, 0]] - 4.0).abs() < 1e-10);
}
