
use super::*;
use crate::mpo::factorize::FactorizeMethod;

#[test]
fn test_contract_naive_identity() {
    // Identity * Identity = Identity
    let mpo_a = MPO::<f64>::identity(&[2, 2]).unwrap();
    let mpo_b = MPO::<f64>::identity(&[2, 2]).unwrap();

    let result = contract_naive(&mpo_a, &mpo_b, None).unwrap();

    assert_eq!(result.len(), 2);

    // The result should be equivalent to identity
    // Check some evaluations
    assert!((result.evaluate(&[0, 0, 0, 0]).unwrap() - 1.0).abs() < 1e-10);
    assert!((result.evaluate(&[0, 1, 0, 0]).unwrap()).abs() < 1e-10);
    assert!((result.evaluate(&[1, 1, 1, 1]).unwrap() - 1.0).abs() < 1e-10);
}

#[test]
fn test_contract_naive_constant() {
    // Constant * Constant
    let mpo_a = MPO::<f64>::constant(&[(2, 2)], 2.0);
    let mpo_b = MPO::<f64>::constant(&[(2, 2)], 3.0);

    let result = contract_naive(&mpo_a, &mpo_b, None).unwrap();

    // Each element of C = sum over k of A[i, k] * B[k, j]
    // = sum over k of 2 * 3 = 6 * (number of k values = 2) = 12
    assert_eq!(result.len(), 1);
    let val = result.evaluate(&[0, 0]).unwrap();
    assert!((val - 12.0).abs() < 1e-10);
}

#[test]
fn test_contract_naive_dimension_mismatch() {
    let mpo_a = MPO::<f64>::constant(&[(2, 3)], 1.0); // s2 = 3
    let mpo_b = MPO::<f64>::constant(&[(2, 2)], 1.0); // s1 = 2 ≠ 3

    let result = contract_naive(&mpo_a, &mpo_b, None);
    assert!(result.is_err());
}

#[test]
fn test_contract_naive_with_compression() {
    let mpo_a = MPO::<f64>::constant(&[(2, 2), (2, 2)], 1.0);
    let mpo_b = MPO::<f64>::constant(&[(2, 2), (2, 2)], 1.0);

    let options = ContractionOptions {
        tolerance: 1e-10,
        max_bond_dim: 2,
        factorize_method: FactorizeMethod::SVD,
    };

    let result = contract_naive(&mpo_a, &mpo_b, Some(options)).unwrap();

    // Bond dimension should be compressed
    assert!(result.rank() <= 2);
}
