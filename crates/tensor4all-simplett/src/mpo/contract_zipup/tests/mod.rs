use super::*;
use crate::mpo::factorize::FactorizeMethod;

#[test]
fn test_contract_zipup_identity() {
    // Identity * Identity = Identity
    let mpo_a = MPO::<f64>::identity(&[2, 2]).unwrap();
    let mpo_b = MPO::<f64>::identity(&[2, 2]).unwrap();

    let options = ContractionOptions {
        tolerance: 1e-12,
        max_bond_dim: 10,
        factorize_method: FactorizeMethod::SVD,
    };

    let result = contract_zipup(&mpo_a, &mpo_b, &options).unwrap();

    assert_eq!(result.len(), 2);

    // The result should be equivalent to identity
    assert!((result.evaluate(&[0, 0, 0, 0]).unwrap() - 1.0).abs() < 1e-10);
    assert!((result.evaluate(&[0, 1, 0, 0]).unwrap()).abs() < 1e-10);
    assert!((result.evaluate(&[1, 1, 1, 1]).unwrap() - 1.0).abs() < 1e-10);
}

#[test]
fn test_contract_zipup_constant() {
    let mpo_a = MPO::<f64>::constant(&[(2, 2)], 2.0);
    let mpo_b = MPO::<f64>::constant(&[(2, 2)], 3.0);

    let options = ContractionOptions::default();

    let result = contract_zipup(&mpo_a, &mpo_b, &options).unwrap();

    // Each element of C = sum over k of A[i, k] * B[k, j]
    // = sum over k of 2 * 3 = 6 * 2 = 12
    let val = result.evaluate(&[0, 0]).unwrap();
    assert!((val - 12.0).abs() < 1e-10);
}

#[test]
fn test_contract_zipup_compresses() {
    // Create MPOs with higher bond dimensions
    let mpo_a = MPO::<f64>::constant(&[(2, 2), (2, 2)], 1.0);
    let mpo_b = MPO::<f64>::constant(&[(2, 2), (2, 2)], 1.0);

    let options = ContractionOptions {
        tolerance: 1e-12,
        max_bond_dim: 2,
        factorize_method: FactorizeMethod::SVD,
    };

    let result = contract_zipup(&mpo_a, &mpo_b, &options).unwrap();

    // Bond dimension should be limited
    assert!(result.rank() <= 2);
}
