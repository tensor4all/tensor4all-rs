use super::*;
use tensor4all_simplett::AbstractTensorTrain;

#[test]
fn test_binary_coeffs_valid() {
    assert!(BinaryCoeffs::new(1, 1).is_ok());
    assert!(BinaryCoeffs::new(1, -1).is_ok());
    assert!(BinaryCoeffs::new(-1, 1).is_ok());
    assert!(BinaryCoeffs::new(0, 1).is_ok());
    assert!(BinaryCoeffs::new(1, 0).is_ok());
    assert!(BinaryCoeffs::new(0, 0).is_ok());
}

#[test]
fn test_binary_coeffs_invalid() {
    assert!(BinaryCoeffs::new(-1, -1).is_err());
    assert!(BinaryCoeffs::new(2, 0).is_err());
    assert!(BinaryCoeffs::new(0, 2).is_err());
}

#[test]
fn test_binaryop_tensor_single() {
    // Test (1, 1) coefficients (sum)
    let tensor = binaryop_tensor_single(1, 1, false, false, 1);
    assert_eq!(tensor.dims(), &[1, 1, 2, 2, 2]); // (cin_size, cout_size, x, y, out)

    // x=0, y=0: result=0, out=0
    assert_eq!(tensor[[0, 0, 0, 0, 0]], Complex64::one());
    // x=0, y=1: result=1, out=1
    assert_eq!(tensor[[0, 0, 0, 1, 1]], Complex64::one());
    // x=1, y=0: result=1, out=1
    assert_eq!(tensor[[0, 0, 1, 0, 1]], Complex64::one());
    // x=1, y=1: result=2, out=0 (with carry)
    assert_eq!(tensor[[0, 0, 1, 1, 0]], Complex64::one());
}

#[test]
fn test_binaryop_tensor_difference() {
    // Test (1, -1) coefficients (difference)
    let tensor = binaryop_tensor_single(1, -1, false, false, 1);

    // x=0, y=0: result=0, out=0
    assert_eq!(tensor[[0, 0, 0, 0, 0]], Complex64::one());
    // x=0, y=1: result=-1, out=1 (|−1| mod 2 = 1)
    assert_eq!(tensor[[0, 0, 0, 1, 1]], Complex64::one());
    // x=1, y=0: result=1, out=1
    assert_eq!(tensor[[0, 0, 1, 0, 1]], Complex64::one());
    // x=1, y=1: result=0, out=0
    assert_eq!(tensor[[0, 0, 1, 1, 0]], Complex64::one());
}

#[test]
fn test_binaryop_single_mpo_structure() {
    let mpo = binaryop_single_mpo(4, 1, 1, BoundaryCondition::Periodic).unwrap();
    // 4 bits per variable × 2 variables = 8 sites
    assert_eq!(mpo.len(), 8);
}

#[test]
fn test_binaryop_single_operator_creation() {
    let op = binaryop_single_operator(4, 1, 1, BoundaryCondition::Periodic);
    assert!(op.is_ok());
}

#[test]
fn test_binaryop_single_identity() {
    // a=1, b=0 should be similar to identity on x
    let mpo = binaryop_single_mpo(2, 1, 0, BoundaryCondition::Periodic).unwrap();
    assert_eq!(mpo.len(), 4);
}

#[test]
fn test_binaryop_error_cases() {
    assert!(binaryop_single_mpo(0, 1, 1, BoundaryCondition::Periodic).is_err());
    assert!(binaryop_single_mpo(4, 2, 0, BoundaryCondition::Periodic).is_err());
    assert!(binaryop_single_mpo(4, -1, -1, BoundaryCondition::Periodic).is_err());
}
