use super::*;

#[test]
fn test_contract_dispatch_naive() {
    let mpo_a = MPO::<f64>::identity(&[2, 2]).unwrap();
    let mpo_b = MPO::<f64>::identity(&[2, 2]).unwrap();
    let options = ContractionOptions::default();

    let result = contract(&mpo_a, &mpo_b, ContractionAlgorithm::Naive, &options).unwrap();
    assert_eq!(result.len(), 2);
}

#[test]
fn test_contract_dispatch_zipup() {
    let mpo_a = MPO::<f64>::identity(&[2, 2]).unwrap();
    let mpo_b = MPO::<f64>::identity(&[2, 2]).unwrap();
    let options = ContractionOptions::default();

    let result = contract(&mpo_a, &mpo_b, ContractionAlgorithm::ZipUp, &options).unwrap();
    assert_eq!(result.len(), 2);
}

#[test]
fn test_contract_dispatch_fit() {
    let mpo_a = MPO::<f64>::identity(&[2, 2]).unwrap();
    let mpo_b = MPO::<f64>::identity(&[2, 2]).unwrap();
    let options = ContractionOptions::default();

    let result = contract(&mpo_a, &mpo_b, ContractionAlgorithm::Fit, &options).unwrap();
    assert_eq!(result.len(), 2);
}

#[test]
fn test_contract_algorithms_consistent() {
    // All algorithms should give the same result for simple case
    // constant takes &[(s1, s2)] - tuple per site
    let mpo_a = MPO::<f64>::constant(&[(2, 2)], 2.0);
    let mpo_b = MPO::<f64>::constant(&[(2, 2)], 3.0);
    let options = ContractionOptions::default();

    let result_naive = contract(&mpo_a, &mpo_b, ContractionAlgorithm::Naive, &options).unwrap();
    let result_zipup = contract(&mpo_a, &mpo_b, ContractionAlgorithm::ZipUp, &options).unwrap();
    let result_fit = contract(&mpo_a, &mpo_b, ContractionAlgorithm::Fit, &options).unwrap();

    // Check all algorithms give the same result
    // evaluate takes &[LocalIndex] - flat list, 2 per site
    let val_naive = result_naive.evaluate(&[0, 0]).unwrap();
    let val_zipup = result_zipup.evaluate(&[0, 0]).unwrap();
    let val_fit = result_fit.evaluate(&[0, 0]).unwrap();

    // All algorithms should give the same result
    assert!((val_naive - val_zipup).abs() < 1e-10);
    assert!((val_naive - val_fit).abs() < 1e-10);
}
