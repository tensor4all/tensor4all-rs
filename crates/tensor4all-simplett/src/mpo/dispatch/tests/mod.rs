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
fn test_contract_algorithms_consistent() {
    // The available algorithms should give the same result for a simple case.
    // constant takes &[(s1, s2)] - tuple per site
    let mpo_a = MPO::<f64>::constant(&[(2, 2)], 2.0).unwrap();
    let mpo_b = MPO::<f64>::constant(&[(2, 2)], 3.0).unwrap();
    let options = ContractionOptions::default();

    let result_naive = contract(&mpo_a, &mpo_b, ContractionAlgorithm::Naive, &options).unwrap();
    let result_zipup = contract(&mpo_a, &mpo_b, ContractionAlgorithm::ZipUp, &options).unwrap();
    // evaluate takes &[LocalIndex] - flat list, 2 per site
    let val_naive = result_naive.evaluate(&[0, 0]).unwrap();
    let val_zipup = result_zipup.evaluate(&[0, 0]).unwrap();
    assert!((val_naive - val_zipup).abs() < 1e-10);
}
