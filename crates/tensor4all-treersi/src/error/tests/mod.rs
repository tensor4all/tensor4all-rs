use super::TreeRsiError;
use std::error::Error as _;

#[test]
fn numerical_retains_the_backend_failure_as_its_source() {
    let error = TreeRsiError::numerical("backend rejected the factorization");
    assert!(matches!(error, TreeRsiError::Numerical { .. }));
    assert_eq!(
        error.to_string(),
        "tree RSI numerical operation failed: backend rejected the factorization"
    );
    assert_eq!(
        error.source().map(|source| source.to_string()).as_deref(),
        Some("backend rejected the factorization")
    );
}
