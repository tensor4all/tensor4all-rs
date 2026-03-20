
use super::*;

#[test]
fn test_unfolding_scheme_default() {
    assert_eq!(UnfoldingScheme::default(), UnfoldingScheme::Fused);
}
