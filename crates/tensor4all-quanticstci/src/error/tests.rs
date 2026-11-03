use super::*;

#[test]
fn variant_displays_are_stable() {
    assert_eq!(
        QuanticsTCIError::InvalidConfiguration {
            message: "grid must be a power of two".to_string(),
        }
        .to_string(),
        "invalid quantics configuration: grid must be a power of two"
    );
    assert_eq!(
        QuanticsTCIError::DiscreteGridRequired.to_string(),
        "original coordinates are only available for discretized grids"
    );
    assert_eq!(
        QuanticsTCIError::from(anyhow::anyhow!("backend failed")).to_string(),
        "quantics TCI operation failed: backend failed"
    );
}

#[test]
fn from_conversions_preserve_source() {
    let e = QuanticsTCIError::from(anyhow::anyhow!("root cause"));
    assert_eq!(
        <dyn std::error::Error>::source(&e).unwrap().to_string(),
        "root cause"
    );
}
