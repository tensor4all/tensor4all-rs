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

#[test]
fn from_treetci_error_preserves_source() {
    let e = QuanticsTCIError::from(tensor4all_treetci::TreeTciError::from(anyhow::anyhow!(
        "treetci root cause"
    )));
    assert!(e.to_string().contains("quantics TCI operation failed"));
    let source = <dyn std::error::Error>::source(&e).unwrap().to_string();
    assert!(source.contains("tree-TCI operation failed: treetci root cause"));
}
