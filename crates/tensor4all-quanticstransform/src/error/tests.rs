use super::*;

#[test]
fn variant_displays_are_stable() {
    assert_eq!(
        QuanticsTransformError::InvalidConfiguration {
            message: "theta must be finite".to_string(),
        }
        .to_string(),
        "invalid quantics transform configuration: theta must be finite"
    );
    assert_eq!(
        QuanticsTransformError::from(anyhow::anyhow!("backend failed")).to_string(),
        "quantics transform operation failed: backend failed"
    );
}

#[test]
fn from_conversions_preserve_source() {
    let e = QuanticsTransformError::from(anyhow::anyhow!("root cause"));
    assert_eq!(
        <dyn std::error::Error>::source(&e).unwrap().to_string(),
        "root cause"
    );

    let dyn_err = QuanticsTransformError::from(tensor4all_core::IdxTensorError::NaNInput {
        operation: "test",
    });
    assert!(dyn_err
        .to_string()
        .contains("quantics transform operation failed"));

    let tree_err = QuanticsTransformError::from(tensor4all_treetn::TreeTNOperationError::from(
        anyhow::anyhow!("tree root"),
    ));
    assert!(tree_err
        .to_string()
        .contains("quantics transform operation failed"));
}
