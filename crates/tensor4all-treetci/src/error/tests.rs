use super::*;

#[test]
fn variant_displays_are_stable() {
    assert_eq!(
        TreeTciError::InvalidGraph {
            message: "disconnected".to_string(),
        }
        .to_string(),
        "invalid tree-TCI graph: disconnected"
    );
    assert_eq!(
        TreeTciError::IndexOutOfBounds {
            message: "site 3".to_string(),
        }
        .to_string(),
        "index out of bounds: site 3"
    );
    assert_eq!(
        TreeTciError::from(anyhow::anyhow!("backend failed")).to_string(),
        "tree-TCI operation failed: backend failed"
    );
}

#[test]
fn from_conversions_preserve_source() {
    let e = TreeTciError::from(anyhow::anyhow!("root cause"));
    assert_eq!(
        <dyn std::error::Error>::source(&e).unwrap().to_string(),
        "root cause"
    );

    let dyn_err =
        TreeTciError::from(tensor4all_core::IdxTensorError::NaNInput { operation: "test" });
    assert!(dyn_err.to_string().contains("tree-TCI operation failed"));
}

#[test]
fn from_treetn_operation_error_preserves_source() {
    let e = TreeTciError::from(tensor4all_treetn::TreeTNOperationError::from(
        anyhow::anyhow!("treetn root cause"),
    ));
    assert!(e.to_string().contains("tree-TCI operation failed"));
    let source = <dyn std::error::Error>::source(&e).unwrap().to_string();
    assert!(source.contains("TreeTN operation failed: treetn root cause"));
}
