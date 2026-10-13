use std::error::Error;

use super::*;

fn assert_display(error: PartitionedTreeTNError, expected: &str) {
    assert_eq!(error.to_string(), expected);
}

#[test]
fn every_variant_has_the_documented_display() {
    assert_display(
        PartitionedTreeTNError::OverlappingProjectors,
        "projectors overlap; use disjoint projector keys or replace the exact key",
    );
    assert_display(
        PartitionedTreeTNError::ProjectorConflict,
        "projector entries conflict; use compatible coordinates",
    );
    assert_display(
        PartitionedTreeTNError::TopologyMismatch,
        "TreeTNs have different named topologies; use the same node names and edges",
    );
    assert_display(
        PartitionedTreeTNError::SiteIndexMismatch,
        "TreeTNs have different site-index assignments; reindex explicitly before the operation",
    );
    assert_display(
        PartitionedTreeTNError::ProjectorMismatch,
        "strict subdomain addition requires identical projector keys",
    );
    assert_display(
        PartitionedTreeTNError::InvalidCenter,
        "the requested TreeTN center is absent; choose an existing node name",
    );
    assert_display(
        PartitionedTreeTNError::InvalidOptions {
            operation: "merge",
            reason: "cutoff must be nonnegative",
        },
        "invalid merge options: cutoff must be nonnegative",
    );
    assert_display(
        PartitionedTreeTNError::ResourceLimit {
            operation: "merge-refine scheduling",
            limit: "max_terms",
            value: 2048,
        },
        "merge-refine scheduling reached its max_terms limit of 2048; raise the limit or reduce the request",
    );
    assert_display(
        PartitionedTreeTNError::VolumeOverflow,
        "partition volume overflowed usize; use smaller site dimensions or fewer patches",
    );
    assert_display(
        PartitionedTreeTNError::LogicalParameterCountOverflow,
        "logical tensor parameter count overflowed usize; use smaller tensors or bond dimensions",
    );
    assert_display(
        PartitionedTreeTNError::NonFiniteAdaptiveValue,
        "adaptive patching encountered a non-finite norm or truncation budget",
    );
    assert_display(
        PartitionedTreeTNError::Empty,
        "partitioned TreeTN is empty; add at least one patch",
    );

    let index = DynIndex::new_dyn(3);
    let index_debug = format!("{index:?}");
    let error = PartitionedTreeTNError::ProjectorIndexNotFound { index };
    assert!(error.to_string().contains(&index_debug));

    let index = DynIndex::new_dyn(3);
    let index_debug = format!("{index:?}");
    let error = PartitionedTreeTNError::ProjectorCoordinateOutOfBounds {
        index,
        value: 4,
        dim: 3,
    };
    let message = error.to_string();
    assert!(message.contains(&index_debug));
    assert!(message.contains("4") && message.contains("dimension 3"));

    let tree_error = TreeTNOperationError::from(anyhow::anyhow!("disconnected"));
    assert_display(
        PartitionedTreeTNError::InvalidTopology { source: tree_error },
        "invalid TreeTN topology: TreeTN operation failed: disconnected",
    );
    assert_display(
        PartitionedTreeTNError::DTypeMismatch {
            expected: "f64".to_string(),
            actual: "c64".to_string(),
        },
        "TreeTN tensors have mixed scalar dtypes: expected f64, found c64",
    );
    assert_display(
        PartitionedTreeTNError::UnsupportedDType {
            dtype: "f32".to_string(),
        },
        "unsupported IdxTensor scalar dtype: f32",
    );

    let tree_error = TreeTNOperationError::from(anyhow::anyhow!("contraction failed"));
    assert_display(
        PartitionedTreeTNError::TreeTN { source: tree_error },
        "TreeTN operation failed: TreeTN operation failed: contraction failed",
    );
    assert_display(
        PartitionedTreeTNError::TensorStorage {
            source: TensorStorageError::UnsupportedDtype { dtype: "f32" },
        },
        "tensor storage operation failed: compact IdxTensor storage does not support dtype f32; the eager payload remains authoritative",
    );
    assert_display(
        PartitionedTreeTNError::TensorConstruction {
            source: anyhow::anyhow!("backend unavailable"),
        },
        "tensor construction or backend operation failed: backend unavailable",
    );
}

#[test]
fn conversions_preserve_typed_sources_and_messages() {
    let tree_source = TreeTNOperationError::from(anyhow::anyhow!("tree failure"));
    let error = PartitionedTreeTNError::from(tree_source);
    assert!(matches!(error, PartitionedTreeTNError::TreeTN { .. }));
    assert_eq!(
        error.to_string(),
        "TreeTN operation failed: TreeTN operation failed: tree failure"
    );
    assert!(Error::source(&error).is_some());

    let storage_source = TensorStorageError::UnsupportedDtype { dtype: "f32" };
    let error = PartitionedTreeTNError::from(IdxTensorError::Storage {
        source: storage_source,
    });
    assert!(matches!(
        error,
        PartitionedTreeTNError::TensorStorage { .. }
    ));
    assert!(error.to_string().contains("compact IdxTensor storage"));

    let error = PartitionedTreeTNError::from(IdxTensorError::NaNInput { operation: "norm" });
    assert!(matches!(
        error,
        PartitionedTreeTNError::TensorConstruction { .. }
    ));
    assert!(error
        .to_string()
        .contains("IdxTensor norm received NaN input"));
    assert!(Error::source(&error).is_some());
}

#[test]
fn private_helpers_wrap_the_requested_diagnostics() {
    let tree = PartitionedTreeTNError::tree("tensor lookup failed");
    assert!(matches!(tree, PartitionedTreeTNError::TreeTN { .. }));
    assert_eq!(
        tree.to_string(),
        "TreeTN operation failed: TreeTN operation failed: tensor lookup failed"
    );
    assert!(Error::source(&tree).is_some());

    let topology_source = TreeTNOperationError::from(anyhow::anyhow!("cycle"));
    let topology = PartitionedTreeTNError::invalid_topology(topology_source);
    assert!(matches!(
        topology,
        PartitionedTreeTNError::InvalidTopology { .. }
    ));
    assert_eq!(
        topology.to_string(),
        "invalid TreeTN topology: TreeTN operation failed: cycle"
    );
    assert!(Error::source(&topology).is_some());
}
