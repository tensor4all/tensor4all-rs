use super::*;

#[test]
fn certified_failures_map_to_the_scheduler_error_kinds() {
    let memory = certified_failure(CertifiedApplyError::MemoryLimit {
        estimated_bytes: 8,
        limit_bytes: 4,
    });
    assert!(matches!(
        memory,
        PartitionedTreeTNError::ResourceLimit {
            operation: "certified operator application",
            limit: "memory_limit (bytes)",
            value: 8,
        }
    ));

    let budget = certified_failure(CertifiedApplyError::BudgetUnreachable {
        budget: 1e-15,
        step: 2,
        step_bound: 2e-15,
        share: 1e-16,
    });
    assert!(matches!(
        budget,
        PartitionedTreeTNError::InvalidOptions {
            operation: "certified operator application",
            reason,
        } if reason.contains("roundoff level")
    ));

    let topology = certified_failure(CertifiedApplyError::UnsupportedTopology {
        reason: "branched".into(),
    });
    assert!(matches!(
        topology,
        PartitionedTreeTNError::InvalidOptions {
            operation: "certified operator application",
            reason,
        } if reason.contains("chain-shaped")
    ));

    let other = certified_failure(CertifiedApplyError::InvalidOptions {
        reason: "negative budget".into(),
    });
    assert!(matches!(other, PartitionedTreeTNError::TreeTN { .. }));
    assert!(other.to_string().contains("certified operator application"));
}
