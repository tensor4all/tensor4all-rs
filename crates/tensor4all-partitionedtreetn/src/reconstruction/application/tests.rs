use super::*;

#[test]
fn certified_failures_map_to_the_scheduler_error_kinds() {
    let memory = certified_failure(CertifiedApplyError::MemoryLimit {
        estimated_bytes: 9_705_853_056,
        limit_bytes: 4_294_967_296,
    });
    assert!(matches!(
        memory,
        PartitionedTreeTNError::CertifiedMemoryLimit {
            operation: "certified operator application",
            estimated_bytes: 9_705_853_056,
            limit_bytes: 4_294_967_296,
        }
    ));
    // The message names both numbers and never calls the estimate the limit.
    let message = memory.to_string();
    assert!(message.contains("estimated 9705853056 bytes"), "{message}");
    assert!(
        message.contains("memory_limit = 4294967296 bytes"),
        "{message}"
    );
    assert!(!message.contains("limit of 9705853056"), "{message}");

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
