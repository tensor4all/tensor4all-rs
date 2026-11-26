//! Tests of the error helpers.

use crate::TreeAciError;

fn callback() -> TreeAciError {
    TreeAciError::Callback {
        message: "stop".into(),
    }
}

/// A successful rollback leaves the step's error as it was.
#[test]
fn after_a_successful_rollback_the_error_is_unchanged() {
    let error = callback().after_rollback(Ok(()));
    assert!(matches!(error, TreeAciError::Callback { message } if message == "stop"));
}

/// A failed rollback is attached to the step's error instead of replacing it.
#[test]
fn a_failed_rollback_is_attached_to_the_error() {
    let error = callback().after_rollback(Err(TreeAciError::InternalInvariant {
        message: "checkpoint",
    }));
    let text = error.to_string();
    match error {
        TreeAciError::RollbackFailed { error, rollback } => {
            assert!(matches!(*error, TreeAciError::Callback { .. }));
            assert!(matches!(
                *rollback,
                TreeAciError::InternalInvariant {
                    message: "checkpoint"
                }
            ));
        }
        other => panic!("unexpected error {other:?}"),
    }
    assert_eq!(
        text,
        "tree ACI callback failed: stop; restoring the state afterwards also failed: \
         tree ACI internal invariant failed: checkpoint"
    );
}
