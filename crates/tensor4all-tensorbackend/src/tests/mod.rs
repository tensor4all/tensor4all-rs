use super::require_invariant;

#[test]
fn require_invariant_returns_success_and_reports_failure_context() {
    assert_eq!(require_invariant::<_, &str>(Ok(7), "valid state"), 7);

    let failure = std::panic::catch_unwind(|| {
        require_invariant::<(), _>(Err("broken state"), "tensor invariant")
    });
    let message = failure
        .unwrap_err()
        .downcast::<String>()
        .map(|message| *message)
        .unwrap_or_default();
    assert!(message.contains("tensor invariant: broken state"));
}
