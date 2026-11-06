use super::invariant_violated;

#[test]
fn invariant_violated_reports_context_and_cause() {
    let failure = std::panic::catch_unwind(|| {
        invariant_violated("tensor invariant", "broken state");
    });
    let message = failure
        .unwrap_err()
        .downcast::<String>()
        .map(|message| *message)
        .unwrap_or_default();
    assert_eq!(message, "tensor invariant: broken state");
}
