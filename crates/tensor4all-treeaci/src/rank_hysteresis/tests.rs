use super::{applies_to_pass, kept_rank, RANK_HYSTERESIS_FIRST_PASS};

// Errors of a search at threshold 1.0 / 4: five pivots, then the residual.
const ERRORS: [f64; 6] = [8.0, 3.0, 1.5, 0.9, 0.3, 0.2];

#[test]
fn growth_follows_the_threshold() {
    // Pivots 8, 3 and 1.5 pass the threshold; a smaller previous rank
    // does not hold the edge back.
    assert_eq!(kept_rank(&ERRORS, 1.0, 1), 3);
    assert_eq!(kept_rank(&ERRORS, 1.0, 3), 3);
}

#[test]
fn previous_rank_keeps_pivots_down_to_the_lowered_threshold() {
    // The plain update would drop to 3; the edge had 4, and pivot 0.9 is
    // above the lowered threshold, so it stays.
    assert_eq!(kept_rank(&ERRORS, 1.0, 4), 4);
    assert_eq!(kept_rank(&ERRORS, 1.0, 5), 5);
}

#[test]
fn floor_never_exceeds_the_lowered_search() {
    assert_eq!(kept_rank(&ERRORS, 1.0, 9), 5);
}

#[test]
fn first_pivot_is_kept_even_below_the_threshold() {
    // The plain update always accepts one pivot.
    assert_eq!(kept_rank(&[0.5, 0.3, 0.1], 1.0, 0), 1);
    assert_eq!(kept_rank(&[0.5, 0.3, 0.1], 1.0, 2), 2);
}

#[test]
fn exhausted_search_keeps_every_pivot_above_the_threshold() {
    // Full rank reached: the trailing residual is zero and every pivot passes.
    assert_eq!(kept_rank(&[4.0, 2.0, 0.0], 1.0, 0), 2);
}

#[test]
fn empty_search_keeps_nothing() {
    assert_eq!(kept_rank(&[0.0], 1.0, 3), 0);
    assert_eq!(kept_rank(&[], 1.0, 3), 0);
}

#[test]
fn applies_from_the_first_hysteresis_pass_on() {
    assert!(!applies_to_pass(1));
    assert!(!applies_to_pass(RANK_HYSTERESIS_FIRST_PASS - 1));
    assert!(applies_to_pass(RANK_HYSTERESIS_FIRST_PASS));
    assert!(applies_to_pass(RANK_HYSTERESIS_FIRST_PASS + 10));
}
