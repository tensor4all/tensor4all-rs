use num_complex::Complex64;

use super::TreeAciOptions;

#[test]
fn default_message_cache_budget_is_bounded() {
    let options = TreeAciOptions::<usize>::default();

    assert_eq!(options.message_cache_max_bytes, 256 << 20);
}

/// The derived ceilings must reproduce the constants they replaced, so a
/// caller who never touches `max_working_bytes` sees no change.
#[test]
fn default_element_ceilings_match_the_historical_constants() {
    let options = TreeAciOptions::<usize>::default();

    assert_eq!(options.max_local_matrix_elements, None);
    assert_eq!(options.max_core_elements, None);
    assert_eq!(options.max_frame_elements, None);
    assert_eq!(options.resolved_max_local_matrix_elements::<f64>(), 1 << 24);
    assert_eq!(options.resolved_max_core_elements::<f64>(), 1 << 24);
    assert_eq!(options.resolved_max_frame_elements::<f64>(), 1 << 24);
}

/// Issue #729: a caller who raises the byte budget must actually get it.
#[test]
fn derived_ceilings_follow_the_working_budget() {
    let options = TreeAciOptions::<usize> {
        max_working_bytes: 10 << 30,
        ..TreeAciOptions::default()
    };

    let expected = (10usize << 30) / 4 / size_of::<f64>();
    assert_eq!(
        options.resolved_max_local_matrix_elements::<f64>(),
        expected
    );
    assert_eq!(options.resolved_max_core_elements::<f64>(), expected);
    assert_eq!(options.resolved_max_frame_elements::<f64>(), expected);
    // The 17,958,192-element local matrix of the reported failure fits a
    // 10 GiB budget, which is what the caller asked for.
    assert!(expected > 17_958_192);
}

/// The ceilings measure bytes, so a complex run gets half the elements and
/// the same memory.
#[test]
fn derived_ceilings_charge_the_run_scalar() {
    let options = TreeAciOptions::<usize>::default();

    assert_eq!(
        options.resolved_max_core_elements::<f64>() * size_of::<f64>(),
        options.resolved_max_core_elements::<Complex64>() * size_of::<Complex64>()
    );
}

/// An explicit ceiling is a pin: it overrides the budget in both
/// directions and does not move when the budget does.
#[test]
fn explicit_ceilings_override_the_derived_value() {
    let mut options = TreeAciOptions::<usize> {
        max_local_matrix_elements: Some(7),
        max_core_elements: Some(1 << 30),
        ..TreeAciOptions::default()
    };

    assert_eq!(options.resolved_max_local_matrix_elements::<f64>(), 7);
    assert_eq!(options.resolved_max_core_elements::<f64>(), 1 << 30);
    options.max_working_bytes = 1 << 40;
    assert_eq!(options.resolved_max_local_matrix_elements::<f64>(), 7);
    assert_eq!(options.resolved_max_core_elements::<f64>(), 1 << 30);
    assert_eq!(
        options.resolved_max_frame_elements::<f64>(),
        (1usize << 40) / 4 / size_of::<f64>()
    );
}

/// A budget below one scalar still derives a usable ceiling; the working
/// budget itself is what refuses the allocation.
#[test]
fn derived_ceiling_never_reaches_zero() {
    let options = TreeAciOptions::<usize> {
        max_working_bytes: 1,
        ..TreeAciOptions::default()
    };

    assert_eq!(options.resolved_max_local_matrix_elements::<f64>(), 1);
}
