
use super::*;

#[test]
fn test_canonicalization_options_default() {
    let opts = CanonicalizationOptions::default();
    assert_eq!(opts.form, CanonicalForm::Unitary);
    assert!(!opts.force);
}

#[test]
fn test_canonicalization_options_new() {
    let opts = CanonicalizationOptions::new();
    assert_eq!(opts.form, CanonicalForm::Unitary);
    assert!(!opts.force);
}

#[test]
fn test_canonicalization_options_forced() {
    let opts = CanonicalizationOptions::forced();
    assert!(opts.force);
}

#[test]
fn test_canonicalization_options_builder() {
    let opts = CanonicalizationOptions::new()
        .with_form(CanonicalForm::LU)
        .force();
    assert_eq!(opts.form, CanonicalForm::LU);
    assert!(opts.force);

    let opts = opts.smart();
    assert!(!opts.force);
}

#[test]
fn test_truncation_options_default() {
    let opts = TruncationOptions::default();
    assert_eq!(opts.form, CanonicalForm::Unitary);
    assert_eq!(opts.rtol(), None);
    assert_eq!(opts.max_rank(), None);
}

#[test]
fn test_truncation_options_new() {
    let opts = TruncationOptions::new();
    assert_eq!(opts.form, CanonicalForm::Unitary);
}

#[test]
fn test_truncation_options_builder() {
    let opts = TruncationOptions::new()
        .with_max_rank(50)
        .with_rtol(1e-10)
        .with_form(CanonicalForm::Unitary);
    assert_eq!(opts.max_rank(), Some(50));
    assert_eq!(opts.rtol(), Some(1e-10));
    assert_eq!(opts.form, CanonicalForm::Unitary);
}

#[test]
fn test_truncation_options_has_truncation_params() {
    let mut opts = TruncationOptions::new();
    assert!(opts.truncation_params().rtol.is_none());
    opts.truncation_params_mut().rtol = Some(1e-8);
    assert_eq!(opts.truncation_params().rtol, Some(1e-8));
}

#[test]
fn test_split_options_default() {
    let opts = SplitOptions::default();
    assert_eq!(opts.form, CanonicalForm::Unitary);
    assert!(!opts.final_sweep);
    assert_eq!(opts.rtol(), None);
    assert_eq!(opts.max_rank(), None);
}

#[test]
fn test_split_options_new() {
    let opts = SplitOptions::new();
    assert_eq!(opts.form, CanonicalForm::Unitary);
}

#[test]
fn test_split_options_builder() {
    let opts = SplitOptions::new()
        .with_max_rank(100)
        .with_rtol(1e-12)
        .with_form(CanonicalForm::Unitary)
        .with_final_sweep(true);
    assert_eq!(opts.max_rank(), Some(100));
    assert_eq!(opts.rtol(), Some(1e-12));
    assert_eq!(opts.form, CanonicalForm::Unitary);
    assert!(opts.final_sweep);
}

#[test]
fn test_split_options_has_truncation_params() {
    let mut opts = SplitOptions::new();
    assert!(opts.truncation_params().max_rank.is_none());
    opts.truncation_params_mut().max_rank = Some(25);
    assert_eq!(opts.truncation_params().max_rank, Some(25));
}
