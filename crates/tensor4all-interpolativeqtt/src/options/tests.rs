use super::*;

#[test]
fn default_matches_julia_style_unbounded_compression() {
    let opts = InterpolativeQttOptions::default();

    assert_eq!(opts.tolerance, 1.0e-12);
    assert_eq!(opts.max_bond_dim, None);
}

#[test]
fn builder_methods_update_only_requested_fields() {
    let opts = InterpolativeQttOptions::default().with_tolerance(1.0e-8);

    assert_eq!(opts.tolerance, 1.0e-8);
    assert_eq!(opts.max_bond_dim, None);

    let capped = opts.with_max_bond_dim(12);

    assert_eq!(capped.tolerance, 1.0e-8);
    assert_eq!(capped.max_bond_dim, Some(12));
}
