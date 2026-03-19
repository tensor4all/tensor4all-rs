
use super::*;

#[test]
fn test_default_options() {
    let opts = QtciOptions::default();
    assert!((opts.tolerance - 1e-8).abs() < 1e-15);
    assert!(opts.maxbonddim.is_none());
    assert_eq!(opts.maxiter, 200);
    assert_eq!(opts.nrandominitpivot, 5);
    assert_eq!(opts.unfoldingscheme, UnfoldingScheme::Interleaved);
    assert_eq!(opts.nsearchglobalpivot, 5);
    assert_eq!(opts.nsearch, 100);
    assert_eq!(opts.pivot_search, PivotSearchStrategy::Full);
}

#[test]
fn test_builder_pattern() {
    let opts = QtciOptions::default()
        .with_tolerance(1e-6)
        .with_maxbonddim(100)
        .with_maxiter(50)
        .with_nsearchglobalpivot(10)
        .with_nsearch(200)
        .with_pivot_search(PivotSearchStrategy::Rook);

    assert!((opts.tolerance - 1e-6).abs() < 1e-15);
    assert_eq!(opts.maxbonddim, Some(100));
    assert_eq!(opts.maxiter, 50);
    assert_eq!(opts.nsearchglobalpivot, 10);
    assert_eq!(opts.nsearch, 200);
    assert_eq!(opts.pivot_search, PivotSearchStrategy::Rook);
}

#[test]
fn test_to_tci2_options() {
    let opts = QtciOptions::default()
        .with_tolerance(1e-6)
        .with_maxbonddim(100)
        .with_nsearchglobalpivot(10)
        .with_nsearch(200);

    let tci_opts = opts.to_tci2_options();
    assert!((tci_opts.tolerance - 1e-6).abs() < 1e-15);
    assert_eq!(tci_opts.max_bond_dim, 100);
    assert_eq!(tci_opts.max_nglobal_pivot, 10);
    assert_eq!(tci_opts.nsearch, 200);
}
