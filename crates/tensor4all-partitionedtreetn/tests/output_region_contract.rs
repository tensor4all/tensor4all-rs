//! Public-surface guard for the accepted contract of #1254 (drift fixed in #1284):
//! `output_regions` is a static bit-prefix list of `OutputRegion`, with no
//! `Projector`, no predicate and no caller-implemented trait, and the report
//! carries `pruned_regions` apart from `error_bound`.
//!
//! Each assertion is a compile-time type check, so reintroducing the old
//! `Option<Vec<Projector>>` shape (or dropping the coverage counter) fails the
//! build of this test, not just a runtime comparison.

use tensor4all_partitionedtreetn::reconstruction::{
    CoverageContract, MergeRefineOptions, MergeRefineReport, OutputRegion,
};

/// Accepts only `Option<Vec<OutputRegion>>`; any other field type is a type error.
fn output_regions_type(options: &MergeRefineOptions<f64>) -> &Option<Vec<OutputRegion>> {
    &options.output_regions
}

/// Accepts only a `usize` coverage counter, distinct from the `f64` accuracy bound.
fn coverage_and_accuracy(report: &MergeRefineReport) -> (usize, f64) {
    (report.pruned_regions, report.error_bound)
}

fn assert_plain_data<T: Clone + Eq + std::hash::Hash + Ord + std::fmt::Debug + Send + Sync>() {}

#[test]
fn output_regions_is_a_static_bit_prefix_list() {
    assert_plain_data::<OutputRegion>();
    let options = MergeRefineOptions::<f64> {
        output_regions: Some(vec![OutputRegion::new(2, 0b10), OutputRegion::new(2, 0b11)]),
        coverage: CoverageContract::Complete,
        ..Default::default()
    };
    let regions = output_regions_type(&options).as_deref().unwrap();
    assert_eq!(regions[0].axes(), &[(2, 0b10)]);
    assert_eq!(regions[1], OutputRegion::new(2, 0b11));
    // The default computes every region.
    assert!(output_regions_type(&MergeRefineOptions::default()).is_none());
    // Plain data: ordered, so a replayed list can be compared with a logged one.
    assert!(regions[0] < regions[1]);
}

#[test]
fn report_separates_coverage_from_accuracy() {
    let _: fn(&MergeRefineReport) -> (usize, f64) = coverage_and_accuracy;
}
