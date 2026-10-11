//! Tests of the report accessors on hand-built reports, one per variant.

use super::{
    GlobalL2Error, L2ErrorReport, L2ReferenceSource, MaxReferenceSource, NormReport,
    ToleranceNotMetBasis,
};

fn certified(rms_error: f64) -> GlobalL2Error {
    GlobalL2Error::Certified {
        rms_error,
        rounding_allowance_rms: None,
        rounding_limited: None,
        relative_error_bound: None,
    }
}

fn error_report(domain_points: f64, global: GlobalL2Error) -> L2ErrorReport {
    L2ErrorReport {
        domain_points,
        global,
        certified_fraction: 1.0,
        approximation_rms: None,
    }
}

fn l2_norm(domain_points: f64, tau: f64, reference_rms: Option<f64>) -> NormReport {
    NormReport::L2 {
        reference_rms,
        source: L2ReferenceSource::Given,
        tau,
        error: error_report(domain_points, certified(0.5)),
    }
}

fn sampled_max() -> NormReport {
    NormReport::SampledMax {
        max_reference: 2.0,
        source: MaxReferenceSource::Given,
        engine_tolerance: 1e-3,
    }
}

#[test]
fn rms_value_reads_the_value_of_every_variant() {
    let variants = [
        (certified(0.25), 0.25),
        (
            GlobalL2Error::Audited {
                rms_error_estimate: 0.5,
                mean_square_rel_std_error: 0.1,
                relative_bound_estimate: None,
            },
            0.5,
        ),
        (
            GlobalL2Error::AcceptanceOnly {
                acceptance_statistic_rms: 0.75,
            },
            0.75,
        ),
        (
            GlobalL2Error::ToleranceNotMet {
                measured_rms: 1.5,
                unmet_fraction: 0.5,
                basis: ToleranceNotMetBasis::AcceptanceOnly,
            },
            1.5,
        ),
    ];
    for (global, expected) in variants {
        assert_eq!(global.rms_value(), expected, "{global:?}");
    }
}

#[test]
fn error_norm_scales_the_rms_by_the_domain_size() {
    // ||f - f~|| = sqrt(|X|) * rms.
    assert_eq!(error_report(16.0, certified(0.5)).error_norm(), Some(2.0));
    assert_eq!(
        error_report(f64::MAX, certified(f64::MAX)).error_norm(),
        None
    );
}

#[test]
fn approximation_norm_needs_a_finite_approximation_rms() {
    let mut report = error_report(9.0, certified(0.0));
    assert_eq!(report.approximation_norm(), None);
    report.approximation_rms = Some(2.0);
    assert_eq!(report.approximation_norm(), Some(6.0));
    report.approximation_rms = Some(f64::INFINITY);
    assert_eq!(report.approximation_norm(), None);
}

#[test]
fn l2_accessors_read_l2_reports() {
    let norm = l2_norm(4.0, 0.25, Some(3.0));
    let error = norm.l2_error().expect("an L2 report has an error report");
    assert_eq!(error.domain_points, 4.0);
    assert_eq!(error.global.rms_value(), 0.5);
    // delta = sqrt(|X|) * tau and ||f|| = sqrt(|X|) * reference_rms.
    assert_eq!(norm.delta(), Some(0.5));
    assert_eq!(norm.reference_norm(), Some(6.0));
}

#[test]
fn l2_accessors_return_none_without_a_finite_value() {
    assert_eq!(l2_norm(4.0, 0.25, None).reference_norm(), None);
    let overflowing = l2_norm(f64::MAX, f64::MAX, Some(f64::MAX));
    assert_eq!(overflowing.delta(), None);
    assert_eq!(overflowing.reference_norm(), None);
}

#[test]
fn l2_accessors_return_none_under_sampled_max() {
    let norm = sampled_max();
    assert!(norm.l2_error().is_none());
    assert_eq!(norm.delta(), None);
    assert_eq!(norm.reference_norm(), None);
}
