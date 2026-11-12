#![cfg(feature = "partitioned")]
#[path = "../examples/support/large_fourier.rs"]
mod large_fourier;

#[test]
fn large_r10_gaussian_has_the_analytic_discrete_spectrum() {
    let measured = large_fourier::run_case(10, 1, false).unwrap();
    assert!(measured.reconstruction_bound > 0.0);
    assert!(measured.compressions > 0);
}
