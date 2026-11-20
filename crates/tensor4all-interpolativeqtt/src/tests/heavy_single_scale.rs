//! Higher-degree approximation evidence, separate from the exact polynomial oracle.

#[test]
#[ignore = "heavy: runs in the scheduled heavy-tests workflow (see CONTRIBUTING.md)"]
fn single_scale_interpolation_n3() {
    super::assert_single_scale_interpolation_n3(15, |x| {
        (-x[0] * x[0] - x[1].powi(3) - 2.0 * x[2] * x[2]).exp()
    });
}
