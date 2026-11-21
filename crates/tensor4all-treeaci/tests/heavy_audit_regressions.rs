//! Broad geometry and seed evidence; ordinary tests pin each structural route.

use num_complex::{Complex32, Complex64};

use crate::audit_regressions::dense_suite;

#[test]
#[ignore = "heavy: runs in the scheduled heavy-tests workflow (see CONTRIBUTING.md)"]
fn dense_oracle_f64() {
    dense_suite::<f64>(true);
}

#[test]
#[ignore = "heavy: runs in the scheduled heavy-tests workflow (see CONTRIBUTING.md)"]
fn dense_oracle_f32() {
    dense_suite::<f32>(true);
}

#[test]
#[ignore = "heavy: runs in the scheduled heavy-tests workflow (see CONTRIBUTING.md)"]
fn dense_oracle_c64() {
    dense_suite::<Complex64>(true);
}

#[test]
#[ignore = "heavy: runs in the scheduled heavy-tests workflow (see CONTRIBUTING.md)"]
fn dense_oracle_c32() {
    dense_suite::<Complex32>(true);
}
