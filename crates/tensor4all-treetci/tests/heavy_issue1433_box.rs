//! All 11 seeds of the #1433 box regressions; the ordinary suite runs a
//! fixed subset in `issue1433_leaf_side_enumeration`.

use crate::common::box_target::{BOX_R6, BOX_R7};
use anyhow::Result;

const SEEDS: [u64; 11] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

#[test]
#[ignore = "heavy: runs in the scheduled heavy-tests workflow (see CONTRIBUTING.md)"]
fn heavy_width_32_box_is_exact_for_every_seed() -> Result<()> {
    BOX_R7.assert_exact(&SEEDS)
}

#[test]
#[ignore = "heavy: runs in the scheduled heavy-tests workflow (see CONTRIBUTING.md)"]
fn heavy_width_16_control_box_stays_exact_for_every_seed() -> Result<()> {
    BOX_R6.assert_exact(&SEEDS)
}
