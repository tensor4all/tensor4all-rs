//! A needle on zero fibers that the global search walks miss (#1433).
//!
//! On Simon's box (see `common::box_target`) the missing face sits several
//! bit flips from every retained pivot, on fibers where both the target and
//! the approximation vanish, so no coordinate walk sees a residual. On `main`
//! before the stage-3 enumeration, 9 of 11 seeds at `R = 7` stopped
//! `Converged` with a dense error of 0.75 or 0.875. The enumeration covers the
//! width-32 box (the low five bits of each axis around a retained pivot). The
//! width-64 box at `R = 8` is outside that budget and is not tested here.
//!
//! The ordinary suite runs two seeds that failed on `main` (seed 2 missed
//! faces on three axes, seed 3 on two) and two control seeds at `R = 6`:
//! seed 1, whose trajectory the enumeration leaves unchanged, and seed 2,
//! where it finds the first pivots. All 11 seeds of both boxes run in
//! `heavy_issue1433_box`. Every case runs under both stop rules (#1405).

use crate::common::box_target::{BOX_R6, BOX_R7};
use anyhow::Result;
use tensor4all_treetci::StopRule;

#[test]
fn width_32_box_is_exact() -> Result<()> {
    [StopRule::Sampled, StopRule::PerBond]
        .into_iter()
        .try_for_each(|rule| BOX_R7.assert_exact(&[2, 3], rule))
}

#[test]
fn width_16_control_box_stays_exact() -> Result<()> {
    [StopRule::Sampled, StopRule::PerBond]
        .into_iter()
        .try_for_each(|rule| BOX_R6.assert_exact(&[1, 2], rule))
}
