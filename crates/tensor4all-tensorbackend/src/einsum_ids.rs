//! Einsum label and axis-to-label helpers shared by both frontends.
//!
//! Compiled under `explicit-context` so the opt-in frontend does not depend on
//! the compatibility module: both frontends validate labels and build binary
//! contraction labels through this one implementation.

use anyhow::{anyhow, ensure, Result};

pub(crate) fn checked_native_einsum_labels(labels: &[usize]) -> Result<Vec<u32>> {
    labels
        .iter()
        .copied()
        .map(|label| {
            u32::try_from(label)
                .map_err(|_| anyhow!("native einsum label {label} exceeds the supported u32 range"))
        })
        .collect()
}

/// Build native einsum ids for a binary contraction.
pub(crate) fn build_binary_einsum_ids(
    lhs_rank: usize,
    axes_a: &[usize],
    rhs_rank: usize,
    axes_b: &[usize],
) -> Result<(Vec<u32>, Vec<u32>, Vec<u32>)> {
    ensure!(
        axes_a.len() == axes_b.len(),
        "contract axis length mismatch: lhs {:?}, rhs {:?}",
        axes_a,
        axes_b
    );

    let mut lhs_ids = vec![u32::MAX; lhs_rank];
    let mut rhs_ids = vec![u32::MAX; rhs_rank];
    let mut next_id = 0u32;

    let mut seen_lhs = vec![false; lhs_rank];
    let mut seen_rhs = vec![false; rhs_rank];

    for (&lhs_axis, &rhs_axis) in axes_a.iter().zip(axes_b.iter()) {
        ensure!(
            lhs_axis < lhs_rank,
            "lhs contract axis {lhs_axis} out of range"
        );
        ensure!(
            rhs_axis < rhs_rank,
            "rhs contract axis {rhs_axis} out of range"
        );
        ensure!(
            !seen_lhs[lhs_axis],
            "duplicate lhs contract axis {lhs_axis}"
        );
        ensure!(
            !seen_rhs[rhs_axis],
            "duplicate rhs contract axis {rhs_axis}"
        );
        seen_lhs[lhs_axis] = true;
        seen_rhs[rhs_axis] = true;
        lhs_ids[lhs_axis] = next_id;
        rhs_ids[rhs_axis] = next_id;
        next_id += 1;
    }

    let mut output_ids = Vec::with_capacity(lhs_rank + rhs_rank - 2 * axes_a.len());
    for (axis, slot) in lhs_ids.iter_mut().enumerate() {
        if *slot == u32::MAX {
            *slot = next_id;
            output_ids.push(next_id);
            next_id += 1;
        } else {
            let _ = axis;
        }
    }
    for slot in &mut rhs_ids {
        if *slot == u32::MAX {
            *slot = next_id;
            output_ids.push(next_id);
            next_id += 1;
        }
    }

    Ok((lhs_ids, rhs_ids, output_ids))
}
