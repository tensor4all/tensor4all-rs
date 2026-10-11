//! Label, axis and dtype helpers shared by both frontends.
//!
//! Compiled under `explicit-context` so the opt-in frontend does not depend on
//! the compatibility module: both frontends validate labels and build binary
//! contraction labels through this one implementation.

use anyhow::{anyhow, ensure, Result};
use tenferro::TensorSessionOpsExt;
use tenferro::{DType, Tensor as NativeTensor};
use tenferro_tensor::BackendSession;

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

/// The dtype both frontends promote a heterogeneous operand set to.
///
/// Homogeneous integer operands promote to `F64`, a mixed real/complex set to `C64`,
/// and a homogeneous `F32` or `Bool` set keeps its dtype.
pub(crate) fn common_dtype(dtypes: &[DType]) -> DType {
    let has_f64 = dtypes.contains(&DType::F64);
    let has_c64 = dtypes.contains(&DType::C64);
    let has_c32 = dtypes.contains(&DType::C32);
    let has_i32 = dtypes.contains(&DType::I32);
    let has_i64 = dtypes.contains(&DType::I64);
    let has_bool = dtypes.contains(&DType::Bool);
    let has_complex = has_c64 || has_c32;
    if has_c64 || (has_f64 && has_complex) {
        DType::C64
    } else if has_c32 {
        DType::C32
    } else if has_f64 || has_i64 || has_i32 {
        DType::F64
    } else if has_bool {
        DType::Bool
    } else {
        DType::F32
    }
}

/// Convert one native tensor to `to` on `session`, duplicating when it is already
/// that dtype so the caller keeps an owned value either way.
pub(crate) fn convert_native_tensor_in(
    session: &mut dyn BackendSession,
    tensor: &NativeTensor,
    to: DType,
) -> tenferro_tensor::Result<NativeTensor> {
    if tensor.dtype() == to {
        tensor.duplicate()
    } else {
        tensor.convert(to, session)
    }
}
