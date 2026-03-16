use anyhow::{anyhow, Result};

/// Compute dense strides for the current backend linearization semantics.
///
/// The current semantics are row-major; the column-major migration will flip
/// this implementation in one place.
pub(crate) fn dense_strides(dims: &[usize]) -> Vec<usize> {
    if dims.is_empty() {
        return vec![];
    }
    let mut strides = vec![1; dims.len()];
    for i in (0..dims.len() - 1).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    strides
}

/// Compute the current dense linearized offset for a logical multi-index.
pub(crate) fn dense_offset(dims: &[usize], idx: &[usize]) -> Result<usize> {
    if dims.len() != idx.len() {
        return Err(anyhow!(
            "dense_offset: dims.len() {} != idx.len() {}",
            dims.len(),
            idx.len()
        ));
    }

    let strides = dense_strides(dims);
    let mut offset = 0usize;
    for (axis, ((&dim, &value), stride)) in dims
        .iter()
        .zip(idx.iter())
        .zip(strides.iter().copied())
        .enumerate()
    {
        if value >= dim {
            return Err(anyhow!(
                "dense_offset: index {} at axis {} is >= dim {}",
                value,
                axis,
                dim
            ));
        }
        offset = offset
            .checked_add(
                value
                    .checked_mul(stride)
                    .ok_or_else(|| anyhow!("dense_offset: overflow"))?,
            )
            .ok_or_else(|| anyhow!("dense_offset: overflow"))?;
    }
    Ok(offset)
}
