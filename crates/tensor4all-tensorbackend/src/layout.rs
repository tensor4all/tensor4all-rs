use anyhow::{anyhow, Result};

/// Compute dense strides for the current backend linearization semantics.
///
/// The current public dense linearization is column-major.
fn dense_linear_strides(dims: &[usize]) -> Vec<usize> {
    if dims.is_empty() {
        return vec![];
    }
    let mut strides = vec![1; dims.len()];
    for i in 1..dims.len() {
        strides[i] = strides[i - 1] * dims[i - 1];
    }
    strides
}

/// Compute strides for the current mdarray-backed physical storage order.
pub(crate) fn storage_strides(dims: &[usize]) -> Vec<usize> {
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
pub fn dense_linear_offset(dims: &[usize], idx: &[usize]) -> Result<usize> {
    if dims.len() != idx.len() {
        return Err(anyhow!(
            "dense_linear_offset: dims.len() {} != idx.len() {}",
            dims.len(),
            idx.len()
        ));
    }

    let strides = dense_linear_strides(dims);
    let mut offset = 0usize;
    for (axis, ((&dim, &value), stride)) in dims
        .iter()
        .zip(idx.iter())
        .zip(strides.iter().copied())
        .enumerate()
    {
        if value >= dim {
            return Err(anyhow!(
                "dense_linear_offset: index {} at axis {} is >= dim {}",
                value,
                axis,
                dim
            ));
        }
        offset = offset
            .checked_add(
                value
                    .checked_mul(stride)
                    .ok_or_else(|| anyhow!("dense_linear_offset: overflow"))?,
            )
            .ok_or_else(|| anyhow!("dense_linear_offset: overflow"))?;
    }
    Ok(offset)
}

/// Recover a logical multi-index from the current dense linearized offset.
pub fn dense_linear_multi_index(dims: &[usize], linear: usize) -> Result<Vec<usize>> {
    let total_size = dims.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| anyhow!("dense_linear_multi_index: overflow"))
    })?;

    if linear >= total_size && !(dims.is_empty() && linear == 0) {
        return Err(anyhow!(
            "dense_linear_multi_index: linear index {} is out of bounds for dims {:?}",
            linear,
            dims
        ));
    }

    let strides = dense_linear_strides(dims);
    let mut remaining = linear;
    let mut multi = vec![0; dims.len()];
    for axis in (0..dims.len()).rev() {
        let dim = dims[axis];
        let stride = strides[axis];
        if dim == 0 {
            return Err(anyhow!(
                "dense_linear_multi_index: zero dimension at axis {}",
                axis
            ));
        }
        multi[axis] = remaining / stride;
        remaining %= stride;
    }
    Ok(multi)
}
