//! Column-major layout adapters around context-scoped backend matrix products.

use tensor4all_tensorbackend::{
    dense_native_tensor_from_col_major, dense_native_tensor_from_col_major_owned,
    CpuExecutionContext, Matrix,
};

use crate::{
    scalar::{component_product_underflows, min_nonzero_component_magnitude},
    Result, TreeRsiError, TreeRsiScalar,
};

/// A dense column-major array with explicit axis sizes.
#[derive(Clone, Debug)]
pub(crate) struct Dense<T> {
    pub(crate) dims: Vec<usize>,
    pub(crate) data: Vec<T>,
}

fn product(dims: &[usize]) -> usize {
    dims.iter().product()
}

fn gemm<T: TreeRsiScalar>(
    context: &CpuExecutionContext,
    a: Matrix<T>,
    b: Matrix<T>,
) -> Result<Matrix<T>> {
    if a.ncols() == b.nrows() {
        ensure_matmul_products_representable(
            a.as_col_major_slice(),
            a.nrows(),
            a.ncols(),
            b.as_col_major_slice(),
            b.ncols(),
        )?;
    }
    context
        .with_concrete_session(|session| session.mat_mul(&a, &b))
        .map_err(TreeRsiError::numerical)?
        .map_err(TreeRsiError::numerical)
}

/// Rejects a contraction if any scalar component product would round to zero.
/// The per-inner-index minima make this linear in the input block sizes rather
/// than repeating the matrix multiplication to inspect its terms.
fn ensure_matmul_products_representable<T: TreeRsiScalar>(
    a: &[T],
    nrows: usize,
    inner: usize,
    b: &[T],
    ncols: usize,
) -> Result<()> {
    for k in 0..inner {
        let min_a = (0..nrows)
            .filter_map(|row| min_nonzero_component_magnitude(a[row + nrows * k]))
            .reduce(f64::min);
        let min_b = (0..ncols)
            .filter_map(|col| min_nonzero_component_magnitude(b[k + inner * col]))
            .reduce(f64::min);
        if let (Some(a), Some(b)) = (min_a, min_b)
            && component_product_underflows::<T>(a, b)
        {
            return Err(TreeRsiError::DynamicRange {
                stage: "matrix multiplication",
            });
        }
    }
    Ok(())
}

fn matrix<T: TreeRsiScalar>(nrows: usize, ncols: usize, data: Vec<T>) -> Result<Matrix<T>> {
    Matrix::try_from_col_major_vec(nrows, ncols, data).map_err(TreeRsiError::numerical)
}

/// Transposes a column-major `nrows x ncols` matrix.
pub(crate) fn transpose<T: Copy>(data: &[T], nrows: usize, ncols: usize) -> Vec<T> {
    let mut out = Vec::with_capacity(data.len());
    for row in 0..nrows {
        for col in 0..ncols {
            out.push(data[row + nrows * col]);
        }
    }
    out
}

/// Permutes axes: output axis `i` is input axis `perm[i]`.
pub(crate) fn permute<T: Copy>(input: &Dense<T>, perm: &[usize]) -> Dense<T> {
    let rank = input.dims.len();
    let mut in_strides = vec![1usize; rank];
    for axis in 1..rank {
        in_strides[axis] = in_strides[axis - 1] * input.dims[axis - 1];
    }
    let dims = perm
        .iter()
        .map(|&axis| input.dims[axis])
        .collect::<Vec<_>>();
    let strides = perm
        .iter()
        .map(|&axis| in_strides[axis])
        .collect::<Vec<_>>();
    let total = product(&dims);
    let mut data = Vec::with_capacity(total);
    let mut counter = vec![0usize; rank];
    let mut offset = 0usize;
    for _ in 0..total {
        data.push(input.data[offset]);
        for axis in 0..rank {
            counter[axis] += 1;
            offset += strides[axis];
            if counter[axis] < dims[axis] {
                break;
            }
            offset -= strides[axis] * dims[axis];
            counter[axis] = 0;
        }
    }
    Dense { dims, data }
}

/// Replaces `axis` (size `n`) by a new axis of size `r`:
/// `out(.., t, ..) = sum_alpha input(.., alpha, ..) frame(t, alpha)`.
///
/// `frame` is a column-major `r x n` matrix.
pub(crate) fn replace_axis<T: TreeRsiScalar>(
    context: &CpuExecutionContext,
    input: Dense<T>,
    axis: usize,
    frame: &[T],
    r: usize,
) -> Result<Dense<T>> {
    let n = input.dims[axis];
    let pre = product(&input.dims[..axis]);
    let post = product(&input.dims[axis + 1..]);
    let mut dims = input.dims.clone();
    dims[axis] = r;
    if pre == 1 {
        // out (r x post) = frame (r x n) * input (n x post).
        let out = gemm(
            context,
            matrix(r, n, frame.to_vec())?,
            matrix(n, post, input.data)?,
        )?;
        return Ok(Dense {
            dims,
            data: out.into_col_major_vec(),
        });
    }
    if post == 1 {
        // out (pre x r) = input (pre x n) * frame^T (n x r).
        let out = gemm(
            context,
            matrix(pre, n, input.data)?,
            matrix(n, r, transpose(frame, r, n))?,
        )?;
        return Ok(Dense {
            dims,
            data: out.into_col_major_vec(),
        });
    }
    // Middle axis: bring it to the front, contract, and restore the order.
    let rank = input.dims.len();
    let mut to_front = vec![axis];
    to_front.extend((0..rank).filter(|&other| other != axis));
    let front = permute(&input, &to_front);
    let contracted = replace_axis(context, front, 0, frame, r)?;
    let mut back = vec![0usize; rank];
    for (position, &original) in to_front.iter().enumerate() {
        back[original] = position;
    }
    Ok(permute(&contracted, &back))
}

/// Contracts `axis` (size `n`) with `weights` (`n x k`), appending the probe
/// axis `l` last: `out(rest.., l) = sum_alpha input(.., alpha, ..) weights(alpha, l)`.
pub(crate) fn contract_axis_to_probe<T: TreeRsiScalar>(
    context: &CpuExecutionContext,
    input: &Dense<T>,
    axis: usize,
    weights: &[T],
    k: usize,
) -> Result<Dense<T>> {
    let n = input.dims[axis];
    let rank = input.dims.len();
    let rest_dims = input
        .dims
        .iter()
        .enumerate()
        .filter(|&(position, _)| position != axis)
        .map(|(_, &dim)| dim)
        .collect::<Vec<_>>();
    let rest = product(&rest_dims);
    let data = if axis + 1 == rank {
        input.data.clone()
    } else {
        let mut to_back = (0..rank).filter(|&other| other != axis).collect::<Vec<_>>();
        to_back.push(axis);
        permute(input, &to_back).data
    };
    let out = gemm(
        context,
        matrix(rest, n, data)?,
        matrix(n, k, weights.to_vec())?,
    )?;
    let mut dims = rest_dims;
    dims.push(k);
    Ok(Dense {
        dims,
        data: out.into_col_major_vec(),
    })
}

/// Contracts `axis` with `weights` (`n x k`) diagonally in the last (probe)
/// axis: `out(.., l) = sum_alpha input(.., alpha, .., l) weights(alpha, l)`.
///
/// This is the Khatri-Rao step of the product-state sketch.
pub(crate) fn contract_axis_diag_probe<T: TreeRsiScalar>(
    context: &CpuExecutionContext,
    input: Dense<T>,
    axis: usize,
    weights: &[T],
) -> Result<Dense<T>> {
    let last = input.dims.len() - 1;
    let k = input.dims[last];
    let n = input.dims[axis];
    let mut order = (0..last).filter(|&a| a != axis).collect::<Vec<_>>();
    order.extend([axis, last]);
    let input = permute(&input, &order);
    let rows = product(&input.dims[..last - 1]);
    let mut dims = input.dims[..last - 1].to_vec();
    dims.push(k);
    let input_batch_len = input.data.len() / k;
    let weights_batch_len = weights.len() / k;
    for batch in 0..k {
        ensure_matmul_products_representable(
            &input.data[batch * input_batch_len..(batch + 1) * input_batch_len],
            rows,
            n,
            &weights[batch * weights_batch_len..(batch + 1) * weights_batch_len],
            1,
        )?;
    }
    // One einsum for every probe label: the input is `rows x n x k` and each
    // probe label has its own weight column, so `r...,ak,al->r...,l` is the
    // whole diag-probe contraction. `ponytail:` the `n x k` weights are copied
    // into a native tensor because this crate does not depend on
    // `tenferro_tensor` and cannot build a borrowed read view; the large input
    // buffer is moved instead of copied.
    let rank = input.dims.len();
    let contracted_label = rank - 2;
    let probe_label = rank - 1;
    let input_ids = (0..rank).collect::<Vec<usize>>();
    let weight_ids = [contracted_label, probe_label];
    let mut output_ids = (0..contracted_label).collect::<Vec<usize>>();
    output_ids.push(probe_label);
    let input_tensor = dense_native_tensor_from_col_major_owned(input.data, &input.dims)
        .map_err(TreeRsiError::numerical)?;
    let weight_tensor =
        dense_native_tensor_from_col_major(weights, &[n, k]).map_err(TreeRsiError::numerical)?;
    let labels = [input_ids.as_slice(), weight_ids.as_slice()];
    let contracted = context
        .with_concrete_session(|session| {
            session.einsum(&[&input_tensor, &weight_tensor], &labels, &output_ids)
        })
        .map_err(TreeRsiError::numerical)?
        .map_err(TreeRsiError::numerical)?;
    let data =
        T::dense_values_from_native_col_major(&contracted).map_err(TreeRsiError::numerical)?;
    Ok(Dense { dims, data })
}

/// Gathers `rows` of a column-major `nrows x ncols` matrix.
pub(crate) fn gather_rows<T: Copy>(
    data: &[T],
    nrows: usize,
    ncols: usize,
    rows: &[usize],
) -> Vec<T> {
    let mut out = Vec::with_capacity(rows.len() * ncols);
    for col in 0..ncols {
        let column = &data[nrows * col..nrows * (col + 1)];
        out.extend(rows.iter().map(|&row| column[row]));
    }
    out
}

/// Multiplies two column-major matrices.
pub(crate) fn mat_mul_cols<T: TreeRsiScalar>(
    context: &CpuExecutionContext,
    a: Vec<T>,
    nrows: usize,
    inner: usize,
    b: Vec<T>,
    ncols: usize,
) -> Result<Vec<T>> {
    Ok(gemm(context, matrix(nrows, inner, a)?, matrix(inner, ncols, b)?)?.into_col_major_vec())
}

#[cfg(test)]
mod tests;
