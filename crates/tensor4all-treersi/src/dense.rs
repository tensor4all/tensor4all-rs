//! Column-major layout adapters around session-scoped backend matrix products.

#[cfg(feature = "tenferro-cuda")]
use tensor4all_tensorbackend::dense_native_tensor_from_col_major_owned;
use tensor4all_tensorbackend::{
    dense_native_tensor_read_input_from_col_major, explicit::Session, Matrix,
};

use crate::{
    scalar::{component_product_underflows, min_nonzero_component_magnitude},
    Result, TreeRsiError, TreeRsiScalar,
};

/// Where the dense helpers run.
///
/// A CPU run always holds one explicit session for the whole product: the
/// session entry receives it from the caller, and a CPU context is entered once
/// by `engine::run_in`. A CUDA context keeps its own route, entering its engine
/// per contraction with explicit transfers.
pub(crate) enum Exec<'a, 'session> {
    /// A caller-held CPU session; nothing else is entered.
    Session(&'a mut Session<'session>),
    /// A CUDA context, entered per contraction.
    #[cfg(feature = "tenferro-cuda")]
    Cuda(&'a tensor4all_tensorbackend::CudaExecutionContext),
}

/// A dense column-major array with explicit axis sizes.
#[derive(Clone, Debug)]
pub(crate) struct Dense<T> {
    pub(crate) dims: Vec<usize>,
    pub(crate) data: Vec<T>,
}

fn product(dims: &[usize]) -> usize {
    dims.iter().product()
}

/// Runs one einsum over host column-major operands and returns host values with
/// the output shape.
///
/// The session arm reads the caller's buffers through borrowed tensor views. A
/// CUDA engine refuses host storage, so the CUDA arm uploads every operand,
/// contracts on the device and downloads the result: one transfer per operand
/// and per result, which is what the naive step 1 of #1158 measures.
fn einsum_host<T: TreeRsiScalar>(
    exec: &mut Exec<'_, '_>,
    operands: &[(&[T], &[usize])],
    input_ids: &[&[usize]],
    output_ids: &[usize],
) -> Result<(Vec<T>, Vec<usize>)> {
    match exec {
        Exec::Session(session) => session_einsum_host(session, operands, input_ids, output_ids),
        #[cfg(feature = "tenferro-cuda")]
        Exec::Cuda(cuda) => cuda_einsum_host(cuda, operands, input_ids, output_ids),
    }
}

fn session_einsum_host<T: TreeRsiScalar>(
    session: &mut Session<'_>,
    operands: &[(&[T], &[usize])],
    input_ids: &[&[usize]],
    output_ids: &[usize],
) -> Result<(Vec<T>, Vec<usize>)> {
    let inputs = operands
        .iter()
        .map(|(data, dims)| {
            dense_native_tensor_read_input_from_col_major(data, dims)
                .map_err(TreeRsiError::numerical)
        })
        .collect::<Result<Vec<_>>>()?;
    let reads = inputs
        .iter()
        .map(|input| input.as_read())
        .collect::<Vec<_>>();
    let contracted = session
        .einsum_reads(&reads, input_ids, output_ids)
        .map_err(TreeRsiError::numerical)?;
    let dims = contracted.shape().to_vec();
    let data =
        T::into_dense_values_from_native_col_major(contracted).map_err(TreeRsiError::numerical)?;
    Ok((data, dims))
}

// Number of einsums a CUDA engine actually returned on this thread.
//
// Test-only observability: a primitive test asserts this advances, so a route
// that keeps the arm but does the arithmetic on the host cannot pass. It is
// counted after the device contraction returns, not on entry to the arm, so the
// count is evidence of device work rather than of reaching a branch. The count
// is thread-local because these primitives run on the caller's thread and the
// test binary runs other CUDA cases in parallel.
#[cfg(all(test, feature = "tenferro-cuda"))]
thread_local! {
    pub(crate) static CUDA_EINSUMS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(feature = "tenferro-cuda")]
fn cuda_einsum_host<T: TreeRsiScalar>(
    cuda: &tensor4all_tensorbackend::CudaExecutionContext,
    operands: &[(&[T], &[usize])],
    input_ids: &[&[usize]],
    output_ids: &[usize],
) -> Result<(Vec<T>, Vec<usize>)> {
    let host = operands
        .iter()
        .map(|(data, dims)| {
            dense_native_tensor_from_col_major_owned(data.to_vec(), dims)
                .map_err(TreeRsiError::numerical)
        })
        .collect::<Result<Vec<_>>>()?;
    let resident = host
        .iter()
        .map(|tensor| cuda.upload_cuda(tensor).map_err(TreeRsiError::numerical))
        .collect::<Result<Vec<_>>>()?;
    let reads = resident.iter().collect::<Vec<_>>();
    let contracted = cuda
        .with_session(|engine| {
            tensor4all_tensorbackend::explicit::Session::new(engine)
                .einsum(&reads, input_ids, output_ids)
        })
        .map_err(TreeRsiError::numerical)?
        .map_err(TreeRsiError::numerical)?;
    // Counted here, where a device contraction has already returned a result:
    // entering the arm is not evidence that the device did the work.
    #[cfg(test)]
    CUDA_EINSUMS.with(|count| count.set(count.get() + 1));
    let host = cuda
        .download(&contracted)
        .map_err(TreeRsiError::numerical)?;
    let dims = host.shape().to_vec();
    let data = T::into_dense_values_from_native_col_major(host).map_err(TreeRsiError::numerical)?;
    Ok((data, dims))
}

fn gemm<T: TreeRsiScalar>(
    exec: &mut Exec<'_, '_>,
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
    match exec {
        // The session product reads both host buffers in place; the einsum
        // route would restage them as tensor views for no gain.
        Exec::Session(session) => session.mat_mul(&a, &b).map_err(TreeRsiError::numerical),
        // A two-operand einsum with one contracted pair takes tenferro-einsum's
        // binary-dot shortcut, so this reaches the device matmul directly.
        #[cfg(feature = "tenferro-cuda")]
        Exec::Cuda(_) => {
            let (data, dims) = einsum_host(
                exec,
                &[
                    (a.as_col_major_slice(), &[a.nrows(), a.ncols()]),
                    (b.as_col_major_slice(), &[b.nrows(), b.ncols()]),
                ],
                &[&[0, 1], &[1, 2]],
                &[0, 2],
            )?;
            matrix(dims[0], dims[1], data)
        }
    }
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
    exec: &mut Exec<'_, '_>,
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
            exec,
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
            exec,
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
    let contracted = replace_axis(exec, front, 0, frame, r)?;
    let mut back = vec![0usize; rank];
    for (position, &original) in to_front.iter().enumerate() {
        back[original] = position;
    }
    Ok(permute(&contracted, &back))
}

/// Contracts `axis` (size `n`) with `weights` (`n x k`), appending the probe
/// axis `l` last: `out(rest.., l) = sum_alpha input(.., alpha, ..) weights(alpha, l)`.
pub(crate) fn contract_axis_to_probe<T: TreeRsiScalar>(
    exec: &mut Exec<'_, '_>,
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
        exec,
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
    exec: &mut Exec<'_, '_>,
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
    // probe label has its own weight column, so the input read `r...,a,l`
    // against the weight read `a,l` gives `r...,l` for the whole diag-probe
    // contraction.
    let rank = input.dims.len();
    let contracted_label = rank - 2;
    let probe_label = rank - 1;
    let input_ids = (0..rank).collect::<Vec<usize>>();
    let weight_ids = [contracted_label, probe_label];
    let mut output_ids = (0..contracted_label).collect::<Vec<usize>>();
    output_ids.push(probe_label);
    let labels = [input_ids.as_slice(), weight_ids.as_slice()];
    let (data, _) = einsum_host(
        exec,
        &[(input.data.as_slice(), &input.dims), (weights, &[n, k])],
        &labels,
        &output_ids,
    )?;
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
    exec: &mut Exec<'_, '_>,
    a: Vec<T>,
    nrows: usize,
    inner: usize,
    b: Vec<T>,
    ncols: usize,
) -> Result<Vec<T>> {
    Ok(gemm(exec, matrix(nrows, inner, a)?, matrix(inner, ncols, b)?)?.into_col_major_vec())
}

#[cfg(test)]
mod tests;
