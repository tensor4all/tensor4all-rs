//! Bridge helpers between tensor4all storage snapshots and tenferro tensors.

use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::HashMap;
use std::env;
use std::time::{Duration, Instant};

use anyhow::{anyhow, ensure, Result};
use num_complex::{Complex32, Complex64};
use tenferro::eager_einsum::eager_einsum_owned;
use tenferro::{DType, Tensor as NativeTensor, TensorBackend};

use crate::any_scalar::promote_scalar_native;
use crate::context::with_default_backend;
use crate::storage::Storage;
#[cfg(test)]
use crate::storage::StorageRepr;
use crate::tensor_element::TensorElement;
use crate::AnyScalar;

#[cfg(test)]
use std::cell::Cell;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
enum NativeEinsumPath {
    FrontendFallback,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct NativeOperandSignature {
    shape: Vec<usize>,
    ids: Vec<u32>,
    dtype: DType,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct NativeEinsumSignature {
    path: NativeEinsumPath,
    operands: Vec<NativeOperandSignature>,
    output_ids: Vec<u32>,
}

#[derive(Debug, Default, Clone)]
struct NativeEinsumProfileEntry {
    calls: usize,
    total_time: Duration,
}

thread_local! {
    static NATIVE_EINSUM_PROFILE_STATE: RefCell<HashMap<NativeEinsumSignature, NativeEinsumProfileEntry>> =
        RefCell::new(HashMap::new());
}

#[cfg(test)]
thread_local! {
    static FORCE_NATIVE_EINSUM_PROFILE: Cell<bool> = const { Cell::new(false) };
}

fn native_einsum_profile_enabled() -> bool {
    #[cfg(test)]
    if FORCE_NATIVE_EINSUM_PROFILE.with(Cell::get) {
        return true;
    }
    env::var("T4A_PROFILE_NATIVE_EINSUM").is_ok()
}

#[cfg(test)]
pub(crate) fn set_native_einsum_profile_enabled_for_tests(enabled: bool) {
    FORCE_NATIVE_EINSUM_PROFILE.with(|slot| slot.set(enabled));
}

fn record_native_einsum_profile(
    path: NativeEinsumPath,
    operands: &[(&NativeTensor, &[usize])],
    output_ids: &[u32],
    elapsed: Duration,
) {
    if !native_einsum_profile_enabled() {
        return;
    }
    let signature = NativeEinsumSignature {
        path,
        operands: operands
            .iter()
            .map(|(tensor, ids)| NativeOperandSignature {
                shape: tensor.shape().to_vec(),
                ids: ids.iter().map(|&id| id as u32).collect(),
                dtype: tensor.dtype(),
            })
            .collect(),
        output_ids: output_ids.to_vec(),
    };
    NATIVE_EINSUM_PROFILE_STATE.with(|state| {
        let mut state = state.borrow_mut();
        let entry = state.entry(signature).or_default();
        entry.calls += 1;
        entry.total_time += elapsed;
    });
}

/// Reset the aggregated native einsum profile.
pub fn reset_native_einsum_profile() {
    NATIVE_EINSUM_PROFILE_STATE.with(|state| state.borrow_mut().clear());
}

/// Print and clear the aggregated native einsum profile.
pub fn print_and_reset_native_einsum_profile() {
    if !native_einsum_profile_enabled() {
        return;
    }
    NATIVE_EINSUM_PROFILE_STATE.with(|state| {
        let mut entries: Vec<_> = state
            .borrow()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        state.borrow_mut().clear();
        entries.sort_by_key(|(_, entry)| Reverse(entry.total_time));

        eprintln!("=== native_einsum Profile ===");
        for (idx, (signature, entry)) in entries.into_iter().take(20).enumerate() {
            eprintln!(
                "#{idx:02} path={:?} calls={} total={:.3}s per_call={:.3}us output_ids={:?}",
                signature.path,
                entry.calls,
                entry.total_time.as_secs_f64(),
                entry.total_time.as_secs_f64() * 1e6 / entry.calls as f64,
                signature.output_ids,
            );
            for operand in signature.operands {
                eprintln!(
                    "     shape={:?} ids={:?} dtype={:?}",
                    operand.shape, operand.ids, operand.dtype
                );
            }
        }
    });
}

fn common_dtype(dtypes: &[DType]) -> DType {
    let has_f64 = dtypes.contains(&DType::F64);
    let has_c64 = dtypes.contains(&DType::C64);
    let has_c32 = dtypes.contains(&DType::C32);
    let has_complex = has_c64 || has_c32;
    if has_c64 || (has_f64 && has_complex) {
        DType::C64
    } else if has_c32 {
        DType::C32
    } else if has_f64 {
        DType::F64
    } else {
        DType::F32
    }
}

fn convert_tensor(tensor: &NativeTensor, to: DType) -> Result<NativeTensor> {
    if tensor.dtype() == to {
        return Ok(tensor.clone());
    }
    with_default_backend(|backend| backend.with_exec_session(|exec| exec.convert(tensor, to)))
        .map_err(|e| anyhow!("tensor conversion to {to:?} failed: {e}"))
}

fn ids_to_subscript(ids: &[u32]) -> Result<String> {
    const LETTERS: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    let mut out = String::with_capacity(ids.len());
    for &id in ids {
        let idx = usize::try_from(id).unwrap_or(usize::MAX);
        let letter = LETTERS
            .get(idx)
            .ok_or_else(|| anyhow!("einsum label {id} exceeds supported label range"))?;
        out.push(char::from(*letter));
    }
    Ok(out)
}

fn build_einsum_subscripts(operands: &[&[u32]], output_ids: &[u32]) -> Result<String> {
    let inputs = operands
        .iter()
        .map(|ids| ids_to_subscript(ids))
        .collect::<Result<Vec<_>>>()?;
    Ok(format!(
        "{}->{}",
        inputs.join(","),
        ids_to_subscript(output_ids)?
    ))
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

/// Build a dense native tensor from column-major data.
pub fn dense_native_tensor_from_col_major<T: TensorElement>(
    data: &[T],
    logical_dims: &[usize],
) -> Result<NativeTensor> {
    T::dense_native_tensor_from_col_major(data, logical_dims)
}

/// Build a dense native tensor whose logical values are diagonal.
pub fn diag_native_tensor_from_col_major<T: TensorElement>(
    data: &[T],
    logical_rank: usize,
) -> Result<NativeTensor> {
    T::diag_native_tensor_from_col_major(data, logical_rank)
}

/// Convert storage to a dense native tensor.
pub fn storage_to_native_tensor(storage: &Storage, logical_dims: &[usize]) -> Result<NativeTensor> {
    if storage.is_c64() {
        dense_native_tensor_from_col_major(
            &storage
                .to_dense_c64_col_major_vec(logical_dims)
                .map_err(|e| anyhow!("dense c64 materialization failed: {e}"))?,
            logical_dims,
        )
    } else {
        dense_native_tensor_from_col_major(
            &storage
                .to_dense_f64_col_major_vec(logical_dims)
                .map_err(|e| anyhow!("dense f64 materialization failed: {e}"))?,
            logical_dims,
        )
    }
}

/// Materialize a native tensor into dense storage.
pub fn native_tensor_primal_to_storage(tensor: &NativeTensor) -> Result<Storage> {
    match tensor.dtype() {
        DType::F32 => Storage::from_dense_col_major(
            tensor
                .as_slice::<f32>()
                .ok_or_else(|| anyhow!("failed to read f32 native tensor"))?
                .iter()
                .map(|&value| value as f64)
                .collect::<Vec<_>>(),
            tensor.shape(),
        ),
        DType::F64 => Storage::from_dense_col_major(
            tensor
                .as_slice::<f64>()
                .ok_or_else(|| anyhow!("failed to read f64 native tensor"))?
                .to_vec(),
            tensor.shape(),
        ),
        DType::C32 => Storage::from_dense_col_major(
            tensor
                .as_slice::<Complex32>()
                .ok_or_else(|| anyhow!("failed to read c32 native tensor"))?
                .iter()
                .map(|&value| Complex64::new(value.re as f64, value.im as f64))
                .collect::<Vec<_>>(),
            tensor.shape(),
        ),
        DType::C64 => Storage::from_dense_col_major(
            tensor
                .as_slice::<Complex64>()
                .ok_or_else(|| anyhow!("failed to read c64 native tensor"))?
                .to_vec(),
            tensor.shape(),
        ),
    }
    .map_err(|e| anyhow!("native tensor snapshot materialization failed: {e}"))
}

/// Materialize dense f64 values from a native tensor.
pub fn native_tensor_primal_to_dense_f64_col_major(tensor: &NativeTensor) -> Result<Vec<f64>> {
    match tensor.dtype() {
        DType::F32 => Ok(tensor
            .as_slice::<f32>()
            .ok_or_else(|| anyhow!("failed to read f32 native tensor"))?
            .iter()
            .map(|&value| value as f64)
            .collect()),
        DType::F64 => <f64 as TensorElement>::dense_values_from_native_col_major(tensor),
        other => Err(anyhow!("expected real native tensor, got dtype {other:?}")),
    }
}

/// Materialize dense Complex64 values from a native tensor.
pub fn native_tensor_primal_to_dense_c64_col_major(
    tensor: &NativeTensor,
) -> Result<Vec<Complex64>> {
    match tensor.dtype() {
        DType::C32 => Ok(tensor
            .as_slice::<Complex32>()
            .ok_or_else(|| anyhow!("failed to read c32 native tensor"))?
            .iter()
            .map(|&value| Complex64::new(value.re as f64, value.im as f64))
            .collect()),
        DType::C64 => <Complex64 as TensorElement>::dense_values_from_native_col_major(tensor),
        other => Err(anyhow!(
            "expected complex native tensor, got dtype {other:?}"
        )),
    }
}

/// Materialize dense column-major values from a native tensor.
pub fn native_tensor_primal_to_dense_col_major<T: TensorElement>(
    tensor: &NativeTensor,
) -> Result<Vec<T>> {
    T::dense_values_from_native_col_major(tensor)
}

/// Extract the diagonal payload from a real native tensor.
pub fn native_tensor_primal_to_diag_f64(tensor: &NativeTensor) -> Result<Vec<f64>> {
    match tensor.dtype() {
        DType::F32 => {
            let promoted = convert_tensor(tensor, DType::F64)?;
            <f64 as TensorElement>::diag_values_from_native_temp(&promoted)
        }
        DType::F64 => <f64 as TensorElement>::diag_values_from_native_temp(tensor),
        other => Err(anyhow!("expected real native tensor, got dtype {other:?}")),
    }
}

/// Extract the diagonal payload from a complex native tensor.
pub fn native_tensor_primal_to_diag_c64(tensor: &NativeTensor) -> Result<Vec<Complex64>> {
    match tensor.dtype() {
        DType::C32 => {
            let promoted = convert_tensor(tensor, DType::C64)?;
            <Complex64 as TensorElement>::diag_values_from_native_temp(&promoted)
        }
        DType::C64 => <Complex64 as TensorElement>::diag_values_from_native_temp(tensor),
        other => Err(anyhow!(
            "expected complex native tensor, got dtype {other:?}"
        )),
    }
}

/// Reshape a native tensor without changing its column-major linearization.
pub fn reshape_col_major_native_tensor(
    tensor: &NativeTensor,
    logical_dims: &[usize],
) -> Result<NativeTensor> {
    with_default_backend(|backend| tensor.reshape(logical_dims, backend))
        .map_err(|e| anyhow!("native reshape failed: {e}"))
}

/// Compute a QR decomposition on a native tensor.
pub fn qr_native_tensor(tensor: &NativeTensor) -> Result<(NativeTensor, NativeTensor)> {
    with_default_backend(|backend| tensor.qr(backend)).map_err(|e| anyhow!("native QR failed: {e}"))
}

/// Compute an SVD on a native tensor.
pub fn svd_native_tensor(
    tensor: &NativeTensor,
) -> Result<(NativeTensor, NativeTensor, NativeTensor)> {
    with_default_backend(|backend| tensor.svd(backend))
        .map_err(|e| anyhow!("native SVD failed: {e}"))
}

/// Sum all elements of a native tensor, returning a dynamic scalar.
pub fn sum_native_tensor(tensor: &NativeTensor) -> Result<AnyScalar> {
    let reduced = if tensor.shape().is_empty() {
        tensor.clone()
    } else {
        let axes: Vec<usize> = (0..tensor.shape().len()).collect();
        with_default_backend(|backend| tensor.reduce_sum(&axes, backend))
            .map_err(|e| anyhow!("native sum failed: {e}"))?
    };
    AnyScalar::from_native(reduced)
}

/// Return the tangent tensor when present.
///
/// Plain `Tensor` values do not carry tangent storage, so this bridge returns
/// `None`.
pub fn tangent_native_tensor(_tensor: &NativeTensor) -> Option<NativeTensor> {
    None
}

/// Multiply a native tensor by a dynamic scalar.
pub fn scale_native_tensor(tensor: &NativeTensor, scalar: &AnyScalar) -> Result<NativeTensor> {
    let target = common_dtype(&[tensor.dtype(), scalar.as_native().dtype()]);
    let tensor = convert_tensor(tensor, target)?;
    let scalar = promote_scalar_native(scalar.as_native(), target)?;

    match target {
        DType::F32 => {
            let factor = scalar
                .as_slice::<f32>()
                .and_then(|values| values.first().copied())
                .ok_or_else(|| anyhow!("failed to read promoted f32 scalar"))?;
            let values = tensor
                .as_slice::<f32>()
                .ok_or_else(|| anyhow!("failed to read promoted f32 tensor"))?
                .iter()
                .map(|&value| value * factor)
                .collect::<Vec<_>>();
            Ok(NativeTensor::from_vec(tensor.shape().to_vec(), values))
        }
        DType::F64 => {
            let factor = scalar
                .as_slice::<f64>()
                .and_then(|values| values.first().copied())
                .ok_or_else(|| anyhow!("failed to read promoted f64 scalar"))?;
            let values = tensor
                .as_slice::<f64>()
                .ok_or_else(|| anyhow!("failed to read promoted f64 tensor"))?
                .iter()
                .map(|&value| value * factor)
                .collect::<Vec<_>>();
            Ok(NativeTensor::from_vec(tensor.shape().to_vec(), values))
        }
        DType::C32 => {
            let factor = scalar
                .as_slice::<Complex32>()
                .and_then(|values| values.first().copied())
                .ok_or_else(|| anyhow!("failed to read promoted c32 scalar"))?;
            let values = tensor
                .as_slice::<Complex32>()
                .ok_or_else(|| anyhow!("failed to read promoted c32 tensor"))?
                .iter()
                .map(|&value| value * factor)
                .collect::<Vec<_>>();
            Ok(NativeTensor::from_vec(tensor.shape().to_vec(), values))
        }
        DType::C64 => {
            let factor = scalar
                .as_slice::<Complex64>()
                .and_then(|values| values.first().copied())
                .ok_or_else(|| anyhow!("failed to read promoted c64 scalar"))?;
            let values = tensor
                .as_slice::<Complex64>()
                .ok_or_else(|| anyhow!("failed to read promoted c64 tensor"))?
                .iter()
                .map(|&value| value * factor)
                .collect::<Vec<_>>();
            Ok(NativeTensor::from_vec(tensor.shape().to_vec(), values))
        }
    }
}

/// Compute `a * lhs + b * rhs`.
pub fn axpby_native_tensor(
    lhs: &NativeTensor,
    a: &AnyScalar,
    rhs: &NativeTensor,
    b: &AnyScalar,
) -> Result<NativeTensor> {
    ensure!(
        lhs.shape() == rhs.shape(),
        "axpby requires matching tensor shapes, got lhs {:?} and rhs {:?}",
        lhs.shape(),
        rhs.shape()
    );

    let target = common_dtype(&[
        lhs.dtype(),
        rhs.dtype(),
        a.as_native().dtype(),
        b.as_native().dtype(),
    ]);
    let lhs = convert_tensor(lhs, target)?;
    let rhs = convert_tensor(rhs, target)?;
    let a = promote_scalar_native(a.as_native(), target)?;
    let b = promote_scalar_native(b.as_native(), target)?;

    match target {
        DType::F32 => {
            let a = a
                .as_slice::<f32>()
                .and_then(|values| values.first().copied())
                .ok_or_else(|| anyhow!("failed to read promoted f32 scalar a"))?;
            let b = b
                .as_slice::<f32>()
                .and_then(|values| values.first().copied())
                .ok_or_else(|| anyhow!("failed to read promoted f32 scalar b"))?;
            let lhs_values = lhs
                .as_slice::<f32>()
                .ok_or_else(|| anyhow!("failed to read promoted f32 lhs"))?;
            let rhs_values = rhs
                .as_slice::<f32>()
                .ok_or_else(|| anyhow!("failed to read promoted f32 rhs"))?;
            let values = lhs_values
                .iter()
                .zip(rhs_values.iter())
                .map(|(&x, &y)| a * x + b * y)
                .collect::<Vec<_>>();
            Ok(NativeTensor::from_vec(lhs.shape().to_vec(), values))
        }
        DType::F64 => {
            let a = a
                .as_slice::<f64>()
                .and_then(|values| values.first().copied())
                .ok_or_else(|| anyhow!("failed to read promoted f64 scalar a"))?;
            let b = b
                .as_slice::<f64>()
                .and_then(|values| values.first().copied())
                .ok_or_else(|| anyhow!("failed to read promoted f64 scalar b"))?;
            let lhs_values = lhs
                .as_slice::<f64>()
                .ok_or_else(|| anyhow!("failed to read promoted f64 lhs"))?;
            let rhs_values = rhs
                .as_slice::<f64>()
                .ok_or_else(|| anyhow!("failed to read promoted f64 rhs"))?;
            let values = lhs_values
                .iter()
                .zip(rhs_values.iter())
                .map(|(&x, &y)| a * x + b * y)
                .collect::<Vec<_>>();
            Ok(NativeTensor::from_vec(lhs.shape().to_vec(), values))
        }
        DType::C32 => {
            let a = a
                .as_slice::<Complex32>()
                .and_then(|values| values.first().copied())
                .ok_or_else(|| anyhow!("failed to read promoted c32 scalar a"))?;
            let b = b
                .as_slice::<Complex32>()
                .and_then(|values| values.first().copied())
                .ok_or_else(|| anyhow!("failed to read promoted c32 scalar b"))?;
            let lhs_values = lhs
                .as_slice::<Complex32>()
                .ok_or_else(|| anyhow!("failed to read promoted c32 lhs"))?;
            let rhs_values = rhs
                .as_slice::<Complex32>()
                .ok_or_else(|| anyhow!("failed to read promoted c32 rhs"))?;
            let values = lhs_values
                .iter()
                .zip(rhs_values.iter())
                .map(|(&x, &y)| a * x + b * y)
                .collect::<Vec<_>>();
            Ok(NativeTensor::from_vec(lhs.shape().to_vec(), values))
        }
        DType::C64 => {
            let a = a
                .as_slice::<Complex64>()
                .and_then(|values| values.first().copied())
                .ok_or_else(|| anyhow!("failed to read promoted c64 scalar a"))?;
            let b = b
                .as_slice::<Complex64>()
                .and_then(|values| values.first().copied())
                .ok_or_else(|| anyhow!("failed to read promoted c64 scalar b"))?;
            let lhs_values = lhs
                .as_slice::<Complex64>()
                .ok_or_else(|| anyhow!("failed to read promoted c64 lhs"))?;
            let rhs_values = rhs
                .as_slice::<Complex64>()
                .ok_or_else(|| anyhow!("failed to read promoted c64 rhs"))?;
            let values = lhs_values
                .iter()
                .zip(rhs_values.iter())
                .map(|(&x, &y)| a * x + b * y)
                .collect::<Vec<_>>();
            Ok(NativeTensor::from_vec(lhs.shape().to_vec(), values))
        }
    }
}

/// Execute an eager einsum over owned native tensors.
///
/// This is the consuming bridge used by higher-level owned contraction APIs.
/// Inputs are promoted to a common dtype before the owned tenferro eager
/// einsum runs.
///
/// # Arguments
/// * `operands` - Native tensors paired with numeric einsum labels for each axis.
/// * `output_ids` - Numeric labels to keep in the result, in output axis order.
///
/// # Returns
/// The contracted native tensor in the promoted common dtype.
///
/// # Errors
/// Returns an error if the operand list is empty, any label list length does
/// not match its tensor rank, label generation exceeds the supported range, or
/// the backend contraction fails.
///
/// # Examples
/// ```
/// use tensor4all_tensorbackend::einsum_native_tensors_owned;
/// use tenferro::Tensor as NativeTensor;
///
/// let lhs = NativeTensor::from_vec(vec![2, 3], vec![1.0_f64; 6]);
/// let rhs = NativeTensor::from_vec(vec![3, 2], vec![1.0_f64; 6]);
/// let result = einsum_native_tensors_owned(vec![(lhs, vec![0, 1]), (rhs, vec![1, 2])], &[0, 2]).unwrap();
///
/// assert_eq!(result.shape(), &[2, 2]);
/// assert_eq!(result.as_slice::<f64>().unwrap(), &[3.0, 3.0, 3.0, 3.0]);
/// ```
pub fn einsum_native_tensors_owned(
    operands: Vec<(NativeTensor, Vec<usize>)>,
    output_ids: &[usize],
) -> Result<NativeTensor> {
    ensure!(
        !operands.is_empty(),
        "native einsum requires at least one operand"
    );

    let target = common_dtype(
        &operands
            .iter()
            .map(|(tensor, _)| tensor.dtype())
            .collect::<Vec<_>>(),
    );

    let mut converted = Vec::with_capacity(operands.len());
    let mut input_ids = Vec::with_capacity(operands.len());
    for (tensor, ids) in operands {
        ensure!(
            tensor.shape().len() == ids.len(),
            "einsum id list {:?} does not match tensor shape {:?}",
            ids,
            tensor.shape()
        );
        let tensor = if tensor.dtype() == target {
            tensor
        } else {
            convert_tensor(&tensor, target)?
        };
        input_ids.push(ids.into_iter().map(|id| id as u32).collect::<Vec<_>>());
        converted.push(tensor);
    }

    let input_slices = input_ids.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let output_ids_u32 = output_ids.iter().map(|&id| id as u32).collect::<Vec<_>>();
    let subscripts = build_einsum_subscripts(&input_slices, &output_ids_u32)?;

    let result =
        with_default_backend(|backend| eager_einsum_owned(backend, converted, &subscripts))
            .map_err(|e| anyhow!("native einsum failed: {e}"))?;
    Ok(result)
}

/// Execute an eager einsum over native tensors.
pub fn einsum_native_tensors(
    operands: &[(&NativeTensor, &[usize])],
    output_ids: &[usize],
) -> Result<NativeTensor> {
    let owned_operands = operands
        .iter()
        .map(|(tensor, ids)| ((*tensor).clone(), ids.to_vec()))
        .collect::<Vec<_>>();
    let output_ids_u32 = output_ids.iter().map(|&id| id as u32).collect::<Vec<_>>();
    let started = Instant::now();
    let result = einsum_native_tensors_owned(owned_operands, output_ids)?;
    record_native_einsum_profile(
        NativeEinsumPath::FrontendFallback,
        operands,
        &output_ids_u32,
        started.elapsed(),
    );
    Ok(result)
}

/// Permute axes of a native tensor.
pub fn permute_native_tensor(tensor: &NativeTensor, perm: &[usize]) -> Result<NativeTensor> {
    with_default_backend(|backend| tensor.transpose(perm, backend))
        .map_err(|e| anyhow!("native permute failed: {e}"))
}

/// Contract two native tensors along matching axes.
pub fn contract_native_tensor(
    lhs: &NativeTensor,
    axes_a: &[usize],
    rhs: &NativeTensor,
    axes_b: &[usize],
) -> Result<NativeTensor> {
    let (lhs_ids, rhs_ids, output_ids) =
        build_binary_einsum_ids(lhs.shape().len(), axes_a, rhs.shape().len(), axes_b)?;
    let lhs_ids_usize = lhs_ids.iter().map(|&id| id as usize).collect::<Vec<_>>();
    let rhs_ids_usize = rhs_ids.iter().map(|&id| id as usize).collect::<Vec<_>>();
    let output_ids_usize = output_ids.iter().map(|&id| id as usize).collect::<Vec<_>>();
    let operands = [
        (lhs, lhs_ids_usize.as_slice()),
        (rhs, rhs_ids_usize.as_slice()),
    ];
    einsum_native_tensors(&operands, &output_ids_usize)
}

/// Compute the outer product of two native tensors.
pub fn outer_product_native_tensor(lhs: &NativeTensor, rhs: &NativeTensor) -> Result<NativeTensor> {
    contract_native_tensor(lhs, &[], rhs, &[])
}

/// Conjugate a native tensor.
pub fn conj_native_tensor(tensor: &NativeTensor) -> Result<NativeTensor> {
    match tensor.dtype() {
        DType::F32 | DType::F64 => Ok(tensor.clone()),
        DType::C32 => Ok(NativeTensor::from_vec(
            tensor.shape().to_vec(),
            tensor
                .as_slice::<Complex32>()
                .ok_or_else(|| anyhow!("failed to read c32 native tensor"))?
                .iter()
                .map(|&value| value.conj())
                .collect::<Vec<_>>(),
        )),
        DType::C64 => Ok(NativeTensor::from_vec(
            tensor.shape().to_vec(),
            tensor
                .as_slice::<Complex64>()
                .ok_or_else(|| anyhow!("failed to read c64 native tensor"))?
                .iter()
                .map(|&value| value.conj())
                .collect::<Vec<_>>(),
        )),
    }
}

/// Permute storage by round-tripping through native tensors.
pub fn permute_storage_native(
    storage: &Storage,
    logical_dims: &[usize],
    perm: &[usize],
) -> Result<Storage> {
    let native = storage_to_native_tensor(storage, logical_dims)?;
    let permuted = permute_native_tensor(&native, perm)?;
    native_tensor_primal_to_storage(&permuted)
}

/// Contract storages via native tensors.
pub fn contract_storage_native(
    storage_a: &Storage,
    dims_a: &[usize],
    axes_a: &[usize],
    storage_b: &Storage,
    dims_b: &[usize],
    axes_b: &[usize],
    _result_dims: &[usize],
) -> Result<Storage> {
    let lhs = storage_to_native_tensor(storage_a, dims_a)?;
    let rhs = storage_to_native_tensor(storage_b, dims_b)?;
    let result = contract_native_tensor(&lhs, axes_a, &rhs, axes_b)?;
    native_tensor_primal_to_storage(&result)
}

/// Outer-product storages via native tensors.
pub fn outer_product_storage_native(
    lhs: &Storage,
    lhs_dims: &[usize],
    rhs: &Storage,
    rhs_dims: &[usize],
    _result_dims: &[usize],
) -> Result<Storage> {
    let lhs = storage_to_native_tensor(lhs, lhs_dims)?;
    let rhs = storage_to_native_tensor(rhs, rhs_dims)?;
    let result = outer_product_native_tensor(&lhs, &rhs)?;
    native_tensor_primal_to_storage(&result)
}

/// Scale storage by a scalar via native tensors.
pub fn scale_storage_native(
    storage: &Storage,
    logical_dims: &[usize],
    scalar: &AnyScalar,
) -> Result<Storage> {
    let native = storage_to_native_tensor(storage, logical_dims)?;
    let scaled = scale_native_tensor(&native, scalar)?;
    native_tensor_primal_to_storage(&scaled)
}

/// Compute `a * lhs + b * rhs` over storages via native tensors.
pub fn axpby_storage_native(
    lhs: &Storage,
    lhs_dims: &[usize],
    a: &AnyScalar,
    rhs: &Storage,
    rhs_dims: &[usize],
    b: &AnyScalar,
) -> Result<Storage> {
    let lhs = storage_to_native_tensor(lhs, lhs_dims)?;
    let rhs = storage_to_native_tensor(rhs, rhs_dims)?;
    let combined = axpby_native_tensor(&lhs, a, &rhs, b)?;
    native_tensor_primal_to_storage(&combined)
}

#[cfg(test)]
mod tests;
