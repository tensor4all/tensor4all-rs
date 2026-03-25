//! Runtime bridge for tenferro-backed execution.
//!
//! This module keeps tensor4all's storage/materialization boundary separate
//! from the canonical compute object: [`tenferro::Tensor`].

use std::collections::HashMap;
use std::env;
use std::sync::{Mutex, OnceLock};

use anyhow::{anyhow, Result};
use num_complex::Complex64;
use tenferro::{set_default_runtime, snapshot, RuntimeContext, Tensor as NativeTensor};
use tenferro_algebra::{Conjugate, Scalar as TfScalar};
use tenferro_device::LogicalMemorySpace;
use tenferro_prims::CpuContext;
use tenferro_tensor::{MemoryOrder, Tensor as TypedTensor};

#[cfg(test)]
use crate::storage::StorageRepr;
use crate::storage::{col_major_strides, NativePayload, Storage};
use crate::tensor_element::TensorElement;
use crate::AnyScalar;

/// Runtime kind for tenferro execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeKind {
    /// CPU runtime.
    Cpu,
    /// CUDA runtime (reserved).
    Cuda,
    /// ROCm runtime (reserved).
    Rocm,
}

fn runtime_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

fn parse_runtime_kind() -> RuntimeKind {
    match env::var("T4A_TENFERRO_RUNTIME") {
        Ok(value) => match value.to_ascii_lowercase().as_str() {
            "cpu" => RuntimeKind::Cpu,
            "cuda" => RuntimeKind::Cuda,
            "rocm" => RuntimeKind::Rocm,
            _ => RuntimeKind::Cpu,
        },
        Err(_) => RuntimeKind::Cpu,
    }
}

fn cpu_threads() -> usize {
    let parsed = env::var("T4A_TENFERRO_CPU_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1);
    parsed.max(1)
}

fn is_missing_default_runtime(err: &anyhow::Error) -> bool {
    err.chain().any(|cause| {
        cause
            .to_string()
            .contains("default runtime is not configured")
    })
}

fn retry_with_default_runtime_if_needed<R>(
    op: &'static str,
    mut f: impl FnMut() -> Result<R>,
) -> Result<R> {
    match f() {
        Ok(value) => Ok(value),
        Err(err) if is_missing_default_runtime(&err) => with_default_runtime(op, f),
        Err(err) => Err(err),
    }
}

/// Run a typed tenferro op against the currently selected runtime.
pub fn with_tenferro_ctx<R>(
    op: &'static str,
    f: impl FnOnce(&mut CpuContext) -> Result<R>,
) -> Result<R> {
    match parse_runtime_kind() {
        RuntimeKind::Cpu => {
            let mut ctx = CpuContext::new(cpu_threads());
            f(&mut ctx)
        }
        RuntimeKind::Cuda => Err(anyhow!(
            "{op}: CUDA runtime is not yet wired in tensor4all tenferro backend"
        )),
        RuntimeKind::Rocm => Err(anyhow!(
            "{op}: ROCm runtime is not yet wired in tensor4all tenferro backend"
        )),
    }
}

pub(crate) fn with_default_runtime<R>(
    op: &'static str,
    f: impl FnOnce() -> Result<R>,
) -> Result<R> {
    let _guard = runtime_lock()
        .lock()
        .map_err(|_| anyhow!("{op}: native runtime lock poisoned"))?;
    match parse_runtime_kind() {
        RuntimeKind::Cpu => {
            let _runtime = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(cpu_threads())));
            f()
        }
        RuntimeKind::Cuda => Err(anyhow!(
            "{op}: CUDA runtime is not yet wired in tensor4all tenferro backend"
        )),
        RuntimeKind::Rocm => Err(anyhow!(
            "{op}: ROCm runtime is not yet wired in tensor4all tenferro backend"
        )),
    }
}

fn native_tensor_from_f64_payload(payload: NativePayload<f64>) -> Result<NativeTensor> {
    let typed = TypedTensor::from_slice(
        &payload.data,
        &payload.payload_dims,
        MemoryOrder::ColumnMajor,
    )
    .map_err(|e| anyhow!("failed to build f64 tensor from storage payload: {e}"))?;
    let native = NativeTensor::from_tensor(typed);
    match payload.axis_classes {
        Some(axis_classes) => NativeTensor::with_axis_classes(native, &axis_classes)
            .map_err(|e| anyhow!("failed to build structured f64 tensor from storage: {e}")),
        None => Ok(native),
    }
}

fn native_tensor_from_c64_payload(payload: NativePayload<Complex64>) -> Result<NativeTensor> {
    let typed = TypedTensor::from_slice(
        &payload.data,
        &payload.payload_dims,
        MemoryOrder::ColumnMajor,
    )
    .map_err(|e| anyhow!("failed to build c64 tensor from storage payload: {e}"))?;
    let native = NativeTensor::from_tensor(typed);
    match payload.axis_classes {
        Some(axis_classes) => NativeTensor::with_axis_classes(native, &axis_classes)
            .map_err(|e| anyhow!("failed to build structured c64 tensor from storage: {e}")),
        None => Ok(native),
    }
}

/// Build a native dense tensor from column-major boundary data.
pub fn dense_native_tensor_from_col_major<T: TensorElement>(
    data: &[T],
    logical_dims: &[usize],
) -> Result<NativeTensor> {
    T::dense_native_tensor_from_col_major(data, logical_dims)
}

/// Build a native diagonal tensor from column-major diagonal payload data.
pub fn diag_native_tensor_from_col_major<T: TensorElement>(
    data: &[T],
    logical_rank: usize,
) -> Result<NativeTensor> {
    T::diag_native_tensor_from_col_major(data, logical_rank)
}

fn dense_f64_storage_from_col_major(
    tensor: &TypedTensor<f64>,
    logical_dims: &[usize],
) -> Result<Storage> {
    let data = materialize_col_major_values(tensor, "f64 dense snapshot materialization")?;
    Storage::from_dense_col_major(data, logical_dims)
}

fn dense_c64_storage_from_col_major(
    tensor: &TypedTensor<Complex64>,
    logical_dims: &[usize],
) -> Result<Storage> {
    let data = materialize_col_major_values(tensor, "c64 dense snapshot materialization")?;
    Storage::from_dense_col_major(data, logical_dims)
}

fn materialize_col_major_values<T>(tensor: &TypedTensor<T>, op: &'static str) -> Result<Vec<T>>
where
    T: TfScalar + Conjugate + Copy,
{
    let col_major = tensor.contiguous(MemoryOrder::ColumnMajor);
    let is_conjugated = col_major.is_conjugated();
    let col_major = if col_major.logical_memory_space() == LogicalMemorySpace::MainMemory {
        col_major
    } else {
        col_major
            .to_memory_space_async(LogicalMemorySpace::MainMemory)
            .map_err(|e| anyhow!("{op}: failed to move tensor to host memory: {e}"))?
    };
    let offset = usize::try_from(col_major.offset())
        .map_err(|_| anyhow!("{op}: negative offset {}", col_major.offset()))?;
    let len = col_major.len();
    let slice = col_major
        .buffer()
        .as_slice()
        .and_then(|values| values.get(offset..offset + len))
        .ok_or_else(|| anyhow!("{op}: expected host-accessible contiguous tensor buffer"))?;
    if is_conjugated {
        Ok(slice.iter().copied().map(Conjugate::conj).collect())
    } else {
        Ok(slice.to_vec())
    }
}

fn snapshot_f64_to_storage(snap: &snapshot::DynTensor) -> Result<Storage> {
    if snap.is_diag() && snap.dims().len() >= 2 {
        let payload = snap
            .payload_f64()
            .ok_or_else(|| anyhow!("expected f64 diagonal payload"))?;
        let data = materialize_col_major_values(payload, "f64 diagonal snapshot materialization")?;
        return Storage::from_diag_col_major(data, snap.dims().len());
    }

    if snap.is_dense() {
        let payload = snap
            .payload_f64()
            .ok_or_else(|| anyhow!("expected f64 dense payload"))?;
        dense_f64_storage_from_col_major(payload, snap.dims())
    } else {
        let payload = snap
            .payload_f64()
            .ok_or_else(|| anyhow!("expected f64 structured payload"))?;
        let data =
            materialize_col_major_values(payload, "f64 structured snapshot materialization")?;
        Storage::new_structured::<f64>(
            data,
            payload.dims().to_vec(),
            col_major_strides(payload.dims()),
            snap.axis_classes().to_vec(),
        )
    }
}

fn snapshot_c64_to_storage(snap: &snapshot::DynTensor) -> Result<Storage> {
    if snap.is_diag() && snap.dims().len() >= 2 {
        let payload = snap
            .payload_c64()
            .ok_or_else(|| anyhow!("expected c64 diagonal payload"))?;
        let data = materialize_col_major_values(payload, "c64 diagonal snapshot materialization")?;
        return Storage::from_diag_col_major(data, snap.dims().len());
    }

    if snap.is_dense() {
        let payload = snap
            .payload_c64()
            .ok_or_else(|| anyhow!("expected c64 dense payload"))?;
        dense_c64_storage_from_col_major(payload, snap.dims())
    } else {
        let payload = snap
            .payload_c64()
            .ok_or_else(|| anyhow!("expected c64 structured payload"))?;
        let data =
            materialize_col_major_values(payload, "c64 structured snapshot materialization")?;
        Storage::new_structured::<Complex64>(
            data,
            payload.dims().to_vec(),
            col_major_strides(payload.dims()),
            snap.axis_classes().to_vec(),
        )
    }
}

fn labels_to_notation(inputs: &[Vec<usize>], output: &[usize]) -> Result<String> {
    let mut id_to_char = HashMap::new();
    let mut next_code = 'a' as u32;

    let mut alloc_label = |id: usize| -> Result<char> {
        if let Some(&ch) = id_to_char.get(&id) {
            return Ok(ch);
        }
        loop {
            let Some(ch) = char::from_u32(next_code) else {
                return Err(anyhow!("ran out of einsum label codepoints"));
            };
            next_code += 1;
            if ch.is_alphanumeric() {
                id_to_char.insert(id, ch);
                return Ok(ch);
            }
        }
    };

    let input_terms: Result<Vec<String>> = inputs
        .iter()
        .map(|ids| ids.iter().map(|&id| alloc_label(id)).collect())
        .collect();
    let output_term: Result<String> = output.iter().map(|&id| alloc_label(id)).collect();

    Ok(format!("{}->{}", input_terms?.join(","), output_term?))
}

fn build_binary_einsum_ids(
    rank_a: usize,
    axes_a: &[usize],
    rank_b: usize,
    axes_b: &[usize],
) -> Result<(Vec<usize>, Vec<usize>, Vec<usize>)> {
    if axes_a.len() != axes_b.len() {
        return Err(anyhow!(
            "binary contraction axes length mismatch: lhs={}, rhs={}",
            axes_a.len(),
            axes_b.len()
        ));
    }

    let mut lhs_ids = vec![usize::MAX; rank_a];
    let mut rhs_ids = vec![usize::MAX; rank_b];
    let mut next_id = 0usize;

    for (&lhs_axis, &rhs_axis) in axes_a.iter().zip(axes_b.iter()) {
        if lhs_axis >= rank_a || rhs_axis >= rank_b {
            return Err(anyhow!(
                "binary contraction axis out of range: lhs_axis={}, rhs_axis={}, lhs_rank={}, rhs_rank={}",
                lhs_axis,
                rhs_axis,
                rank_a,
                rank_b
            ));
        }
        if lhs_ids[lhs_axis] != usize::MAX || rhs_ids[rhs_axis] != usize::MAX {
            return Err(anyhow!(
                "duplicate contraction axis in native binary contraction: lhs_axis={}, rhs_axis={}",
                lhs_axis,
                rhs_axis
            ));
        }
        lhs_ids[lhs_axis] = next_id;
        rhs_ids[rhs_axis] = next_id;
        next_id += 1;
    }

    let mut output_ids = Vec::with_capacity(rank_a + rank_b - 2 * axes_a.len());
    for slot in &mut lhs_ids {
        if *slot == usize::MAX {
            *slot = next_id;
            output_ids.push(next_id);
            next_id += 1;
        }
    }
    for slot in &mut rhs_ids {
        if *slot == usize::MAX {
            *slot = next_id;
            output_ids.push(next_id);
            next_id += 1;
        }
    }

    Ok((lhs_ids, rhs_ids, output_ids))
}

/// Convert legacy [`Storage`] into a primal-mode [`tenferro::Tensor`].
pub fn storage_to_native_tensor(storage: &Storage, logical_dims: &[usize]) -> Result<NativeTensor> {
    if storage.is_c64() {
        native_tensor_from_c64_payload(storage.native_payload_c64(logical_dims)?)
    } else {
        native_tensor_from_f64_payload(storage.native_payload_f64(logical_dims)?)
    }
}

/// Materialize the primal payload of a native tensor back into [`Storage`].
///
/// AD metadata is intentionally dropped at this bridge boundary.
pub fn native_tensor_primal_to_storage(tensor: &NativeTensor) -> Result<Storage> {
    match tensor.primal_snapshot() {
        snapshot::DynTensor::F32(_) | snapshot::DynTensor::C32(_) => Err(anyhow!(
            "tensor4all native bridge currently supports only f64/Complex64 tensors"
        )),
        snap @ snapshot::DynTensor::F64(_) => snapshot_f64_to_storage(&snap),
        snap @ snapshot::DynTensor::C64(_) => snapshot_c64_to_storage(&snap),
    }
}

/// Materialize the dense primal payload of a native tensor as column-major `f64`.
pub fn native_tensor_primal_to_dense_f64_col_major(tensor: &NativeTensor) -> Result<Vec<f64>> {
    retry_with_default_runtime_if_needed("native_tensor_primal_to_dense_f64_col_major", || {
        <f64 as TensorElement>::dense_values_from_native_col_major(tensor)
    })
}

/// Materialize the dense primal payload of a native tensor as column-major `Complex64`.
pub fn native_tensor_primal_to_dense_c64_col_major(
    tensor: &NativeTensor,
) -> Result<Vec<Complex64>> {
    retry_with_default_runtime_if_needed("native_tensor_primal_to_dense_c64_col_major", || {
        <Complex64 as TensorElement>::dense_values_from_native_col_major(tensor)
    })
}

/// Materialize the dense primal payload of a native tensor (generic over element type).
///
/// This is the generic equivalent of [`native_tensor_primal_to_dense_f64_col_major`]
/// and [`native_tensor_primal_to_dense_c64_col_major`].
pub fn native_tensor_primal_to_dense_col_major<T: TensorElement>(
    tensor: &NativeTensor,
) -> Result<Vec<T>> {
    retry_with_default_runtime_if_needed("native_tensor_primal_to_dense_col_major", || {
        T::dense_values_from_native_col_major(tensor)
    })
}

/// Materialize the diagonal payload of a native diagonal tensor as `f64`.
pub fn native_tensor_primal_to_diag_f64(tensor: &NativeTensor) -> Result<Vec<f64>> {
    retry_with_default_runtime_if_needed("native_tensor_primal_to_diag_f64", || {
        <f64 as TensorElement>::diag_values_from_native_temp(tensor)
    })
}

/// Materialize the diagonal payload of a native diagonal tensor as `Complex64`.
pub fn native_tensor_primal_to_diag_c64(tensor: &NativeTensor) -> Result<Vec<Complex64>> {
    retry_with_default_runtime_if_needed("native_tensor_primal_to_diag_c64", || {
        <Complex64 as TensorElement>::diag_values_from_native_temp(tensor)
    })
}

/// Extract the forward tangent of a native tensor as a primal tensor.
pub fn tangent_native_tensor(tensor: &NativeTensor) -> Option<NativeTensor> {
    match tensor {
        NativeTensor::F32(value) => value.tangent().cloned().map(NativeTensor::from_tensor),
        NativeTensor::F64(value) => value.tangent().cloned().map(NativeTensor::from_tensor),
        NativeTensor::C32(value) => value.tangent().cloned().map(NativeTensor::from_tensor),
        NativeTensor::C64(value) => value.tangent().cloned().map(NativeTensor::from_tensor),
    }
}

/// Reshape a native tensor using tensor4all's column-major semantics.
pub fn reshape_col_major_native_tensor(
    tensor: &NativeTensor,
    new_dims: &[usize],
) -> Result<NativeTensor> {
    tensor
        .contiguous(MemoryOrder::ColumnMajor)
        .map_err(|e| anyhow!("native column-major contiguous conversion failed: {e}"))?
        .reshape(new_dims)
        .map_err(|e| anyhow!("native reshape failed: {e}"))
}

/// Compute native QR while preserving AD metadata when supported by upstream.
pub fn qr_native_tensor(tensor: &NativeTensor) -> Result<(NativeTensor, NativeTensor)> {
    with_default_runtime("native_qr", || {
        let out = tensor.qr().map_err(|e| anyhow!("native qr failed: {e}"))?;
        Ok((out.q, out.r))
    })
}

/// Compute native SVD while preserving AD metadata when supported by upstream.
pub fn svd_native_tensor(
    tensor: &NativeTensor,
) -> Result<(NativeTensor, NativeTensor, NativeTensor)> {
    with_default_runtime("native_svd", || {
        let out = tensor
            .svd()
            .map_err(|e| anyhow!("native svd failed: {e}"))?;
        Ok((out.u, out.s, out.vt))
    })
}

/// Sum all elements of a native tensor, preserving AD metadata.
pub fn sum_native_tensor(tensor: &NativeTensor) -> Result<AnyScalar> {
    with_default_runtime("native_sum", || {
        let reduced = tensor
            .sum()
            .map_err(|e| anyhow!("native sum failed: {e}"))?;
        AnyScalar::from_native(reduced)
    })
}

/// Scale a native tensor with a tensor4all scalar while preserving AD metadata.
pub fn scale_native_tensor(tensor: &NativeTensor, scalar: &AnyScalar) -> Result<NativeTensor> {
    with_default_runtime("native_scale", || {
        tensor
            .scale(scalar.as_native())
            .map_err(|e| anyhow!("native scale failed: {e}"))
    })
}

/// Compute `a * lhs + b * rhs` on native tensors while preserving AD metadata.
pub fn axpby_native_tensor(
    lhs: &NativeTensor,
    a: &AnyScalar,
    rhs: &NativeTensor,
    b: &AnyScalar,
) -> Result<NativeTensor> {
    with_default_runtime("native_axpby", || {
        lhs.axpby(a.as_native(), rhs, b.as_native())
            .map_err(|e| anyhow!("native axpby failed: {e}"))
    })
}

/// Execute native structured einsum on multiple tensors.
pub fn einsum_native_tensors(
    operands: &[(&NativeTensor, &[usize])],
    output_ids: &[usize],
) -> Result<NativeTensor> {
    if operands.is_empty() {
        return Err(anyhow!("native einsum requires at least one operand"));
    }

    let input_ids: Vec<Vec<usize>> = operands.iter().map(|(_, ids)| ids.to_vec()).collect();
    let final_operands: Vec<&NativeTensor> = operands.iter().map(|(tensor, _)| *tensor).collect();
    let notation = labels_to_notation(&input_ids, output_ids)?;

    with_default_runtime("native_einsum", || {
        NativeTensor::einsum(&notation, &final_operands)
            .map_err(|e| anyhow!("native einsum failed: {e}"))
    })
}

/// Permute a native tensor through tenferro frontend operations.
pub fn permute_native_tensor(tensor: &NativeTensor, perm: &[usize]) -> Result<NativeTensor> {
    tensor
        .permute(perm)
        .map_err(|e| anyhow!("native permute failed: {e}"))
}

/// Contract two native tensors with AD-preserving einsum execution.
pub fn contract_native_tensor(
    lhs: &NativeTensor,
    axes_lhs: &[usize],
    rhs: &NativeTensor,
    axes_rhs: &[usize],
) -> Result<NativeTensor> {
    let (lhs_ids, rhs_ids, output_ids) =
        build_binary_einsum_ids(lhs.ndim(), axes_lhs, rhs.ndim(), axes_rhs)?;
    einsum_native_tensors(&[(lhs, &lhs_ids), (rhs, &rhs_ids)], &output_ids)
}

/// Compute outer product of two native tensors.
pub fn outer_product_native_tensor(lhs: &NativeTensor, rhs: &NativeTensor) -> Result<NativeTensor> {
    contract_native_tensor(lhs, &[], rhs, &[])
}

/// Conjugate a native tensor while preserving AD metadata.
pub fn conj_native_tensor(tensor: &NativeTensor) -> Result<NativeTensor> {
    Ok(tensor.conj())
}

/// Permute storage through native tenferro execution.
pub fn permute_storage_native(
    storage: &Storage,
    logical_dims: &[usize],
    perm: &[usize],
) -> Result<Storage> {
    let native = storage_to_native_tensor(storage, logical_dims)?;
    let permuted = permute_native_tensor(&native, perm)?;
    native_tensor_primal_to_storage(&permuted)
}

/// Contract two storages through native tenferro execution.
pub fn contract_storage_native(
    storage_a: &Storage,
    dims_a: &[usize],
    axes_a: &[usize],
    storage_b: &Storage,
    dims_b: &[usize],
    axes_b: &[usize],
    result_dims: &[usize],
) -> Result<Storage> {
    let lhs = storage_to_native_tensor(storage_a, dims_a)?;
    let rhs = storage_to_native_tensor(storage_b, dims_b)?;
    let result = contract_native_tensor(&lhs, axes_a, &rhs, axes_b)?;
    let storage = native_tensor_primal_to_storage(&result)?;
    let _ = result_dims;
    Ok(storage)
}

/// Compute an outer product through native tenferro execution.
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

/// Apply native tenferro mixed scalar/tensor scaling at the storage boundary.
pub fn scale_storage_native(
    storage: &Storage,
    logical_dims: &[usize],
    scalar: &AnyScalar,
) -> Result<Storage> {
    let native = storage_to_native_tensor(storage, logical_dims)?;
    let scaled = scale_native_tensor(&native, scalar)?;
    native_tensor_primal_to_storage(&scaled)
}

/// Apply native tenferro fused `a * lhs + b * rhs` at the storage boundary.
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
