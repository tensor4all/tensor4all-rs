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

use crate::layout::{dense_linear_multi_index, storage_strides};
use crate::storage::{
    col_major_strides, DenseStorageC64, DenseStorageF64, DiagStorageC64, DiagStorageF64, Storage,
    StructuredStorage,
};
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

fn legacy_row_major_to_col_major<T: Clone>(data: &[T], dims: &[usize]) -> Result<Vec<T>> {
    let total_len: usize = dims.iter().product();
    anyhow::ensure!(
        data.len() == total_len,
        "legacy row-major payload length {} does not match dims {:?} (expected {})",
        data.len(),
        dims,
        total_len
    );
    if total_len == 0 {
        return Ok(Vec::new());
    }

    let row_major_strides = storage_strides(dims);
    let mut out = Vec::with_capacity(total_len);
    for linear in 0..total_len {
        let index = dense_linear_multi_index(dims, linear)?;
        let offset: usize = index
            .iter()
            .zip(row_major_strides.iter())
            .map(|(&coord, &stride)| coord * stride)
            .sum();
        out.push(data[offset].clone());
    }
    Ok(out)
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

fn typed_f64_from_storage(storage: &Storage, logical_dims: &[usize]) -> Result<TypedTensor<f64>> {
    match storage {
        Storage::DenseF64(ds) => {
            let logical_len: usize = logical_dims.iter().product();
            if logical_len != ds.len() {
                return Err(anyhow!(
                    "logical dims {:?} (len={}) do not match dense f64 storage len {}",
                    logical_dims,
                    logical_len,
                    ds.len()
                ));
            }
            let col_major = legacy_row_major_to_col_major(ds.as_slice(), logical_dims)?;
            TypedTensor::from_slice(&col_major, logical_dims, MemoryOrder::ColumnMajor)
                .map_err(|e| anyhow!("failed to build f64 tensor from storage: {e}"))
        }
        Storage::DiagF64(ds) => {
            TypedTensor::from_slice(ds.as_slice(), &[ds.len()], MemoryOrder::ColumnMajor)
                .map_err(|e| anyhow!("failed to build f64 diagonal payload from storage: {e}"))
        }
        Storage::StructuredF64(ds) => {
            anyhow::ensure!(
                ds.logical_dims() == logical_dims,
                "logical dims {:?} do not match structured f64 storage logical dims {:?}",
                logical_dims,
                ds.logical_dims()
            );
            TypedTensor::from_slice(
                &ds.payload_col_major_vec(),
                ds.payload_dims(),
                MemoryOrder::ColumnMajor,
            )
            .map_err(|e| anyhow!("failed to build structured f64 tensor from storage: {e}"))
        }
        Storage::DenseC64(_) | Storage::DiagC64(_) => Err(anyhow!(
            "complex storage cannot be converted to f64 tenferro tensor"
        )),
        Storage::StructuredC64(_) => Err(anyhow!(
            "complex structured storage cannot be converted to f64 tenferro tensor"
        )),
    }
}

fn typed_c64_from_storage(
    storage: &Storage,
    logical_dims: &[usize],
) -> Result<TypedTensor<Complex64>> {
    match storage {
        Storage::DenseC64(ds) => {
            let logical_len: usize = logical_dims.iter().product();
            if logical_len != ds.len() {
                return Err(anyhow!(
                    "logical dims {:?} (len={}) do not match dense c64 storage len {}",
                    logical_dims,
                    logical_len,
                    ds.len()
                ));
            }
            let col_major = legacy_row_major_to_col_major(ds.as_slice(), logical_dims)?;
            TypedTensor::from_slice(&col_major, logical_dims, MemoryOrder::ColumnMajor)
                .map_err(|e| anyhow!("failed to build c64 tensor from storage: {e}"))
        }
        Storage::DiagC64(ds) => {
            TypedTensor::from_slice(ds.as_slice(), &[ds.len()], MemoryOrder::ColumnMajor)
                .map_err(|e| anyhow!("failed to build c64 diagonal payload from storage: {e}"))
        }
        Storage::DenseF64(ds) => {
            let logical_len: usize = logical_dims.iter().product();
            if logical_len != ds.len() {
                return Err(anyhow!(
                    "logical dims {:?} (len={}) do not match dense f64 storage len {} for promotion",
                    logical_dims,
                    logical_len,
                    ds.len()
                ));
            }
            let promoted: Vec<Complex64> =
                legacy_row_major_to_col_major(ds.as_slice(), logical_dims)?
                    .into_iter()
                    .map(|value| Complex64::new(value, 0.0))
                    .collect();
            TypedTensor::from_slice(&promoted, logical_dims, MemoryOrder::ColumnMajor)
                .map_err(|e| anyhow!("failed to promote dense f64 tensor to c64: {e}"))
        }
        Storage::DiagF64(ds) => {
            let promoted: Vec<Complex64> = ds
                .as_slice()
                .iter()
                .copied()
                .map(|value| Complex64::new(value, 0.0))
                .collect();
            TypedTensor::from_slice(&promoted, &[ds.len()], MemoryOrder::ColumnMajor)
                .map_err(|e| anyhow!("failed to promote diag f64 tensor to c64: {e}"))
        }
        Storage::StructuredC64(ds) => {
            anyhow::ensure!(
                ds.logical_dims() == logical_dims,
                "logical dims {:?} do not match structured c64 storage logical dims {:?}",
                logical_dims,
                ds.logical_dims()
            );
            TypedTensor::from_slice(
                &ds.payload_col_major_vec(),
                ds.payload_dims(),
                MemoryOrder::ColumnMajor,
            )
            .map_err(|e| anyhow!("failed to build structured c64 tensor from storage: {e}"))
        }
        Storage::StructuredF64(ds) => {
            anyhow::ensure!(
                ds.logical_dims() == logical_dims,
                "logical dims {:?} do not match structured f64 storage logical dims {:?} for promotion",
                logical_dims,
                ds.logical_dims()
            );
            let promoted: Vec<Complex64> = ds
                .payload_col_major_vec()
                .into_iter()
                .map(|value| Complex64::new(value, 0.0))
                .collect();
            TypedTensor::from_slice(&promoted, ds.payload_dims(), MemoryOrder::ColumnMajor)
                .map_err(|e| anyhow!("failed to promote structured f64 tensor to c64: {e}"))
        }
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

fn row_major_f64_storage(tensor: &TypedTensor<f64>, logical_dims: &[usize]) -> Result<Storage> {
    let data = materialize_row_major_values(tensor, "f64 row-major storage materialization")?;
    Ok(Storage::DenseF64(DenseStorageF64::from_vec_with_shape(
        data,
        logical_dims,
    )))
}

fn row_major_c64_storage(
    tensor: &TypedTensor<Complex64>,
    logical_dims: &[usize],
) -> Result<Storage> {
    let data = materialize_row_major_values(tensor, "c64 row-major storage materialization")?;
    Ok(Storage::DenseC64(DenseStorageC64::from_vec_with_shape(
        data,
        logical_dims,
    )))
}

fn materialize_row_major_values<T>(tensor: &TypedTensor<T>, op: &'static str) -> Result<Vec<T>>
where
    T: TfScalar + Conjugate + Copy,
{
    let row_major = tensor.contiguous(MemoryOrder::RowMajor);
    let is_conjugated = row_major.is_conjugated();
    let row_major = if row_major.logical_memory_space() == LogicalMemorySpace::MainMemory {
        row_major
    } else {
        row_major
            .to_memory_space_async(LogicalMemorySpace::MainMemory)
            .map_err(|e| anyhow!("{op}: failed to move tensor to host memory: {e}"))?
    };
    let offset = usize::try_from(row_major.offset())
        .map_err(|_| anyhow!("{op}: negative offset {}", row_major.offset()))?;
    let len = row_major.len();
    let slice = row_major
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
        let data = materialize_row_major_values(payload, "f64 diagonal snapshot materialization")?;
        return Ok(Storage::DiagF64(DiagStorageF64::from_vec(data)));
    }

    if snap.is_dense() {
        let payload = snap
            .payload_f64()
            .ok_or_else(|| anyhow!("expected f64 dense payload"))?;
        row_major_f64_storage(payload, snap.dims())
    } else {
        let payload = snap
            .payload_f64()
            .ok_or_else(|| anyhow!("expected f64 structured payload"))?;
        let data =
            materialize_col_major_values(payload, "f64 structured snapshot materialization")?;
        Ok(Storage::StructuredF64(StructuredStorage::new(
            data,
            payload.dims().to_vec(),
            col_major_strides(payload.dims()),
            snap.axis_classes().to_vec(),
        )?))
    }
}

fn snapshot_c64_to_storage(snap: &snapshot::DynTensor) -> Result<Storage> {
    if snap.is_diag() && snap.dims().len() >= 2 {
        let payload = snap
            .payload_c64()
            .ok_or_else(|| anyhow!("expected c64 diagonal payload"))?;
        let data = materialize_row_major_values(payload, "c64 diagonal snapshot materialization")?;
        return Ok(Storage::DiagC64(DiagStorageC64::from_vec(data)));
    }

    if snap.is_dense() {
        let payload = snap
            .payload_c64()
            .ok_or_else(|| anyhow!("expected c64 dense payload"))?;
        row_major_c64_storage(payload, snap.dims())
    } else {
        let payload = snap
            .payload_c64()
            .ok_or_else(|| anyhow!("expected c64 structured payload"))?;
        let data =
            materialize_col_major_values(payload, "c64 structured snapshot materialization")?;
        Ok(Storage::StructuredC64(StructuredStorage::new(
            data,
            payload.dims().to_vec(),
            col_major_strides(payload.dims()),
            snap.axis_classes().to_vec(),
        )?))
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
    match storage {
        Storage::DenseF64(_) => Ok(NativeTensor::from_tensor(typed_f64_from_storage(
            storage,
            logical_dims,
        )?)),
        Storage::DiagF64(_) => {
            let payload = NativeTensor::from_tensor(typed_f64_from_storage(storage, logical_dims)?);
            payload
                .diag_embed(logical_dims.len())
                .map_err(|e| anyhow!("failed to build f64 diag tensor from storage: {e}"))
        }
        Storage::DenseC64(_) | Storage::DiagC64(_) => {
            let payload = NativeTensor::from_tensor(typed_c64_from_storage(storage, logical_dims)?);
            if storage.is_diag() {
                payload
                    .diag_embed(logical_dims.len())
                    .map_err(|e| anyhow!("failed to build c64 diag tensor from storage: {e}"))
            } else {
                Ok(payload)
            }
        }
        Storage::StructuredF64(value) => {
            let payload = NativeTensor::from_tensor(typed_f64_from_storage(storage, logical_dims)?);
            if value.is_dense() {
                Ok(payload)
            } else {
                NativeTensor::with_axis_classes(payload, value.axis_classes())
                    .map_err(|e| anyhow!("failed to build structured f64 tensor from storage: {e}"))
            }
        }
        Storage::StructuredC64(value) => {
            let payload = NativeTensor::from_tensor(typed_c64_from_storage(storage, logical_dims)?);
            if value.is_dense() {
                Ok(payload)
            } else {
                NativeTensor::with_axis_classes(payload, value.axis_classes())
                    .map_err(|e| anyhow!("failed to build structured c64 tensor from storage: {e}"))
            }
        }
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
    <f64 as TensorElement>::dense_values_from_native_col_major(tensor)
}

/// Materialize the dense primal payload of a native tensor as column-major `Complex64`.
pub fn native_tensor_primal_to_dense_c64_col_major(
    tensor: &NativeTensor,
) -> Result<Vec<Complex64>> {
    <Complex64 as TensorElement>::dense_values_from_native_col_major(tensor)
}

/// Materialize the diagonal payload of a native diagonal tensor as `f64`.
pub fn native_tensor_primal_to_diag_f64(tensor: &NativeTensor) -> Result<Vec<f64>> {
    <f64 as TensorElement>::diag_values_from_native_temp(tensor)
}

/// Materialize the diagonal payload of a native diagonal tensor as `Complex64`.
pub fn native_tensor_primal_to_diag_c64(tensor: &NativeTensor) -> Result<Vec<Complex64>> {
    <Complex64 as TensorElement>::diag_values_from_native_temp(tensor)
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
mod tests {
    use super::*;
    use crate::storage::{DenseStorageC64, DenseStorageF64, DiagStorageF64, Storage};

    fn assert_storage_eq(lhs: &Storage, rhs: &Storage) {
        match (lhs, rhs) {
            (Storage::DenseF64(a), Storage::DenseF64(b)) => {
                assert_eq!(a.dims(), b.dims());
                assert_eq!(a.as_slice(), b.as_slice());
            }
            (Storage::DenseC64(a), Storage::DenseC64(b)) => {
                assert_eq!(a.dims(), b.dims());
                assert_eq!(a.as_slice(), b.as_slice());
            }
            (Storage::DiagF64(a), Storage::DiagF64(b)) => {
                assert_eq!(a.as_slice(), b.as_slice());
            }
            (Storage::DiagC64(a), Storage::DiagC64(b)) => {
                assert_eq!(a.as_slice(), b.as_slice());
            }
            (Storage::StructuredF64(a), Storage::StructuredF64(b)) => {
                assert_eq!(a.payload_dims(), b.payload_dims());
                assert_eq!(a.strides(), b.strides());
                assert_eq!(a.axis_classes(), b.axis_classes());
                assert_eq!(a.data(), b.data());
            }
            (Storage::StructuredC64(a), Storage::StructuredC64(b)) => {
                assert_eq!(a.payload_dims(), b.payload_dims());
                assert_eq!(a.strides(), b.strides());
                assert_eq!(a.axis_classes(), b.axis_classes());
                assert_eq!(a.data(), b.data());
            }
            _ => panic!(
                "storage mismatch: lhs variant {:?}, rhs variant {:?}",
                std::mem::discriminant(lhs),
                std::mem::discriminant(rhs)
            ),
        }
    }

    #[test]
    fn storage_native_roundtrip_dense_f64() {
        let storage = Storage::DenseF64(DenseStorageF64::from_vec_with_shape(
            vec![1.0, 2.0, 3.0, 4.0],
            &[2, 2],
        ));

        let native = storage_to_native_tensor(&storage, &[2, 2]).unwrap();
        let roundtrip = native_tensor_primal_to_storage(&native).unwrap();

        assert_storage_eq(&roundtrip, &storage);
    }

    #[test]
    fn storage_native_roundtrip_diag_preserves_diag_layout() {
        let storage = Storage::DiagF64(DiagStorageF64::from_vec(vec![2.0, -1.0, 4.0]));

        let native = storage_to_native_tensor(&storage, &[3, 3]).unwrap();
        let roundtrip = native_tensor_primal_to_storage(&native).unwrap();

        assert!(native.is_diag());
        assert_storage_eq(&roundtrip, &storage);
    }

    #[test]
    fn storage_native_roundtrip_structured_preserves_axis_classes() {
        let payload = NativeTensor::from_slice(&[1.0_f64, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let native = NativeTensor::with_axis_classes(payload, &[0, 1, 1]).unwrap();

        let storage = native_tensor_primal_to_storage(&native).unwrap();
        let roundtrip = storage_to_native_tensor(&storage, &[2, 2, 2]).unwrap();

        match &storage {
            Storage::StructuredF64(value) => {
                assert_eq!(value.axis_classes(), &[0, 1, 1]);
                assert_eq!(value.payload_dims(), &[2, 2]);
            }
            other => panic!("expected StructuredF64 storage, got {other:?}"),
        }
        assert_eq!(roundtrip.dims(), &[2, 2, 2]);
        assert_eq!(roundtrip.axis_classes(), &[0, 1, 1]);
        assert!(!roundtrip.is_dense());
        assert!(!roundtrip.is_diag());
    }

    #[test]
    fn sum_native_tensor_returns_rank0_scalar() {
        let storage = Storage::DenseC64(DenseStorageC64::from_vec_with_shape(
            vec![Complex64::new(1.0, -1.0), Complex64::new(-0.5, 2.0)],
            &[2],
        ));
        let native = storage_to_native_tensor(&storage, &[2]).unwrap();

        let sum = sum_native_tensor(&native).unwrap();

        assert!(sum.is_complex());
        assert_eq!(sum.as_c64(), Some(Complex64::new(0.5, 1.0)));
    }

    #[test]
    fn native_snapshot_materializes_lazy_conjugation() {
        let storage = Storage::DenseC64(DenseStorageC64::from_vec_with_shape(
            vec![
                Complex64::new(1.0, 2.0),
                Complex64::new(-3.0, 4.5),
                Complex64::new(0.0, -1.0),
                Complex64::new(2.5, 0.25),
            ],
            &[2, 2],
        ));
        let native = storage_to_native_tensor(&storage, &[2, 2]).unwrap();

        let conjugated = conj_native_tensor(&native).unwrap();
        let snapshot = native_tensor_primal_to_storage(&conjugated).unwrap();

        let expected = Storage::DenseC64(DenseStorageC64::from_vec_with_shape(
            vec![
                Complex64::new(1.0, -2.0),
                Complex64::new(-3.0, -4.5),
                Complex64::new(0.0, 1.0),
                Complex64::new(2.5, -0.25),
            ],
            &[2, 2],
        ));
        assert_storage_eq(&snapshot, &expected);
    }

    #[test]
    fn native_einsum_accepts_unsorted_nonfirst_operand_labels() {
        let lhs = storage_to_native_tensor(
            &Storage::DenseF64(DenseStorageF64::from_vec_with_shape(
                vec![1.0, 2.0, 3.0, 4.0],
                &[2, 2],
            )),
            &[2, 2],
        )
        .unwrap();
        let rhs = storage_to_native_tensor(
            &Storage::DenseF64(DenseStorageF64::from_vec_with_shape(
                vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
                &[3, 2],
            )),
            &[3, 2],
        )
        .unwrap();

        let out = einsum_native_tensors(&[(&lhs, &[0, 1]), (&rhs, &[2, 1])], &[0, 2]).unwrap();
        let snapshot = native_tensor_primal_to_storage(&out).unwrap();

        let expected = Storage::DenseF64(DenseStorageF64::from_vec_with_shape(
            vec![50.0, 110.0, 170.0, 110.0, 250.0, 390.0],
            &[2, 3],
        ));
        assert_storage_eq(&snapshot, &expected);
    }

    #[test]
    fn contract_native_tensor_restores_rhs_free_axis_order() {
        let lhs = storage_to_native_tensor(
            &Storage::DenseF64(DenseStorageF64::from_vec_with_shape(
                vec![1.0, 2.0, 3.0, 4.0],
                &[2, 2],
            )),
            &[2, 2],
        )
        .unwrap();
        let rhs = storage_to_native_tensor(
            &Storage::DenseF64(DenseStorageF64::from_vec_with_shape(
                vec![10.0, 20.0, 30.0, 40.0],
                &[2, 2],
            )),
            &[2, 2],
        )
        .unwrap();

        let out = contract_native_tensor(&lhs, &[1], &rhs, &[1]).unwrap();
        let snapshot = native_tensor_primal_to_storage(&out).unwrap();

        let expected = Storage::DenseF64(DenseStorageF64::from_vec_with_shape(
            vec![50.0, 110.0, 110.0, 250.0],
            &[2, 2],
        ));
        assert_storage_eq(&snapshot, &expected);
    }

    #[test]
    fn dense_native_tensor_column_major_roundtrip_preserves_linearization() {
        let native =
            dense_native_tensor_from_col_major(&[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
                .unwrap();

        let values = native_tensor_primal_to_dense_f64_col_major(&native).unwrap();

        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn permute_storage_native_dense_matches_expected_data() {
        let storage = Storage::DenseF64(DenseStorageF64::from_vec_with_shape(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 3],
        ));

        let native = permute_storage_native(&storage, &[2, 3], &[1, 0]).unwrap();

        let expected = Storage::DenseF64(DenseStorageF64::from_vec_with_shape(
            vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
            &[3, 2],
        ));
        assert_storage_eq(&native, &expected);
    }

    #[test]
    fn reshape_col_major_native_tensor_handles_noncontiguous_permuted_input() {
        let native = dense_native_tensor_from_col_major(
            &(1..=24).map(|x| x as f64).collect::<Vec<_>>(),
            &[2, 3, 2, 2],
        )
        .unwrap();
        let permuted = permute_native_tensor(&native, &[0, 2, 1, 3]).unwrap();
        let permuted_values = native_tensor_primal_to_dense_f64_col_major(&permuted).unwrap();

        let reshaped = reshape_col_major_native_tensor(&permuted, &[4, 6]).unwrap();
        let reshaped_values = native_tensor_primal_to_dense_f64_col_major(&reshaped).unwrap();

        assert_eq!(reshaped_values, permuted_values);
    }
}
