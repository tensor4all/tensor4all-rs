//! Runtime bridge for tenferro backend execution.
//!
//! This module centralizes runtime selection so backend code stays
//! implementation-agnostic (CPU today, GPU-ready extension point).

use anyhow::{anyhow, Result};
use num_complex::{Complex32, Complex64};
use std::collections::HashMap;
use std::env;
use std::sync::{Mutex, OnceLock};
use tenferro_algebra::Scalar;
use tenferro_dyadtensor::{
    ad, set_default_runtime, AdTensor, AdValue, DynAdScalar, DynAdTensor, RuntimeContext,
    ScalarType, StructuredTensor,
};
use tenferro_prims::{CpuBackend, CpuContext};
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::einsum::{einsum_storage, EinsumInput as BackendEinsumInput};
use crate::storage::{
    contract_storage, DenseStorageC64, DenseStorageF64, DiagStorageC64, DiagStorageF64, Storage,
};
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

/// Active tenferro prims backend used by tensor4all.
///
/// This alias keeps backend selection localized to this bridge module.
pub(crate) type ActivePrimsBackend = CpuBackend;

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

/// Run a tenferro op against currently selected runtime.
///
/// Current implementation executes on CPU and returns explicit errors for GPU
/// runtime requests until tenferro GPU runtime wiring is enabled in tensor4all.
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
            "{}: CUDA runtime is not yet wired in tensor4all tenferro backend",
            op
        )),
        RuntimeKind::Rocm => Err(anyhow!(
            "{}: ROCm runtime is not yet wired in tensor4all tenferro backend",
            op
        )),
    }
}

fn with_default_runtime<R>(op: &'static str, f: impl FnOnce() -> Result<R>) -> Result<R> {
    let _guard = runtime_lock()
        .lock()
        .map_err(|_| anyhow!("{op}: native runtime lock poisoned"))?;
    match parse_runtime_kind() {
        RuntimeKind::Cpu => {
            let _runtime = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(cpu_threads())));
            f()
        }
        RuntimeKind::Cuda => Err(anyhow!(
            "{}: CUDA runtime is not yet wired in tensor4all tenferro backend",
            op
        )),
        RuntimeKind::Rocm => Err(anyhow!(
            "{}: ROCm runtime is not yet wired in tensor4all tenferro backend",
            op
        )),
    }
}

fn dense_f64_to_tensor(storage: &Storage, logical_dims: &[usize]) -> Result<Tensor<f64>> {
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
            Tensor::from_slice(ds.as_slice(), logical_dims, MemoryOrder::RowMajor)
                .map_err(|e| anyhow!("failed to build f64 tensor from storage: {e}"))
        }
        Storage::DiagF64(_) => {
            dense_f64_to_tensor(&storage.to_dense_storage(logical_dims), logical_dims)
        }
        Storage::DenseC64(_) | Storage::DiagC64(_) => Err(anyhow!(
            "complex storage cannot be converted to f64 DynAdTensor"
        )),
    }
}

fn diag_f64_to_structured(
    storage: &Storage,
    logical_dims: &[usize],
) -> Result<StructuredTensor<f64>> {
    match storage {
        Storage::DiagF64(ds) => {
            let payload = Tensor::from_slice(ds.as_slice(), &[ds.len()], MemoryOrder::RowMajor)
                .map_err(|e| anyhow!("failed to build f64 diagonal payload from storage: {e}"))?;
            StructuredTensor::from_diagonal_vector(payload, logical_dims.len())
                .map_err(|e| anyhow!("failed to build f64 structured diagonal tensor: {e}"))
        }
        Storage::DenseF64(_) => Ok(StructuredTensor::from_dense(dense_f64_to_tensor(
            storage,
            logical_dims,
        )?)),
        Storage::DenseC64(_) | Storage::DiagC64(_) => Err(anyhow!(
            "complex storage cannot be converted to f64 DynAdTensor"
        )),
    }
}

fn dense_c64_to_tensor(
    storage: &Storage,
    logical_dims: &[usize],
) -> Result<Tensor<num_complex::Complex64>> {
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
            Tensor::from_slice(ds.as_slice(), logical_dims, MemoryOrder::RowMajor)
                .map_err(|e| anyhow!("failed to build c64 tensor from storage: {e}"))
        }
        Storage::DiagC64(_) => {
            dense_c64_to_tensor(&storage.to_dense_storage(logical_dims), logical_dims)
        }
        Storage::DenseF64(_) | Storage::DiagF64(_) => Err(anyhow!(
            "real storage cannot be converted to c64 DynAdTensor without promotion"
        )),
    }
}

fn diag_c64_to_structured(
    storage: &Storage,
    logical_dims: &[usize],
) -> Result<StructuredTensor<num_complex::Complex64>> {
    match storage {
        Storage::DiagC64(ds) => {
            let payload = Tensor::from_slice(ds.as_slice(), &[ds.len()], MemoryOrder::RowMajor)
                .map_err(|e| anyhow!("failed to build c64 diagonal payload from storage: {e}"))?;
            StructuredTensor::from_diagonal_vector(payload, logical_dims.len())
                .map_err(|e| anyhow!("failed to build c64 structured diagonal tensor: {e}"))
        }
        Storage::DenseC64(_) => Ok(StructuredTensor::from_dense(dense_c64_to_tensor(
            storage,
            logical_dims,
        )?)),
        Storage::DenseF64(_) | Storage::DiagF64(_) => Err(anyhow!(
            "real storage cannot be converted to c64 DynAdTensor without promotion"
        )),
    }
}

fn tensor_f64_to_storage(tensor: &Tensor<f64>) -> Result<Storage> {
    let row_major = tensor.contiguous(MemoryOrder::RowMajor);
    let data = row_major
        .buffer()
        .as_slice()
        .ok_or_else(|| anyhow!("expected host-accessible f64 tensor buffer"))?
        .to_vec();
    Ok(Storage::DenseF64(DenseStorageF64::from_vec_with_shape(
        data,
        tensor.dims(),
    )))
}

fn tensor_c64_to_storage(tensor: &Tensor<num_complex::Complex64>) -> Result<Storage> {
    let row_major = tensor.contiguous(MemoryOrder::RowMajor);
    let data = row_major
        .buffer()
        .as_slice()
        .ok_or_else(|| anyhow!("expected host-accessible c64 tensor buffer"))?
        .to_vec();
    Ok(Storage::DenseC64(DenseStorageC64::from_vec_with_shape(
        data,
        tensor.dims(),
    )))
}

fn structured_f64_to_storage(tensor: &StructuredTensor<f64>) -> Result<Storage> {
    if tensor.is_diag() && tensor.logical_dims().len() >= 2 {
        let row_major = tensor.payload().contiguous(MemoryOrder::RowMajor);
        let data = row_major
            .buffer()
            .as_slice()
            .ok_or_else(|| anyhow!("expected host-accessible f64 diagonal tensor buffer"))?
            .to_vec();
        return Ok(Storage::DiagF64(DiagStorageF64::from_vec(data)));
    }
    if tensor.is_dense() || tensor.logical_dims().len() <= 1 {
        return tensor_f64_to_storage(
            &tensor
                .payload()
                .contiguous(MemoryOrder::RowMajor)
                .reshape(tensor.logical_dims())
                .map_err(|e| anyhow!("failed to reshape dense f64 structured tensor: {e}"))?,
        );
    }
    Err(anyhow!(
        "tensor4all native bridge does not yet support snapshotting non-dense/non-diag structured f64 tensors"
    ))
}

fn structured_c64_to_storage(tensor: &StructuredTensor<num_complex::Complex64>) -> Result<Storage> {
    if tensor.is_diag() && tensor.logical_dims().len() >= 2 {
        let row_major = tensor.payload().contiguous(MemoryOrder::RowMajor);
        let data = row_major
            .buffer()
            .as_slice()
            .ok_or_else(|| anyhow!("expected host-accessible c64 diagonal tensor buffer"))?
            .to_vec();
        return Ok(Storage::DiagC64(DiagStorageC64::from_vec(data)));
    }
    if tensor.is_dense() || tensor.logical_dims().len() <= 1 {
        return tensor_c64_to_storage(
            &tensor
                .payload()
                .contiguous(MemoryOrder::RowMajor)
                .reshape(tensor.logical_dims())
                .map_err(|e| anyhow!("failed to reshape dense c64 structured tensor: {e}"))?,
        );
    }
    Err(anyhow!(
        "tensor4all native bridge does not yet support snapshotting non-dense/non-diag structured c64 tensors"
    ))
}

fn scalar_from_structured_tensor_element<T>(tensor: &StructuredTensor<T>) -> Result<T>
where
    T: Scalar + Copy,
{
    let len: usize = tensor.logical_dims().iter().product();
    if len != 1 {
        return Err(anyhow!(
            "tensor-to-scalar conversion requires exactly one element, got dims {:?}",
            tensor.logical_dims()
        ));
    }
    let row_major = tensor.payload().contiguous(MemoryOrder::RowMajor);
    row_major
        .buffer()
        .as_slice()
        .and_then(|slice| slice.first().copied())
        .ok_or_else(|| anyhow!("expected host-accessible scalar tensor buffer"))
}

fn ad_tensor_to_scalar_typed<T>(tensor: &AdTensor<T>) -> Result<AdValue<T>>
where
    T: Scalar + Copy,
{
    match tensor.clone().into_value() {
        AdValue::Primal(primal) => Ok(AdValue::Primal(scalar_from_structured_tensor_element(
            &primal,
        )?)),
        AdValue::Forward { primal, tangent } => Ok(AdValue::Forward {
            primal: scalar_from_structured_tensor_element(&primal)?,
            tangent: scalar_from_structured_tensor_element(&tangent)?,
        }),
        AdValue::Reverse {
            primal,
            node,
            tape,
            tangent,
        } => Ok(AdValue::Reverse {
            primal: scalar_from_structured_tensor_element(&primal)?,
            node,
            tape,
            tangent: tangent
                .as_ref()
                .map(scalar_from_structured_tensor_element)
                .transpose()?,
        }),
    }
}

fn scalar_type_to_minus_one(scalar_type: ScalarType) -> DynAdScalar {
    match scalar_type {
        ScalarType::F32 => DynAdScalar::from(-1.0_f32),
        ScalarType::F64 => DynAdScalar::from(-1.0_f64),
        ScalarType::C32 => DynAdScalar::from(Complex32::new(-1.0, 0.0)),
        ScalarType::C64 => DynAdScalar::from(Complex64::new(-1.0, 0.0)),
    }
}

fn promote_real_tensor_to_complex(tensor: &DynAdTensor) -> Result<DynAdTensor> {
    match tensor.scalar_type() {
        ScalarType::F32 | ScalarType::F64 => {
            let imag = tensor.imag_part()?;
            DynAdTensor::compose_complex(tensor.clone(), imag)
                .map_err(|e| anyhow!("native real->complex promotion failed: {e}"))
        }
        ScalarType::C32 | ScalarType::C64 => Ok(tensor.clone()),
    }
}

fn promote_native_operands(
    lhs: &DynAdTensor,
    rhs: &DynAdTensor,
) -> Result<(DynAdTensor, DynAdTensor)> {
    match (lhs.scalar_type(), rhs.scalar_type()) {
        (ScalarType::F32, ScalarType::F32)
        | (ScalarType::F64, ScalarType::F64)
        | (ScalarType::C32, ScalarType::C32)
        | (ScalarType::C64, ScalarType::C64) => Ok((lhs.clone(), rhs.clone())),
        (ScalarType::F32, ScalarType::C32) | (ScalarType::C32, ScalarType::F32) => Ok((
            promote_real_tensor_to_complex(lhs)?,
            promote_real_tensor_to_complex(rhs)?,
        )),
        (ScalarType::F64, ScalarType::C64) | (ScalarType::C64, ScalarType::F64) => Ok((
            promote_real_tensor_to_complex(lhs)?,
            promote_real_tensor_to_complex(rhs)?,
        )),
        _ => Err(anyhow!(
            "native tensor promotion does not support lhs={:?}, rhs={:?}",
            lhs.scalar_type(),
            rhs.scalar_type()
        )),
    }
}

fn target_scalar_type_for_operands(operands: &[&DynAdTensor]) -> Result<ScalarType> {
    let mut saw_f32 = false;
    let mut saw_f64 = false;
    let mut saw_c32 = false;
    let mut saw_c64 = false;

    for operand in operands {
        match operand.scalar_type() {
            ScalarType::F32 => saw_f32 = true,
            ScalarType::F64 => saw_f64 = true,
            ScalarType::C32 => saw_c32 = true,
            ScalarType::C64 => saw_c64 = true,
        }
    }

    match (saw_f32, saw_f64, saw_c32, saw_c64) {
        (true, false, false, false) => Ok(ScalarType::F32),
        (false, true, false, false) => Ok(ScalarType::F64),
        (false, false, true, false) | (true, false, true, false) => Ok(ScalarType::C32),
        (false, false, false, true) | (false, true, false, true) => Ok(ScalarType::C64),
        _ => Err(anyhow!(
            "native tensor promotion does not support operand scalar types {:?}",
            operands
                .iter()
                .map(|operand| operand.scalar_type())
                .collect::<Vec<_>>()
        )),
    }
}

fn promote_native_operands_many(
    operands: &[&DynAdTensor],
) -> Result<(Vec<DynAdTensor>, ScalarType)> {
    let target = target_scalar_type_for_operands(operands)?;
    let promoted = operands
        .iter()
        .map(|operand| match (operand.scalar_type(), target) {
            (src, dst) if src == dst => Ok((*operand).clone()),
            (ScalarType::F32, ScalarType::C32) | (ScalarType::F64, ScalarType::C64) => {
                promote_real_tensor_to_complex(operand)
            }
            _ => Err(anyhow!(
                "native tensor promotion does not support operand {:?} to {:?}",
                operand.scalar_type(),
                target
            )),
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((promoted, target))
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

fn dyn_ad_einsum_typed(
    notation: &str,
    operands: &[&DynAdTensor],
    scalar_type: ScalarType,
) -> Result<DynAdTensor> {
    with_default_runtime("dyn_ad_einsum", || match scalar_type {
        ScalarType::F32 => {
            let typed: Result<Vec<&AdTensor<f32>>> = operands
                .iter()
                .map(|operand| {
                    operand
                        .as_f32()
                        .ok_or_else(|| anyhow!("expected f32 operand in native einsum"))
                })
                .collect();
            Ok(DynAdTensor::from(ad::einsum(notation, &typed?).map_err(
                |e| anyhow!("native einsum failed for f32 tensor: {e}"),
            )?))
        }
        ScalarType::F64 => {
            let typed: Result<Vec<&AdTensor<f64>>> = operands
                .iter()
                .map(|operand| {
                    operand
                        .as_f64()
                        .ok_or_else(|| anyhow!("expected f64 operand in native einsum"))
                })
                .collect();
            Ok(DynAdTensor::from(ad::einsum(notation, &typed?).map_err(
                |e| anyhow!("native einsum failed for f64 tensor: {e}"),
            )?))
        }
        ScalarType::C32 => {
            let typed: Result<Vec<&AdTensor<Complex32>>> = operands
                .iter()
                .map(|operand| {
                    operand
                        .as_c32()
                        .ok_or_else(|| anyhow!("expected c32 operand in native einsum"))
                })
                .collect();
            Ok(DynAdTensor::from(ad::einsum(notation, &typed?).map_err(
                |e| anyhow!("native einsum failed for c32 tensor: {e}"),
            )?))
        }
        ScalarType::C64 => {
            let typed: Result<Vec<&AdTensor<Complex64>>> = operands
                .iter()
                .map(|operand| {
                    operand
                        .as_c64()
                        .ok_or_else(|| anyhow!("expected c64 operand in native einsum"))
                })
                .collect();
            Ok(DynAdTensor::from(ad::einsum(notation, &typed?).map_err(
                |e| anyhow!("native einsum failed for c64 tensor: {e}"),
            )?))
        }
    })
}

/// Execute native structured einsum on multiple AD tensors.
pub fn einsum_dyn_ad_tensors_native(
    operands: &[(&DynAdTensor, &[usize])],
    output_ids: &[usize],
) -> Result<DynAdTensor> {
    if operands.is_empty() {
        return Err(anyhow!("native einsum requires at least one operand"));
    }

    let operand_refs: Vec<&DynAdTensor> = operands.iter().map(|(tensor, _)| *tensor).collect();
    let (promoted, scalar_type) = promote_native_operands_many(&operand_refs)?;
    let input_ids: Vec<Vec<usize>> = operands.iter().map(|(_, ids)| ids.to_vec()).collect();
    let notation = labels_to_notation(&input_ids, output_ids)?;
    let promoted_refs: Vec<&DynAdTensor> = promoted.iter().collect();

    dyn_ad_einsum_typed(&notation, &promoted_refs, scalar_type)
}

fn dyn_ad_tensor_to_scalar_native(tensor: &DynAdTensor) -> Result<DynAdScalar> {
    match tensor {
        DynAdTensor::F32(t) => Ok(DynAdScalar::F32(ad_tensor_to_scalar_typed(t)?)),
        DynAdTensor::F64(t) => Ok(DynAdScalar::F64(ad_tensor_to_scalar_typed(t)?)),
        DynAdTensor::C32(t) => Ok(DynAdScalar::C32(ad_tensor_to_scalar_typed(t)?)),
        DynAdTensor::C64(t) => Ok(DynAdScalar::C64(ad_tensor_to_scalar_typed(t)?)),
    }
}

/// Convert legacy [`Storage`] into a primal-mode [`DynAdTensor`].
pub fn storage_to_dyn_ad_tensor(storage: &Storage, logical_dims: &[usize]) -> Result<DynAdTensor> {
    match storage {
        Storage::DenseF64(_) => Ok(DynAdTensor::from(AdTensor::new_primal(
            dense_f64_to_tensor(storage, logical_dims)?,
        ))),
        Storage::DiagF64(_) => Ok(DynAdTensor::from(AdTensor::new_primal(
            diag_f64_to_structured(storage, logical_dims)?,
        ))),
        Storage::DenseC64(_) => Ok(DynAdTensor::from(AdTensor::new_primal(
            dense_c64_to_tensor(storage, logical_dims)?,
        ))),
        Storage::DiagC64(_) => Ok(DynAdTensor::from(AdTensor::new_primal(
            diag_c64_to_structured(storage, logical_dims)?,
        ))),
    }
}

/// Materialize the primal payload of a [`DynAdTensor`] back into dense [`Storage`].
///
/// AD metadata is intentionally dropped at this bridge boundary.
pub fn dyn_ad_tensor_primal_to_storage(tensor: &DynAdTensor) -> Result<Storage> {
    match tensor {
        DynAdTensor::F32(_) | DynAdTensor::C32(_) => Err(anyhow!(
            "tensor4all native bridge currently supports only f64/Complex64 tensors"
        )),
        DynAdTensor::F64(t) => structured_f64_to_storage(t.structured_primal()),
        DynAdTensor::C64(t) => structured_c64_to_storage(t.structured_primal()),
    }
}

/// Reshape a raw `Tensor<T>` to `new_dims` with guaranteed row-major strides.
///
/// Calls `contiguous(RowMajor)` then rebuilds the tensor via `from_slice`
/// to avoid tenferro's `reshape()` column-major priority bug when
/// dimensions contain 1.
fn reshape_tensor_row_major<T: Scalar + Copy>(
    tensor: &Tensor<T>,
    new_dims: &[usize],
) -> Result<Tensor<T>> {
    let c = tensor.contiguous(MemoryOrder::RowMajor);
    let buf = c
        .buffer()
        .as_slice()
        .ok_or_else(|| anyhow!("reshape_row_major: cannot get buffer slice"))?;
    let off = c.offset() as usize;
    let len: usize = new_dims.iter().product();
    let total = buf.len() - off;
    if len != total {
        return Err(anyhow!(
            "reshape_row_major: size mismatch: tensor has {total} elements but new shape requires {len}"
        ));
    }
    Tensor::from_slice(&buf[off..off + len], new_dims, MemoryOrder::RowMajor)
        .map_err(|e| anyhow!("reshape_row_major from_slice failed: {e}"))
}

/// Reshape a `StructuredTensor<T>` to `new_dims` with guaranteed row-major strides.
fn reshape_structured_row_major<T: Scalar + Copy>(
    st: &StructuredTensor<T>,
    new_dims: &[usize],
) -> Result<StructuredTensor<T>> {
    let new_payload = reshape_tensor_row_major(st.payload(), new_dims)?;
    Ok(StructuredTensor::from_dense(new_payload))
}

/// Reshape an `AdTensor<T>` to `new_dims` preserving AD mode and tangent.
fn reshape_ad_tensor_row_major<T: Scalar + Copy>(
    ad: &AdTensor<T>,
    new_dims: &[usize],
) -> Result<AdTensor<T>> {
    let new_primal = reshape_structured_row_major(ad.structured_primal(), new_dims)?;
    match ad.as_value() {
        AdValue::Primal(_) => Ok(AdTensor::new_primal(new_primal)),
        AdValue::Forward { tangent, .. } => {
            let new_tangent = reshape_structured_row_major(tangent, new_dims)?;
            AdTensor::new_forward(new_primal, new_tangent)
                .map_err(|e| anyhow!("reshape_ad forward failed: {e}"))
        }
        AdValue::Reverse {
            node,
            tape,
            tangent,
            ..
        } => {
            let new_tangent = tangent
                .as_ref()
                .map(|t| reshape_structured_row_major(t, new_dims))
                .transpose()?;
            AdTensor::new_reverse(new_primal, *node, *tape, new_tangent)
                .map_err(|e| anyhow!("reshape_ad reverse failed: {e}"))
        }
    }
}

/// Reshape a [`DynAdTensor`] to `new_dims` with guaranteed row-major layout,
/// preserving AD metadata (forward-mode tangent, reverse-mode graph info).
///
/// This avoids tenferro's `Tensor::reshape()` which can incorrectly assign
/// column-major strides when dimensions contain 1.
pub fn reshape_row_major_dyn_ad_tensor(
    tensor: &DynAdTensor,
    new_dims: &[usize],
) -> Result<DynAdTensor> {
    match tensor {
        DynAdTensor::F64(ad) => Ok(DynAdTensor::from(reshape_ad_tensor_row_major(
            ad, new_dims,
        )?)),
        DynAdTensor::C64(ad) => Ok(DynAdTensor::from(reshape_ad_tensor_row_major(
            ad, new_dims,
        )?)),
        _ => Err(anyhow!(
            "reshape_row_major_dyn_ad_tensor: unsupported scalar type {:?}",
            tensor.scalar_type()
        )),
    }
}

/// Compute native QR while preserving AD metadata.
pub fn qr_dyn_ad_tensor_native(tensor: &DynAdTensor) -> Result<(DynAdTensor, DynAdTensor)> {
    with_default_runtime("native_qr", || match tensor {
        DynAdTensor::F32(t) => {
            let out = ad::qr(t).map_err(|e| anyhow!("native qr failed for f32 tensor: {e}"))?;
            Ok((DynAdTensor::from(out.q), DynAdTensor::from(out.r)))
        }
        DynAdTensor::F64(t) => {
            let out = ad::qr(t).map_err(|e| anyhow!("native qr failed for f64 tensor: {e}"))?;
            Ok((DynAdTensor::from(out.q), DynAdTensor::from(out.r)))
        }
        DynAdTensor::C32(_) | DynAdTensor::C64(_) => Err(anyhow!(
            "native qr bridge currently supports only real tensors"
        )),
    })
}

/// Compute native SVD while preserving AD metadata.
///
/// Returns `(u, s, vt)` with the same conventions as tenferro linalg:
/// `u` is `(m, k)`, `s` is rank-1 `(k)`, `vt` is `(k, n)`.
pub fn svd_dyn_ad_tensor_native(
    tensor: &DynAdTensor,
) -> Result<(DynAdTensor, DynAdTensor, DynAdTensor)> {
    with_default_runtime("native_svd", || match tensor {
        DynAdTensor::F32(t) => {
            let out = ad::svd(t).map_err(|e| anyhow!("native svd failed for f32 tensor: {e}"))?;
            Ok((
                DynAdTensor::from(out.u),
                DynAdTensor::from(out.s),
                DynAdTensor::from(out.vt),
            ))
        }
        DynAdTensor::F64(t) => {
            let out = ad::svd(t).map_err(|e| anyhow!("native svd failed for f64 tensor: {e}"))?;
            Ok((
                DynAdTensor::from(out.u),
                DynAdTensor::from(out.s),
                DynAdTensor::from(out.vt),
            ))
        }
        DynAdTensor::C32(_) | DynAdTensor::C64(_) => Err(anyhow!(
            "native svd bridge currently supports only real tensors"
        )),
    })
}

/// Sum all elements of a native AD tensor while preserving AD mode.
pub fn sum_dyn_ad_tensor_native(tensor: &DynAdTensor) -> Result<DynAdScalar> {
    if tensor.ndim() == 0 {
        return dyn_ad_tensor_to_scalar_native(tensor);
    }
    let input_ids = vec![(0..tensor.ndim()).collect::<Vec<_>>()];
    let notation = labels_to_notation(&input_ids, &[])?;
    let reduced = dyn_ad_einsum_typed(&notation, &[tensor], tensor.scalar_type())?;
    dyn_ad_tensor_to_scalar_native(&reduced)
}

/// Permute a native AD tensor through AD-aware einsum.
pub fn permute_dyn_ad_tensor_native(tensor: &DynAdTensor, perm: &[usize]) -> Result<DynAdTensor> {
    if perm.len() != tensor.ndim() {
        return Err(anyhow!(
            "native permute rank mismatch: rank={}, perm={:?}",
            tensor.ndim(),
            perm
        ));
    }
    if perm.iter().enumerate().all(|(i, &p)| i == p) {
        return Ok(tensor.clone());
    }
    let input_ids: Vec<usize> = (0..tensor.ndim()).collect();
    let output_ids: Vec<usize> = perm
        .iter()
        .map(|&axis| {
            input_ids.get(axis).copied().ok_or_else(|| {
                anyhow!(
                    "native permute axis {} out of range for rank {}",
                    axis,
                    tensor.ndim()
                )
            })
        })
        .collect::<Result<_>>()?;
    let notation = labels_to_notation(&[input_ids], &output_ids)?;
    dyn_ad_einsum_typed(&notation, &[tensor], tensor.scalar_type())
}

/// Contract two native AD tensors with AD-preserving mixed real/complex promotion.
pub fn contract_dyn_ad_tensor_native(
    lhs: &DynAdTensor,
    axes_lhs: &[usize],
    rhs: &DynAdTensor,
    axes_rhs: &[usize],
) -> Result<DynAdTensor> {
    let (lhs, rhs) = promote_native_operands(lhs, rhs)?;
    let (lhs_ids, rhs_ids, output_ids) =
        build_binary_einsum_ids(lhs.ndim(), axes_lhs, rhs.ndim(), axes_rhs)?;
    let notation = labels_to_notation(&[lhs_ids, rhs_ids], &output_ids)?;
    dyn_ad_einsum_typed(&notation, &[&lhs, &rhs], lhs.scalar_type())
}

/// Compute outer product of two native AD tensors.
pub fn outer_product_dyn_ad_tensor_native(
    lhs: &DynAdTensor,
    rhs: &DynAdTensor,
) -> Result<DynAdTensor> {
    contract_dyn_ad_tensor_native(lhs, &[], rhs, &[])
}

/// Conjugate a native AD tensor while preserving AD metadata.
pub fn conj_dyn_ad_tensor_native(tensor: &DynAdTensor) -> Result<DynAdTensor> {
    if tensor.is_real() {
        return Ok(tensor.clone());
    }
    let real = tensor.real_part()?;
    let imag = tensor.imag_part()?;
    let neg_imag = imag
        .scale(&scalar_type_to_minus_one(imag.scalar_type()))
        .map_err(|e| anyhow!("native conjugation failed while negating imag part: {e}"))?;
    DynAdTensor::compose_complex(real, neg_imag)
        .map_err(|e| anyhow!("native conjugation failed: {e}"))
}

/// Permute storage through tenferro-native execution for dense tensors.
///
/// Diagonal storage only changes metadata, so it is returned unchanged.
pub fn permute_storage_native(
    storage: &Storage,
    logical_dims: &[usize],
    perm: &[usize],
) -> Result<Storage> {
    match storage {
        Storage::DiagF64(_) | Storage::DiagC64(_) => {
            Ok(storage.permute_storage(logical_dims, perm))
        }
        Storage::DenseF64(_) => {
            let tensor = dense_f64_to_tensor(storage, logical_dims)?;
            let permuted = tensor
                .permute(perm)
                .map_err(|e| anyhow!("native permute failed for f64 tensor: {e}"))?;
            tensor_f64_to_storage(&permuted)
        }
        Storage::DenseC64(_) => {
            let tensor = dense_c64_to_tensor(storage, logical_dims)?;
            let permuted = tensor
                .permute(perm)
                .map_err(|e| anyhow!("native permute failed for c64 tensor: {e}"))?;
            tensor_c64_to_storage(&permuted)
        }
    }
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

/// Contract two storages through tenferro-native execution for dense tensors.
///
/// Cases involving diagonal storage fall back to the existing structured path.
pub fn contract_storage_native(
    storage_a: &Storage,
    dims_a: &[usize],
    axes_a: &[usize],
    storage_b: &Storage,
    dims_b: &[usize],
    axes_b: &[usize],
    result_dims: &[usize],
) -> Result<Storage> {
    if storage_a.is_diag() || storage_b.is_diag() {
        return Ok(contract_storage(
            storage_a,
            dims_a,
            axes_a,
            storage_b,
            dims_b,
            axes_b,
            result_dims,
        ));
    }

    let (lhs_ids, rhs_ids, output_ids) =
        build_binary_einsum_ids(dims_a.len(), axes_a, dims_b.len(), axes_b)?;
    let inputs = [
        BackendEinsumInput {
            ids: lhs_ids.as_slice(),
            storage: storage_a,
            dims: dims_a,
        },
        BackendEinsumInput {
            ids: rhs_ids.as_slice(),
            storage: storage_b,
            dims: dims_b,
        },
    ];
    einsum_storage(&inputs, &output_ids)
}

/// Compute an outer product through tenferro-native execution for dense tensors.
pub fn outer_product_storage_native(
    lhs: &Storage,
    lhs_dims: &[usize],
    rhs: &Storage,
    rhs_dims: &[usize],
    result_dims: &[usize],
) -> Result<Storage> {
    contract_storage_native(lhs, lhs_dims, &[], rhs, rhs_dims, &[], result_dims)
}

/// Apply native tenferro mixed scalar/tensor scaling at the storage boundary.
pub fn scale_storage_native(
    storage: &Storage,
    logical_dims: &[usize],
    scalar: &AnyScalar,
) -> Result<Storage> {
    if storage.is_diag() {
        return Ok(storage.scale(scalar));
    }
    let native = storage_to_dyn_ad_tensor(storage, logical_dims)?;
    let scaled = native
        .scale(scalar)
        .map_err(|e| anyhow!("native scale failed: {e}"))?;
    dyn_ad_tensor_primal_to_storage(&scaled)
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
    match (lhs, rhs) {
        (Storage::DiagF64(x), Storage::DiagF64(y)) if a.is_real() && b.is_real() => {
            let result: Vec<f64> = x
                .as_slice()
                .iter()
                .zip(y.as_slice().iter())
                .map(|(&xi, &yi)| a.real() * xi + b.real() * yi)
                .collect();
            return Ok(Storage::DiagF64(DiagStorageF64::from_vec(result)));
        }
        (Storage::DiagF64(x), Storage::DiagF64(y)) => {
            let a_c: num_complex::Complex64 = a.clone().into();
            let b_c: num_complex::Complex64 = b.clone().into();
            let result: Vec<num_complex::Complex64> = x
                .as_slice()
                .iter()
                .zip(y.as_slice().iter())
                .map(|(&xi, &yi)| {
                    a_c * num_complex::Complex64::new(xi, 0.0)
                        + b_c * num_complex::Complex64::new(yi, 0.0)
                })
                .collect();
            return Ok(Storage::DiagC64(DiagStorageC64::from_vec(result)));
        }
        (Storage::DiagF64(x), Storage::DiagC64(y)) => {
            let a_c: num_complex::Complex64 = a.clone().into();
            let b_c: num_complex::Complex64 = b.clone().into();
            let result: Vec<num_complex::Complex64> = x
                .as_slice()
                .iter()
                .zip(y.as_slice().iter())
                .map(|(&xi, &yi)| a_c * num_complex::Complex64::new(xi, 0.0) + b_c * yi)
                .collect();
            return Ok(Storage::DiagC64(DiagStorageC64::from_vec(result)));
        }
        (Storage::DiagC64(x), Storage::DiagF64(y)) => {
            let a_c: num_complex::Complex64 = a.clone().into();
            let b_c: num_complex::Complex64 = b.clone().into();
            let result: Vec<num_complex::Complex64> = x
                .as_slice()
                .iter()
                .zip(y.as_slice().iter())
                .map(|(&xi, &yi)| a_c * xi + b_c * num_complex::Complex64::new(yi, 0.0))
                .collect();
            return Ok(Storage::DiagC64(DiagStorageC64::from_vec(result)));
        }
        (Storage::DiagC64(x), Storage::DiagC64(y)) => {
            let a_c: num_complex::Complex64 = a.clone().into();
            let b_c: num_complex::Complex64 = b.clone().into();
            let result: Vec<num_complex::Complex64> = x
                .as_slice()
                .iter()
                .zip(y.as_slice().iter())
                .map(|(&xi, &yi)| a_c * xi + b_c * yi)
                .collect();
            return Ok(Storage::DiagC64(DiagStorageC64::from_vec(result)));
        }
        _ => {}
    }

    let lhs_native = storage_to_dyn_ad_tensor(lhs, lhs_dims)?;
    let rhs_native = storage_to_dyn_ad_tensor(rhs, rhs_dims)?;
    let combined = lhs_native
        .axpby(a, &rhs_native, b)
        .map_err(|e| anyhow!("native axpby failed: {e}"))?;
    dyn_ad_tensor_primal_to_storage(&combined)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{DenseStorageC64, DenseStorageF64, DiagStorageF64, Storage};
    use num_complex::Complex64;
    use std::sync::{Mutex, OnceLock};
    use tenferro_dyadtensor::AdMode;
    use tenferro_tensor::MemoryOrder;

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn with_env(runtime: Option<&str>, threads: Option<&str>, f: impl FnOnce()) {
        let _guard = env_lock().lock().unwrap();
        let prev_runtime = env::var("T4A_TENFERRO_RUNTIME").ok();
        let prev_threads = env::var("T4A_TENFERRO_CPU_THREADS").ok();

        match runtime {
            Some(v) => env::set_var("T4A_TENFERRO_RUNTIME", v),
            None => env::remove_var("T4A_TENFERRO_RUNTIME"),
        }
        match threads {
            Some(v) => env::set_var("T4A_TENFERRO_CPU_THREADS", v),
            None => env::remove_var("T4A_TENFERRO_CPU_THREADS"),
        }

        f();

        match prev_runtime {
            Some(v) => env::set_var("T4A_TENFERRO_RUNTIME", v),
            None => env::remove_var("T4A_TENFERRO_RUNTIME"),
        }
        match prev_threads {
            Some(v) => env::set_var("T4A_TENFERRO_CPU_THREADS", v),
            None => env::remove_var("T4A_TENFERRO_CPU_THREADS"),
        }
    }

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
            _ => panic!(
                "storage mismatch: lhs variant {:?}, rhs variant {:?}",
                std::mem::discriminant(lhs),
                std::mem::discriminant(rhs)
            ),
        }
    }

    #[test]
    fn parse_runtime_kind_defaults_to_cpu() {
        with_env(None, None, || {
            assert_eq!(parse_runtime_kind(), RuntimeKind::Cpu);
        });
    }

    #[test]
    fn parse_runtime_kind_accepts_known_values_and_fallback() {
        with_env(Some("cpu"), None, || {
            assert_eq!(parse_runtime_kind(), RuntimeKind::Cpu)
        });
        with_env(Some("CUDA"), None, || {
            assert_eq!(parse_runtime_kind(), RuntimeKind::Cuda);
        });
        with_env(Some("rocm"), None, || {
            assert_eq!(parse_runtime_kind(), RuntimeKind::Rocm);
        });
        with_env(Some("unknown"), None, || {
            assert_eq!(parse_runtime_kind(), RuntimeKind::Cpu);
        });
    }

    #[test]
    fn cpu_threads_parsing_and_clamp() {
        with_env(None, None, || assert_eq!(cpu_threads(), 1));
        with_env(None, Some("8"), || assert_eq!(cpu_threads(), 8));
        with_env(None, Some("0"), || assert_eq!(cpu_threads(), 1));
        with_env(None, Some("bad"), || assert_eq!(cpu_threads(), 1));
    }

    #[test]
    fn with_tenferro_ctx_cpu_executes_closure_and_propagates_error() {
        with_env(Some("cpu"), Some("2"), || {
            let value = with_tenferro_ctx("cpu-op", |_ctx| Ok::<usize, anyhow::Error>(42)).unwrap();
            assert_eq!(value, 42);

            let err = with_tenferro_ctx("cpu-op", |_ctx| {
                Err::<(), anyhow::Error>(anyhow!("inner failure"))
            })
            .unwrap_err();
            assert!(err.to_string().contains("inner failure"));
        });
    }

    #[test]
    fn with_tenferro_ctx_gpu_runtimes_return_explicit_errors() {
        with_env(Some("cuda"), None, || {
            let err = with_tenferro_ctx("einsum", |_ctx| Ok::<(), anyhow::Error>(())).unwrap_err();
            let msg = err.to_string();
            assert!(msg.contains("CUDA runtime is not yet wired"));
            assert!(msg.contains("einsum"));
        });

        with_env(Some("rocm"), None, || {
            let err = with_tenferro_ctx("linalg", |_ctx| Ok::<(), anyhow::Error>(())).unwrap_err();
            let msg = err.to_string();
            assert!(msg.contains("ROCm runtime is not yet wired"));
            assert!(msg.contains("linalg"));
        });
    }

    #[test]
    fn storage_dyn_ad_tensor_roundtrip_dense_f64() {
        let storage = Storage::DenseF64(DenseStorageF64::from_vec_with_shape(
            vec![1.0, 2.0, 3.0, 4.0],
            &[2, 2],
        ));

        let native = storage_to_dyn_ad_tensor(&storage, &[2, 2]).unwrap();
        let roundtrip = dyn_ad_tensor_primal_to_storage(&native).unwrap();

        assert_storage_eq(&roundtrip, &storage);
    }

    #[test]
    fn storage_dyn_ad_tensor_roundtrip_dense_c64() {
        let storage = Storage::DenseC64(DenseStorageC64::from_vec_with_shape(
            vec![
                Complex64::new(1.0, -1.0),
                Complex64::new(2.0, 0.5),
                Complex64::new(-3.0, 4.0),
                Complex64::new(0.0, -2.0),
            ],
            &[2, 2],
        ));

        let native = storage_to_dyn_ad_tensor(&storage, &[2, 2]).unwrap();
        let roundtrip = dyn_ad_tensor_primal_to_storage(&native).unwrap();

        assert_storage_eq(&roundtrip, &storage);
    }

    #[test]
    fn storage_dyn_ad_tensor_roundtrip_diag_preserves_diag_layout() {
        let storage = Storage::DiagF64(DiagStorageF64::from_vec(vec![2.0, -1.0, 4.0]));

        let native = storage_to_dyn_ad_tensor(&storage, &[3, 3]).unwrap();
        let roundtrip = dyn_ad_tensor_primal_to_storage(&native).unwrap();

        assert!(native.is_diag());
        assert_storage_eq(&roundtrip, &storage);
    }

    #[test]
    fn dyn_ad_tensor_axpby_accepts_real_scalars_for_complex_tensor() {
        let storage = Storage::DenseC64(DenseStorageC64::from_vec_with_shape(
            vec![Complex64::new(1.0, 2.0), Complex64::new(-3.0, 0.5)],
            &[2],
        ));
        let native = storage_to_dyn_ad_tensor(&storage, &[2]).unwrap();

        let combined = native
            .axpby(
                &crate::AnyScalar::new_real(2.0),
                &native,
                &crate::AnyScalar::new_real(-1.0),
            )
            .unwrap();
        let roundtrip = dyn_ad_tensor_primal_to_storage(&combined).unwrap();

        let expected = Storage::DenseC64(DenseStorageC64::from_vec_with_shape(
            vec![Complex64::new(1.0, 2.0), Complex64::new(-3.0, 0.5)],
            &[2],
        ));
        assert_storage_eq(&roundtrip, &expected);
    }

    #[test]
    fn permute_storage_native_dense_matches_legacy() {
        with_env(None, None, || {
            let storage = Storage::DenseF64(DenseStorageF64::from_vec_with_shape(
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                &[2, 3],
            ));

            let native = permute_storage_native(&storage, &[2, 3], &[1, 0]).unwrap();
            let legacy = storage.permute_storage(&[2, 3], &[1, 0]);

            assert_storage_eq(&native, &legacy);
        });
    }

    #[test]
    fn contract_storage_native_dense_mixed_matches_legacy() {
        with_env(None, None, || {
            let lhs = Storage::DenseF64(DenseStorageF64::from_vec_with_shape(
                vec![1.0, 2.0, 3.0, 4.0],
                &[2, 2],
            ));
            let rhs = Storage::DenseC64(DenseStorageC64::from_vec_with_shape(
                vec![
                    Complex64::new(1.0, -1.0),
                    Complex64::new(2.0, 0.5),
                    Complex64::new(3.0, 1.5),
                    Complex64::new(-4.0, 2.0),
                ],
                &[2, 2],
            ));

            let native =
                contract_storage_native(&lhs, &[2, 2], &[1], &rhs, &[2, 2], &[0], &[2, 2]).unwrap();
            let legacy = contract_storage(&lhs, &[2, 2], &[1], &rhs, &[2, 2], &[0], &[2, 2]);

            assert_storage_eq(&native, &legacy);
        });
    }

    #[test]
    fn outer_product_storage_native_dense_matches_legacy() {
        with_env(None, None, || {
            let lhs = Storage::DenseC64(DenseStorageC64::from_vec_with_shape(
                vec![Complex64::new(1.0, 0.25), Complex64::new(-2.0, 1.0)],
                &[2],
            ));
            let rhs = Storage::DenseF64(DenseStorageF64::from_vec_with_shape(
                vec![3.0, -1.0, 0.5],
                &[3],
            ));

            let native = outer_product_storage_native(&lhs, &[2], &rhs, &[3], &[2, 3]).unwrap();
            let legacy = contract_storage(&lhs, &[2], &[], &rhs, &[3], &[], &[2, 3]);

            assert_storage_eq(&native, &legacy);
        });
    }

    #[test]
    fn contract_storage_native_diag_falls_back_to_legacy() {
        let lhs = Storage::DiagF64(DiagStorageF64::from_vec(vec![1.0, 2.0, 3.0]));
        let rhs = Storage::DiagF64(DiagStorageF64::from_vec(vec![4.0, 5.0, 6.0]));

        let native =
            contract_storage_native(&lhs, &[3, 3], &[1], &rhs, &[3, 3], &[0], &[3, 3]).unwrap();
        let legacy = contract_storage(&lhs, &[3, 3], &[1], &rhs, &[3, 3], &[0], &[3, 3]);

        assert_storage_eq(&native, &legacy);
        assert!(matches!(native, Storage::DiagF64(_)));
    }

    #[test]
    fn sum_dyn_ad_tensor_native_preserves_forward_mode() {
        let primal =
            tenferro_tensor::Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::RowMajor)
                .unwrap();
        let tangent =
            tenferro_tensor::Tensor::<f64>::from_slice(&[0.25, -0.75], &[2], MemoryOrder::RowMajor)
                .unwrap();
        let native = DynAdTensor::from(AdTensor::new_forward(primal, tangent).unwrap());

        let sum = sum_dyn_ad_tensor_native(&native).unwrap();

        assert_eq!(sum.mode(), AdMode::Forward);
        assert_eq!(sum.primal().as_f64(), Some(3.0));
        assert_eq!(sum.tangent().and_then(|x| x.as_f64()), Some(-0.5));
    }

    #[test]
    fn permute_dyn_ad_tensor_native_matches_primal_parity() {
        with_env(None, None, || {
            let native = storage_to_dyn_ad_tensor(
                &Storage::DenseF64(DenseStorageF64::from_vec_with_shape(
                    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    &[2, 3],
                )),
                &[2, 3],
            )
            .unwrap();

            let permuted = permute_dyn_ad_tensor_native(&native, &[1, 0]).unwrap();
            let storage = dyn_ad_tensor_primal_to_storage(&permuted).unwrap();
            let expected = Storage::DenseF64(DenseStorageF64::from_vec_with_shape(
                vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
                &[3, 2],
            ));
            assert_storage_eq(&storage, &expected);
        });
    }

    #[test]
    fn einsum_dyn_ad_tensors_native_preserves_diag_and_forward_mode() {
        with_env(None, None, || {
            let diag = Storage::DiagF64(DiagStorageF64::from_vec(vec![2.0, 3.0]));
            let native_diag = storage_to_dyn_ad_tensor(&diag, &[2, 2]).unwrap();

            let primal = tenferro_tensor::Tensor::<f64>::from_slice(
                &[1.0, 4.0],
                &[2],
                MemoryOrder::RowMajor,
            )
            .unwrap();
            let tangent = tenferro_tensor::Tensor::<f64>::from_slice(
                &[0.5, -1.0],
                &[2],
                MemoryOrder::RowMajor,
            )
            .unwrap();
            let native_vec = DynAdTensor::from(AdTensor::new_forward(primal, tangent).unwrap());

            let result =
                einsum_dyn_ad_tensors_native(&[(&native_diag, &[0, 1]), (&native_vec, &[1])], &[0])
                    .unwrap();
            let roundtrip = dyn_ad_tensor_primal_to_storage(&result).unwrap();

            assert_eq!(result.mode(), AdMode::Forward);
            assert_storage_eq(
                &roundtrip,
                &Storage::DenseF64(DenseStorageF64::from_vec_with_shape(vec![2.0, 12.0], &[2])),
            );
            let tangent = sum_dyn_ad_tensor_native(&result)
                .unwrap()
                .tangent()
                .and_then(|x| x.as_f64())
                .unwrap();
            assert!(
                (tangent + 2.0).abs() < 1e-12,
                "unexpected native einsum tangent: {tangent}"
            );
        });
    }

    #[test]
    fn contract_dyn_ad_tensor_native_matches_primal_parity() {
        with_env(None, None, || {
            let lhs = storage_to_dyn_ad_tensor(
                &Storage::DenseF64(DenseStorageF64::from_vec_with_shape(vec![1.0; 6], &[2, 3])),
                &[2, 3],
            )
            .unwrap();
            let rhs = storage_to_dyn_ad_tensor(
                &Storage::DenseF64(DenseStorageF64::from_vec_with_shape(vec![1.0; 12], &[3, 4])),
                &[3, 4],
            )
            .unwrap();

            let contracted = contract_dyn_ad_tensor_native(&lhs, &[1], &rhs, &[0]).unwrap();
            let storage = dyn_ad_tensor_primal_to_storage(&contracted).unwrap();
            let expected =
                Storage::DenseF64(DenseStorageF64::from_vec_with_shape(vec![3.0; 8], &[2, 4]));
            assert_storage_eq(&storage, &expected);
        });
    }

    #[test]
    fn storage_dyn_ad_tensor_roundtrip_diag_c64_preserves_diag_layout() {
        let storage = Storage::DiagC64(crate::storage::DiagStorageC64::from_vec(vec![
            Complex64::new(1.0, -0.5),
            Complex64::new(-2.0, 3.0),
        ]));

        let native = storage_to_dyn_ad_tensor(&storage, &[2, 2]).unwrap();
        let roundtrip = dyn_ad_tensor_primal_to_storage(&native).unwrap();

        assert!(native.is_diag());
        assert_storage_eq(&roundtrip, &storage);
    }

    #[test]
    fn scale_storage_native_accepts_real_scalar_for_complex_dense() {
        let storage = Storage::DenseC64(DenseStorageC64::from_vec_with_shape(
            vec![Complex64::new(1.0, 2.0), Complex64::new(-3.0, 0.5)],
            &[2],
        ));

        let scaled =
            scale_storage_native(&storage, &[2], &crate::AnyScalar::new_real(2.0)).unwrap();
        let expected = Storage::DenseC64(DenseStorageC64::from_vec_with_shape(
            vec![Complex64::new(2.0, 4.0), Complex64::new(-6.0, 1.0)],
            &[2],
        ));
        assert_storage_eq(&scaled, &expected);
    }

    #[test]
    fn axpby_storage_native_promotes_mixed_diag_operands_to_complex_diag() {
        let lhs = Storage::DiagF64(DiagStorageF64::from_vec(vec![1.0, 2.0]));
        let rhs = Storage::DiagC64(crate::storage::DiagStorageC64::from_vec(vec![
            Complex64::new(0.0, 1.0),
            Complex64::new(2.0, -1.0),
        ]));

        let combined = axpby_storage_native(
            &lhs,
            &[2, 2],
            &crate::AnyScalar::new_real(2.0),
            &rhs,
            &[2, 2],
            &crate::AnyScalar::new_complex(-1.0, 0.5),
        )
        .unwrap();

        let expected = Storage::DiagC64(crate::storage::DiagStorageC64::from_vec(vec![
            Complex64::new(1.5, -1.0),
            Complex64::new(2.5, 2.0),
        ]));
        assert_storage_eq(&combined, &expected);
    }

    #[test]
    fn qr_and_svd_native_helpers_preserve_forward_mode() {
        with_env(None, None, || {
            let primal = tenferro_tensor::Tensor::<f64>::from_slice(
                &[3.0, 1.0, 0.0, 2.0],
                &[2, 2],
                MemoryOrder::RowMajor,
            )
            .unwrap();
            let tangent = tenferro_tensor::Tensor::<f64>::from_slice(
                &[0.5, -0.25, 0.0, 1.0],
                &[2, 2],
                MemoryOrder::RowMajor,
            )
            .unwrap();
            let native = DynAdTensor::from(AdTensor::new_forward(primal, tangent).unwrap());

            let (q, r) = qr_dyn_ad_tensor_native(&native).unwrap();
            assert_eq!(q.mode(), AdMode::Forward);
            assert_eq!(r.mode(), AdMode::Forward);
            assert_eq!(q.dims(), vec![2, 2]);
            assert_eq!(r.dims(), vec![2, 2]);

            let (u, s, vt) = svd_dyn_ad_tensor_native(&native).unwrap();
            assert_eq!(u.mode(), AdMode::Forward);
            assert_eq!(s.mode(), AdMode::Forward);
            assert_eq!(vt.mode(), AdMode::Forward);
            assert_eq!(u.dims(), vec![2, 2]);
            assert_eq!(s.dims(), vec![2]);
            assert_eq!(vt.dims(), vec![2, 2]);
        });
    }

    #[test]
    fn reshape_row_major_primal_column_vector() {
        // A [4,1] column vector reshaped to [2,2] — this is the exact case
        // that triggered the bug (reshape assigned column-major strides).
        let t = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[4, 1], MemoryOrder::RowMajor)
            .unwrap();
        let native = DynAdTensor::from(AdTensor::new_primal(t));
        let reshaped = reshape_row_major_dyn_ad_tensor(&native, &[2, 2]).unwrap();

        assert_eq!(reshaped.dims(), &[2, 2]);
        let storage = dyn_ad_tensor_primal_to_storage(&reshaped).unwrap();
        match &storage {
            Storage::DenseF64(d) => {
                assert_eq!(d.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
            }
            _ => panic!("expected DenseF64"),
        }
    }

    #[test]
    fn reshape_row_major_primal_with_unit_dim() {
        // [2,2,1] reshaped to [4] — another unit-dim case.
        let t =
            Tensor::<f64>::from_slice(&[10.0, 20.0, 30.0, 40.0], &[2, 2, 1], MemoryOrder::RowMajor)
                .unwrap();
        let native = DynAdTensor::from(AdTensor::new_primal(t));
        let reshaped = reshape_row_major_dyn_ad_tensor(&native, &[4]).unwrap();

        assert_eq!(reshaped.dims(), &[4]);
        let storage = dyn_ad_tensor_primal_to_storage(&reshaped).unwrap();
        match &storage {
            Storage::DenseF64(d) => {
                assert_eq!(d.as_slice(), &[10.0, 20.0, 30.0, 40.0]);
            }
            _ => panic!("expected DenseF64"),
        }
    }

    #[test]
    fn reshape_row_major_preserves_forward_mode() {
        // Forward-mode AD: reshape should preserve both primal and tangent.
        let primal =
            Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[4, 1], MemoryOrder::RowMajor)
                .unwrap();
        let tangent =
            Tensor::<f64>::from_slice(&[0.1, 0.2, 0.3, 0.4], &[4, 1], MemoryOrder::RowMajor)
                .unwrap();
        let native = DynAdTensor::from(AdTensor::new_forward(primal, tangent).unwrap());

        let reshaped = reshape_row_major_dyn_ad_tensor(&native, &[2, 2]).unwrap();

        assert_eq!(reshaped.dims(), &[2, 2]);
        assert_eq!(reshaped.mode(), AdMode::Forward);

        // Check primal values
        let p_storage = dyn_ad_tensor_primal_to_storage(&reshaped).unwrap();
        match &p_storage {
            Storage::DenseF64(d) => assert_eq!(d.as_slice(), &[1.0, 2.0, 3.0, 4.0]),
            _ => panic!("expected DenseF64"),
        }

        // Check tangent values via sum
        let sum = sum_dyn_ad_tensor_native(&reshaped).unwrap();
        let tangent_sum = sum.tangent().and_then(|x| x.as_f64()).unwrap();
        assert!(
            (tangent_sum - 1.0).abs() < 1e-12,
            "tangent sum should be 1.0, got {tangent_sum}"
        );
    }

    #[test]
    fn reshape_row_major_mismatched_size_errors() {
        let t =
            Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[4], MemoryOrder::RowMajor).unwrap();
        let native = DynAdTensor::from(AdTensor::new_primal(t));
        let result = reshape_row_major_dyn_ad_tensor(&native, &[3]);
        assert!(result.is_err(), "reshape with mismatched size should fail");
    }

    #[test]
    fn conj_dyn_ad_tensor_native_preserves_complex_forward_mode() {
        let primal = tenferro_tensor::Tensor::<Complex64>::from_slice(
            &[Complex64::new(1.0, 2.0), Complex64::new(-3.0, 0.5)],
            &[2],
            MemoryOrder::RowMajor,
        )
        .unwrap();
        let tangent = tenferro_tensor::Tensor::<Complex64>::from_slice(
            &[Complex64::new(0.25, -0.5), Complex64::new(1.0, 1.5)],
            &[2],
            MemoryOrder::RowMajor,
        )
        .unwrap();
        let native = DynAdTensor::from(AdTensor::new_forward(primal, tangent).unwrap());

        let conjugated = conj_dyn_ad_tensor_native(&native).unwrap();
        assert_eq!(conjugated.mode(), AdMode::Forward);
        let storage = dyn_ad_tensor_primal_to_storage(&conjugated).unwrap();
        let expected = Storage::DenseC64(DenseStorageC64::from_vec_with_shape(
            vec![Complex64::new(1.0, -2.0), Complex64::new(-3.0, -0.5)],
            &[2],
        ));
        assert_storage_eq(&storage, &expected);
    }
}
