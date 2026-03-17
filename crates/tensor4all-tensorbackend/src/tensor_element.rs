use anyhow::{anyhow, Result};
use num_complex::{Complex32, Complex64};
use tenferro::{snapshot, Tensor as NativeTensor};
use tenferro_algebra::Conjugate;
use tenferro_device::LogicalMemorySpace;
use tenferro_tensor::{MemoryOrder, Tensor as TypedTensor};

/// Public scalar element types supported by tensor4all dense/diag constructors.
pub trait TensorElement: Copy + Send + Sync + 'static {
    /// Build a native tensor from row-major dense data.
    ///
    /// This is a temporary low-level bridge helper. High-level tensor4all APIs
    /// use column-major semantics and must convert explicitly at the boundary.
    fn dense_native_tensor_from_row_major_temp(
        data: &[Self],
        dims: &[usize],
    ) -> Result<NativeTensor>;

    /// Build a native diagonal tensor from row-major diagonal payload data.
    ///
    /// This is a temporary low-level bridge helper. Diagonal payloads are 1D,
    /// but the name remains explicit because callers typically pair it with
    /// row-major dense bridge logic.
    fn diag_native_tensor_from_row_major_temp(
        data: &[Self],
        logical_rank: usize,
    ) -> Result<NativeTensor>;

    /// Build a rank-0 native tensor.
    fn scalar_native_tensor(value: Self) -> Result<NativeTensor>;

    /// Materialize dense row-major primal values from a native tensor.
    ///
    /// This is a temporary low-level bridge helper. High-level tensor4all APIs
    /// convert these values to column-major before exposing them publicly.
    fn dense_values_from_native_row_major_temp(tensor: &NativeTensor) -> Result<Vec<Self>>;

    /// Materialize diagonal payload values from a native diagonal tensor.
    fn diag_values_from_native_temp(tensor: &NativeTensor) -> Result<Vec<Self>>;
}

fn materialize_typed_values<T>(tensor: &TypedTensor<T>, op: &'static str) -> Result<Vec<T>>
where
    T: tenferro_algebra::Scalar + Copy + Conjugate,
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
        .and_then(|values: &[T]| values.get(offset..offset + len))
        .ok_or_else(|| anyhow!("{op}: expected host-accessible contiguous tensor buffer"))?;
    if is_conjugated {
        Ok(slice.iter().copied().map(Conjugate::conj).collect())
    } else {
        Ok(slice.to_vec())
    }
}

macro_rules! impl_tensor_element {
    ($ty:ty, $variant:ident, $payload:ident) => {
        impl TensorElement for $ty {
            fn dense_native_tensor_from_row_major_temp(
                data: &[Self],
                dims: &[usize],
            ) -> Result<NativeTensor> {
                let typed = TypedTensor::<Self>::from_slice(data, dims, MemoryOrder::RowMajor)
                    .map_err(|e| anyhow!("failed to build native dense tensor: {e}"))?;
                Ok(NativeTensor::from_tensor(typed))
            }

            fn diag_native_tensor_from_row_major_temp(
                data: &[Self],
                logical_rank: usize,
            ) -> Result<NativeTensor> {
                if logical_rank == 0 {
                    return Err(anyhow!(
                        "diagonal tensor construction requires at least one logical axis"
                    ));
                }

                let payload =
                    TypedTensor::<Self>::from_slice(data, &[data.len()], MemoryOrder::RowMajor)
                        .map_err(|e| anyhow!("failed to build native diagonal payload: {e}"))?;
                NativeTensor::from_tensor(payload)
                    .diag_embed(logical_rank)
                    .map_err(|e| anyhow!("failed to build native diagonal tensor: {e}"))
            }

            fn scalar_native_tensor(value: Self) -> Result<NativeTensor> {
                let typed =
                    TypedTensor::<Self>::from_slice(&[value], &[], MemoryOrder::ColumnMajor)
                        .map_err(|e| anyhow!("failed to build native rank-0 tensor: {e}"))?;
                Ok(NativeTensor::from_tensor(typed))
            }

            fn dense_values_from_native_row_major_temp(tensor: &NativeTensor) -> Result<Vec<Self>> {
                let snap = tensor.primal_snapshot();
                let dense = if snap.is_dense() {
                    snap
                } else {
                    snap.to_dense()
                        .map_err(|e| anyhow!("failed to densify native tensor snapshot: {e}"))?
                };
                match dense {
                    snapshot::DynTensor::$variant(value) => {
                        materialize_typed_values(value.$payload(), "dense native tensor extraction")
                    }
                    other => Err(anyhow!(
                        "expected {:?} tensor snapshot, got {:?}",
                        stringify!($variant),
                        other.scalar_type()
                    )),
                }
            }

            fn diag_values_from_native_temp(tensor: &NativeTensor) -> Result<Vec<Self>> {
                let snap = tensor.primal_snapshot();
                anyhow::ensure!(snap.is_diag(), "expected diagonal native tensor snapshot");
                match snap {
                    snapshot::DynTensor::$variant(value) => materialize_typed_values(
                        value.$payload(),
                        "diagonal native tensor extraction",
                    ),
                    other => Err(anyhow!(
                        "expected {:?} diagonal tensor snapshot, got {:?}",
                        stringify!($variant),
                        other.scalar_type()
                    )),
                }
            }
        }
    };
}

impl_tensor_element!(f32, F32, payload);
impl_tensor_element!(f64, F64, payload);
impl_tensor_element!(Complex32, C32, payload);
impl_tensor_element!(Complex64, C64, payload);
