use anyhow::{anyhow, Result};
use num_complex::{Complex32, Complex64};
use tenferro::Tensor as NativeTensor;
use tenferro_tensor::{MemoryOrder, Tensor as TypedTensor};

/// Public scalar element types supported by tensor4all dense/diag constructors.
pub trait TensorElement: Copy + Send + Sync + 'static {
    /// Build a native tensor from column-major dense data.
    fn dense_native_tensor_from_col_major(data: &[Self], dims: &[usize]) -> Result<NativeTensor>;

    /// Build a native diagonal tensor from column-major diagonal payload data.
    fn diag_native_tensor_from_col_major(
        data: &[Self],
        logical_rank: usize,
    ) -> Result<NativeTensor>;

    /// Build a rank-0 native tensor.
    fn scalar_native_tensor(value: Self) -> Result<NativeTensor>;

    /// Materialize dense column-major primal values from a native tensor.
    fn dense_values_from_native_col_major(tensor: &NativeTensor) -> Result<Vec<Self>>;

    /// Materialize diagonal payload values from a native diagonal tensor.
    fn diag_values_from_native_temp(tensor: &NativeTensor) -> Result<Vec<Self>>;
}

fn diagonal_multi_index(rank: usize, value: usize) -> Vec<usize> {
    vec![value; rank]
}

fn dense_diagonal_values<T: Copy + Default>(diag: &[T], logical_rank: usize) -> Result<Vec<T>> {
    anyhow::ensure!(
        logical_rank >= 1,
        "diagonal tensor construction requires at least one logical axis"
    );
    let diag_len = diag.len();
    let dims = vec![diag_len; logical_rank];
    let total_len = dims.iter().product::<usize>();
    let mut dense = vec![T::default(); total_len];
    let stride_prefix = (0..logical_rank)
        .scan(1usize, |state, _| {
            let current = *state;
            *state = state.saturating_mul(diag_len);
            Some(current)
        })
        .collect::<Vec<_>>();
    let diagonal_stride = stride_prefix.iter().sum::<usize>();
    for (i, value) in diag.iter().copied().enumerate() {
        dense[i * diagonal_stride] = value;
    }
    Ok(dense)
}

macro_rules! impl_tensor_element {
    ($ty:ty, $variant:ident, $payload:ident) => {
        impl TensorElement for $ty {
            fn dense_native_tensor_from_col_major(
                data: &[Self],
                dims: &[usize],
            ) -> Result<NativeTensor> {
                let typed = TypedTensor::<Self>::from_slice(data, dims, MemoryOrder::ColumnMajor)
                    .map_err(|e| anyhow!("failed to build native dense tensor: {e}"))?;
                Ok(NativeTensor::from(typed))
            }

            fn diag_native_tensor_from_col_major(
                data: &[Self],
                logical_rank: usize,
            ) -> Result<NativeTensor> {
                let dims = vec![data.len(); logical_rank];
                let dense = dense_diagonal_values(data, logical_rank)?;
                Self::dense_native_tensor_from_col_major(&dense, &dims)
            }

            fn scalar_native_tensor(value: Self) -> Result<NativeTensor> {
                let typed =
                    TypedTensor::<Self>::from_slice(&[value], &[], MemoryOrder::ColumnMajor)
                        .map_err(|e| anyhow!("failed to build native rank-0 tensor: {e}"))?;
                Ok(NativeTensor::from(typed))
            }

            fn dense_values_from_native_col_major(tensor: &NativeTensor) -> Result<Vec<Self>> {
                let dense = if tensor.is_dense() {
                    tensor.try_to_vec::<Self>()
                } else {
                    tensor.to_dense()?.try_to_vec::<Self>()
                }
                .map_err(|e| anyhow!("dense native tensor extraction failed: {e}"))?;
                Ok(dense)
            }

            fn diag_values_from_native_temp(tensor: &NativeTensor) -> Result<Vec<Self>> {
                let rank = tensor.ndim();
                anyhow::ensure!(rank >= 1, "diagonal native tensor rank must be at least 1");
                let diag_len = tensor.dims()[0];
                anyhow::ensure!(
                    tensor.dims().iter().all(|&dim| dim == diag_len),
                    "expected square/equal logical dims for diagonal extraction, got {:?}",
                    tensor.dims()
                );
                let mut values = Vec::with_capacity(diag_len);
                for i in 0..diag_len {
                    let index = diagonal_multi_index(rank, i);
                    values.push(
                        tensor.try_get::<Self>(&index).map_err(|e| {
                            anyhow!("diagonal native tensor extraction failed: {e}")
                        })?,
                    );
                }
                Ok(values)
            }
        }
    };
}

impl_tensor_element!(f32, F32, payload);
impl_tensor_element!(f64, F64, payload);
impl_tensor_element!(Complex32, C32, payload);
impl_tensor_element!(Complex64, C64, payload);
