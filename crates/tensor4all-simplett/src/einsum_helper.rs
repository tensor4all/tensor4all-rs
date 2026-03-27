use tenferro_tensor::{MemoryOrder, Tensor as TfTensor};

pub(crate) fn einsum_tensors<T: EinsumScalar>(
    subscripts: &str,
    operands: &[&TfTensor<T>],
) -> TfTensor<T> {
    T::einsum_dispatch(subscripts, operands)
}

pub(crate) fn tensor_to_row_major_vec<T: tenferro_algebra::Scalar>(tensor: &TfTensor<T>) -> Vec<T> {
    tensor
        .contiguous(MemoryOrder::RowMajor)
        .buffer()
        .as_slice()
        .expect("einsum helper requires CPU-accessible tensors")
        .to_vec()
}

pub(crate) fn row_vector_times_matrix<T: EinsumScalar>(
    vector: &[T],
    matrix: &[T],
    rows: usize,
    cols: usize,
) -> Vec<T> {
    assert_eq!(vector.len(), rows);
    assert_eq!(matrix.len(), rows * cols);

    let vector_tf = TfTensor::from_slice(vector, &[rows], MemoryOrder::RowMajor)
        .expect("row vector dimensions should match einsum input");
    let matrix_tf = TfTensor::from_slice(matrix, &[rows, cols], MemoryOrder::RowMajor)
        .expect("matrix dimensions should match einsum input");

    tensor_to_row_major_vec(&einsum_tensors("l,lr->r", &[&vector_tf, &matrix_tf]))
}

pub(crate) fn matrix_times_col_vector<T: EinsumScalar>(
    matrix: &[T],
    rows: usize,
    cols: usize,
    vector: &[T],
) -> Vec<T> {
    assert_eq!(matrix.len(), rows * cols);
    assert_eq!(vector.len(), cols);

    let matrix_tf = TfTensor::from_slice(matrix, &[rows, cols], MemoryOrder::RowMajor)
        .expect("matrix dimensions should match einsum input");
    let vector_tf = TfTensor::from_slice(vector, &[cols], MemoryOrder::RowMajor)
        .expect("column vector dimensions should match einsum input");

    tensor_to_row_major_vec(&einsum_tensors("lr,r->l", &[&matrix_tf, &vector_tf]))
}

/// Scalar types supported by the tenferro-backed einsum helpers used in
/// `tensor4all-simplett`.
pub trait EinsumScalar: tenferro_algebra::Scalar + Sized {
    /// Execute an einsum on typed tenferro tensors.
    fn einsum_dispatch(subscripts: &str, operands: &[&TfTensor<Self>]) -> TfTensor<Self>;
}

macro_rules! impl_einsum_scalar {
    ($($t:ty),*) => {
        $(
            impl EinsumScalar for $t {
                fn einsum_dispatch(subscripts: &str, operands: &[&TfTensor<Self>]) -> TfTensor<Self> {
                    use tenferro_tensor_compute::{einsum, CpuBackend, CpuContext, Standard};

                    let mut ctx = CpuContext::new(1);
                    einsum::<Standard<$t>, CpuBackend>(&mut ctx, subscripts, operands, None)
                        .expect("einsum failed")
                }
            }
        )*
    };
}

impl_einsum_scalar!(f64, f32, num_complex::Complex64, num_complex::Complex32);
