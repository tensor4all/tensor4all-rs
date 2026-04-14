use tenferro_einsum::typed_eager_einsum;
use tenferro_tensor::{TensorScalar, TypedTensor};
use tensor4all_tensorbackend::with_default_backend;

pub(crate) fn einsum_tensors<T: EinsumScalar>(
    subscripts: &str,
    operands: &[&TypedTensor<T>],
) -> TypedTensor<T> {
    with_default_backend(|backend| typed_eager_einsum(backend, operands, subscripts))
        .expect("einsum failed")
}

fn row_major_index_from_linear(shape: &[usize], mut linear: usize) -> Vec<usize> {
    let mut idx = vec![0usize; shape.len()];
    for axis in (0..shape.len()).rev() {
        let dim = shape[axis];
        if dim == 0 {
            return idx;
        }
        idx[axis] = linear % dim;
        linear /= dim;
    }
    idx
}

fn column_major_offset(indices: &[usize], shape: &[usize]) -> usize {
    let mut offset = 0usize;
    let mut stride = 1usize;
    for (axis, &idx) in indices.iter().enumerate() {
        offset += idx * stride;
        stride *= shape[axis];
    }
    offset
}

fn row_major_to_col_major<T: Clone>(shape: &[usize], data: &[T]) -> Vec<T> {
    assert_eq!(shape.iter().product::<usize>(), data.len());
    let mut col_major = data.to_vec();
    for (row_offset, value) in data.iter().cloned().enumerate() {
        let indices = row_major_index_from_linear(shape, row_offset);
        let col_offset = column_major_offset(&indices, shape);
        col_major[col_offset] = value;
    }
    col_major
}

fn col_major_to_row_major<T: Clone>(shape: &[usize], data: &[T]) -> Vec<T> {
    assert_eq!(shape.iter().product::<usize>(), data.len());
    let mut row_major = data.to_vec();
    for (row_offset, slot) in row_major.iter_mut().enumerate() {
        let indices = row_major_index_from_linear(shape, row_offset);
        let col_offset = column_major_offset(&indices, shape);
        *slot = data[col_offset].clone();
    }
    row_major
}

pub(crate) fn typed_tensor_from_row_major_slice<T: TensorScalar>(
    data: &[T],
    shape: &[usize],
) -> TypedTensor<T> {
    TypedTensor::from_vec(shape.to_vec(), row_major_to_col_major(shape, data))
}

pub(crate) fn typed_tensor_reshape<T: TensorScalar>(
    tensor: &TypedTensor<T>,
    shape: &[usize],
) -> TypedTensor<T> {
    assert_eq!(
        shape.iter().product::<usize>(),
        tensor.shape.iter().product::<usize>()
    );
    TypedTensor::from_vec(shape.to_vec(), tensor.host_data().to_vec())
}

pub(crate) fn tensor_to_row_major_vec<T: TensorScalar>(tensor: &TypedTensor<T>) -> Vec<T> {
    col_major_to_row_major(&tensor.shape, tensor.host_data())
}

pub(crate) fn row_vector_times_matrix<T: EinsumScalar>(
    vector: &[T],
    matrix: &[T],
    rows: usize,
    cols: usize,
) -> Vec<T> {
    assert_eq!(vector.len(), rows);
    assert_eq!(matrix.len(), rows * cols);

    let vector_tf = typed_tensor_from_row_major_slice(vector, &[rows]);
    let matrix_tf = typed_tensor_from_row_major_slice(matrix, &[rows, cols]);

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

    let matrix_tf = typed_tensor_from_row_major_slice(matrix, &[rows, cols]);
    let vector_tf = typed_tensor_from_row_major_slice(vector, &[cols]);

    tensor_to_row_major_vec(&einsum_tensors("lr,r->l", &[&matrix_tf, &vector_tf]))
}

/// Scalar types supported by the tenferro-backed einsum helpers used in
/// `tensor4all-simplett`.
pub trait EinsumScalar: tenferro_algebra::Scalar + TensorScalar + Sized {}

macro_rules! impl_einsum_scalar {
    ($($t:ty),*) => {
        $(impl EinsumScalar for $t {})*
    };
}

impl_einsum_scalar!(f64, f32, num_complex::Complex64, num_complex::Complex32);
