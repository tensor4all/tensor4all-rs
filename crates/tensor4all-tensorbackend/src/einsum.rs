//! Backend-neutral einsum facade for tensor contraction.
//!
//! This module provides a unified interface for einsum operations that routes
//! to either mdarray-einsum (default) or torch.einsum (when `backend-libtorch` is enabled).
//!
//! ## Backend Selection
//!
//! - If all inputs are mdarray-based (Dense/Diag), uses mdarray-einsum
//! - If any input is Torch storage, uses torch.einsum (converts mdarray inputs to torch)
//!
//! ## Usage
//!
//! ```ignore
//! use tensor4all_tensorbackend::einsum::{einsum_storage, EinsumInput};
//!
//! let inputs = vec![
//!     EinsumInput { ids: &[0, 1], storage: &storage_a, dims: &[2, 3] },
//!     EinsumInput { ids: &[1, 2], storage: &storage_b, dims: &[3, 4] },
//! ];
//! let result = einsum_storage(&inputs, &[0, 2])?;  // Contract index 1
//! ```

use anyhow::Result;
use num_complex::Complex64;
use std::collections::HashMap;

use crate::storage::{DenseStorageC64, DenseStorageF64, Storage};

/// Input for einsum operation.
#[derive(Debug, Clone)]
pub struct EinsumInput<'a> {
    /// Axis IDs for this tensor (unique identifiers for each axis).
    pub ids: &'a [usize],
    /// Reference to the storage.
    pub storage: &'a Storage,
    /// Dimensions of the tensor.
    pub dims: &'a [usize],
}

/// Perform einsum contraction on storage tensors.
///
/// This is the main entry point for backend-neutral einsum. It automatically
/// selects the appropriate backend based on the input storage types.
///
/// # Arguments
///
/// * `inputs` - Slice of einsum inputs, each containing axis IDs, storage, and dimensions
/// * `output_ids` - Axis IDs for the output tensor (uncontracted indices)
///
/// # Returns
///
/// The contracted tensor as Storage. The result type (F64 or C64) is determined
/// by the input types - if any input is complex, the result is complex.
///
/// # Backend Selection
///
/// - All mdarray (Dense/Diag): uses mdarray-einsum
/// - Any Torch: uses torch.einsum (with automatic conversion)
pub fn einsum_storage(inputs: &[EinsumInput<'_>], output_ids: &[usize]) -> Result<Storage> {
    // Check if any input is torch storage
    #[cfg(feature = "backend-libtorch")]
    {
        let has_torch = inputs.iter().any(|input| input.storage.is_torch());
        if has_torch {
            return einsum_torch(inputs, output_ids);
        }
    }

    // Default: use mdarray-einsum
    einsum_mdarray(inputs, output_ids)
}

/// Perform einsum using mdarray-einsum backend.
fn einsum_mdarray(inputs: &[EinsumInput<'_>], output_ids: &[usize]) -> Result<Storage> {
    use mdarray::{Dense, DynRank, Slice, Tensor};
    use mdarray_einsum::{einsum_optimized, Naive, TypedTensor};

    // Determine if we need complex output
    let has_complex = inputs.iter().any(|input| {
        matches!(
            input.storage,
            Storage::DenseC64(_) | Storage::DiagC64(_)
        )
    });

    // Build sizes map
    let mut sizes: HashMap<usize, usize> = HashMap::new();
    for input in inputs {
        for (&id, &dim) in input.ids.iter().zip(input.dims.iter()) {
            sizes.entry(id).or_insert(dim);
        }
    }

    // Convert to TypedTensor and handle Diag tensors
    let mut typed_tensors: Vec<TypedTensor> = Vec::new();
    let mut einsum_ids: Vec<Vec<usize>> = Vec::new();

    for input in inputs {
        let is_diag = input.storage.is_diag();

        if is_diag {
            // Diag tensor: extract diagonal as 1D tensor with single hyperedge ID
            let typed = match input.storage {
                Storage::DiagC64(ds) => {
                    let diag_len = ds.as_slice().len();
                    let tensor_1d = Tensor::from(ds.as_slice().to_vec())
                        .into_shape([diag_len].as_slice())
                        .into_dyn();
                    TypedTensor::C64(tensor_1d)
                }
                Storage::DiagF64(ds) => {
                    let diag_len = ds.as_slice().len();
                    let tensor_1d = Tensor::from(ds.as_slice().to_vec())
                        .into_shape([diag_len].as_slice())
                        .into_dyn();
                    TypedTensor::F64(tensor_1d)
                }
                _ => unreachable!(),
            };
            typed_tensors.push(typed);
            // Diag uses first ID as hyperedge (all axes unified)
            einsum_ids.push(vec![input.ids[0]]);
        } else {
            // Dense tensor
            let typed = match input.storage {
                Storage::DenseC64(ds) => TypedTensor::C64(ds.tensor().clone()),
                Storage::DenseF64(ds) => TypedTensor::F64(ds.tensor().clone()),
                #[cfg(feature = "backend-libtorch")]
                Storage::TorchF64(_) | Storage::TorchC64(_) => {
                    unreachable!("Torch storage should be handled by einsum_torch")
                }
                _ => unreachable!(),
            };
            typed_tensors.push(typed);
            einsum_ids.push(input.ids.to_vec());
        }
    }

    // Perform einsum based on type
    type EinsumInputRef<'a, T> = (&'a [usize], &'a Slice<T, DynRank, Dense>);

    let result_typed = if has_complex {
        // Convert all to C64
        let c64_tensors: Vec<Tensor<Complex64, DynRank>> =
            typed_tensors.iter().map(|t| t.to_c64()).collect();
        let einsum_inputs: Vec<EinsumInputRef<Complex64>> = einsum_ids
            .iter()
            .zip(c64_tensors.iter())
            .map(|(ids, tensor)| (ids.as_slice(), tensor.as_ref()))
            .collect();
        let result = einsum_optimized(&Naive, &einsum_inputs, output_ids, &sizes);
        TypedTensor::C64(result)
    } else {
        // All F64
        let f64_tensors: Vec<&Tensor<f64, DynRank>> = typed_tensors
            .iter()
            .map(|t| t.as_f64().expect("Expected F64 tensor"))
            .collect();
        let einsum_inputs: Vec<EinsumInputRef<f64>> = einsum_ids
            .iter()
            .zip(f64_tensors.iter())
            .map(|(ids, tensor)| (ids.as_slice(), tensor.as_ref()))
            .collect();
        let result = einsum_optimized(&Naive, &einsum_inputs, output_ids, &sizes);
        TypedTensor::F64(result)
    };

    // Convert result to Storage
    let result_dims: Vec<usize> = result_typed.dims().to_vec();
    let (final_dims, storage) = if output_ids.is_empty() && result_dims == vec![1] {
        // Scalar output
        (vec![], typed_tensor_to_storage(result_typed, &[]))
    } else {
        (result_dims.clone(), typed_tensor_to_storage(result_typed, &result_dims))
    };

    // Adjust for scalar case
    if final_dims.is_empty() {
        Ok(storage)
    } else {
        Ok(storage)
    }
}

/// Perform einsum using torch backend.
#[cfg(feature = "backend-libtorch")]
fn einsum_torch(inputs: &[EinsumInput<'_>], output_ids: &[usize]) -> Result<Storage> {
    use crate::torch::TorchStorage;
    use tch::Tensor;

    // Determine if we need complex output
    let has_complex = inputs.iter().any(|input| {
        matches!(
            input.storage,
            Storage::DenseC64(_) | Storage::DiagC64(_) | Storage::TorchC64(_)
        )
    });

    // Convert all inputs to torch tensors
    let mut torch_tensors: Vec<Tensor> = Vec::new();
    let mut einsum_ids: Vec<Vec<usize>> = Vec::new();

    for input in inputs {
        let is_diag = input.storage.is_diag();

        if is_diag {
            // Diag tensor: densify first, then convert to torch
            let dense = input.storage.to_dense_storage(input.dims);
            let tensor = storage_to_torch_tensor(&dense, has_complex)?;
            torch_tensors.push(tensor);
            einsum_ids.push(input.ids.to_vec());
        } else {
            let tensor = storage_to_torch_tensor(input.storage, has_complex)?;
            torch_tensors.push(tensor);
            einsum_ids.push(input.ids.to_vec());
        }
    }

    // Build einsum equation string
    let equation = ids_to_equation(
        &einsum_ids.iter().map(|v| v.as_slice()).collect::<Vec<_>>(),
        output_ids,
    )?;

    // Perform einsum
    let result = Tensor::einsum(&equation, &torch_tensors, None::<&[i64]>);

    // Convert result back to Storage
    if has_complex {
        let storage = TorchStorage::<Complex64>::from_tensor(result);
        Ok(Storage::TorchC64(storage))
    } else {
        let storage = TorchStorage::<f64>::from_tensor(result);
        Ok(Storage::TorchF64(storage))
    }
}

/// Convert Storage to torch Tensor.
#[cfg(feature = "backend-libtorch")]
fn storage_to_torch_tensor(storage: &Storage, promote_to_complex: bool) -> Result<tch::Tensor> {
    use tch::{Kind, Tensor};

    match storage {
        Storage::DenseF64(ds) => {
            let data = ds.as_slice();
            let dims: Vec<i64> = ds.dims().iter().map(|&d| d as i64).collect();
            let tensor = Tensor::from_slice(data).view(&dims[..]).to_kind(Kind::Double);
            if promote_to_complex {
                let imag = Tensor::zeros_like(&tensor);
                Ok(Tensor::complex(&tensor, &imag))
            } else {
                Ok(tensor)
            }
        }
        Storage::DenseC64(ds) => {
            let data = ds.as_slice();
            let dims: Vec<i64> = ds.dims().iter().map(|&d| d as i64).collect();
            let real: Vec<f64> = data.iter().map(|c| c.re).collect();
            let imag: Vec<f64> = data.iter().map(|c| c.im).collect();
            let real_tensor = Tensor::from_slice(&real).view(&dims[..]).to_kind(Kind::Double);
            let imag_tensor = Tensor::from_slice(&imag).view(&dims[..]).to_kind(Kind::Double);
            Ok(Tensor::complex(&real_tensor, &imag_tensor))
        }
        Storage::DiagF64(_) | Storage::DiagC64(_) => {
            anyhow::bail!("Diag storage should be densified before calling storage_to_torch_tensor")
        }
        #[cfg(feature = "backend-libtorch")]
        Storage::TorchF64(ts) => {
            if promote_to_complex {
                let tensor = ts.tensor();
                let imag = tch::Tensor::zeros_like(tensor);
                Ok(tch::Tensor::complex(tensor, &imag))
            } else {
                Ok(ts.tensor().shallow_clone())
            }
        }
        #[cfg(feature = "backend-libtorch")]
        Storage::TorchC64(ts) => Ok(ts.tensor().shallow_clone()),
    }
}

/// Convert axis IDs to einsum equation string.
///
/// Maps usize IDs to letters a-zA-Z (max 52 unique labels).
#[cfg(feature = "backend-libtorch")]
fn ids_to_equation(input_ids: &[&[usize]], output_ids: &[usize]) -> Result<String> {
    let mut id_to_char: HashMap<usize, char> = HashMap::new();
    let mut next_char = 0u8;

    let mut get_char = |id: usize| -> Result<char> {
        if let Some(&c) = id_to_char.get(&id) {
            return Ok(c);
        }
        if next_char >= 52 {
            anyhow::bail!(
                "Too many unique axis IDs ({}). Torch einsum supports max 52 (a-zA-Z).",
                id_to_char.len() + 1
            );
        }
        let c = if next_char < 26 {
            (b'a' + next_char) as char
        } else {
            (b'A' + next_char - 26) as char
        };
        id_to_char.insert(id, c);
        next_char += 1;
        Ok(c)
    };

    // Build input part of equation
    let mut parts: Vec<String> = Vec::new();
    for ids in input_ids {
        let chars: String = ids.iter().map(|&id| get_char(id)).collect::<Result<_>>()?;
        parts.push(chars);
    }

    // Build output part
    let output_chars: String = output_ids
        .iter()
        .map(|&id| get_char(id))
        .collect::<Result<_>>()?;

    Ok(format!("{}->{}", parts.join(","), output_chars))
}

/// Convert TypedTensor to Storage.
fn typed_tensor_to_storage(typed: mdarray_einsum::TypedTensor, dims: &[usize]) -> Storage {
    match typed {
        mdarray_einsum::TypedTensor::F64(t) => {
            let data: Vec<f64> = t.into_vec();
            Storage::DenseF64(DenseStorageF64::from_vec_with_shape(data, dims))
        }
        mdarray_einsum::TypedTensor::C64(t) => {
            let data: Vec<Complex64> = t.into_vec();
            Storage::DenseC64(DenseStorageC64::from_vec_with_shape(data, dims))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_dense_f64(data: Vec<f64>, dims: &[usize]) -> Storage {
        Storage::DenseF64(DenseStorageF64::from_vec_with_shape(data, dims))
    }

    fn make_dense_c64(data: Vec<Complex64>, dims: &[usize]) -> Storage {
        Storage::DenseC64(DenseStorageC64::from_vec_with_shape(data, dims))
    }

    #[test]
    fn test_einsum_matmul_f64() {
        // A[i,j] * B[j,k] -> C[i,k]
        // A = [[1,2],[3,4]], B = [[5,6],[7,8]]
        // C = [[19,22],[43,50]]
        let a = make_dense_f64(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = make_dense_f64(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]);

        let inputs = vec![
            EinsumInput {
                ids: &[0, 1],
                storage: &a,
                dims: &[2, 2],
            },
            EinsumInput {
                ids: &[1, 2],
                storage: &b,
                dims: &[2, 2],
            },
        ];

        let result = einsum_storage(&inputs, &[0, 2]).unwrap();

        match &result {
            Storage::DenseF64(ds) => {
                let data = ds.as_slice();
                assert_eq!(data.len(), 4);
                assert!((data[0] - 19.0).abs() < 1e-10);
                assert!((data[1] - 22.0).abs() < 1e-10);
                assert!((data[2] - 43.0).abs() < 1e-10);
                assert!((data[3] - 50.0).abs() < 1e-10);
            }
            _ => panic!("Expected DenseF64"),
        }
    }

    #[test]
    fn test_einsum_inner_product() {
        // A[i] * B[i] -> scalar
        let a = make_dense_f64(vec![1.0, 2.0, 3.0], &[3]);
        let b = make_dense_f64(vec![4.0, 5.0, 6.0], &[3]);

        let inputs = vec![
            EinsumInput {
                ids: &[0],
                storage: &a,
                dims: &[3],
            },
            EinsumInput {
                ids: &[0],
                storage: &b,
                dims: &[3],
            },
        ];

        let result = einsum_storage(&inputs, &[]).unwrap();

        match &result {
            Storage::DenseF64(ds) => {
                let data = ds.as_slice();
                // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
                assert!((data[0] - 32.0).abs() < 1e-10);
            }
            _ => panic!("Expected DenseF64"),
        }
    }

    #[test]
    fn test_einsum_mixed_types() {
        // F64 * C64 -> C64
        let a = make_dense_f64(vec![1.0, 2.0], &[2]);
        let b = make_dense_c64(
            vec![Complex64::new(3.0, 1.0), Complex64::new(4.0, 2.0)],
            &[2],
        );

        let inputs = vec![
            EinsumInput {
                ids: &[0],
                storage: &a,
                dims: &[2],
            },
            EinsumInput {
                ids: &[0],
                storage: &b,
                dims: &[2],
            },
        ];

        let result = einsum_storage(&inputs, &[]).unwrap();

        match &result {
            Storage::DenseC64(ds) => {
                let data = ds.as_slice();
                // (1+0i)*(3+1i) + (2+0i)*(4+2i) = (3+1i) + (8+4i) = (11+5i)
                assert!((data[0].re - 11.0).abs() < 1e-10);
                assert!((data[0].im - 5.0).abs() < 1e-10);
            }
            _ => panic!("Expected DenseC64"),
        }
    }

    #[test]
    fn test_einsum_three_tensors() {
        // A[i,j] * B[j,k] * C[k,l] -> D[i,l]
        let a = make_dense_f64(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = make_dense_f64(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]); // Identity
        let c = make_dense_f64(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]);

        let inputs = vec![
            EinsumInput {
                ids: &[0, 1],
                storage: &a,
                dims: &[2, 2],
            },
            EinsumInput {
                ids: &[1, 2],
                storage: &b,
                dims: &[2, 2],
            },
            EinsumInput {
                ids: &[2, 3],
                storage: &c,
                dims: &[2, 2],
            },
        ];

        let result = einsum_storage(&inputs, &[0, 3]).unwrap();

        // A * I * C = A * C
        match &result {
            Storage::DenseF64(ds) => {
                let data = ds.as_slice();
                assert_eq!(data.len(), 4);
                // Same as A * C
                assert!((data[0] - 19.0).abs() < 1e-10);
                assert!((data[1] - 22.0).abs() < 1e-10);
                assert!((data[2] - 43.0).abs() < 1e-10);
                assert!((data[3] - 50.0).abs() < 1e-10);
            }
            _ => panic!("Expected DenseF64"),
        }
    }
}
