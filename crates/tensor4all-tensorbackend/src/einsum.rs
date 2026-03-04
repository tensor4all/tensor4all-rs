//! Backend-neutral einsum facade for tensor contraction.
//!
//! This implementation routes contraction to `tenferro-einsum`.

use anyhow::{anyhow, Result};
use num_complex::Complex64;
use tenferro_algebra::Standard;
use tenferro_einsum::{einsum_with_subscripts, Subscripts};
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::storage::{DenseStorageC64, DenseStorageF64, Storage};
use crate::tenferro_bridge::{with_tenferro_ctx, ActivePrimsBackend};

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

fn id_to_u32(id: usize) -> Result<u32> {
    u32::try_from(id).map_err(|_| anyhow!("axis id {} does not fit in u32", id))
}

fn dense_f64_to_tensor(storage: &Storage) -> Result<Tensor<f64>> {
    match storage {
        Storage::DenseF64(ds) => {
            Tensor::from_slice(ds.as_slice(), &ds.dims(), MemoryOrder::RowMajor)
                .map_err(|e| anyhow!("failed to build f64 tensor from dense storage: {}", e))
        }
        Storage::DiagF64(ds) => {
            Tensor::from_slice(ds.as_slice(), &[ds.len()], MemoryOrder::RowMajor)
                .map_err(|e| anyhow!("failed to build f64 tensor from diag storage: {}", e))
        }
        Storage::DenseC64(_) | Storage::DiagC64(_) => {
            Err(anyhow!("complex storage cannot be converted to f64 tensor"))
        }
    }
}

fn dense_c64_to_tensor(storage: &Storage) -> Result<Tensor<Complex64>> {
    match storage {
        Storage::DenseC64(ds) => {
            Tensor::from_slice(ds.as_slice(), &ds.dims(), MemoryOrder::RowMajor)
                .map_err(|e| anyhow!("failed to build c64 tensor from dense storage: {}", e))
        }
        Storage::DiagC64(ds) => {
            Tensor::from_slice(ds.as_slice(), &[ds.len()], MemoryOrder::RowMajor)
                .map_err(|e| anyhow!("failed to build c64 tensor from diag storage: {}", e))
        }
        Storage::DenseF64(ds) => {
            let promoted: Vec<Complex64> = ds
                .as_slice()
                .iter()
                .copied()
                .map(|x| Complex64::new(x, 0.0))
                .collect();
            Tensor::from_slice(&promoted, &ds.dims(), MemoryOrder::RowMajor)
                .map_err(|e| anyhow!("failed to promote dense f64 tensor to c64: {}", e))
        }
        Storage::DiagF64(ds) => {
            let promoted: Vec<Complex64> = ds
                .as_slice()
                .iter()
                .copied()
                .map(|x| Complex64::new(x, 0.0))
                .collect();
            Tensor::from_slice(&promoted, &[ds.len()], MemoryOrder::RowMajor)
                .map_err(|e| anyhow!("failed to promote diag f64 tensor to c64: {}", e))
        }
    }
}

fn tensor_f64_to_storage(tensor: Tensor<f64>, output_ids: &[usize]) -> Result<Storage> {
    let dims = tensor.dims().to_vec();
    let final_dims = if output_ids.is_empty() && dims == vec![1] {
        vec![]
    } else {
        dims
    };
    let row = tensor.into_contiguous(MemoryOrder::RowMajor);
    let data = row
        .buffer()
        .as_slice()
        .ok_or_else(|| anyhow!("expected host-accessible einsum output"))?
        .to_vec();
    Ok(Storage::DenseF64(DenseStorageF64::from_vec_with_shape(
        data,
        &final_dims,
    )))
}

fn tensor_c64_to_storage(tensor: Tensor<Complex64>, output_ids: &[usize]) -> Result<Storage> {
    let dims = tensor.dims().to_vec();
    let final_dims = if output_ids.is_empty() && dims == vec![1] {
        vec![]
    } else {
        dims
    };
    let row = tensor.into_contiguous(MemoryOrder::RowMajor);
    let data = row
        .buffer()
        .as_slice()
        .ok_or_else(|| anyhow!("expected host-accessible einsum output"))?
        .to_vec();
    Ok(Storage::DenseC64(DenseStorageC64::from_vec_with_shape(
        data,
        &final_dims,
    )))
}

/// Perform einsum contraction on storage tensors.
pub fn einsum_storage(inputs: &[EinsumInput<'_>], output_ids: &[usize]) -> Result<Storage> {
    if inputs.is_empty() {
        return Err(anyhow!("einsum_storage requires at least one input"));
    }

    let has_complex = inputs
        .iter()
        .any(|input| matches!(input.storage, Storage::DenseC64(_) | Storage::DiagC64(_)));

    let input_labels: Vec<Vec<u32>> = inputs
        .iter()
        .map(|input| {
            input
                .ids
                .iter()
                .copied()
                .map(id_to_u32)
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<_>>>()?;
    let output_labels: Vec<u32> = output_ids
        .iter()
        .copied()
        .map(id_to_u32)
        .collect::<Result<Vec<_>>>()?;
    let input_label_slices: Vec<&[u32]> = input_labels.iter().map(Vec::as_slice).collect();
    let subscripts = Subscripts::new(&input_label_slices, &output_labels);

    if has_complex {
        let operands: Vec<Tensor<Complex64>> = inputs
            .iter()
            .map(|input| dense_c64_to_tensor(input.storage))
            .collect::<Result<_>>()?;
        let operand_refs: Vec<&Tensor<Complex64>> = operands.iter().collect();
        let result = with_tenferro_ctx("einsum(c64)", |ctx| {
            einsum_with_subscripts::<Standard<Complex64>, ActivePrimsBackend>(
                ctx,
                &subscripts,
                &operand_refs,
                None,
            )
            .map_err(|e| anyhow!("tenferro einsum (c64) failed: {}", e))
        })?;
        tensor_c64_to_storage(result, output_ids)
    } else {
        let operands: Vec<Tensor<f64>> = inputs
            .iter()
            .map(|input| dense_f64_to_tensor(input.storage))
            .collect::<Result<_>>()?;
        let operand_refs: Vec<&Tensor<f64>> = operands.iter().collect();
        let result = with_tenferro_ctx("einsum(f64)", |ctx| {
            einsum_with_subscripts::<Standard<f64>, ActivePrimsBackend>(
                ctx,
                &subscripts,
                &operand_refs,
                None,
            )
            .map_err(|e| anyhow!("tenferro einsum (f64) failed: {}", e))
        })?;
        tensor_f64_to_storage(result, output_ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{contract_storage, DenseStorageF64};

    #[test]
    fn test_einsum_storage_respects_output_axis_order() {
        // A[i,j] with dims [2,3], row-major data:
        // [[1,2,3],
        //  [4,5,6]]
        let storage = Storage::DenseF64(DenseStorageF64::from_vec_with_shape(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 3],
        ));
        let input = EinsumInput {
            ids: &[0, 1],
            storage: &storage,
            dims: &[2, 3],
        };

        // Reorder output axes: [j, i]
        let result = einsum_storage(&[input], &[1, 0]).expect("einsum should succeed");
        match result {
            Storage::DenseF64(ds) => {
                assert_eq!(ds.dims(), vec![3, 2]);
                // Expected transpose:
                // [[1,4],
                //  [2,5],
                //  [3,6]]
                assert_eq!(ds.as_slice(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
            }
            other => panic!("expected DenseF64, got {other:?}"),
        }
    }

    #[test]
    fn test_einsum_storage_respects_output_axis_order_after_contraction() {
        // A[i,j] (2x3)
        let a = Storage::DenseF64(DenseStorageF64::from_vec_with_shape(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 3],
        ));
        // B[j,k] (3x4)
        let b = Storage::DenseF64(DenseStorageF64::from_vec_with_shape(
            vec![
                1.0, 2.0, 3.0, 4.0, //
                5.0, 6.0, 7.0, 8.0, //
                9.0, 10.0, 11.0, 12.0,
            ],
            &[3, 4],
        ));

        let inputs = vec![
            EinsumInput {
                ids: &[0, 1], // i,j
                storage: &a,
                dims: &[2, 3],
            },
            EinsumInput {
                ids: &[1, 2], // j,k
                storage: &b,
                dims: &[3, 4],
            },
        ];

        // Output in reversed remaining axis order: [k, i]
        let result = einsum_storage(&inputs, &[2, 0]).expect("einsum should succeed");
        match result {
            Storage::DenseF64(ds) => {
                assert_eq!(ds.dims(), vec![4, 2]);
                assert_eq!(
                    ds.as_slice(),
                    &[38.0, 83.0, 44.0, 98.0, 50.0, 113.0, 56.0, 128.0]
                );
            }
            other => panic!("expected DenseF64, got {other:?}"),
        }
    }

    #[test]
    fn test_einsum_storage_matches_binary_contract_for_rank3_rank2() {
        // A[a,b,c] with dims [3,2,3]
        let a = Storage::DenseF64(DenseStorageF64::from_vec_with_shape(
            (1..=18).map(|x| x as f64).collect(),
            &[3, 2, 3],
        ));
        // B[c,d] with dims [3,2]
        let b = Storage::DenseF64(DenseStorageF64::from_vec_with_shape(
            (1..=6).map(|x| x as f64).collect(),
            &[3, 2],
        ));

        // Einsum path: contract on c => output [a,b,d]
        let inputs = vec![
            EinsumInput {
                ids: &[0, 1, 2],
                storage: &a,
                dims: &[3, 2, 3],
            },
            EinsumInput {
                ids: &[2, 3],
                storage: &b,
                dims: &[3, 2],
            },
        ];
        let einsum = einsum_storage(&inputs, &[0, 1, 3]).expect("einsum should succeed");

        // Binary dense contraction path (reference)
        let expected = contract_storage(&a, &[3, 2, 3], &[2], &b, &[3, 2], &[0], &[3, 2, 2]);

        match (einsum, expected) {
            (Storage::DenseF64(e), Storage::DenseF64(x)) => {
                assert_eq!(e.dims(), x.dims());
                assert_eq!(e.as_slice(), x.as_slice());
            }
            (e, x) => panic!("expected DenseF64/DenseF64, got {e:?} and {x:?}"),
        }
    }

    #[test]
    fn test_rank3_dense_roundtrip_preserves_row_major_order() {
        let src = Storage::DenseF64(DenseStorageF64::from_vec_with_shape(
            (1..=18).map(|x| x as f64).collect(),
            &[3, 2, 3],
        ));
        let t = dense_f64_to_tensor(&src).expect("to tensor");
        let roundtrip = tensor_f64_to_storage(t, &[0, 1, 2]).expect("to storage");
        match roundtrip {
            Storage::DenseF64(ds) => {
                assert_eq!(ds.dims(), vec![3, 2, 3]);
                assert_eq!(
                    ds.as_slice(),
                    &(1..=18).map(|x| x as f64).collect::<Vec<_>>()[..]
                );
            }
            other => panic!("expected DenseF64, got {other:?}"),
        }
    }

    #[test]
    fn test_einsum_storage_three_inputs_matches_sequential_binary_contract() {
        let t0 = Storage::DenseF64(DenseStorageF64::from_vec_with_shape(
            (1..=6).map(|x| x as f64).collect(),
            &[2, 3],
        )); // [i,a]
        let t1 = Storage::DenseF64(DenseStorageF64::from_vec_with_shape(
            (1..=18).map(|x| x as f64).collect(),
            &[3, 2, 3],
        )); // [a,b,c]
        let t2 = Storage::DenseF64(DenseStorageF64::from_vec_with_shape(
            (1..=6).map(|x| x as f64).collect(),
            &[3, 2],
        )); // [c,k]

        let inputs = vec![
            EinsumInput {
                ids: &[0, 1], // i,a
                storage: &t0,
                dims: &[2, 3],
            },
            EinsumInput {
                ids: &[1, 2, 3], // a,b,c
                storage: &t1,
                dims: &[3, 2, 3],
            },
            EinsumInput {
                ids: &[3, 4], // c,k
                storage: &t2,
                dims: &[3, 2],
            },
        ];

        let einsum = einsum_storage(&inputs, &[0, 2, 4]).expect("einsum should succeed");

        // Sequential binary contraction reference:
        // ((t0 * t1 over a) * t2 over c)
        let tmp = contract_storage(&t0, &[2, 3], &[1], &t1, &[3, 2, 3], &[0], &[2, 2, 3]);
        let expected = contract_storage(&tmp, &[2, 2, 3], &[2], &t2, &[3, 2], &[0], &[2, 2, 2]);

        match (einsum, expected) {
            (Storage::DenseF64(e), Storage::DenseF64(x)) => {
                assert_eq!(e.dims(), x.dims());
                assert_eq!(e.as_slice(), x.as_slice());
            }
            (e, x) => panic!("expected DenseF64/DenseF64, got {e:?} and {x:?}"),
        }
    }
}
