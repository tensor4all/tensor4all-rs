//! Runtime bridge for tenferro backend execution.
//!
//! This module centralizes runtime selection so backend code stays
//! implementation-agnostic (CPU today, GPU-ready extension point).

use anyhow::{anyhow, Result};
use std::env;
use tenferro_dyadtensor::{AdTensor, DynAdTensor};
use tenferro_prims::{CpuBackend, CpuContext};
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::storage::{DenseStorageC64, DenseStorageF64, DiagStorageC64, DiagStorageF64, Storage};
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

/// Convert legacy [`Storage`] into a primal-mode [`DynAdTensor`].
///
/// Diagonal storage is materialized to dense storage for now.
pub fn storage_to_dyn_ad_tensor(storage: &Storage, logical_dims: &[usize]) -> Result<DynAdTensor> {
    match storage {
        Storage::DenseF64(_) | Storage::DiagF64(_) => Ok(DynAdTensor::from(AdTensor::new_primal(
            dense_f64_to_tensor(storage, logical_dims)?,
        ))),
        Storage::DenseC64(_) | Storage::DiagC64(_) => Ok(DynAdTensor::from(AdTensor::new_primal(
            dense_c64_to_tensor(storage, logical_dims)?,
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
        DynAdTensor::F64(t) => tensor_f64_to_storage(t.primal()),
        DynAdTensor::C64(t) => tensor_c64_to_storage(t.primal()),
    }
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
    fn storage_dyn_ad_tensor_roundtrip_diag_materializes_to_dense() {
        let storage = Storage::DiagF64(DiagStorageF64::from_vec(vec![2.0, -1.0, 4.0]));

        let native = storage_to_dyn_ad_tensor(&storage, &[3, 3]).unwrap();
        let roundtrip = dyn_ad_tensor_primal_to_storage(&native).unwrap();

        let expected = storage.to_dense_storage(&[3, 3]);
        assert_storage_eq(&roundtrip, &expected);
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
}
