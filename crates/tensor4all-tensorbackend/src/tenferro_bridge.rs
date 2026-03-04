//! Runtime bridge for tenferro backend execution.
//!
//! This module centralizes runtime selection so backend code stays
//! implementation-agnostic (CPU today, GPU-ready extension point).

use anyhow::{anyhow, Result};
use std::env;
use tenferro_prims::{CpuBackend, CpuContext};

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

#[cfg(test)]
mod tests {
    use super::*;
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
}
