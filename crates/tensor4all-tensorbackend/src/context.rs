//! Process-global tenferro CPU execution helpers.
//!
//! tensor4all-rs routes tenferro CPU execution through one process-global
//! `CpuContext`, matching tenferro's `cpu:0` default-global thread-pool model.
//! Plain tensor operations and eager AD currently use separate `CpuBackend`
//! values because tenferro does not yet expose a public API for borrowing the
//! backend owned by an `EagerContext<CpuBackend>`. Both backends are created
//! from the same global CPU context, so thread-pool configuration is shared.

use std::sync::{Arc, Mutex, OnceLock};

use tenferro::{CpuBackend, EagerContext};
use tenferro_tensor::cpu::CpuContext;

static DEFAULT_CPU_CONTEXT: OnceLock<Arc<CpuContext>> = OnceLock::new();
static DEFAULT_BACKEND: OnceLock<Mutex<CpuBackend>> = OnceLock::new();
static DEFAULT_EAGER_CTX: OnceLock<Arc<EagerContext<CpuBackend>>> = OnceLock::new();

fn default_cpu_context() -> Arc<CpuContext> {
    DEFAULT_CPU_CONTEXT
        .get_or_init(|| Arc::new(CpuContext::from_env()))
        .clone()
}

fn default_backend() -> &'static Mutex<CpuBackend> {
    DEFAULT_BACKEND.get_or_init(|| Mutex::new(CpuBackend::from_context(default_cpu_context())))
}

/// Run a closure against the process-global CPU backend.
///
/// This is the canonical entry point for typed and untyped tenferro tensor
/// operations inside `tensor4all-tensorbackend`.
pub fn with_default_backend<R>(f: impl FnOnce(&mut CpuBackend) -> R) -> R {
    let mut backend = match default_backend().lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    f(&mut backend)
}

/// Return the process-global eager context used for reverse-mode AD.
///
/// This context owns a separate `CpuBackend` from [`with_default_backend`], but
/// both backends share the same process-global tenferro CPU context.
pub fn default_eager_ctx() -> Arc<EagerContext<CpuBackend>> {
    DEFAULT_EAGER_CTX
        .get_or_init(|| EagerContext::with_backend(CpuBackend::from_context(default_cpu_context())))
        .clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eager_context_is_process_global() {
        let first = default_eager_ctx();
        let second = default_eager_ctx();

        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn eager_context_is_shared_across_threads() {
        let main_context = default_eager_ctx();
        let worker_context = std::thread::spawn(default_eager_ctx)
            .join()
            .expect("worker thread should complete");

        assert!(Arc::ptr_eq(&main_context, &worker_context));
    }

    #[test]
    fn default_backend_is_shared_across_threads() {
        let main_threads = with_default_backend(|backend| backend.num_threads());
        let worker_threads =
            std::thread::spawn(|| with_default_backend(|backend| backend.num_threads()))
                .join()
                .expect("worker thread should complete");

        assert_eq!(main_threads, worker_threads);
    }
}
