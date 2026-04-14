//! Thread-local tenferro execution helpers.
//!
//! tensor4all-rs uses plain typed/untyped tensor ops and eager AD through
//! separate thread-local objects. The current tenferro public API does not
//! expose a way to share one exact `CpuBackend` instance with an
//! `EagerContext<CpuBackend>`, so this module intentionally keeps them as
//! paired but independent thread-local resources.

use std::cell::RefCell;
use std::sync::Arc;

use tenferro::{CpuBackend, EagerContext};

thread_local! {
    static DEFAULT_BACKEND: RefCell<CpuBackend> = RefCell::new(CpuBackend::new());
    static DEFAULT_EAGER_CTX: RefCell<Arc<EagerContext<CpuBackend>>> =
        RefCell::new(EagerContext::with_backend(CpuBackend::new()));
}

/// Run a closure against the thread-local CPU backend.
///
/// This is the canonical entry point for typed and untyped tenferro tensor
/// operations inside `tensor4all-tensorbackend`.
pub fn with_default_backend<R>(f: impl FnOnce(&mut CpuBackend) -> R) -> R {
    DEFAULT_BACKEND.with(|slot| {
        let mut backend = slot.borrow_mut();
        f(&mut backend)
    })
}

/// Return the thread-local eager context used for reverse-mode AD.
///
/// This context is separate from the backend used by [`with_default_backend`].
pub fn default_eager_ctx() -> Arc<EagerContext<CpuBackend>> {
    DEFAULT_EAGER_CTX.with(|slot| slot.borrow().clone())
}
