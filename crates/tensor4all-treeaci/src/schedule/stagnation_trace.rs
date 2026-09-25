//! Test-only, opt-in per-pass trace of local updates and global-guard work.
//!
//! Disabled unless [`enable`] was called on the current thread; every hook is
//! then a thread-local check. Snapshots clone the output once per pass, so the
//! trace is for bounded diagnostic fixtures only.

use std::{any::Any, cell::RefCell};

#[derive(Clone, Debug)]
pub(crate) struct GuardSearchTrace {
    /// Maximum `|f|` over the random starts (the guard's scale estimate).
    pub(crate) max_start_output: f64,
    /// Absolute acceptance threshold used by every walk.
    pub(crate) threshold: f64,
    /// Final `(error, point)` of every walk, above threshold or not.
    pub(crate) walks: Vec<(f64, Vec<usize>)>,
    /// Unique above-threshold points returned to the scheduler.
    pub(crate) pivots: Vec<Vec<usize>>,
    /// Physical indices of each point coordinate, in coordinate order.
    pub(crate) coordinate_indices: Vec<Vec<tensor4all_core::DynIndex>>,
}

#[derive(Clone, Debug)]
pub(crate) struct InjectionTrace {
    pub(crate) found: usize,
    pub(crate) injected: usize,
    pub(crate) ranks_after: Vec<usize>,
}

#[derive(Debug)]
pub(crate) struct PassTrace {
    pub(crate) pass: usize,
    pub(crate) edge_ranks: Vec<usize>,
    pub(crate) edge_errors: Vec<f64>,
    pub(crate) edge_scales: Vec<f64>,
    pub(crate) max_error_metric: f64,
    pub(crate) rank_limited: bool,
    pub(crate) stable_rank_passes: usize,
    /// Output (`TreeTN<IdxTensor, V>`) after the pass, before guard injection.
    pub(crate) output: Option<Box<dyn Any>>,
    pub(crate) search: Option<GuardSearchTrace>,
    pub(crate) injection: Option<InjectionTrace>,
}

thread_local! {
    static TRACE: RefCell<Option<Vec<PassTrace>>> = const { RefCell::new(None) };
}

pub(crate) fn enable() {
    TRACE.with(|trace| *trace.borrow_mut() = Some(Vec::new()));
}

pub(crate) fn take() -> Vec<PassTrace> {
    TRACE.with(|trace| trace.borrow_mut().take().unwrap_or_default())
}

pub(crate) fn is_enabled() -> bool {
    TRACE.with(|trace| trace.borrow().is_some())
}

pub(crate) fn with_passes(update: impl FnOnce(&mut Vec<PassTrace>)) {
    TRACE.with(|trace| {
        if let Some(passes) = trace.borrow_mut().as_mut() {
            update(passes);
        }
    });
}

pub(crate) fn record_search(search: GuardSearchTrace) {
    with_passes(|passes| {
        if let Some(last) = passes.last_mut() {
            last.search = Some(search);
        }
    });
}

pub(crate) fn record_injection(injection: InjectionTrace) {
    with_passes(|passes| {
        if let Some(last) = passes.last_mut() {
            last.injection = Some(injection);
        }
    });
}
