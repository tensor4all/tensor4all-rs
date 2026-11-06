//! Per-thread instrumentation of the actual public entry's tree edge loop.

use std::cell::Cell;

thread_local! {
    static FACTORIZATIONS: Cell<usize> = const { Cell::new(0) };
}

pub(in crate::treetn::contraction) fn record_factorization() {
    FACTORIZATIONS.with(|count| count.set(count.get() + 1));
}

pub(in crate::treetn::contraction) fn measure<R>(run: impl FnOnce() -> R) -> (R, usize) {
    FACTORIZATIONS.with(|count| count.set(0));
    let output = run();
    (output, FACTORIZATIONS.with(Cell::get))
}
