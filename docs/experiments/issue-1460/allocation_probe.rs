//! Temporary instrumentation included only in the allocation-probe binaries.
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::{Cell, RefCell};

#[derive(Clone, Copy, Debug, Default)]
struct Counts {
    entries: usize,
    allocating_entries: usize,
    allocations: usize,
    reallocations: usize,
}

thread_local! {
    static ACTIVE: Cell<Option<usize>> = const { Cell::new(None) };
    static COUNTS: RefCell<[Counts; 4]> = RefCell::new([Counts::default(); 4]);
}

struct Allocator;

#[global_allocator]
static ALLOCATOR: Allocator = Allocator;

// SAFETY: All requests are forwarded unchanged to System. TLS accounting uses
// fixed-size, non-allocating cells and is disabled outside index preparation.
unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        account(|counts| counts.allocations += 1);
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        account(|counts| counts.allocations += 1);
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        account(|counts| counts.reallocations += 1);
        unsafe { System.realloc(ptr, layout, size) }
    }
}

fn account(update: impl FnOnce(&mut Counts)) {
    let _ = ACTIVE.try_with(|active| {
        if let Some(slot) = active.get() {
            COUNTS.with(|counts| update(&mut counts.borrow_mut()[slot]));
        }
    });
}

pub(super) fn measure<R>(slot: usize, prepare_index: impl FnOnce() -> R) -> R {
    let operations = || {
        COUNTS.with(|counts| {
            let counts = counts.borrow()[slot];
            counts.allocations + counts.reallocations
        })
    };
    let before = operations();
    ACTIVE.set(Some(slot));
    let index = prepare_index();
    ACTIVE.set(None);
    let allocated = operations() > before;
    COUNTS.with(|counts| {
        let counts = &mut counts.borrow_mut()[slot];
        counts.entries += 1;
        counts.allocating_entries += usize::from(allocated);
    });
    index
}

pub fn report() {
    COUNTS.with(|counts| {
        for (loop_name, counts) in ["cleanup", "pi1", "p_mat", "two_site"]
            .into_iter()
            .zip(counts.borrow().iter())
        {
            eprintln!("{loop_name}: {counts:?}");
        }
    });
}
