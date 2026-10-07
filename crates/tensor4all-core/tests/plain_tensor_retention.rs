//! Tag test-owned allocations so unrelated harness cleanup cannot affect the count.
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicIsize, Ordering};
use std::sync::Arc;

use num_complex::{Complex32, Complex64};
use tenferro_cpu::CpuBackend;
use tensor4all_core::{DynIndex, ExecutionContext, IdxTensor};
use tensor4all_tensorbackend::CpuExecutionContext;

struct CountingAllocator;
static LIVE_BYTES: AtomicIsize = AtomicIsize::new(0);

thread_local! {
    static MEASURING: Cell<bool> = const { Cell::new(false) };
}

fn measuring() -> bool {
    MEASURING.try_with(Cell::get).unwrap_or(false)
}

fn tagged_layout(layout: Layout) -> Option<(Layout, usize)> {
    layout.extend(Layout::new::<usize>()).ok()
}

// SAFETY: Payload alignment is preserved by Layout::extend. A trailing word
// records whether this allocation belongs to the measurement. System receives
// the same extended layout on allocation, reallocation, and deallocation.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let Some((allocation, tag_offset)) = tagged_layout(layout) else {
            return std::ptr::null_mut();
        };
        // SAFETY: The checked extended layout includes the payload and tag.
        let ptr = unsafe { System.alloc(allocation) };
        if !ptr.is_null() {
            let counted = measuring();
            // SAFETY: Layout::extend places an aligned usize inside the allocation.
            unsafe {
                ptr.add(tag_offset)
                    .cast::<usize>()
                    .write(usize::from(counted))
            };
            if counted {
                LIVE_BYTES.fetch_add(layout.size() as isize, Ordering::Relaxed);
            }
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let (allocation, tag_offset) =
            tagged_layout(layout).unwrap_or_else(|| std::process::abort());
        // SAFETY: alloc/realloc initialized this tag in the same allocation.
        if unsafe { ptr.add(tag_offset).cast::<usize>().read() } != 0 {
            LIVE_BYTES.fetch_sub(layout.size() as isize, Ordering::Relaxed);
        }
        // SAFETY: This is the exact allocation layout used for this payload.
        unsafe { System.dealloc(ptr, allocation) };
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let (old_allocation, old_tag_offset) =
            tagged_layout(layout).unwrap_or_else(|| std::process::abort());
        let Ok(new_payload) = Layout::from_size_align(new_size, layout.align()) else {
            return std::ptr::null_mut();
        };
        let Some((new_allocation, new_tag_offset)) = tagged_layout(new_payload) else {
            return std::ptr::null_mut();
        };
        // SAFETY: The old allocation has the initialized tag written by alloc/realloc.
        let was_counted = unsafe { ptr.add(old_tag_offset).cast::<usize>().read() } != 0;
        // SAFETY: Extended layouts have identical alignment; System preserves
        // the old payload prefix, and null leaves the original allocation intact.
        let result = unsafe { System.realloc(ptr, old_allocation, new_allocation.size()) };
        if !result.is_null() {
            let counted = was_counted || measuring();
            // SAFETY: The resized allocation contains this aligned trailing tag.
            unsafe {
                result
                    .add(new_tag_offset)
                    .cast::<usize>()
                    .write(usize::from(counted));
            }
            let old_bytes = if was_counted { layout.size() } else { 0 };
            let new_bytes = if counted { new_size } else { 0 };
            LIVE_BYTES.fetch_add(new_bytes as isize - old_bytes as isize, Ordering::Relaxed);
        }
        result
    }
}

struct Measurement;

impl Measurement {
    fn start() -> Self {
        MEASURING.with(|flag| flag.set(true));
        Self
    }
}

impl Drop for Measurement {
    fn drop(&mut self) {
        MEASURING.with(|flag| flag.set(false));
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn assert_no_retention(name: &str, mut construct_and_drop: impl FnMut()) {
    for _ in 0..16 {
        construct_and_drop();
    }
    let before = LIVE_BYTES.load(Ordering::Relaxed);
    for _ in 0..256 {
        construct_and_drop();
    }
    let retained = LIVE_BYTES.load(Ordering::Relaxed) - before;
    assert_eq!(
        retained, 0,
        "{name} retained {retained} bytes after dropping plain tensors"
    );
}

fn check_allocator_accounting() {
    // Harness allocations made before the loop may be released during it.
    let unrelated = vec![1_u8; 16_384];
    let mut resized = vec![9_u8; 15];
    let measurement = Measurement::start();
    drop(unrelated);
    assert_eq!(LIVE_BYTES.load(Ordering::Relaxed), 0);
    resized.resize(32, 9);
    assert_eq!(
        LIVE_BYTES.load(Ordering::Relaxed),
        resized.capacity() as isize
    );
    assert_eq!(resized, vec![9; 32]);
    drop(resized);
    assert_eq!(LIVE_BYTES.load(Ordering::Relaxed), 0);
    let mut values = vec![7_u8; 31];
    values.reserve_exact(33);
    assert_eq!(
        LIVE_BYTES.load(Ordering::Relaxed),
        values.capacity() as isize
    );
    values.shrink_to_fit();
    assert_eq!(values, vec![7; 31]);
    assert_eq!(LIVE_BYTES.load(Ordering::Relaxed), 31);
    drop(values);
    #[repr(align(64))]
    struct Aligned(u8);
    let aligned = Box::new(Aligned(7));
    assert_eq!(aligned.0, 7);
    assert_eq!((&*aligned as *const Aligned as usize) % 64, 0);
    assert_eq!(LIVE_BYTES.load(Ordering::Relaxed), 64);
    drop(aligned);
    let mut moved = vec![3_u8; 47];
    drop(measurement);
    moved.reserve_exact(32);
    assert_eq!(
        LIVE_BYTES.load(Ordering::Relaxed),
        moved.capacity() as isize
    );
    assert_eq!(moved, vec![3; 47]);
    // The tag follows ownership, so another thread can free measured data.
    std::thread::spawn(move || drop(moved)).join().unwrap();
    assert_eq!(LIVE_BYTES.load(Ordering::Relaxed), 0);
}

#[test]
fn plain_tensor_construction_does_not_accumulate_runtime_records() {
    check_allocator_accounting();
    let _measurement = Measurement::start();
    let index = DynIndex::new_dyn(2);
    let context = ExecutionContext::Cpu(Arc::new(CpuExecutionContext::from_backend(
        CpuBackend::new(),
    )));
    macro_rules! check_dtype {
        ($ty:ty, $data:expr) => {{
            let data: Vec<$ty> = $data;
            assert_no_retention(concat!("global ", stringify!($ty)), || {
                let tensor = IdxTensor::from_dense(vec![index.clone()], data.clone()).unwrap();
                assert!(!tensor.tracks_grad());
                assert_eq!(tensor.to_vec::<$ty>().unwrap(), data);
            });
            assert_no_retention(concat!("explicit ", stringify!($ty)), || {
                let tensor =
                    IdxTensor::from_dense_in(&context, vec![index.clone()], data.clone()).unwrap();
                tensor.validate_context(&context).unwrap();
                assert!(!tensor.tracks_grad());
                assert_eq!(tensor.to_vec::<$ty>().unwrap(), data);
            });
        }};
    }
    check_dtype!(f64, vec![1.0, -2.0]);
    check_dtype!(f32, vec![1.0, -2.0]);
    check_dtype!(
        Complex64,
        vec![Complex64::new(1.0, 2.0), Complex64::new(-2.0, 3.0)]
    );
    check_dtype!(
        Complex32,
        vec![Complex32::new(1.0, 2.0), Complex32::new(-2.0, 3.0)]
    );
    let second = DynIndex::new_dyn(2);
    assert_no_retention("diagonal", || {
        let tensor =
            IdxTensor::from_diag(vec![index.clone(), second.clone()], vec![1.0_f64, 2.0]).unwrap();
        assert_eq!(tensor.to_vec::<f64>().unwrap(), vec![1.0, 0.0, 0.0, 2.0]);
    });
}
