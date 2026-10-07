//! Keep the allocator measurement isolated in its own single-test process.
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicIsize, Ordering};
use std::sync::Arc;

use num_complex::{Complex32, Complex64};
use tenferro_cpu::CpuBackend;
use tensor4all_core::{DynIndex, ExecutionContext, IdxTensor};
use tensor4all_tensorbackend::CpuExecutionContext;

struct CountingAllocator;
static LIVE_BYTES: AtomicIsize = AtomicIsize::new(0);

// SAFETY: All allocation operations delegate unchanged to System; the counter
// only observes the size of successful allocations and their deallocations.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: The caller supplies the valid allocation layout required by System.
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            LIVE_BYTES.fetch_add(layout.size() as isize, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE_BYTES.fetch_sub(layout.size() as isize, Ordering::Relaxed);
        // SAFETY: The original System allocation and layout are forwarded unchanged.
        unsafe { System.dealloc(ptr, layout) };
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: The original allocation and caller-provided resize are forwarded.
        let result = unsafe { System.realloc(ptr, layout, new_size) };
        if !result.is_null() {
            LIVE_BYTES.fetch_add(
                new_size as isize - layout.size() as isize,
                Ordering::Relaxed,
            );
        }
        result
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

#[test]
fn plain_tensor_construction_does_not_accumulate_runtime_records() {
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
