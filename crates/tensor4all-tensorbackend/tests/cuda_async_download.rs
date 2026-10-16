#![cfg(feature = "tenferro-cuda")]

//! The pending device-to-host route (#553 / #859 F5).
//!
//! The point of this route is that submission and completion are separable, so wait and overlap can be measured
//! instead of assumed. These tests assert correctness and print the figures; the assertions are the correctness
//! ones, because a timing assertion on a shared machine would be a flake.

use std::time::{Duration, Instant};

use tenferro::Tensor;
use tensor4all_tensorbackend::{CudaExecutionContext, CudaExecutionContextError};

fn host_tensor(elements: usize) -> Tensor {
    Tensor::from_vec_col_major(
        vec![elements],
        (0..elements)
            .map(|index| (index % 97) as f64 * 0.25 - 1.0)
            .collect(),
    )
    .expect("host tensor")
}

fn wait_ready(pending: &mut tensor4all_tensorbackend::PendingCudaDownload) {
    for _ in 0..200_000 {
        if pending.is_ready().expect("readiness") {
            return;
        }
        std::thread::sleep(Duration::from_micros(50));
    }
    panic!("the copy did not complete within the bounded readiness loop");
}

fn decode_f64(buffer: &tensor4all_tensorbackend::CudaPinnedBuffer) -> Vec<f64> {
    buffer
        .as_slice()
        .as_chunks::<{ std::mem::size_of::<f64>() }>()
        .0
        .iter()
        .map(|bytes| f64::from_ne_bytes(*bytes))
        .collect()
}

#[test]
fn a_pending_download_returns_what_the_synchronous_route_returns() {
    let context = CudaExecutionContext::new().expect("CUDA ordinal 0");
    let elements = 4096;
    let host = host_tensor(elements);
    let device = context.upload_cuda(&host).expect("upload");

    let buffer = context
        .allocate_pinned(elements * std::mem::size_of::<f64>())
        .expect("pinned buffer");
    assert_eq!(buffer.byte_len(), elements * std::mem::size_of::<f64>());

    let mut pending = context
        .download_pending(device, buffer)
        .expect("submit pending download");
    wait_ready(&mut pending);
    let (_device, buffer) = pending.wait().expect("wait");

    let values = decode_f64(&buffer);
    let expected = host.as_slice::<f64>().expect("host slice");
    assert_eq!(values.len(), expected.len());
    assert_eq!(values, expected);
}

#[test]
fn two_pending_downloads_are_submitted_before_either_completes() {
    let context = CudaExecutionContext::new().expect("CUDA ordinal 0");
    let elements = 8 * 1024 * 1024 / std::mem::size_of::<f64>();
    let bytes = elements * std::mem::size_of::<f64>();
    let host = host_tensor(elements);

    // Five independent device sources, uploaded before anything is timed: one is the warm-up, because an
    // upload is not part of a download's
    // cost, and `download_pending` consumes its source, so the sources cannot be reused.
    let sources: Vec<Tensor> = (0..5)
        .map(|_| context.upload_cuda(&host).expect("upload"))
        .collect();
    let mut queue = sources.into_iter();

    // Warm-up: one throwaway download of the same shape, so neither timed region pays the first transfer's
    // initialization. What makes the comparison controlled is that both regions then do the same work: allocate
    // two pinned buffers, submit, poll, wait and decode two copies.
    {
        let mut pending = context
            .download_pending(
                queue.next().expect("warm-up source"),
                context.allocate_pinned(bytes).expect("buffer"),
            )
            .expect("warm-up submit");
        wait_ready(&mut pending);
        let (_device, buffer) = pending.wait().expect("warm-up wait");
        assert_eq!(decode_f64(&buffer).len(), elements);
    }

    // Serial: submit, wait, submit, wait. The sources are consumed, so the queue is walked by value; the results
    // are kept alive past the timer so neither region pays a destruction the other does not.
    let mut serial_results = Vec::new();
    let started = Instant::now();
    for _ in 0..2 {
        let source = queue.next().expect("source");
        let mut pending = context
            .download_pending(source, context.allocate_pinned(bytes).expect("buffer"))
            .expect("submit");
        wait_ready(&mut pending);
        let result = pending.wait().expect("wait");
        // Decoding is inside this timer because it is inside the batched one: both regions do the same work.
        assert_eq!(decode_f64(&result.1).len(), elements);
        serial_results.push(result);
    }
    let serial = started.elapsed();

    // Overlapped: submit both, then wait for both.
    let started = Instant::now();
    let mut first = context
        .download_pending(
            queue.next().expect("source"),
            context.allocate_pinned(bytes).expect("buffer"),
        )
        .expect("submit first");
    let mut second = context
        .download_pending(
            queue.next().expect("source"),
            context.allocate_pinned(bytes).expect("buffer"),
        )
        .expect("submit second");
    wait_ready(&mut first);
    wait_ready(&mut second);
    let (_first, first_buffer) = first.wait().expect("wait first");
    let (_second, second_buffer) = second.wait().expect("wait second");
    // Decoding is inside this timer because it is inside the serial one: both regions do the same work.
    assert_eq!(decode_f64(&first_buffer).len(), elements);
    assert_eq!(decode_f64(&second_buffer).len(), elements);
    let overlapped = started.elapsed();

    // The serial results are still alive here, so neither region paid a destruction the other did not.
    assert_eq!(serial_results.len(), 2);

    println!(
        "cuda_async_download: bytes={bytes} serial_two_downloads={:.3}ms overlapped_two_downloads={:.3}ms \
         ratio={:.2}x (both regions warmed and doing the same work: two pinned allocations, submission, readiness \
         polling, both waits and both decodes, with the sources and results kept past the timer; uploads are \
         excluded. Each figure is the end-to-end time of two copies, so the ratio is an end-to-end ratio, not a \
         single transfer's latency. Printed, not asserted)",
        serial.as_secs_f64() * 1e3,
        overlapped.as_secs_f64() * 1e3,
        serial.as_secs_f64() / overlapped.as_secs_f64()
    );
}

#[test]
fn a_short_pinned_buffer_is_rejected_with_a_typed_error() {
    let context = CudaExecutionContext::new().expect("CUDA ordinal 0");
    let host = host_tensor(64);
    let device = context.upload_cuda(&host).expect("upload");
    let too_small = context.allocate_pinned(8).expect("pinned buffer");

    let error = context
        .download_pending(device, too_small)
        .expect_err("a byte-count mismatch must be refused");
    match error {
        CudaExecutionContextError::Transfer { operation, source } => {
            assert_eq!(operation, "download submit");
            assert!(
                format!("{source}").contains("bytes"),
                "the diagnostic must name the byte mismatch: {source}"
            );
        }
        other => panic!("expected a transfer error, got {other:?}"),
    }
}
