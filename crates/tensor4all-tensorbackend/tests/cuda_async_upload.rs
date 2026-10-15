#![cfg(feature = "tenferro-cuda")]

//! The pending host-to-device route (#553 / #859 F5).
//!
//! The device-to-host half landed first, and until now an upload could not be told apart from its wait: the
//! synchronous `upload_cuda` submits and completes in one call. This route separates them, so an upload's
//! submission and its completion can be measured. These tests assert correctness and print the figures; the
//! assertions are the correctness ones, because a timing assertion on a shared machine would be a flake.
//!
//! The printed figures are **not** a controlled overlap comparison. The serial region runs first, so it also
//! carries first-use effects that the overlapped region does not, and both regions include placement validation,
//! the runtime's allocation and event work, readiness polling and the wait's cleanup. What the batched test
//! demonstrates is the *ordering*: two uploads are submitted and outstanding before either is waited on.
//!
//! The pinned revision consumes a pending transfer's destination, so a caller that wants to keep its tensor
//! should not use this route - the value comes back from `PendingCudaUpload::wait`.

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

fn wait_ready(pending: &mut tensor4all_tensorbackend::PendingCudaUpload) {
    for _ in 0..200_000 {
        if pending.is_ready().expect("readiness") {
            return;
        }
        std::thread::sleep(Duration::from_micros(50));
    }
    panic!("the copy did not complete within the bounded readiness loop");
}

/// Stage `values` into a pinned buffer's bytes, which is the caller's step on this route.
fn encode_f64(buffer: &mut tensor4all_tensorbackend::CudaPinnedBuffer, values: &[f64]) {
    let bytes = buffer.as_mut_slice();
    assert_eq!(bytes.len(), std::mem::size_of_val(values));
    for (chunk, value) in bytes
        .as_chunks_mut::<{ std::mem::size_of::<f64>() }>()
        .0
        .iter_mut()
        .zip(values)
    {
        *chunk = value.to_ne_bytes();
    }
}

fn staged_values(elements: usize) -> Vec<f64> {
    (0..elements)
        .map(|index| (index % 53) as f64 * 0.5 - 3.0)
        .collect()
}

#[test]
fn a_pending_upload_returns_what_the_synchronous_route_returns() {
    let context = CudaExecutionContext::new().expect("CUDA ordinal 0");
    let elements = 4096;
    let host = host_tensor(elements);
    let destination = context.upload_cuda(&host).expect("upload");

    let mut buffer = context
        .allocate_pinned(elements * std::mem::size_of::<f64>())
        .expect("pinned buffer");
    assert_eq!(buffer.byte_len(), elements * std::mem::size_of::<f64>());
    let values = staged_values(elements);
    encode_f64(&mut buffer, &values);

    let mut pending = context
        .upload_pending(buffer, destination)
        .expect("submit pending upload");
    wait_ready(&mut pending);
    let (destination, buffer) = pending.wait().expect("wait");

    // Read the written device value back through the synchronous route: the pending route hands back the device
    // tensor and the source buffer, not a host value.
    assert_eq!(buffer.byte_len(), elements * std::mem::size_of::<f64>());
    let downloaded = context.download(&destination).expect("download");
    assert_eq!(
        downloaded.as_slice::<f64>().expect("host slice"),
        values.as_slice()
    );
}

#[test]
fn two_pending_uploads_are_submitted_before_either_is_waited_on() {
    let context = CudaExecutionContext::new().expect("CUDA ordinal 0");
    let elements = 8 * 1024 * 1024 / std::mem::size_of::<f64>();
    let bytes = elements * std::mem::size_of::<f64>();
    let host = host_tensor(elements);
    let values = staged_values(elements);

    // Four independent destinations, created before anything is timed: `upload_pending` consumes its
    // destination and hands it back from `wait`, so a destination can only carry one submission at a time.
    let destinations: Vec<Tensor> = (0..4)
        .map(|_| context.upload_cuda(&host).expect("destination"))
        .collect();

    // Serial: stage, submit, wait, stage, submit, wait. The results are kept alive past the timer so that the
    // two regions have the same resource lifetime: destroying a returned buffer inside one region and not the
    // other would compare a transfer against a transfer plus a destruction.
    let mut queue = destinations.into_iter();
    let mut serial_results = Vec::new();
    let started = Instant::now();
    for _ in 0..2 {
        let mut buffer = context.allocate_pinned(bytes).expect("buffer");
        encode_f64(&mut buffer, &values);
        let mut pending = context
            .upload_pending(buffer, queue.next().expect("destination"))
            .expect("submit");
        wait_ready(&mut pending);
        serial_results.push(pending.wait().expect("wait"));
    }
    let serial = started.elapsed();

    // Batched: stage and submit both, then wait for both.
    let started = Instant::now();
    let mut first_buffer = context.allocate_pinned(bytes).expect("buffer");
    encode_f64(&mut first_buffer, &values);
    let mut first = context
        .upload_pending(first_buffer, queue.next().expect("destination"))
        .expect("submit first");
    let mut second_buffer = context.allocate_pinned(bytes).expect("buffer");
    encode_f64(&mut second_buffer, &values);
    let mut second = context
        .upload_pending(second_buffer, queue.next().expect("destination"))
        .expect("submit second");
    wait_ready(&mut first);
    wait_ready(&mut second);
    let (first_destination, _first_buffer) = first.wait().expect("wait first");
    let (second_destination, _second_buffer) = second.wait().expect("wait second");
    let overlapped = started.elapsed();

    for destination in [first_destination, second_destination] {
        let downloaded = context.download(&destination).expect("download");
        assert_eq!(downloaded.as_slice::<f64>().expect("host slice"), values);
    }

    println!(
        "cuda_async_upload: bytes={bytes} serial_two_uploads={:.3}ms batched_two_uploads={:.3}ms \
         (figures only; the serial region also carries first-use effects, so this is not a controlled overlap \
         comparison. Each figure includes two pinned allocations, the caller's staging, placement validation, \
         the runtime's allocation/event work, readiness polling and the wait's cleanup; destinations are \
         excluded, and the returned values are dropped after the timer in both regions)",
        serial.as_secs_f64() * 1e3,
        overlapped.as_secs_f64() * 1e3
    );
}

#[test]
fn a_short_pinned_buffer_is_rejected_with_a_typed_error() {
    let context = CudaExecutionContext::new().expect("CUDA ordinal 0");
    let host = host_tensor(64);
    let destination = context.upload_cuda(&host).expect("destination");
    let too_small = context.allocate_pinned(8).expect("pinned buffer");

    let error = context
        .upload_pending(too_small, destination)
        .expect_err("a byte-count mismatch must be refused");
    match error {
        CudaExecutionContextError::Transfer { operation, source } => {
            assert_eq!(operation, "upload submit");
            assert!(
                format!("{source}").contains("bytes"),
                "the diagnostic must name the byte mismatch: {source}"
            );
        }
        other => panic!("expected a transfer error, got {other:?}"),
    }
}

#[test]
fn a_host_destination_is_rejected_before_submission() {
    let context = CudaExecutionContext::new().expect("CUDA ordinal 0");
    let host = host_tensor(64);
    let buffer = context
        .allocate_pinned(64 * std::mem::size_of::<f64>())
        .expect("pinned buffer");

    let error = context
        .upload_pending(buffer, host)
        .expect_err("a host destination must be refused");
    match error {
        CudaExecutionContextError::PlacementMismatch { operation } => {
            assert_eq!(operation, "validate CUDA placement");
        }
        other => panic!("expected a placement mismatch, got {other:?}"),
    }
}
