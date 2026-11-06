#![cfg(feature = "tenferro-cuda")]

//! The TreeTN-level pending download (#553 / #859 F5).
//!
//! F5 asks for wait and overlap to be measurable rather than assumed. The tensor level has had a pending
//! device-to-host route since #932; this slice exposes it for a whole network, so a caller can submit every
//! node's transfer and only then decide when to wait. These tests pin the parity with the synchronous route, the
//! ordering the route exists for, and the preflight that keeps a refusal from transferring anything.

use std::time::{Duration, Instant};

use tensor4all_core::{CudaExecutionContext, DynIndex, IdxTensor, IndexLike};
use tensor4all_treetn::{CudaTreeTNError, TreeTN};

fn dense(indices: Vec<DynIndex>, seed: usize) -> IdxTensor {
    let elements = indices
        .iter()
        .map(IndexLike::dim)
        .try_fold(1usize, |size, dim| size.checked_mul(dim))
        .expect("element count");
    IdxTensor::from_dense(
        indices,
        (0..elements)
            .map(|offset| ((seed + offset) % 19) as f64 / 19.0 + 0.25)
            .collect(),
    )
    .expect("dense tensor")
}

/// A four-node chain whose second bond is a primed copy of another bond, so the route is exercised on bonds
/// whose identity differs from a site index's even where a dimension coincides.
fn chain() -> TreeTN<IdxTensor, usize> {
    let sites: Vec<_> = (0..4).map(|_| DynIndex::new_dyn(2)).collect();
    let mut bonds: Vec<_> = (0..3).map(|_| DynIndex::new_dyn(3)).collect();
    bonds[1] = bonds[1].prime();
    let mut tensors = Vec::new();
    for (node, site) in sites.iter().enumerate() {
        let mut indices = Vec::new();
        if let Some(left) = node.checked_sub(1).and_then(|index| bonds.get(index)) {
            indices.push(left.clone());
        }
        indices.push(site.clone());
        if let Some(right) = bonds.get(node) {
            indices.push(right.clone());
        }
        tensors.push(dense(indices, node * 31 + 1));
    }
    TreeTN::from_tensors(tensors, (0..4).collect()).expect("chain")
}

/// Per node, in name order: the node's graph index and the ordered identity of every index on its tensor.
///
/// The indices are compared as `DynIndex` values, so a relabelling between two indices that share a dimension and
/// a prime level is visible, and the graph index pins the node itself.
fn structure(
    tn: &TreeTN<IdxTensor, usize>,
) -> Vec<(usize, petgraph::stable_graph::NodeIndex, Vec<DynIndex>)> {
    let mut names = tn.node_names();
    names.sort();
    names
        .into_iter()
        .map(|name| {
            let node = tn.node_index(&name).expect("node");
            let tensor = tn.tensor(node).expect("tensor");
            (name, node, tensor.indices().to_vec())
        })
        .collect()
}

fn values_at(tn: &TreeTN<IdxTensor, usize>, name: usize) -> Vec<f64> {
    let node = tn.node_index(&name).expect("node");
    tn.tensor(node)
        .expect("tensor")
        .to_vec::<f64>()
        .expect("host values")
}

fn context() -> CudaExecutionContext {
    CudaExecutionContext::new().expect("CUDA ordinal 0")
}

fn wait_ready(pending: &mut tensor4all_treetn::CudaTreeTNPendingDownload<usize>) {
    for _ in 0..200_000 {
        if pending.is_ready().expect("readiness") {
            return;
        }
        std::thread::sleep(Duration::from_micros(50));
    }
    panic!("the transfers did not complete within the bounded readiness loop");
}

#[test]
fn a_pending_download_returns_what_the_synchronous_route_returns() {
    let context = context();
    let host = chain();
    let device = host.upload_cuda(&context).expect("upload");

    let mut pending = device.download_pending(&context).expect("submit");
    wait_ready(&mut pending);
    let pending_host = pending.wait().expect("wait");
    let synchronous_host = device.download(&context).expect("synchronous download");

    assert_eq!(structure(&pending_host), structure(&synchronous_host));
    for name in 0..4 {
        assert_eq!(
            values_at(&pending_host, name),
            values_at(&synchronous_host, name)
        );
        assert_eq!(values_at(&pending_host, name), values_at(&host, name));
    }
}

#[test]
fn two_networks_are_submitted_before_either_is_waited_on() {
    let context = context();
    let host = chain();
    let first_device = host.upload_cuda(&context).expect("upload first");
    let second_device = host.upload_cuda(&context).expect("upload second");

    // Warm-up: one throwaway download of the same shape, so neither timed region pays the first transfer's
    // initialization. The warm-up is not what makes the comparison controlled - what does is that both regions
    // below do the same work (two submissions, readiness polling, both waits, both host reconstructions), so
    // neither pays a first-use cost and neither pays a destruction the other does not.
    {
        let mut pending = first_device
            .download_pending(&context)
            .expect("warm-up submit");
        wait_ready(&mut pending);
        let _ = pending.wait().expect("warm-up wait");
    }

    // Serial: submit, wait, submit, wait. Both results are kept alive past the timer, so neither region pays a
    // destruction the other does not.
    let mut serial_hosts = Vec::new();
    let started = Instant::now();
    for device in [&first_device, &second_device] {
        let mut pending = device.download_pending(&context).expect("submit");
        wait_ready(&mut pending);
        serial_hosts.push(pending.wait().expect("wait"));
    }
    let serial = started.elapsed();

    // Batched: submit both, then wait for both.
    let started = Instant::now();
    let mut first = first_device
        .download_pending(&context)
        .expect("submit first");
    let mut second = second_device
        .download_pending(&context)
        .expect("submit second");
    wait_ready(&mut first);
    wait_ready(&mut second);
    let first_host = first.wait().expect("wait first");
    let second_host = second.wait().expect("wait second");
    let batched = started.elapsed();

    // Correctness outside both timers: the two routes must produce the same host networks.
    for network in serial_hosts.iter().chain([&first_host, &second_host]) {
        assert_eq!(structure(network), structure(&host));
    }

    // Each figure is the end-to-end time of *two* network reconstructions, so the ratio is an end-to-end ratio:
    // warming both regions removes the first-use asymmetry, and both do the same work, so the difference is where
    // the waits sit relative to the submissions. Not a statement about a single transfer's latency.
    println!(
        "cuda_pending_download: nodes={} serial_two_networks={:.3}ms batched_two_networks={:.3}ms ratio={:.2}x \
         (both regions warmed and doing the same work: two submissions, readiness polling, both waits and both \
         host reconstructions; printed, not asserted)",
        host.node_names().len(),
        serial.as_secs_f64() * 1e3,
        batched.as_secs_f64() * 1e3,
        serial.as_secs_f64() / batched.as_secs_f64()
    );
}

#[test]
fn a_host_network_is_refused_before_anything_is_submitted() {
    let context = context();
    let host = chain();

    let error = host
        .download_pending(&context)
        .expect_err("a host network has no device value to download");
    match error {
        CudaTreeTNError::Download { .. } => {}
        other => panic!("expected a node-local download error, got {other:?}"),
    }
}

/// A two-node chain in `f32`, with the site index written first on the first node and second on the last, so the
/// reconstruction is exercised on an index order that is not the one a fresh fixture would produce.
fn f32_pair() -> TreeTN<IdxTensor, usize> {
    let site = DynIndex::new_dyn(2);
    let bond = DynIndex::new_dyn(3);
    let first = IdxTensor::from_dense(
        vec![site.clone(), bond.clone()],
        vec![0.5_f32, 1.5, 2.5, 3.5, 4.5, 5.5],
    )
    .expect("first");
    let second = IdxTensor::from_dense(vec![bond, site], vec![6.5_f32, 7.5, 8.5, 9.5, 10.5, 11.5])
        .expect("second");
    TreeTN::from_tensors(vec![first, second], vec![0, 1]).expect("chain")
}

/// A two-node chain in `Complex64` whose real and imaginary parts are unequal and nonzero, so a decoder that
/// swapped them or read one for the other would fail.
fn c64_pair() -> TreeTN<IdxTensor, usize> {
    let site = DynIndex::new_dyn(2);
    let bond = DynIndex::new_dyn(3);
    let values = |offset: f64| {
        (0..6)
            .map(|step| {
                num_complex::Complex64::new(
                    offset + step as f64 + 0.25,
                    offset + step as f64 + 0.75,
                )
            })
            .collect::<Vec<_>>()
    };
    let first =
        IdxTensor::from_dense(vec![bond.clone(), site.clone()], values(0.0)).expect("first");
    let second = IdxTensor::from_dense(vec![site, bond], values(10.0)).expect("second");
    TreeTN::from_tensors(vec![first, second], vec![0, 1]).expect("chain")
}

#[test]
fn the_float_and_complex_dtypes_round_trip_through_the_pending_route() {
    let context = context();

    // f32, with an index order that differs between the two nodes.
    let f32_host = f32_pair();
    let f32_device = f32_host.upload_cuda(&context).expect("upload f32");
    let mut pending = f32_device.download_pending(&context).expect("submit f32");
    wait_ready(&mut pending);
    let f32_back = pending.wait().expect("wait f32");
    assert_eq!(structure(&f32_back), structure(&f32_host));
    for name in 0..2 {
        let node = f32_back.node_index(&name).expect("node");
        assert_eq!(
            f32_back
                .tensor(node)
                .expect("tensor")
                .to_vec::<f32>()
                .expect("values"),
            f32_host
                .tensor(node)
                .expect("tensor")
                .to_vec::<f32>()
                .expect("values"),
        );
    }

    // Complex64, with unequal real and imaginary parts.
    let c64_host = c64_pair();
    let c64_device = c64_host.upload_cuda(&context).expect("upload c64");
    let mut pending = c64_device.download_pending(&context).expect("submit c64");
    wait_ready(&mut pending);
    let c64_back = pending.wait().expect("wait c64");
    assert_eq!(structure(&c64_back), structure(&c64_host));
    for name in 0..2 {
        let node = c64_back.node_index(&name).expect("node");
        assert_eq!(
            c64_back
                .tensor(node)
                .expect("tensor")
                .to_vec::<num_complex::Complex64>()
                .expect("values"),
            c64_host
                .tensor(node)
                .expect("tensor")
                .to_vec::<num_complex::Complex64>()
                .expect("values"),
        );
    }
}
