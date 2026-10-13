use super::PrefixCache;
use crate::treetn::contraction::src_probe::ProbeBank;
use tensor4all_core::{DynIndex, IdxTensor, IndexLike};

fn two_site_local(dim: usize) -> (IdxTensor, IdxTensor, IdxTensor, IdxTensor) {
    let s0_out = DynIndex::new_dyn(dim);
    let s0_in = DynIndex::new_dyn(dim);
    let s1_out = DynIndex::new_dyn(dim);
    let elements = dim * dim;
    let a0 = IdxTensor::from_dense(vec![s0_out, s0_in.clone()], vec![0.1; elements]).unwrap();
    let b0 = IdxTensor::from_dense(vec![s0_in], vec![0.2; dim]).unwrap();
    let a1 = IdxTensor::from_dense(vec![s1_out.clone()], vec![0.3; dim]).unwrap();
    let b1 = IdxTensor::from_dense(vec![s1_out], vec![0.4; dim]).unwrap();
    (a0, b0, a1, b1)
}

#[test]
fn request_grows_a_fresh_segment_and_reuses_a_previously_cached_one() {
    let context = tensor4all_tensorbackend::ExecutionContext::Cpu(
        tensor4all_tensorbackend::default_cpu_execution_context(),
    );
    let (a0, b0, a1, b1) = two_site_local(3);
    let local = vec![(&a0, &b0), (&a1, &b1)];
    let outputs = vec![vec![a0.indices()[0].clone()], vec![a1.indices()[0].clone()]];
    let mut probes = ProbeBank::from_seed(
        outputs.iter().flat_map(|o| o.iter().cloned()).collect(),
        1,
        42,
    )
    .unwrap();
    let mut cache = PrefixCache::new(&local, &outputs, &mut probes, 3, &context);

    let (first, first_batch) = cache.request(0, 0, 3).unwrap();
    assert_eq!(first_batch.dim(), 3);
    // Site 0's own output index (`s0_out`) is one of `outputs[0]`, so it
    // is contracted away against the probe columns rather than kept --
    // like `s0_in` (the a0/b0 shared bond), it does not survive into the
    // prefix. Only the batch axis remains external.
    assert_eq!(first.dims().len(), 1); // [batch]

    // Requesting the same already-cached range again must not recompute
    // it -- assert the returned tensor is bit-identical (a fresh
    // recomputation of the same probes would also be numerically
    // identical here since ProbeBank is deterministic, so this alone
    // isn't proof of reuse; the ragged-boundary test below adds a
    // segment-count assertion that actually proves it).
    let (again, again_batch) = cache.request(0, 0, 3).unwrap();
    assert_eq!(again_batch.dim(), first_batch.dim());
    assert_eq!(
        again.to_vec::<f64>().unwrap(),
        first.to_vec::<f64>().unwrap()
    );
}

#[test]
fn request_overgenerates_a_full_post_initial_segment_for_reuse() {
    let context = tensor4all_tensorbackend::ExecutionContext::Cpu(
        tensor4all_tensorbackend::default_cpu_execution_context(),
    );
    let (a0, b0, a1, b1) = two_site_local(3);
    let local = vec![(&a0, &b0), (&a1, &b1)];
    let outputs = vec![vec![a0.indices()[0].clone()], vec![a1.indices()[0].clone()]];
    let mut probes = ProbeBank::from_seed(
        outputs.iter().flat_map(|o| o.iter().cloned()).collect(),
        1,
        42,
    )
    .unwrap();
    // batch_size 3, first caller only needs width 4. The cache computes
    // a full second increment so a later width-3 request can reuse it.
    let mut cache = PrefixCache::new(&local, &outputs, &mut probes, 3, &context);
    let (_first, _) = cache.request(0, 0, 4).unwrap();
    assert_eq!(cache.segments[0].len(), 2, "expected two width-3 segments");
    assert_eq!(
        cache.segments[0][1].2, 3,
        "post-initial growth should use the full rank increment"
    );

    // A later caller can reuse that full segment directly.
    let (_second, second_batch) = cache.request(0, 3, 3).unwrap();
    assert_eq!(second_batch.dim(), 3);
    assert_eq!(
        cache.segments[0].len(),
        2,
        "re-reading an existing aligned segment must not grow a new one"
    );
}

#[test]
fn request_extends_only_needed_segments_to_requested_site() {
    let context = tensor4all_tensorbackend::ExecutionContext::Cpu(
        tensor4all_tensorbackend::default_cpu_execution_context(),
    );
    let outputs = (0..3).map(|_| DynIndex::new_dyn(3)).collect::<Vec<_>>();
    let inner = (0..3).map(|_| DynIndex::new_dyn(3)).collect::<Vec<_>>();
    let tensors = (0..3)
        .map(|site| {
            (
                IdxTensor::from_dense(
                    vec![outputs[site].clone(), inner[site].clone()],
                    vec![0.1; 9],
                )
                .unwrap(),
                IdxTensor::from_dense(vec![inner[site].clone()], vec![0.2; 3]).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    let local = tensors.iter().map(|(a, b)| (a, b)).collect::<Vec<_>>();
    let site_outputs = outputs
        .iter()
        .cloned()
        .map(|output| vec![output])
        .collect::<Vec<_>>();
    let mut probes = ProbeBank::from_seed(outputs.clone(), 1, 42).unwrap();
    let mut cache = PrefixCache::new(&local, &site_outputs, &mut probes, 2, &context);

    cache.request(0, 0, 2).unwrap();
    cache.request(0, 2, 2).unwrap();
    assert_eq!(cache.segments[0].len(), 2);
    assert!(
        cache.segments[1].is_empty(),
        "site 1 must stay lazy while only site 0 is requested"
    );

    let (lazy, _) = cache.request(1, 0, 2).unwrap();
    assert_eq!(
        cache.segments[1].len(),
        1,
        "requesting the first range must not extend a later segment"
    );

    let mut direct_probes = ProbeBank::from_seed(outputs, 1, 42).unwrap();
    let mut direct = PrefixCache::new(&local, &site_outputs, &mut direct_probes, 2, &context);
    let (expected, _) = direct.request(1, 0, 2).unwrap();
    assert_eq!(
        lazy.to_vec::<f64>().unwrap(),
        expected.to_vec::<f64>().unwrap()
    );

    let (later, _) = cache.request(1, 4, 2).unwrap();
    assert_eq!(
        cache.segments[1].len(),
        3,
        "growing while ragged must fill the missing segment before appending"
    );
    let (middle, _) = cache.request(1, 2, 2).unwrap();

    let (expected_middle, _) = direct.request(1, 2, 2).unwrap();
    let (expected_later, _) = direct.request(1, 4, 2).unwrap();
    assert_eq!(
        middle.to_vec::<f64>().unwrap(),
        expected_middle.to_vec::<f64>().unwrap()
    );
    assert_eq!(
        later.to_vec::<f64>().unwrap(),
        expected_later.to_vec::<f64>().unwrap()
    );
}

#[test]
fn request_misaligned_range_spanning_a_segment_boundary_matches_a_direct_reference() {
    let context = tensor4all_tensorbackend::ExecutionContext::Cpu(
        tensor4all_tensorbackend::default_cpu_execution_context(),
    );
    let (a0, b0, a1, b1) = two_site_local(3);
    let local = vec![(&a0, &b0), (&a1, &b1)];
    let outputs = vec![vec![a0.indices()[0].clone()], vec![a1.indices()[0].clone()]];
    let index_list = outputs
        .iter()
        .flat_map(|o| o.iter().cloned())
        .collect::<Vec<_>>();
    let mut probes = ProbeBank::from_seed(index_list.clone(), 1, 42).unwrap();
    // batch_size 2: the first request exactly fills one aligned segment
    // [0,2); the second request [1,3) straddles that segment's right
    // boundary and the freshly grown [2,3) segment, so neither stored
    // segment alone covers [1,3) -- this must hit the misaligned
    // fallback (unlike the ragged-reuse test above, whose second
    // request re-reads an existing segment exactly and so never
    // exercises this branch).
    let mut cache = PrefixCache::new(&local, &outputs, &mut probes, 2, &context);
    let (_first, _) = cache.request(0, 0, 2).unwrap();
    assert_eq!(
        cache.segments[0].len(),
        1,
        "first request should be a single aligned segment"
    );

    let (misaligned, misaligned_batch) = cache.request(0, 1, 2).unwrap();
    assert_eq!(misaligned_batch.dim(), 2);
    assert_eq!(
        cache.segments[0].len(),
        2,
        "the misaligned request should grow a second segment covering [2,3) \
         but still needs to fall back to splitting/restacking, since [1,3) \
         does not align with either stored segment's boundaries"
    );

    // Ground truth: an independent probe bank/prefix computation for the
    // same [1, 3) range, built without going through `PrefixCache` at
    // all.
    let mut reference_probes = ProbeBank::from_seed(index_list, 1, 42).unwrap();
    reference_probes.extend_to(3).unwrap();
    let reference_batch = DynIndex::new_link(2).unwrap();
    let context = tensor4all_tensorbackend::ExecutionContext::Cpu(
        tensor4all_tensorbackend::default_cpu_execution_context(),
    );
    let reference = crate::treetn::contraction::src_probe::probed_site_pair_batch_range(
        &a0,
        &b0,
        &outputs[0],
        &reference_probes,
        1,
        2,
        &reference_batch,
        &context,
    )
    .unwrap();

    assert_eq!(
        misaligned.to_vec::<f64>().unwrap(),
        reference.to_vec::<f64>().unwrap(),
        "misaligned fallback result must match a directly computed reference"
    );
}
