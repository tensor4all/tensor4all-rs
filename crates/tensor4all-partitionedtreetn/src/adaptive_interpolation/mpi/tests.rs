use super::super::cache::PatchCache;
use super::super::layout::SiteLayout;
use super::*;
use std::collections::BTreeMap;
use tensor4all_treetn::NodeNameNetwork;

/// A layout of a two-node chain, each node with one binary site.
fn chain_layout() -> SiteLayout<String> {
    let mut topology = NodeNameNetwork::new();
    topology.add_node("a".to_string()).unwrap();
    topology.add_node("b".to_string()).unwrap();
    topology
        .add_edge(&"a".to_string(), &"b".to_string())
        .unwrap();
    let node_sites = BTreeMap::from([
        ("a".to_string(), vec![DynIndex::new_dyn(2)]),
        ("b".to_string(), vec![DynIndex::new_dyn(2)]),
    ]);
    let pivots = ColMajorArray::new(vec![0, 0], vec![2, 1]).unwrap();
    SiteLayout::validated(
        topology,
        node_sites,
        &pivots,
        &PatchedInterpolationOptions::new(2)
            .with_error_norm(super::super::ErrorNorm::sampled_max()),
    )
    .unwrap()
}

/// The site layout of a one-node problem with three binary sites.
fn chain_cache_layout() -> SiteLayout<String> {
    let mut topology = NodeNameNetwork::new();
    topology.add_node("a".to_string()).unwrap();
    let node_sites = BTreeMap::from([(
        "a".to_string(),
        vec![
            DynIndex::new_dyn(2),
            DynIndex::new_dyn(2),
            DynIndex::new_dyn(2),
        ],
    )]);
    let pivots = ColMajorArray::new(vec![0, 0, 0], vec![3, 1]).unwrap();
    SiteLayout::validated(
        topology,
        node_sites,
        &pivots,
        &PatchedInterpolationOptions::new(2)
            .with_error_norm(super::super::ErrorNorm::sampled_max()),
    )
    .unwrap()
}

/// A cache split at its first active slot keeps the parent's word offsets,
/// so the child packs a coordinate at a different shift than a compact
/// layout of its own dimensions would; the wire must carry the layout.
#[test]
fn a_split_cache_crosses_the_wire_with_its_holes() {
    let big = 1usize << 40;
    let mut cache = PatchCache::new(vec![2, big, 2]);
    let key = cache.layout().encode([1usize, 5, 1].into_iter());
    cache.insert(key, 1.5_f64);
    let children = cache.split(0).unwrap();
    let child = children.into_iter().nth(1).unwrap();
    let layout = child.layout_description();
    // The first slot of the parent occupied bit 0; the split clears it and
    // keeps the remaining slots at their shifts.
    assert_eq!(layout.dims, vec![big, 2]);
    assert_eq!(layout.slots, vec![(0, 1, 40), (0, 41, 1)]);
    let entries = child.into_entries();
    // The point (5, 1) keeps the parent's shifts: `5` still sits at bit 1
    // and the child's second slot at bit 41, with the split slot's bit 0
    // cleared.
    assert_eq!(entries, vec![(vec![(5u64 << 1) | (1u64 << 41)], 1.5_f64)]);

    let decoded = PatchCache::from_entries(layout.clone(), entries.clone()).unwrap();
    assert_eq!(decoded.layout_description(), layout);
    assert_eq!(decoded.get(&entries[0].0), Some(1.5));
    // A compact packing of the child's own dimensions places the same
    // coordinates at other bits, so a receiver that inferred the layout
    // from the dimensions would look up the wrong key.
    let wrong = PatchCache::<f64>::new(vec![big, 2])
        .layout()
        .encode([5usize, 1].into_iter());
    assert_ne!(wrong.as_ref(), entries[0].0.as_slice());
    assert_ne!(wrong.as_ref(), entries[0].0.as_slice());
    assert_eq!(decoded.get(&wrong), None);

    // A malformed description is rejected instead of misread.
    assert!(PatchCache::from_entries(
        CacheLayout {
            dims: vec![big, 2],
            slots: vec![(1, 1, 40), (0, 41, 1)],
            n_words: 1,
        },
        entries,
    )
    .is_err());
}

#[test]
fn references_cross_the_wire_with_their_provenance() {
    let references = References {
        l2: Some(L2Pin {
            reference_rms: Some(0.25),
            source: L2ReferenceSource::MonteCarlo {
                samples: 17,
                mean_square_rel_std_error: 0.125,
            },
            tau: 0.5,
        }),
        max_reference: Some((3.5, MaxReferenceSource::SearchedRootMaximum)),
        max_search: Some(MaxReferenceSearch {
            starts: vec![vec![0, 1], vec![2, 0]],
            maximizer: vec![2, 1],
            sweeps: 2,
            evaluations: 7,
            cache_hits: 3,
            seconds: 0.25,
        }),
    };
    let wire = encode_references(references.clone());
    let decoded = decode_references(&wire);
    assert_eq!(encode_references(decoded.clone()), wire);
    assert_eq!(decoded.l2.map(|pin| pin.tau), Some(0.5));
    assert_eq!(decoded.l2.map(|pin| pin.reference_rms), Some(Some(0.25)));
    assert_eq!(decoded.max_search, references.max_search);

    let empty = encode_references(References::default());
    assert!(decode_references(&empty).l2.is_none());
    assert!(decode_references(&empty).max_reference.is_none());
}

#[test]
fn a_patch_crosses_the_wire_with_its_inherited_state() {
    let big = 1usize << 40;
    let mut cache = PatchCache::new(vec![2, big, 2]);
    cache.insert(cache.layout().encode([1usize, 7, 0].into_iter()), 2.5_f64);
    let children = cache.split(0).unwrap();
    let cache = children.into_iter().nth(1).unwrap();
    let patch = Patch {
        path: vec![(0, 1)],
        fixed: vec![Some(1), None, None],
        cache,
        recycled: vec![vec![1, 7 % 2, 0]],
        worst: vec![vec![1, 3 % 2, 1]],
    };
    let wire = encode_patch(patch);
    let layout = chain_cache_layout();
    let decoded = decode_patch(&layout, wire.clone()).unwrap();
    assert_eq!(decoded.path, vec![(0, 1)]);
    assert_eq!(decoded.fixed, vec![Some(1), None, None]);
    assert_eq!(decoded.recycled, vec![vec![1, 1, 0]]);
    assert_eq!(decoded.worst, vec![vec![1, 1, 1]]);
    // A point outside the receiver's site dimensions is rejected.
    let mut bad = wire.clone();
    bad.recycled = vec![vec![1, 7, 0]];
    assert!(decode_patch(&layout, bad).is_err());
    let expected = CacheLayout {
        dims: wire.cache.layout.dims.clone(),
        slots: wire.cache.layout.slots.clone(),
        n_words: wire.cache.layout.n_words,
    };
    assert_eq!(decoded.cache.layout_description(), expected);

    // The record survives a serde round trip, key packing included.
    let json = serde_json::to_string(&wire).unwrap();
    let back: WirePatch<f64> = serde_json::from_str(&json).unwrap();
    assert_eq!(format!("{back:?}"), format!("{wire:?}"));
}

#[test]
fn a_patch_network_crosses_the_wire_with_its_axes() {
    let layout = chain_layout();
    let wire = WireSubdomain {
        nodes: vec![
            WireNode {
                node: 0,
                axes: vec![WireAxis::Site { position: 0 }, WireAxis::Bond { id: 0 }],
                shape: vec![2, 3],
                values: WireArray::F64(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            },
            WireNode {
                node: 1,
                axes: vec![WireAxis::Bond { id: 0 }, WireAxis::Site { position: 1 }],
                shape: vec![3, 2],
                values: WireArray::F64(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
            },
        ],
    };
    let projector = layout.projector(&[]).unwrap();
    let subdomain = decode_subdomain(&layout, projector, wire.clone()).unwrap();
    assert_eq!(subdomain.node_count(), 2);
    assert_eq!(subdomain.max_bond_dim(), 3);
    let back = encode_subdomain(&layout, &subdomain).unwrap();
    assert_eq!(format!("{back:?}"), format!("{wire:?}"));
}

#[test]
fn an_outcome_crosses_the_wire_as_a_whole() {
    let subdomain = WireSubdomain {
        nodes: vec![
            WireNode {
                node: 0,
                axes: vec![WireAxis::Site { position: 0 }],
                shape: vec![2],
                values: WireArray::F32(vec![1.0, 2.0]),
            },
            WireNode {
                node: 1,
                axes: vec![WireAxis::Site { position: 1 }],
                shape: vec![2],
                values: WireArray::C64(vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)]),
            },
        ],
    };
    let outcome = WireOutcome {
        path: Vec::new(),
        verdict: WireVerdict::<f64>::Accepted(Box::new(WireAccepted {
            record: WirePatchRecord {
                termination: WireTermination::Converged,
                engine_error_estimate: 1e-9,
                max_sample_magnitude: 2.0,
                max_bond_dim: 1,
                retries_used: 1,
                acceptance: Some(WireMeasurement {
                    method: WireMethod::Exhaustive,
                    points: 4,
                    patch_points: 4.0,
                    rms: 0.0,
                    mean_square_rel_std_error: 0.0,
                    max_residual: 0.0,
                    residual_rms: None,
                    residual_saturated: false,
                    sample_rms: None,
                }),
                audit: None,
                status: WireStatus::WithinTolerance,
            },
            subdomain,
            patch_points: 4.0,
            log_norm: Some(1.5),
        })),
        totals: WireTotals {
            evaluations: 9,
            cache_hits: 2,
            measurement_evaluations: 7,
            audit_evaluations: 0,
            residual_evaluations: 3,
            residual_saturations: 1,
            verification_failures: 0,
            engine_retries: 1,
        },
        references: encode_references(References::default()),
    };
    let json = serde_json::to_string(&outcome).unwrap();
    let back: WireOutcome<f64> = serde_json::from_str(&json).unwrap();
    assert_eq!(format!("{back:?}"), format!("{outcome:?}"));
    // The verdict keeps the pieces a distributed report is assembled from.
    match &back.verdict {
        WireVerdict::Accepted(accepted) => {
            assert_eq!(
                accepted.record.acceptance.as_ref().map(|m| m.points),
                Some(4)
            );
            assert_eq!(accepted.record.retries_used, 1);
            assert_eq!(accepted.patch_points, 4.0);
            assert_eq!(accepted.log_norm, Some(1.5));
            assert_eq!(accepted.subdomain.nodes.len(), 2);
        }
        other => panic!("expected an accepted outcome, got {other:?}"),
    }
    assert_eq!(back.totals.evaluations, 9);
}

#[test]
fn a_residual_tt_measurement_and_its_counters_cross_the_wire() {
    let measurement = WireMeasurement {
        method: WireMethod::ResidualTt,
        points: 8,
        patch_points: 64.0,
        rms: 0.25,
        mean_square_rel_std_error: 0.5,
        max_residual: 0.75,
        residual_rms: Some(0.25),
        residual_saturated: true,
        sample_rms: Some(0.125),
    };
    let json = serde_json::to_string(&measurement).unwrap();
    let back: WireMeasurement = serde_json::from_str(&json).unwrap();
    assert_eq!(format!("{back:?}"), format!("{measurement:?}"));
    assert!(matches!(
        decode_method(back.method),
        MeasurementMethod::ResidualTt
    ));

    let totals = WireTotals {
        evaluations: 21,
        cache_hits: 3,
        measurement_evaluations: 11,
        audit_evaluations: 2,
        residual_evaluations: 9,
        residual_saturations: 2,
        verification_failures: 1,
        engine_retries: 1,
    };
    let json = serde_json::to_string(&totals).unwrap();
    let back: WireTotals = serde_json::from_str(&json).unwrap();
    assert_eq!(format!("{back:?}"), format!("{totals:?}"));
    let decoded = decode_totals(back);
    assert_eq!(decoded.residual_evaluations, 9);
    assert_eq!(decoded.residual_saturations, 2);
    assert_eq!(decoded.measurement_evaluations, 11);
}

/// A root-only result converges through the agreed control: a failure the root alone observed fails
/// every rank - including peers whose own placeholder result is `Ok` - and a success keeps each rank's
/// own result.
#[test]
fn a_root_only_result_converges_through_the_agreed_control() {
    // The root succeeded: the control keeps its decoded outcomes.
    assert_eq!(
        converge_root_outcome(Ok::<u8, _>(7), WireControl::Continue).ok(),
        Some(7)
    );
    // A peer keeps its placeholder when the root succeeded.
    assert_eq!(
        converge_root_outcome(
            Ok::<Option<u8>, PatchedInterpolationError>(None),
            WireControl::Continue,
        )
        .ok(),
        Some(None)
    );
    // A root-only decode or assembly failure reaches every rank as the converged message, whatever
    // their local placeholder says.
    for local in [
        Ok::<u8, PatchedInterpolationError>(0),
        Err(PatchedInterpolationError::FanOut {
            message: "local placeholder".to_string(),
        }),
    ] {
        match converge_root_outcome(local, WireControl::Fail("root decode failed".to_string())) {
            Err(PatchedInterpolationError::FanOut { message }) => {
                assert_eq!(message, "root decode failed");
            }
            other => panic!("expected the converged failure, got {other:?}"),
        }
    }
    // A local failure is never masked by an agreeing control.
    assert!(matches!(
        converge_root_outcome(
            Err::<u8, _>(PatchedInterpolationError::FanOut {
                message: "local".to_string(),
            }),
            WireControl::Continue,
        ),
        Err(PatchedInterpolationError::FanOut { .. })
    ));
    // `Stop` is not a result control.
    assert!(matches!(
        converge_root_outcome(Ok(0_u8), WireControl::Stop),
        Err(PatchedInterpolationError::FanOut { .. })
    ));
}

#[test]
fn every_known_termination_crosses_the_wire() {
    for termination in [
        InterpolationTermination::Converged,
        InterpolationTermination::BondCapReached,
        InterpolationTermination::IterationLimit,
        InterpolationTermination::ErrorNotMeasured,
    ] {
        let wire = encode_termination(termination).unwrap();
        assert_eq!(decode_termination(wire), termination);
    }
}
